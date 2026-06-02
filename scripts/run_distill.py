"""离线 KD 训练：读教师 shard(含学生 new-id input/labels + 教师 top-k) 训学生。

shard 自带对齐好的一切，本脚本不分词。批处理前向(右 padding + attention_mask)，
但只把 target 位置的 hidden 过 lm_head(避开整 [B,L,V] logits 的显存大头)，
再把全 batch 的 target 摊平成一个大 N 算 KD+CE。shard 内打乱、按 shard 迭代避免 I/O 抖动。

用法:
  python scripts/run_distill.py --config configs/distill.yaml [--smoke]
"""
import argparse
import random
from pathlib import Path

import torch

from streamtrans.config import DistillConfig, load_config
from streamtrans.distill.kd_losses import combined_loss
from streamtrans.distill.teacher_logits_io import ShardReader


def build_optimizer(params, lr: float):
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 8bit 优化器不可用({e})，回退 torch AdamW")
        return torch.optim.AdamW(params, lr=lr)


def batch_loss(student, batch, dev, cfg):
    """一个 batch 的 KD+CE。右 padding 批前向取 hidden，只对 target 位置过 lm_head。"""
    B = len(batch)
    lens = [r["input_ids"].shape[0] for r in batch]
    Lmax = max(lens)
    input_ids = torch.zeros((B, Lmax), dtype=torch.long)   # pad=0；mask 掉故值无关紧要
    attn = torch.zeros((B, Lmax), dtype=torch.long)
    for b, r in enumerate(batch):
        L = lens[b]
        input_ids[b, :L] = r["input_ids"]
        attn[b, :L] = 1

    hidden = student.model(
        input_ids=input_ids.to(dev), attention_mask=attn.to(dev), use_cache=False
    ).last_hidden_state                                    # [B, Lmax, H]

    sel_b, sel_p, golds, tids, tlps = [], [], [], [], []
    for b, r in enumerate(batch):
        labels = r["labels"]
        tp = [p for p in range(1, labels.shape[0]) if int(labels[p]) != -100]
        if not tp:
            continue
        if r["t_ids"].shape[0] != len(tp):
            raise RuntimeError(f"对齐错位: target {len(tp)} vs teacher {r['t_ids'].shape[0]}")
        sel_b.extend([b] * len(tp))
        sel_p.extend([p - 1 for p in tp])                  # 预测 p 位 token 的是 p-1 位 hidden
        golds.append(labels[tp].to(dev))
        tids.append(r["t_ids"].to(dev))
        tlps.append(r["t_logp"].float().to(dev))
    if not golds:
        return None

    sel = hidden[sel_b, sel_p]                             # [N, H]
    pred = student.lm_head(sel)                            # [N, V] —— 只对 target 位置算 logits
    gold = torch.cat(golds)
    t_ids = torch.cat(tids)
    t_logp = torch.cat(tlps)
    return combined_loss(pred, gold, t_ids, t_logp, cfg.alpha_ce, cfg.beta_kd, cfg.temperature)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 config 的 steps")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 config 的 batch_size")
    args = ap.parse_args()
    cfg = load_config(args.config, DistillConfig)

    from transformers import Qwen3_5ForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[distill] 加载学生 {cfg.student_ckpt}")
    student = Qwen3_5ForCausalLM.from_pretrained(cfg.student_ckpt, trust_remote_code=True).to(dev)
    if cfg.grad_checkpointing:
        student.config.use_cache = False
        student.gradient_checkpointing_enable()
    student.train()

    logits_dir = cfg.teacher_logits_dir + ("_smoke" if args.smoke else "")
    reader = ShardReader(logits_dir)
    n_ex = len(reader)
    steps = 30 if args.smoke else (args.steps if args.steps is not None else cfg.steps)
    B = args.batch_size if args.batch_size is not None else cfg.batch_size
    print(f"[distill] {n_ex} 条教师样本, batch={B} grad_accum={cfg.grad_accum} "
          f"(有效 batch {B * cfg.grad_accum}), 目标 {steps} step")

    opt = build_optimizer(student.parameters(), cfg.lr)
    opt.zero_grad()

    from transformers import get_cosine_schedule_with_warmup

    warmup = int(steps * cfg.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps)
    print(f"[distill] lr={cfg.lr} warmup={warmup} step, cosine 衰减")

    from collections import deque

    from tqdm import tqdm

    step, micro = 0, 0
    losses: list[float] = []
    recent = deque(maxlen=50)   # 近 50 step 滑动均值，压噪声看真实趋势
    win: list[float] = []       # 当前 grad_accum 窗口内各 batch 的 loss
    shard_ids = list(range(reader.num_shards))
    pbar = tqdm(total=steps, desc="[distill]", unit="step")
    done = False
    while not done:
        random.shuffle(shard_ids)                          # shard 间打乱
        for si in shard_ids:
            recs = reader.load_shard(si)
            idx = list(range(len(recs)))
            random.shuffle(idx)                            # shard 内打乱
            for bs in range(0, len(idx), B):
                batch = [recs[i] for i in idx[bs: bs + B]]
                res = batch_loss(student, batch, dev, cfg)
                if res is None:
                    continue
                total, ce, kd = res
                (total / cfg.grad_accum).backward()
                win.append(total.detach().item())
                micro += 1
                if micro % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                    step += 1
                    step_loss = sum(win) / len(win)
                    win.clear()
                    losses.append(step_loss)
                    recent.append(step_loss)
                    pbar.update(1)
                    pbar.set_postfix(lr=f"{sched.get_last_lr()[0]:.1e}",
                                     avg50=f"{sum(recent) / len(recent):.3f}", loss=f"{step_loss:.3f}")
                    if step >= steps:
                        done = True
                        break
            if done:
                break
    pbar.close()

    out_dir = Path("checkpoints") / ("distilled_smoke" if args.smoke else "distilled_20260530")
    student.config.use_cache = True
    student.save_pretrained(out_dir)
    print(f"[distill] 保存 -> {out_dir}")
    if len(losses) >= 4:
        head = sum(losses[:3]) / 3
        tail = sum(losses[-3:]) / 3
        print(f"[distill] loss 首段均值 {head:.4f} -> 末段均值 {tail:.4f}")
        if args.smoke and not all(map(torch.isfinite, map(torch.tensor, losses))):
            raise RuntimeError("出现非有限 loss")


if __name__ == "__main__":
    main()
