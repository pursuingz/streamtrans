"""离线 KD 训练：读教师 shard(含学生 new-id input/labels + 教师 top-k) 训学生。

shard 已自带对齐好的一切，本脚本不分词。逐样本累积梯度(grad_accum)再 step，
对齐简单可靠(target 位置内部摊平，无需跨样本 padding)；首版重正确性，提速后置。

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill.yaml")
    ap.add_argument("--smoke", action="store_true")
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
    steps = 30 if args.smoke else cfg.steps
    print(f"[distill] {n_ex} 条教师样本, 目标 {steps} step")

    opt = build_optimizer(student.parameters(), cfg.lr)
    opt.zero_grad()

    step, micro = 0, 0
    order = list(range(n_ex))
    losses: list[float] = []
    while step < steps:
        random.shuffle(order)
        for gi in order:
            rec = reader.get(gi)
            input_ids = rec["input_ids"].unsqueeze(0).to(dev)
            labels = rec["labels"]
            out = student(input_ids).logits[0]  # [L, V]

            tp = [p for p in range(1, len(labels)) if int(labels[p]) != -100]
            if not tp:
                continue
            pred = out[[p - 1 for p in tp]]                       # [N, V]
            gold = labels[tp].to(dev)                             # [N]
            t_ids = rec["t_ids"].to(dev)                          # [N, K]
            t_logp = rec["t_logp"].float().to(dev)               # [N, K]
            if t_ids.shape[0] != pred.shape[0]:
                raise RuntimeError(f"对齐错位: pred {pred.shape[0]} vs teacher {t_ids.shape[0]} (gi={gi})")

            total, ce, kd = combined_loss(
                pred, gold, t_ids, t_logp, cfg.alpha_ce, cfg.beta_kd, cfg.temperature
            )
            (total / cfg.grad_accum).backward()
            micro += 1
            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                losses.append(float(total))
                if step % 10 == 0 or args.smoke:
                    print(f"  step {step}/{steps}  loss={float(total):.4f} ce={float(ce):.4f} kd={float(kd):.4f}")
                if step >= steps:
                    break

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
