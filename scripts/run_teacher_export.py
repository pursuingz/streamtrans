"""教师(9B, 4-bit)离线导出 target 段 top-k logits → shard。

对齐策略：教师在 old-id(248320) 空间 forward；导出时把学生 new-id 的 input_ids/labels
与教师 top-k(已 old→new、丢集外、重归一化、补齐到 K) **一起**存入 shard，run_distill 直接读、
不再分词，杜绝位置错位（Phase 2 头号风险）。

用法:
  python scripts/run_teacher_export.py --config configs/distill.yaml [--smoke]
"""
import argparse
import json
from pathlib import Path

import torch

from streamtrans.config import DistillConfig, load_config
from streamtrans.data.vocab_remap import VocabRemapper
from streamtrans.distill.prompt import assemble_labels, render_prompt
from streamtrans.distill.teacher_logits_io import ShardWriter, map_and_renormalize_topk


def pad_topk(new_ids, logp, k: int):
    """把单位置变长 top-k 补齐到固定 K：id 补 0、logp 补 -inf(exp=0，不影响归一与 KL)。"""
    cur = new_ids.shape[0]
    if cur >= k:
        return new_ids[:k], logp[:k]
    pad_n = k - cur
    ids = torch.cat([new_ids, torch.zeros(pad_n, dtype=torch.long)])
    # 用有限大负数而非 -inf：exp(-30)≈0(不影响归一与 KL)，且避免 0*(-inf)=nan
    lp = torch.cat([logp, torch.full((pad_n,), -30.0, dtype=torch.float16)])
    return ids, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, DistillConfig)

    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(cfg.teacher_model, trust_remote_code=True)
    remapper = VocabRemapper.from_file(cfg.vocab_map)
    old_eos = tok.eos_token_id
    eos_new = remapper.old2new.get(int(old_eos))
    if eos_new is None:
        raise SystemExit("eos 不在 keep 集；检查 vocab_map 是否含特殊 token")
    remapper.unk_new_id = eos_new  # 输入序列 OOV 兜底(中英下几乎不触发)，保证长度对齐

    print(f"[teacher] 4-bit 加载 {cfg.teacher_model}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.teacher_4bit, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    teacher = AutoModelForImageTextToText.from_pretrained(
        cfg.teacher_model, quantization_config=bnb, trust_remote_code=True, device_map="auto"
    )
    teacher.eval()
    dev = next(teacher.parameters()).device

    rows = [json.loads(l) for l in Path(cfg.corpus_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.smoke:
        rows = rows[:32]
    out_dir = Path(cfg.teacher_logits_dir + ("_smoke" if args.smoke else ""))
    writer = ShardWriter(out_dir, shard_size=cfg.shard_size)

    n = 0
    from tqdm import tqdm

    for r in tqdm(rows, desc="[teacher] export", unit="ex"):
        # old-id 序列（教师空间）
        p_old = tok.encode(render_prompt(r["src"], r["direction"]), add_special_tokens=False)
        t_old = tok.encode(r["tgt"], add_special_tokens=False) + [old_eos]
        inp_old, lab_old = assemble_labels(p_old, t_old)
        inp_old, lab_old = inp_old[: cfg.max_len], lab_old[: cfg.max_len]

        ids = torch.tensor([inp_old], device=dev)
        with torch.no_grad():
            logits = teacher(ids).logits[0]  # [L, 248320]

        # target 位置：labels != -100；预测该位置 token 的是上一位 logits[p-1]
        t_ids_rows, t_logp_rows = [], []
        for p in range(len(lab_old)):
            if lab_old[p] == -100 or p == 0:
                continue
            topv, topi = logits[p - 1].topk(cfg.topk)
            nids, nlogp = map_and_renormalize_topk(topi.tolist(), topv, remapper.old2new)
            nids, nlogp = pad_topk(nids, nlogp, cfg.topk)
            t_ids_rows.append(nids)
            t_logp_rows.append(nlogp)

        if not t_ids_rows:
            continue
        # 学生 new-id 序列（位置对齐，unk 兜底不丢 token）
        inp_new = remapper.map_ids(inp_old)
        lab_new = [-100 if x == -100 else remapper.old2new.get(int(x), eos_new) for x in lab_old]
        writer.add(
            {
                "input_ids": torch.tensor(inp_new, dtype=torch.long),
                "labels": torch.tensor(lab_new, dtype=torch.long),
                "t_ids": torch.stack(t_ids_rows),     # [L_t, K]
                "t_logp": torch.stack(t_logp_rows),   # [L_t, K]
            }
        )
        n += 1
    writer.close()
    print(f"[teacher] 导出 {n} 条 -> {out_dir}")


if __name__ == "__main__":
    main()
