"""在中英测试集上评学生翻译质量(BLEU/chrF)。

学生在 new-id 空间：prompt 经 VocabRemapper 编码 → generate → 解码回文本。
用法:
  python scripts/run_eval.py --ckpt checkpoints/distilled_20260530 \
         --test data/test_zh_en.jsonl --base Qwen/Qwen3.5-2B \
         --vocab-map checkpoints/pruned_20260530/vocab_map.json
"""
import argparse
import json
from pathlib import Path

import torch

from streamtrans.data.vocab_remap import VocabRemapper
from streamtrans.distill.prompt import render_prompt
from streamtrans.eval.quality import corpus_bleu, corpus_chrf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="学生 checkpoint")
    ap.add_argument("--test", required=True, help="测试 jsonl: {src,tgt,direction}")
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B", help="取 tokenizer")
    ap.add_argument("--vocab-map", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--min-new", type=int, default=0, help="强制最少生成 token(诊断过早 eos)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条 src/hyp/ref 供诊断")
    ap.add_argument("--out", default=None, help="把每条 src/hyp/ref 写 JSONL,末行写 summary")
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    remapper = VocabRemapper.from_file(args.vocab_map)
    eos_new = remapper.old2new.get(int(tok.eos_token_id))
    model = Qwen3_5ForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True).to(dev).eval()

    rows = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    from tqdm import tqdm

    hyps, refs, recs = [], [], []
    for i, r in enumerate(tqdm(rows, desc="[eval] generate", unit="ex")):
        p_new = remapper.encode(tok, render_prompt(r["src"], r["direction"]))
        ids = torch.tensor([p_new], device=dev)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new, min_new_tokens=args.min_new,
                                 do_sample=False, eos_token_id=eos_new, pad_token_id=eos_new)
        out_new = gen[0, ids.shape[1]:].tolist()
        hyp = remapper.decode(tok, out_new, skip_special_tokens=True)
        hyps.append(hyp)
        refs.append(r["tgt"])
        recs.append({"direction": r["direction"], "src": r["src"], "hyp": hyp, "ref": r["tgt"],
                     "gen_tok": len(out_new)})
        if i < args.show:
            print(f"--- [{i}] {r['direction']}  (prompt {len(p_new)} tok, gen {len(out_new)} tok)")
            print(f"  SRC : {r['src']}")
            print(f"  HYP : {hyp!r}")
            print(f"  REF : {r['tgt']}")

    def score(idxs, tokenize):
        # 目标中文用 sacrebleu 'zh' 字符级分词,否则默认空格切词会把整句中文当一个词、BLEU≈0
        h = [hyps[i] for i in idxs]
        rf = [refs[i] for i in idxs]
        al = sum(len(x) for x in h) / max(1, len(h))
        return {"samples": len(h), "BLEU": round(corpus_bleu(h, rf, tokenize=tokenize), 2),
                "chrF": round(corpus_chrf(h, rf), 2), "avg_hyp_chars": round(al, 1)}

    # 按方向分别评(混合方向的 BLEU 因分词器不同无意义,故只分方向报)
    summary = {}
    for d in sorted({r["direction"] for r in recs}):
        tok_d = "zh" if d.endswith("2zh") else None   # 目标语言决定分词器
        summary[d] = score([i for i, r in enumerate(recs) if r["direction"] == d], tok_d)
    summary["chrF_all"] = round(corpus_chrf(hyps, refs), 2)   # chrF 字符级,跨语言可比,给个总览

    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"[{k:8}] " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
        else:
            print(f"[{k:8}] {v}")

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
        print(f"-> {args.out} ({len(recs)} 条译文 + summary)")


if __name__ == "__main__":
    main()
