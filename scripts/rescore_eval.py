"""从 run_eval --out 的 JSONL 重新算分,不重跑生成。

贪心解码确定性,改了 metric(如中文 BLEU 分词)只需重算、无需再 generate(省 ~20min)。

用法:
  python scripts/rescore_eval.py --in data/eval_5000step.jsonl
"""
import argparse
import json
from pathlib import Path

from streamtrans.eval.quality import corpus_bleu, corpus_chrf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="run_eval --out 产出的 JSONL")
    args = ap.parse_args()

    recs = []
    for l in Path(args.inp).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        o = json.loads(l)
        if "summary" in o:          # 跳过末行旧 summary
            continue
        recs.append(o)

    def score(rows, tokenize):
        h = [r["hyp"] for r in rows]
        rf = [r["ref"] for r in rows]
        al = sum(len(x) for x in h) / max(1, len(h))
        return {"samples": len(h), "BLEU": round(corpus_bleu(h, rf, tokenize=tokenize), 2),
                "chrF": round(corpus_chrf(h, rf), 2), "avg_hyp_chars": round(al, 1)}

    for d in sorted({r["direction"] for r in recs}):
        tok_d = "zh" if d.endswith("2zh") else None
        v = score([r for r in recs if r["direction"] == d], tok_d)
        print(f"[{d:8}] " + "  ".join(f"{kk}={vv}" for kk, vv in v.items()))
    print(f"[chrF_all] {round(corpus_chrf([r['hyp'] for r in recs], [r['ref'] for r in recs]), 2)}")


if __name__ == "__main__":
    main()
