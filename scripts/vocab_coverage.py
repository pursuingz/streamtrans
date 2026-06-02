"""统计真实训练语料的 token 使用情况，为重建 vocab_map 定型。

回答两件事：
  1. 语料(src+tgt 双侧)用了多少 unique token → 新 keep 集需多大才能 OOV≈0；
  2. 当前 keep 集(旧 vocab_map)有多少 token 其实没被语料用到 → 暴露预算浪费。
纯 stdlib + tokenizer，不需权重。

用法:
  python scripts/vocab_coverage.py --corpus data/train_zh_en.jsonl \
         --base Qwen/Qwen3.5-2B --vocab-map checkpoints/pruned_20260530/vocab_map.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--vocab-map", default=None, help="旧 vocab_map.json(可选,算预算浪费)")
    ap.add_argument("--budget", type=int, default=110000, help="目标词表大小,看 unique 是否塞得下")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    rows = [json.loads(l) for l in Path(args.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]

    freq: Counter[int] = Counter()
    for r in rows:
        # 双侧都统计：src 与 tgt 都会作为某方向的输入/输出出现
        freq.update(tok.encode(r["src"], add_special_tokens=False))
        freq.update(tok.encode(r["tgt"], add_special_tokens=False))

    uniq = len(freq)
    total = sum(freq.values())
    print(f"语料句对: {len(rows)}   总 token: {total}   unique token: {uniq}")
    print(f"目标预算: {args.budget}  -> {'塞得下,中英可 OOV≈0,余量留预留语言' if uniq <= args.budget else '塞不下,需取舍'}")

    # 覆盖率：top-N 频次 token 覆盖语料的百分比(看长尾)
    ranked = [c for _, c in freq.most_common()]
    for N in [50000, 80000, 100000, 110000]:
        cov = sum(ranked[:N]) / max(1, total)
        print(f"  top-{N:6d} token 覆盖语料 {100*cov:.3f}%   (尾部 {max(0, uniq-N)} 个 token 占 {100*(1-cov):.3f}%)")

    if args.vocab_map:
        vm = json.loads(Path(args.vocab_map).read_text(encoding="utf-8"))
        keep = set(int(k) for k in vm.keys())
        used = set(freq.keys())
        wasted = len(keep - used)
        missing = len(used - keep)
        print(f"\n旧 keep 集: {len(keep)}")
        print(f"  其中语料没用到(浪费): {wasted}  ({100*wasted/len(keep):.1f}%)")
        print(f"  语料用到但被裁掉(OOV源头): {missing}  token")
        print(f"  -> 把这 {wasted} 个浪费名额让给那 {missing} 个被裁 token,即可基本消除 OOV")


if __name__ == "__main__":
    main()
