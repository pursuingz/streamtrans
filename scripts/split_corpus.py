"""把双向平行语料切成 train/test，按 (zh,en) 唯一句对去重防泄漏。

build_parallel_corpus 每对写两行(zh2en + en2zh)。本脚本先归并回唯一句对，
再整对划入 train 或 test —— 同一句对的两个方向**绝不**跨集，杜绝评测泄漏。
test 仍写双向(评 zh->en 与 en->zh 两个方向)。

用法:
  python scripts/split_corpus.py --in data/parallel_zh_en.jsonl \
      --train data/train_zh_en.jsonl --test data/test_zh_en.jsonl --test-size 2000
"""
import argparse
import json
import random
from pathlib import Path


def pair_key(rec):
    """统一成 (zh, en)，无论该行是哪个方向。"""
    if rec["direction"] == "zh2en":
        return rec["src"], rec["tgt"]
    return rec["tgt"], rec["src"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/parallel_zh_en.jsonl")
    ap.add_argument("--train", default="data/train_zh_en.jsonl")
    ap.add_argument("--test", default="data/test_zh_en.jsonl")
    ap.add_argument("--test-size", type=int, default=2000, help="held-out 句对数(双向各写一条)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    # 归并回唯一句对 (zh,en)
    pairs = {}
    for rec in lines:
        pairs.setdefault(pair_key(rec), True)
    keys = list(pairs.keys())
    random.Random(args.seed).shuffle(keys)

    n_test = min(args.test_size, len(keys) // 5)  # test 不超过 1/5，避免小语料把 train 切空
    test_keys = set(keys[:n_test])

    def write(path, key_filter):
        cnt = 0
        with Path(path).open("w", encoding="utf-8") as f:
            for zh, en in keys:
                if not key_filter(zh, en):
                    continue
                f.write(json.dumps({"src": zh, "tgt": en, "direction": "zh2en"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"src": en, "tgt": zh, "direction": "en2zh"}, ensure_ascii=False) + "\n")
                cnt += 1
        return cnt

    n_tr = write(args.train, lambda zh, en: (zh, en) not in test_keys)
    n_te = write(args.test, lambda zh, en: (zh, en) in test_keys)
    print(f"unique pairs={len(keys)}  train={n_tr}  test={n_te}  (各双向2行)")
    print(f"-> {args.train} / {args.test}")
    # 自检：train/test 句对零交集
    tr_keys = set(keys) - test_keys
    assert tr_keys.isdisjoint(test_keys), "train/test 句对泄漏！"


if __name__ == "__main__":
    main()
