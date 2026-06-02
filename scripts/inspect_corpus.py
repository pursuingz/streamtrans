"""勘查平行语料质量：空/超短/同语言污染/源==目标 的占比，分方向统计。

诊断 en2zh 过早 eos 的数据根因——若 en2zh 目标侧大量为空/纯标点/实为英文，
模型就会学到"见 en2zh prompt → 高概率 eos"。纯 stdlib，服务器直接跑。

用法:
  python scripts/inspect_corpus.py data/train_zh_en.jsonl
  python scripts/inspect_corpus.py data/train_zh_en.jsonl --dump-bad 20
"""
import argparse
import json
import unicodedata
from collections import defaultdict


def has_cjk(s: str) -> bool:
    for ch in s:
        if "一" <= ch <= "鿿":
            return True
    return False


def is_punct_only(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    for ch in t:
        if ch.isalnum():
            return False
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--dump-bad", type=int, default=0, help="每类问题打印前 N 条样例")
    args = ap.parse_args()

    stat = defaultdict(lambda: defaultdict(int))   # direction -> metric -> count
    bad = defaultdict(list)                          # tag -> samples
    total = 0
    with open(args.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src, tgt, d = r.get("src", ""), r.get("tgt", ""), r.get("direction", "?")
            total += 1
            S = stat[d]
            S["n"] += 1
            S["tgt_chars"] += len(tgt.strip())

            if not tgt.strip():
                S["empty_tgt"] += 1
                if len(bad["empty_tgt"]) < args.dump_bad:
                    bad["empty_tgt"].append((d, src, tgt))
            if is_punct_only(tgt):
                S["punct_only_tgt"] += 1
                if len(bad["punct_only_tgt"]) < args.dump_bad:
                    bad["punct_only_tgt"].append((d, src, tgt))
            # 目标侧语言污染：*2zh 目标无 CJK；*2en 目标含 CJK
            if d.endswith("2zh") and tgt.strip() and not has_cjk(tgt):
                S["tgt_not_zh"] += 1
                if len(bad["tgt_not_zh"]) < args.dump_bad:
                    bad["tgt_not_zh"].append((d, src, tgt))
            if d.endswith("2en") and has_cjk(tgt):
                S["tgt_has_cjk"] += 1
                if len(bad["tgt_has_cjk"]) < args.dump_bad:
                    bad["tgt_has_cjk"].append((d, src, tgt))
            if src.strip() == tgt.strip() and src.strip():
                S["src_eq_tgt"] += 1
                if len(bad["src_eq_tgt"]) < args.dump_bad:
                    bad["src_eq_tgt"].append((d, src, tgt))
            # 超短目标（中文<2字 / 英文<2词）
            if d.endswith("2zh"):
                if len(tgt.strip()) <= 1:
                    S["ultrashort_tgt"] += 1
            else:
                if len(tgt.split()) <= 1:
                    S["ultrashort_tgt"] += 1

    print(f"总句对: {total}\n")
    metrics = ["empty_tgt", "punct_only_tgt", "ultrashort_tgt", "tgt_not_zh",
               "tgt_has_cjk", "src_eq_tgt"]
    for d in sorted(stat):
        S = stat[d]
        n = S["n"]
        print(f"=== {d}  (n={n}, 目标均长 {S['tgt_chars']/max(1,n):.1f} 字符) ===")
        for m in metrics:
            c = S.get(m, 0)
            print(f"  {m:16s} {c:7d}  {100*c/max(1,n):5.1f}%")
        print()

    if args.dump_bad:
        for tag, items in bad.items():
            print(f"--- 样例 [{tag}] ---")
            for d, src, tgt in items:
                print(f"  [{d}] SRC={src[:60]!r}")
                print(f"        TGT={tgt[:60]!r}")
            print()


if __name__ == "__main__":
    main()
