"""测目标侧 token 的 OOV 率（不在 vocab_map keep 集里的比例），分方向。

诊断过早 eos 根因：teacher export 把 OOV token 映射成 eos(run_teacher_export.py 第97行),
于是 OOV 目标 token 在训练标签里变成假 eos,教模型早停。本脚本量化:
  - 目标 token 总 OOV 率(每方向)
  - 句首 target token 为 OOV 的样本比例 → 直接训出 step-0 eos 的样本占比
若 en2zh(中文目标) OOV 率显著高于 zh2en,即坐实方向不对称的根因。

用法:
  python scripts/check_oov.py --corpus data/train_zh_en.jsonl \
         --base Qwen/Qwen3.5-2B --vocab-map checkpoints/pruned_20260530/vocab_map.json --limit 20000
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--vocab-map", required=True)
    ap.add_argument("--limit", type=int, default=20000, help="抽样条数(全量慢)")
    ap.add_argument("--dump", type=int, default=10, help="打印前 N 条句首 OOV 样例")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    vm = json.loads(Path(args.vocab_map).read_text(encoding="utf-8"))
    # vocab_map.json: old->new（键可能是 str）。keep 集 = 其键集合
    keep = set(int(k) for k in vm.keys())
    print(f"keep 集大小: {len(keep)}")

    rows = [json.loads(l) for l in Path(args.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    stat = defaultdict(lambda: {"n": 0, "tok": 0, "oov": 0, "first_oov": 0})
    dumped = []
    for r in rows:
        d = r.get("direction", "?")
        ids = tok.encode(r["tgt"], add_special_tokens=False)
        if not ids:
            continue
        S = stat[d]
        S["n"] += 1
        S["tok"] += len(ids)
        S["oov"] += sum(1 for x in ids if x not in keep)
        if ids[0] not in keep:
            S["first_oov"] += 1
            if len(dumped) < args.dump:
                dumped.append((d, r["tgt"][:50], tok.convert_ids_to_tokens([ids[0]])[0]))

    for d in sorted(stat):
        S = stat[d]
        n, t = S["n"], max(1, S["tok"])
        print(f"=== {d} (n={S['n']}) ===")
        print(f"  目标 token OOV 率   : {S['oov']}/{t} = {100*S['oov']/t:.2f}%")
        print(f"  句首 token 为 OOV   : {S['first_oov']}/{n} = {100*S['first_oov']/max(1,n):.2f}%  <- 直接训出 step-0 eos 的样本占比")
        print()

    if dumped:
        print("--- 句首 OOV 样例 (方向 / 目标 / 句首token) ---")
        for d, tgt, t0 in dumped:
            print(f"  [{d}] {tgt!r}  首token={t0!r}")


if __name__ == "__main__":
    main()
