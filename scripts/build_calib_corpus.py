"""拉多语言单语文本，作词表裁剪的校准语料（只用于统计 token 频次，无需平行）。

数据源：wikimedia/wikipedia（干净、按语言分 config、可流式，不必全量下载）。
用法：
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/build_calib_corpus.py --out data/calib_multiling.txt --per-lang 20000

产物喂给 run_prune：
  python scripts/run_prune.py --config configs/prune.yaml --calib-file data/calib_multiling.txt --out checkpoints/pruned_<date>

注意：覆盖的语言应与 configs/prune.yaml 的 keep_languages + reserve_languages 一致，
保证裁出的 110k 词表覆盖目标语言全集（扩语言时只补数据不动结构）。
"""
import argparse
from pathlib import Path

# 与 prune.yaml keep_languages + reserve_languages 对齐：中英 + 日韩 + 主要欧洲语言
DEFAULT_LANGS = ["zh", "en", "ja", "ko", "fr", "de", "es", "ru"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/calib_multiling.txt")
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--snapshot", default="20231101", help="wikipedia 快照日期(config 前缀)")
    ap.add_argument("--per-lang", type=int, default=20000, help="每语言保留的文本段数")
    ap.add_argument("--min-chars", type=int, default=16, help="过滤过短段")
    args = ap.parse_args()

    from datasets import load_dataset

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    summary: list[str] = []
    with out.open("w", encoding="utf-8") as f:
        for lang in args.langs:
            cfg = f"{args.snapshot}.{lang}"
            try:
                ds = load_dataset("wikimedia/wikipedia", cfg, split="train", streaming=True)
            except Exception as e:  # noqa: BLE001
                msg = f"[skip] {lang} ({cfg}): {type(e).__name__}: {e}"
                print(msg)
                summary.append(msg)
                continue
            n = 0
            for ex in ds:
                for seg in ex.get("text", "").split("\n"):
                    seg = " ".join(seg.split())
                    if len(seg) >= args.min_chars:
                        f.write(seg + "\n")
                        n += 1
                        total += 1
                        if n >= args.per_lang:
                            break
                if n >= args.per_lang:
                    break
            line = f"[ok] {lang}: {n} segments"
            print(line)
            summary.append(line)

    print("---")
    for s in summary:
        print(s)
    print(f"total {total} segments -> {out}")
    if total == 0:
        raise SystemExit("没拉到任何文本：检查网络 / HF_ENDPOINT / snapshot 日期是否有效")


if __name__ == "__main__":
    main()
