"""拉中英平行语料，口语/字幕为主 + 通用混合(约 7:3)，写 jsonl(双向)。

源用 HF datasets 的 OPUS 系（schema 多为 ex["translation"]={"en":..,"zh":..}）。
确切 config 名/可用性以服务器为准——每源 try/except 并汇报，缺的换源即可。
清洗复用 data/parallel_corpus.clean_pair。

用法:
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/build_parallel_corpus.py --out data/parallel_zh_en.jsonl --total 200000
"""
import argparse
import json
import os
import sys
from pathlib import Path

from streamtrans.data.parallel_corpus import clean_pair

# (id, load_args, load_kwargs, kind)
SOURCES = [
    ("open_subtitles", ("open_subtitles",), {"lang1": "en", "lang2": "zh"}, "spoken"),
    ("opus-100", ("Helsinki-NLP/opus-100", "en-zh"), {}, "general"),
]


def keep(src: str, tgt: str, min_len=1, max_len=200, max_ratio=4.0) -> bool:
    if not src or not tgt:
        return False
    ls, lt = len(src), len(tgt)
    if ls < min_len or lt < min_len or ls > max_len or lt > max_len:
        return False
    return max(ls, lt) / max(1, min(ls, lt)) <= max_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/parallel_zh_en.jsonl")
    ap.add_argument("--total", type=int, default=200000, help="目标 zh-en 句对数(双向各写一条)")
    ap.add_argument("--spoken-ratio", type=float, default=0.7)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.total = 200

    from datasets import load_dataset

    targets = {
        "spoken": int(args.total * args.spoken_ratio),
        "general": args.total - int(args.total * args.spoken_ratio),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary, written = [], 0
    with out.open("w", encoding="utf-8") as f:
        for sid, load_args, load_kwargs, kind in SOURCES:
            want = targets.get(kind, 0)
            if want <= 0:
                continue
            try:
                ds = load_dataset(*load_args, **load_kwargs, split="train",
                                  streaming=True, trust_remote_code=True)
            except Exception as e:  # noqa: BLE001
                msg = f"[skip] {sid} ({kind}): {type(e).__name__}: {e}"
                print(msg); summary.append(msg); continue
            got = 0
            for ex in ds:
                tr = ex.get("translation") or {}
                en, zh = clean_pair(tr.get("en", ""), tr.get("zh", ""))
                if not keep(zh, en):
                    continue
                f.write(json.dumps({"src": zh, "tgt": en, "direction": "zh2en"}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"src": en, "tgt": zh, "direction": "en2zh"}, ensure_ascii=False) + "\n")
                got += 1
                written += 1
                if got >= want:
                    break
            line = f"[ok] {sid} ({kind}): {got} pairs"
            print(line); summary.append(line)

    print("---")
    for s in summary:
        print(s)
    print(f"total {written} pairs ({written*2} jsonl 行) -> {out}")
    if written == 0:
        raise SystemExit("没拉到任何句对：检查源 config / 网络 / HF_ENDPOINT")
    sys.stdout.flush()
    os._exit(0)  # 绕过 datasets 流式后台线程退出期 GIL 崩


if __name__ == "__main__":
    main()
