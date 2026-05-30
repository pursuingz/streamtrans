"""平行语料清洗与过滤（数据源无关）。"""
import re

_WS = re.compile(r"\s+")


def clean_pair(src: str, tgt: str) -> tuple[str, str]:
    return _WS.sub(" ", src).strip(), _WS.sub(" ", tgt).strip()


def filter_pairs(pairs, min_len: int, max_len: int, max_ratio: float):
    kept = []
    for src, tgt in pairs:
        src, tgt = clean_pair(src, tgt)
        if not src or not tgt:
            continue
        ls, lt = len(src), len(tgt)
        if ls < min_len or lt < min_len or ls > max_len or lt > max_len:
            continue
        ratio = max(ls, lt) / max(1, min(ls, lt))
        if ratio > max_ratio:
            continue
        kept.append((src, tgt))
    return kept
