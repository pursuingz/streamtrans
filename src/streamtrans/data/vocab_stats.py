"""词表裁剪前的 token 频次统计与保留集选择。"""
from collections import Counter
from typing import Callable, Iterable


def count_token_freq(texts: Iterable[str], encode: Callable[[str], list[int]]) -> dict[int, int]:
    c: Counter[int] = Counter()
    for t in texts:
        c.update(encode(t))
    return dict(c)


def select_keep_tokens(freq: dict[int, int], special_ids: list[int], max_vocab: int) -> set[int]:
    """保留全部 special + 按频次补满到 max_vocab。"""
    keep = set(special_ids)
    remaining = max_vocab - len(keep)
    if remaining <= 0:
        return keep
    ranked = sorted((tid for tid in freq if tid not in keep),
                    key=lambda t: freq[t], reverse=True)
    keep.update(ranked[:remaining])
    return keep
