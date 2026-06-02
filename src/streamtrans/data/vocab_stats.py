"""词表裁剪前的 token 频次统计与保留集选择。"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable


def count_token_freq(texts: Iterable[str], encode: Callable[[str], list[int]]) -> dict[int, int]:
    c: Counter[int] = Counter()
    for t in texts:
        c.update(encode(t))
    return dict(c)


def select_keep_tokens(
    freq: dict[int, int],
    special_ids: list[int],
    max_vocab: int,
    force_keep: set[int] | None = None,
) -> set[int]:
    """保留 全部 special + force_keep(训练语料必用 token,强制保留以保证 OOV≈0)
    + 按频次(freq,通常是预留语言)补满到 max_vocab。

    force_keep 优先级最高:即使加上它会略超 max_vocab 也全保(正确性 > 预算),
    防止重蹈"中英常用 token 被裁→OOV→过早 eos"的覆辙。
    """
    keep = set(special_ids)
    if force_keep:
        keep |= set(force_keep)
    remaining = max_vocab - len(keep)
    if remaining <= 0:
        return keep
    ranked = sorted((tid for tid in freq if tid not in keep),
                    key=lambda t: freq[t], reverse=True)
    keep.update(ranked[:remaining])
    return keep
