"""词表裁剪（torch-free 部分）：构造 keep id 列表与 old->new 映射。

复用 data/vocab_stats.select_keep_tokens 选 keep 集（全部 special + 按频次补满）。
切 embedding 行的张量操作在 surgery.py。映射可逆：存 old->new + 原始 embedding。
"""
from __future__ import annotations

from streamtrans.data.vocab_stats import select_keep_tokens


def build_keep_ids(
    freq: dict[int, int],
    special_ids: list[int],
    max_vocab: int,
    force_keep: set[int] | None = None,
) -> list[int]:
    """返回升序的保留 token 原始 id 列表。

    force_keep(训练语料必用 token)强制保留,故长度可能略超 max_vocab(正确性优先)。
    升序保证 new_id = 列表下标 时映射确定、可复现。
    """
    keep = select_keep_tokens(freq, special_ids=special_ids, max_vocab=max_vocab,
                              force_keep=force_keep)
    return sorted(keep)


def build_vocab_map(keep_ids: list[int]) -> dict[int, int]:
    """old_id -> new_id（new_id 为 keep_ids 中的位置，连续 0..len-1）。"""
    return {old: new for new, old in enumerate(keep_ids)}
