import pytest

from streamtrans.distill.teacher_logits_io import locate, remap_topk_ids


def test_locate_across_shards():
    sizes = [3, 2, 4]   # 全局 0..8
    assert locate(sizes, 0) == (0, 0)
    assert locate(sizes, 2) == (0, 2)
    assert locate(sizes, 3) == (1, 0)
    assert locate(sizes, 4) == (1, 1)
    assert locate(sizes, 5) == (2, 0)
    assert locate(sizes, 8) == (2, 3)
    with pytest.raises(IndexError):
        locate(sizes, 9)


def test_remap_topk_drops_oov():
    old2new = {100: 0, 250: 1, 7: 2}
    keep_pos, new_ids = remap_topk_ids([100, 999, 7, 250], old2new)
    assert keep_pos == [0, 2, 3]      # 999 集外被丢
    assert new_ids == [0, 2, 1]
