from streamtrans.data.vocab_remap import VocabRemapper


def test_map_unmap_roundtrip():
    rm = VocabRemapper({10: 0, 11: 1, 250: 2})
    new = rm.map_ids([10, 250, 11])
    assert new == [0, 2, 1]
    assert rm.unmap_ids(new) == [10, 250, 11]   # 可逆


def test_oov_dropped_without_unk():
    rm = VocabRemapper({10: 0, 11: 1})
    assert rm.map_ids([10, 999, 11]) == [0, 1]   # 999 集外，丢弃


def test_oov_mapped_to_unk():
    rm = VocabRemapper({10: 0, 11: 1}, unk_new_id=0)
    assert rm.map_ids([10, 999, 11]) == [0, 0, 1]  # 999 → unk(0)
