from streamtrans.prune.vocab_prune import build_keep_ids, build_vocab_map


def test_build_keep_ids_sorted_and_includes_special():
    freq = {10: 100, 11: 50, 12: 30, 99: 1}
    keep = build_keep_ids(freq, special_ids=[0, 1], max_vocab=4)
    assert keep == sorted(keep)              # 升序
    assert 0 in keep and 1 in keep           # special 必留
    assert len(keep) == 4                    # 补满到 max_vocab
    assert 10 in keep and 11 in keep         # 高频补入


def test_build_vocab_map_contiguous_roundtrip():
    keep = [0, 1, 10, 11]
    vmap = build_vocab_map(keep)
    assert vmap == {0: 0, 1: 1, 10: 2, 11: 3}
    # new_id 连续，且 keep[new] == old
    for old, new in vmap.items():
        assert keep[new] == old
