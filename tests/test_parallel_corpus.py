from streamtrans.data.parallel_corpus import clean_pair, filter_pairs


def test_clean_pair_strips_and_normalizes_space():
    src, tgt = clean_pair("  Hello   world ", "你好  世界")
    assert src == "Hello world"
    assert tgt == "你好 世界"


def test_filter_pairs_drops_empty_and_ratio_outliers():
    pairs = [
        ("hello", "你好"),          # ok
        ("", "你好"),               # 空源 -> drop
        ("a", "这是一个非常非常长的句子超过比例上限了哦哦哦"),  # 长度比异常 -> drop
        ("a normal sentence here", "一个正常长度的句子"),  # ok
    ]
    kept = filter_pairs(pairs, min_len=1, max_len=200, max_ratio=4.0)
    assert ("hello", "你好") in kept
    assert ("", "你好") not in kept
    assert len(kept) == 2
