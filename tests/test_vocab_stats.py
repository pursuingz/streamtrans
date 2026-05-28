from streamtrans.data.vocab_stats import count_token_freq, select_keep_tokens


def test_count_token_freq():
    # 假 tokenizer：按空格切，token->id 用长度模拟
    def fake_encode(text):
        return [len(w) for w in text.split()]
    freq = count_token_freq(["a bb ccc", "bb bb"], fake_encode)
    assert freq[1] == 1   # "a"
    assert freq[2] == 3   # "bb" x3
    assert freq[3] == 1   # "ccc"


def test_select_keep_tokens_respects_special_and_topk():
    freq = {10: 100, 11: 50, 12: 1, 13: 1}
    keep = select_keep_tokens(freq, special_ids=[0, 1], max_vocab=4)
    # 必保留 special，再按频次补到 max_vocab
    assert 0 in keep and 1 in keep
    assert 10 in keep and 11 in keep
    assert len(keep) == 4
    assert 12 not in keep or 13 not in keep  # 低频被裁
