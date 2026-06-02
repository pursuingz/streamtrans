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


def test_force_keep_always_retained():
    # force_keep(语料必用 token)必须全保,即便频次为 0、即便挤占预算
    freq = {10: 100, 11: 50}        # 预留语言候选(填剩余预算用)
    force = {500, 501, 502}          # 语料 token,freq 里没有也要保
    keep = select_keep_tokens(freq, special_ids=[0, 1], max_vocab=6, force_keep=force)
    assert force <= keep             # 全部强制保留
    assert 0 in keep and 1 in keep   # special 仍保
    # 剩余预算(6-2special-3force=1)给最高频预留 token
    assert 10 in keep and 11 not in keep


def test_force_keep_overflow_wins_over_budget():
    # force_keep + special 已超 max_vocab 时,正确性优先,全保(长度超 max_vocab)
    keep = select_keep_tokens({}, special_ids=[0], max_vocab=2, force_keep={500, 501, 502})
    assert {0, 500, 501, 502} <= keep
    assert len(keep) == 4            # 超 max_vocab=2,但语料 token 不丢
