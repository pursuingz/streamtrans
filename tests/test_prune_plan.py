from streamtrans.prune.plan import (
    make_layer_types,
    select_layer_indices,
    target_config_overrides,
)


def test_select_layer_indices_24_to_12():
    # 端点+中间均匀：保留块 0,2,5（含首块与末块）
    idx = select_layer_indices(num_layers=24, target_layers=12, interval=4)
    assert idx == [0, 1, 2, 3, 8, 9, 10, 11, 20, 21, 22, 23]


def test_kept_layers_preserve_full_attn_rhythm():
    # 原 full attention 在 idx%4==3；保留的块整块搬运，新序列里 full 仍每 4 层一次
    idx = select_layer_indices(24, 12, 4)
    full_positions_old = [i for i in idx if (i + 1) % 4 == 0]
    assert full_positions_old == [3, 11, 23]  # 每个保留块的末层
    types = make_layer_types(12, 4)
    assert [i for i, t in enumerate(types) if t == "full_attention"] == [3, 7, 11]


def test_make_layer_types():
    assert make_layer_types(12, 4) == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ]


def test_select_layer_indices_validates_multiples():
    import pytest
    with pytest.raises(ValueError):
        select_layer_indices(24, 10, 4)   # target 非 4 倍数
    with pytest.raises(ValueError):
        select_layer_indices(25, 12, 4)   # num 非 4 倍数


def test_target_config_overrides():
    ov = target_config_overrides(target_layers=12, target_ffn=3072, target_vocab=110000, interval=4)
    assert ov["num_hidden_layers"] == 12
    assert ov["intermediate_size"] == 3072
    assert ov["vocab_size"] == 110000
    assert ov["full_attention_interval"] == 4
    assert len(ov["layer_types"]) == 12
