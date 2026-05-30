import torch

from streamtrans.prune.ffn import select_ffn_neurons, slice_mlp


def test_select_ffn_neurons_picks_high_norm_sorted():
    inter, hidden = 10, 4
    gate = torch.zeros(inter, hidden)
    up = torch.zeros(inter, hidden)
    # 行范数随行号递增 -> top-6 应为行 4..9
    for i in range(inter):
        gate[i] = float(i)
    idx = select_ffn_neurons(gate, up, keep=6)
    assert idx.tolist() == [4, 5, 6, 7, 8, 9]
    assert torch.equal(idx, torch.sort(idx).values)  # 升序


def test_slice_mlp_shapes():
    inter, hidden = 10, 4
    gate = torch.randn(inter, hidden)
    up = torch.randn(inter, hidden)
    down = torch.randn(hidden, inter)
    idx = torch.tensor([1, 3, 5, 7])
    g, u, d = slice_mlp(gate, up, down, idx)
    assert g.shape == (4, hidden)
    assert u.shape == (4, hidden)
    assert d.shape == (hidden, 4)
    assert torch.equal(g[0], gate[1])     # 行选对
    assert torch.equal(d[:, 0], down[:, 1])  # 列选对
