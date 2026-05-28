import torch.nn as nn
from streamtrans.prune.inspect_model import param_breakdown


def test_param_breakdown_groups_by_prefix():
    model = nn.ModuleDict({
        "embed_tokens": nn.Embedding(100, 8),      # 800
        "layers": nn.ModuleList([nn.Linear(8, 8)]),  # 64 + 8 bias = 72
        "vision": nn.Linear(8, 4),                  # 32 + 4 = 36
    })
    bd = param_breakdown(model, group_prefixes=["embed_tokens", "layers", "vision"])
    assert bd["embed_tokens"] == 800
    assert bd["layers"] == 72
    assert bd["vision"] == 36
    assert bd["__total__"] == 800 + 72 + 36
