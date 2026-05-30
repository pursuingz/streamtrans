"""FFN 中间维剪枝：按神经元重要度选 top-k，切 gate/up 行、down 列。

SwiGLU MLP：down(silu(gate(x)) * up(x))。中间维的每个神经元 j 对应
gate_proj 第 j 行、up_proj 第 j 行、down_proj 第 j 列，三者绑定，整体保留或丢弃，
不破坏逐神经元的门控语义。
"""
import torch


def select_ffn_neurons(gate_w: torch.Tensor, up_w: torch.Tensor, keep: int) -> torch.Tensor:
    """选 keep 个中间神经元，返回升序的行/列索引（LongTensor）。

    重要度 = ||gate 行||₂ + ||up 行||₂（magnitude 策略）。
    gate_w/up_w 形状均为 [intermediate, hidden]。
    """
    inter = gate_w.shape[0]
    if keep > inter:
        raise ValueError(f"keep={keep} 超过中间维 {inter}")
    score = gate_w.float().norm(dim=1) + up_w.float().norm(dim=1)  # [intermediate]
    idx = torch.topk(score, keep).indices
    return torch.sort(idx).values


def slice_mlp(
    gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor, idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按神经元索引切片：gate/up 留行 idx，down 留列 idx。"""
    return gate_w[idx, :], up_w[idx, :], down_w[:, idx]
