"""剪枝规划（torch-free 纯逻辑）：选层、生成目标 layer_types、配置改写。

与权重无关，可在无 torch 环境单测。实际权重手术在 surgery.py。
"""
from typing import Any


def select_layer_indices(num_layers: int, target_layers: int, interval: int) -> list[int]:
    """从 num_layers 层中等距保留 target_layers 层，保持 [linear×(interval-1), full] 节律。

    原层按 interval 分块（每块末层是 full attention）。等距挑 target_layers/interval 个
    完整块，整块保留，从而新层序列仍满足 full_attention_interval=interval。

    例：num=24, target=12, interval=4 → 保留块 0,2,5（含首块与末块）
        → idx [0,1,2,3, 8,9,10,11, 20,21,22,23]。
    """
    if num_layers % interval != 0:
        raise ValueError(f"num_layers={num_layers} 不是 interval={interval} 的整数倍")
    if target_layers % interval != 0:
        raise ValueError(f"target_layers={target_layers} 不是 interval={interval} 的整数倍")
    n_blocks = num_layers // interval
    keep_blocks = target_layers // interval
    if keep_blocks > n_blocks:
        raise ValueError(f"target_layers={target_layers} 超过 num_layers={num_layers}")
    # 在 n_blocks 个块里等距挑 keep_blocks 个块（含首块，尽量均匀铺开）
    if keep_blocks == 1:
        block_ids = [0]
    else:
        step = (n_blocks - 1) / (keep_blocks - 1)
        block_ids = [round(i * step) for i in range(keep_blocks)]
    kept: list[int] = []
    for b in block_ids:
        start = b * interval
        kept.extend(range(start, start + interval))
    return kept


def make_layer_types(target_layers: int, interval: int) -> list[str]:
    """生成目标 layer_types：每 interval 层的最后一层是 full_attention，其余 linear_attention。"""
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(target_layers)
    ]


def target_config_overrides(
    target_layers: int, target_ffn: int, target_vocab: int, interval: int
) -> dict[str, Any]:
    """返回需要写到 Qwen3_5TextConfig 上的字段覆盖（其余字段从原 text_config 继承）。"""
    return {
        "num_hidden_layers": target_layers,
        "intermediate_size": target_ffn,
        "vocab_size": target_vocab,
        "layer_types": make_layer_types(target_layers, interval),
        "full_attention_interval": interval,
    }
