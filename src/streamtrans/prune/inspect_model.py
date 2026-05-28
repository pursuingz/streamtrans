"""模型结构勘探：参数量分解 + module 命名导出。"""
from typing import Iterable
import torch.nn as nn


def param_breakdown(model: nn.Module, group_prefixes: Iterable[str]) -> dict[str, int]:
    """按 named_parameters 的前缀分组统计参数量（meta tensor 也可，numel 不依赖数据）。"""
    prefixes = list(group_prefixes)
    out: dict[str, int] = {p: 0 for p in prefixes}
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        for pref in prefixes:
            if pref in name:
                out[pref] += n
                break
    out["__total__"] = total
    return out


def load_meta_model(model_id: str):
    """在 meta device 上实例化模型（不下载权重，看结构/数参数用）。

    Qwen3.5 是 ConditionalGeneration（多模态）架构，AutoModelForCausalLM 可能不认，
    故按 多模态生成 -> CausalLM -> 通用 顺序尝试。返回 (model, cfg, loader_name)。
    """
    import torch
    import transformers as tf
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    last_err = None
    for name in ["AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"]:
        cls = getattr(tf, name, None)
        if cls is None:
            continue
        try:
            with torch.device("meta"):
                model = cls.from_config(cfg, trust_remote_code=True)
            return model, cfg, name
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"无法用任何 Auto 类加载 {model_id}: {last_err}")


def dump_module_names(model: nn.Module) -> list[str]:
    """所有 module 的限定名（用于发现 SSM/full-attn/vision/MTP 的真实命名）。"""
    return [name for name, _ in model.named_modules() if name]
