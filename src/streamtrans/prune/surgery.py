"""state_dict 级剪枝手术（torch）。

数据流：full_sd → strip(剥视觉+改前缀) → prune_layers(选层重编号)
        → prune_ffn(切中间维) → prune_vocab(切词表行) → 灌入新建 Qwen3_5ForCausalLM。

约定的 key 布局：
  源（Qwen3_5ForConditionalGeneration）：model.language_model.{embed_tokens,layers.N.*,norm}, lm_head.*, model.visual.*
  目标（Qwen3_5ForCausalLM）：           model.{embed_tokens,layers.N.*,norm}, lm_head.*
"""
import re
import torch

from streamtrans.prune.ffn import select_ffn_neurons, slice_mlp

_TEXT_PREFIX = "model.language_model."
_TGT_PREFIX = "model."
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def strip_to_text_sd(full_sd: dict) -> dict:
    """只保留文本主干 + lm_head，并把 model.language_model. 前缀改成 model.。丢弃 visual 等。"""
    out: dict = {}
    for k, v in full_sd.items():
        if k.startswith(_TEXT_PREFIX):
            out[_TGT_PREFIX + k[len(_TEXT_PREFIX):]] = v
        elif k.startswith("lm_head."):
            out[k] = v
        # 其余（model.visual.* 等）丢弃
    return out


def prune_layers_sd(sd: dict, kept_idx: list[int]) -> dict:
    """保留 kept_idx 指定的层并重编号为 0..len-1；非层参数原样保留。"""
    remap = {old: new for new, old in enumerate(kept_idx)}
    out: dict = {}
    for k, v in sd.items():
        m = _LAYER_RE.match(k)
        if m is None:
            out[k] = v
            continue
        old = int(m.group(1))
        if old not in remap:
            continue
        out[f"model.layers.{remap[old]}.{m.group(2)}"] = v
    return out


def prune_ffn_sd(sd: dict, num_layers: int, keep: int) -> dict:
    """逐层按重要度切 MLP 中间维到 keep。要求 sd 已是目标层编号（0..num_layers-1）。"""
    out = dict(sd)
    for j in range(num_layers):
        g = f"model.layers.{j}.mlp.gate_proj.weight"
        u = f"model.layers.{j}.mlp.up_proj.weight"
        d = f"model.layers.{j}.mlp.down_proj.weight"
        if g not in out:  # 该层可能无 mlp（理论不会），跳过
            continue
        idx = select_ffn_neurons(out[g], out[u], keep)
        out[g], out[u], out[d] = slice_mlp(out[g], out[u], out[d], idx)
    return out


def prune_vocab_sd(sd: dict, keep_ids: list[int]) -> dict:
    """按 keep_ids 切 embed_tokens（及 lm_head，如未 tie）的行。"""
    out = dict(sd)
    idx = torch.tensor(keep_ids, dtype=torch.long)
    emb_k = "model.embed_tokens.weight"
    if emb_k in out:
        out[emb_k] = out[emb_k][idx, :]
    lm_k = "lm_head.weight"
    if lm_k in out:
        out[lm_k] = out[lm_k][idx, :]
    return out
