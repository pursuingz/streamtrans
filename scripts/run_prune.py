"""Phase 1 结构化剪枝入口：Qwen3.5-2B → ~0.69B 纯文本学生底座。

用法:
  python scripts/run_prune.py --config configs/prune.yaml --smoke
  python scripts/run_prune.py --config configs/prune.yaml --calib-file data/calib_multiling.txt --out checkpoints/pruned_20260530

链路：load full(ConditionalGeneration) → strip视觉 → 选层 → 切FFN → 切词表
     → 灌入新建 Qwen3_5ForCausalLM → save_pretrained + 可逆产物(vocab_map.json, embed_original.pt)。
质量恢复(OPD)属 Phase 2；本脚本只产出结构有效、可加载、可前向的 student。
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from streamtrans.config import PruneConfig, load_config
from streamtrans.data.vocab_stats import count_token_freq
from streamtrans.prune.plan import select_layer_indices, target_config_overrides
from streamtrans.prune.surgery import (
    prune_ffn_sd,
    prune_layers_sd,
    prune_vocab_sd,
    strip_to_text_sd,
)
from streamtrans.prune.vocab_prune import build_keep_ids, build_vocab_map

# 覆盖 reserve_languages 的极小校准样本（仅用于 --smoke 的 token 频次命中，不求质量）
SMOKE_CALIB = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Machine translation needs both source and target language coverage.",
    "今天天气很好，我们一起去公园散步，然后吃晚饭。",
    "流式同声传译要求低延迟和高质量的平衡。",
    "本日はとても良い天気ですね、一緒に公園へ行きましょう。",
    "오늘 날씨가 정말 좋아서 공원에 산책하러 갑니다.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
]


def gather_special_ids(tokenizer, full_cfg) -> list[int]:
    special = set(int(i) for i in (tokenizer.all_special_ids or []))
    for attr in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "eos_token_id",
    ):
        v = getattr(full_cfg, attr, None)
        if isinstance(v, int):
            special.add(v)
    decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    special |= set(int(k) for k in decoder.keys())
    return sorted(special)


def build_freq(tokenizer, texts):
    return count_token_freq(texts, lambda t: tokenizer.encode(t, add_special_tokens=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prune.yaml")
    ap.add_argument("--out", default=None, help="输出目录；默认 checkpoints/pruned_<date>")
    ap.add_argument("--calib-file", default=None, help="多语言校准语料(一行一句)；--smoke 时忽略")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, PruneConfig)
    out_dir = Path(args.out) if args.out else Path("checkpoints") / "pruned_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import (  # 延迟导入，torch-free 模块的单测不受影响
        AutoConfig,
        AutoModelForImageTextToText,
        AutoTokenizer,
        Qwen3_5ForCausalLM,
    )

    print(f"[1/6] 加载 config / tokenizer: {cfg.starter_model}")
    full_cfg = AutoConfig.from_pretrained(cfg.starter_model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(cfg.starter_model, trust_remote_code=True)
    src_layers = full_cfg.text_config.num_hidden_layers

    print("[2/6] 统计词频 + 构造 keep 集")
    if args.smoke:
        texts = SMOKE_CALIB
    else:
        if not args.calib_file:
            raise SystemExit("非 smoke 模式必须给 --calib-file 多语言校准语料")
        texts = [ln.strip() for ln in Path(args.calib_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    special_ids = gather_special_ids(tok, full_cfg)
    freq = build_freq(tok, texts)
    keep_ids = build_keep_ids(freq, special_ids, cfg.target_vocab_size)
    vmap = build_vocab_map(keep_ids)
    print(f"      special={len(special_ids)}  keep={len(keep_ids)} / target={cfg.target_vocab_size}"
          f"  (smoke 下 keep 远小于 target 属正常，样本小)")

    print("[3/6] 加载完整权重 (ConditionalGeneration, fp16)")
    full = AutoModelForImageTextToText.from_pretrained(
        cfg.starter_model, trust_remote_code=True, dtype=torch.float16
    )
    full_sd = full.state_dict()
    del full

    print("[4/6] state_dict 手术: strip → 选层 → 切FFN → 切词表")
    kept_idx = select_layer_indices(src_layers, cfg.target_layers, cfg.full_attention_interval)
    print(f"      保留层 {kept_idx}")
    sd = strip_to_text_sd(full_sd)
    sd = prune_layers_sd(sd, kept_idx)
    sd = prune_ffn_sd(sd, cfg.target_layers, cfg.target_ffn)
    # 可逆凭据：切词表前存原始(已剥视觉/已选层后的)全词表 embedding
    orig_embed = sd["model.embed_tokens.weight"].clone()
    sd = prune_vocab_sd(sd, keep_ids)

    print("[5/6] 新建 Qwen3_5ForCausalLM 并加载")
    tcfg = copy.deepcopy(full_cfg.text_config)
    for k, v in target_config_overrides(
        cfg.target_layers, cfg.target_ffn, len(keep_ids), cfg.full_attention_interval
    ).items():
        setattr(tcfg, k, v)
    # 裁词表后特殊 token id 必须经 vocab_map 重映射到新 id（否则越界，generation 失效）
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        v = getattr(tcfg, attr, None)
        if isinstance(v, int):
            setattr(tcfg, attr, vmap.get(v))  # 不在 keep 集则置 None
    student = Qwen3_5ForCausalLM(tcfg)
    gc = getattr(student, "generation_config", None)
    if gc is not None:
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            v = getattr(gc, attr, None)
            if isinstance(v, int):
                setattr(gc, attr, vmap.get(v))
    missing, unexpected = student.load_state_dict(sd, strict=False)
    student.tie_weights()  # tie 后 lm_head 指向(已切词表的) embed
    # tie 模式下 lm_head.weight 落在 missing 还是 unexpected 取决于 transformers 内部，两边都容忍
    missing = [m for m in missing if not m.startswith("lm_head.")]
    unexpected = [u for u in unexpected if not u.startswith("lm_head.")]
    if missing or unexpected:
        raise RuntimeError(f"state_dict 不匹配 missing={missing[:8]} unexpected={unexpected[:8]}")

    n_params = sum(p.numel() for p in student.parameters())
    print(f"      student 参数量 = {n_params / 1e9:.3f}B")

    print(f"[6/6] 落盘 -> {out_dir}")
    student.save_pretrained(out_dir)
    # tokenizer 仍是原 248320 词表；学生模型用 110k 新 id。二者通过 vocab_map.json 桥接：
    # 训练/推理时用 streamtrans.data.vocab_remap.VocabRemapper(原 tokenizer + vocab_map) 编解码。
    tok.save_pretrained(out_dir)
    (out_dir / "vocab_map.json").write_text(
        json.dumps({str(o): n for o, n in vmap.items()}, ensure_ascii=False), encoding="utf-8"
    )
    if cfg.reversible:
        torch.save(orig_embed, out_dir / "embed_original.pt")

    # smoke 自检：重新加载 + 前向
    print("[smoke] 重新加载 + 前向自检")
    reloaded = Qwen3_5ForCausalLM.from_pretrained(out_dir, trust_remote_code=True)
    reloaded.eval()
    with torch.no_grad():
        ids = torch.randint(0, len(keep_ids), (1, 8))
        logits = reloaded(ids).logits
    assert logits.shape == (1, 8, len(keep_ids)), logits.shape
    assert torch.isfinite(logits).all(), "logits 含 NaN/Inf"
    print(f"      OK  logits.shape={tuple(logits.shape)}  params={n_params/1e9:.3f}B")
    if not args.smoke and not (0.66e9 <= n_params <= 0.72e9):
        raise RuntimeError(f"参数量 {n_params/1e9:.3f}B 不在 [0.66,0.72]B")


if __name__ == "__main__":
    main()
