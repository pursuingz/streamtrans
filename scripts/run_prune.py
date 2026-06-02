"""Phase 1 结构化剪枝入口：Qwen3.5-2B → ~0.69B 纯文本学生底座。

用法:
  python scripts/run_prune.py --config configs/prune.yaml --smoke
  python scripts/run_prune.py --config configs/prune.yaml \
      --corpus-file data/train_zh_en.jsonl --test-file data/test_zh_en.jsonl \
      --out checkpoints/pruned_20260602
  # 频次铁律：训练语料(src+tgt)token 全部强制保留(OOV≈0)；预留语言可选 --calib-file 填剩余预算。

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


def corpus_texts_from_jsonl(path: str) -> list[str]:
    """读平行语料 jsonl，src 与 tgt 双侧都收集（两个方向都会作输入/输出）。"""
    out: list[str] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        out.append(r["src"])
        out.append(r["tgt"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prune.yaml")
    ap.add_argument("--out", default=None, help="输出目录；默认 checkpoints/pruned_<date>")
    ap.add_argument("--corpus-file", default=None,
                    help="训练语料 jsonl(含 train+可加 test)；其 src+tgt 全部 token 强制保留(OOV≈0)")
    ap.add_argument("--test-file", default=None, help="测试语料 jsonl(可选)；token 一并强制保留,eval 零 OOV")
    ap.add_argument("--calib-file", default=None,
                    help="预留语言校准语料(一行一句,可选)；只用其频次填中英之外的剩余预算")
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
    # keep 集铁律(见 CLAUDE.md 决策2)：训练语料(src+tgt)用到的 token 全部强制保留(force_keep)，
    # 中英 OOV≈0；预留语言只用 calib 频次填剩余预算，绝不挤占中英。
    if args.smoke:
        corpus_texts = SMOKE_CALIB
        calib_texts: list[str] = []
    else:
        if not args.corpus_file:
            raise SystemExit("非 smoke 模式必须给 --corpus-file 训练语料(其 token 强制保留)")
        corpus_texts = corpus_texts_from_jsonl(args.corpus_file)
        if args.test_file:
            corpus_texts += corpus_texts_from_jsonl(args.test_file)
        calib_texts = ([ln.strip() for ln in Path(args.calib_file).read_text(encoding="utf-8").splitlines()
                        if ln.strip()] if args.calib_file else [])
    special_ids = gather_special_ids(tok, full_cfg)
    force_keep = set(build_freq(tok, corpus_texts).keys())   # 语料必用 token,强制保留
    fill_freq = build_freq(tok, calib_texts) if calib_texts else {}   # 仅填剩余预算(预留语言)
    keep_ids = build_keep_ids(fill_freq, special_ids, cfg.target_vocab_size, force_keep=force_keep)
    vmap = build_vocab_map(keep_ids)
    n_reserve = max(0, len(keep_ids) - len(force_keep | set(special_ids)))
    print(f"      special={len(special_ids)}  force_keep(语料)={len(force_keep)}  "
          f"keep={len(keep_ids)} / target={cfg.target_vocab_size}  预留语言填入≈{n_reserve}"
          f"  (smoke 下样本小,数值仅自检用)")

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
    if not args.smoke and not (0.55e9 <= n_params <= 0.72e9):
        raise RuntimeError(f"参数量 {n_params/1e9:.3f}B 不在 [0.55,0.72]B(80k 词表约 0.62B,110k 约 0.685B)")


if __name__ == "__main__":
    main()
