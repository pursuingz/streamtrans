# Phase 2 实现计划 —— 离线知识蒸馏（heal 剪枝 + 强离线中英翻译器）

> 前置：Phase 1 完成，学生 `checkpoints/pruned_20260530/`（0.685B，12 层，词表 110k new-id，`Qwen3_5ForCausalLM`）。
> 决策（2026-05-30）：**Phase 2 只做离线 KD**；on-policy OPD + wait-k 留 Phase 3。数据**口语/字幕为主 7:3 通用**。
> 产物：`checkpoints/distilled_<date>/`，一个全句（非流式）中英翻译能力恢复/增强的 0.685B 学生。

## 0. 目标与边界
- **目标**：用 9B 教师离线 top-k logits 蒸馏，heal 12 层剪枝的损伤，得到强离线中英翻译器。
- **不做**：流式/wait-k（Phase 3）、on-policy/reverse-KL（Phase 3）、量化导出（Phase 4）。
- **方向**：第一版 **zh↔en 双向**单模型，prompt 区分方向，两向数据都喂。

## 1. 关键约束与设计决策

### 1.1 教师离线（24GB 约束）
9B 教师以 **4-bit** 离线 forward，导出每 token top-k logits 落盘；训练时只读盘、不持有教师。
24GB 仅训 0.685B 学生（8-bit 优化器 + 梯度检查点，绰绰有余）。教师导出是一次性、可复用跨 epoch。

### 1.2 师生词表空间桥接（核心难点）
教师输出 248320 维，学生 110k（new-id）。**导出时**把教师 top-k 的 old-id 经 `vocab_map.json`→new-id，
落集外的 token 丢弃、对剩余 top-k 重新归一化 → 落盘即 new-id 空间，训练端零映射成本。
（CONCLUSIONS 已确认 2B/9B 同 tokenizer、同 248320、特殊 id 一致，logits 可对齐。）

### 1.3 任务格式（教师/学生必须完全一致）
因果 LM 形式，统一 prompt 模板（如 `<bos>{src}\n[zh2en]\n{tgt}<eos>`），**loss 只在 target 段**。
教师在**同一 prompt、同一 tokenization**上 forward，取 target 段每位置 top-k。prompt/分词不一致会导致 logits 错位——列为头号风险。

### 1.4 损失
`L = α·CE(student, ref) + β·KL_forward(teacher_topk ‖ student) / T²`（温度 T 软化）。
forward-KL（mean-seeking，覆盖教师分布）适合 warm-up/heal；reverse-KL 留 Phase 3 on-policy。

## 2. 数据流
```
build_parallel_corpus(口语/字幕 7:3 通用) → clean/filter(parallel_corpus.py)
  → 套 prompt 模板 + VocabRemapper 编码(new-id)
  → teacher_export(4bit 9B, top-k logits, old→new id, 落盘 shards)
  → run_distill(读 ref + 对齐的 teacher top-k, CE+KD, 训学生, 8bit优化器+梯度检查点)
  → eval(BLEU/COMET, 对比基线)
```

## 3. 任务拆解

### Task 1 — 平行语料 `scripts/build_parallel_corpus.py`
- 拉中英平行：口语/字幕(OpenSubtitles、TED via OPUS/datasets) 为主，news-commentary、UN 为辅，约 7:3。
- 清洗用已有 `data/parallel_corpus.py`（clean_pair/filter_pairs：去空、长度比异常）。
- 写 `data/parallel_zh_en.jsonl`（{src, tgt, direction}）。流式拉取后同样 `os._exit(0)` 防退出期崩。

### Task 2 — prompt 与 target mask `src/streamtrans/distill/prompt.py`
- `build_example(src, tgt, direction) -> (input_ids, labels)`，labels 中 prompt/src 段置 -100，仅 target 段计 loss。
- 纯逻辑部分（mask 位置计算）torch-free 可单测。

### Task 3 — KD 损失 `src/streamtrans/distill/kd_losses.py`（torch，可单测）
- `kd_forward_kl_topk(student_logits, teacher_ids, teacher_logprobs, T)`：在教师 top-k 子集上算 forward-KL。
- `combined_loss(student_logits, labels, teacher_*, alpha, beta, T)`：CE + KD。
- 小张量单测：KL≥0、师生分布相同时 KL≈0、mask 生效。

### Task 4 — 教师 logits 落盘/读盘 `src/streamtrans/distill/teacher_logits_io.py`
- shard 格式：`new_ids[int32]`、`logprobs[fp16]`、每序列长度/偏移索引。分 shard（如每 shard 1万序列）。
- 磁盘量估算放注释：top-k=64 × tokens × (4B+2B)。控制语料规模或降 top-k。

### Task 5 — 教师离线导出 `scripts/run_teacher_export.py --config configs/distill.yaml [--smoke]`
- 4-bit 加载 9B（`AutoModelForImageTextToText`，bitsandbytes nf4），对每 batch (src,tgt) forward，取 target 段 top-k logits → old→new id → 重归一化 → 写 shard。
- `--smoke`：几十条样本，校验 shard 形状、new-id ∈ [0,110000)、与 ref 对齐。

### Task 6 — 蒸馏数据集 `src/streamtrans/distill/dataset.py`
- 对齐平行样本与其 teacher top-k shard；collate 成 batch（pad + teacher 稀疏 logits 对齐）。

### Task 7 — 训练 `scripts/run_distill.py --config configs/distill.yaml [--smoke]`
- 加载 `pruned_20260530` 学生；8-bit Adam + 梯度检查点；`combined_loss`。
- 训练前装 `flash-linear-attention` + `causal-conv1d`（线性注意力 fast path 提速）。
- `--smoke`：几十 step 跑通，断言 loss 有限、整体下降；存 `checkpoints/distilled_smoke`。
- 全量：存 `checkpoints/distilled_<date>`。

### Task 8 — 评测 `src/streamtrans/eval/quality.py` + `scripts/run_eval.py`
- BLEU（sacrebleu，已在依赖）；COMET（unbabel-comet，重依赖，标**可选**）。
- 基线对比：pruned 学生(蒸馏前) vs distilled vs 原 2B vs 9B 教师(上界)。

## 4. configs/distill.yaml（草案字段）
```
teacher_model: Qwen/Qwen3.5-9B
student_ckpt: checkpoints/pruned_20260530
vocab_map: checkpoints/pruned_20260530/vocab_map.json
topk: 64
temperature: 2.0
alpha_ce: 0.5
beta_kd: 0.5
directions: [zh2en, en2zh]
corpus: { spoken: [...], general: [...], ratio: 0.7 }
max_len: 256
batch_size: 8
grad_accum: 4
lr: 2.0e-4
steps: ...
optim: adamw_8bit
grad_checkpointing: true
teacher_4bit: true
```

## 5. 验证（服务器 24GB）
```bash
pytest -v tests/test_kd_losses.py tests/test_prompt.py        # torch/逻辑单测
python scripts/build_parallel_corpus.py --smoke                # 小样本拉取+清洗
python scripts/run_teacher_export.py --config configs/distill.yaml --smoke
python scripts/run_distill.py --config configs/distill.yaml --smoke   # loss 下降、无 NaN
python scripts/run_eval.py --ckpt checkpoints/distilled_smoke         # BLEU 烟雾
```

## 6. 风险与对策
1. **prompt/tokenization 师生不一致 → logits 错位**（头号风险）：导出与训练共用同一 `build_example`，同一 tokenizer，单测对齐。
2. **教师导出磁盘量**：top-k=64 控制；语料分批导出、分 shard；必要时降 top-k 或限语料规模。
3. **4-bit 教师软标签质量损失**：先 4-bit 起步，质量不足再 8-bit；导出是离线一次性，可承受更慢。
4. **heal 不充分**（12 层剪枝较激进）：加数据/步数；CE 权重保底；仍不足则回看 Phase 1 取向（如 16 层）。
5. **双向稀释**：zh↔en 双向若互相干扰，退单向或调配比。

## 7. 产物与解耦
- `checkpoints/distilled_<date>/`（强离线翻译学生）+ eval 报告 + teacher logits shards（gitignore）。
- Phase 3 以此为起点：wait-k 流式数据构造 + 流式微调 + on-policy OPD（reverse-KL）。
