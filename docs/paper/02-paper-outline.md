# StreamTrans 论文结构（端侧流式同传系统）

> 定位：**系统/应用向**论文。核心贡献是"把通用大模型压成端侧可部署、低延迟流式同传系统"的
> 完整 pipeline 与端侧可行性证据，而非单点算法新颖性。
> 叙事主线：通用 2B 混合架构 LLM → 结构化剪枝+可逆词表裁剪 → 离线KD heal →
> wait-k 流式改造 → 端侧量化部署，每一环都服务于"端侧 + 低延迟 + 流式"三个约束。
>
> 配套：事实数据见 `01-worklog.md`，相关工作见 `03-related-work.md`。

---

## 标题（候选）
- *StreamTrans: A 0.7B On-Device Streaming Simultaneous Translation System via Structured Pruning and Distillation*

## Abstract
问题 → 方法（压缩 pipeline + wait-k 流式 + 端侧）→ 结果（0.685B、质量、延迟 AL/LAAL、端侧延迟/内存）→ 意义（首个/少见的端侧流式同传完整系统）。

## 1. Introduction
- **痛点**：同声传译要求低延迟、流式、隐私（端侧、不上云）；云端大模型延迟与隐私不满足，端侧又装不下通用大模型。
- **挑战三连**：① 通用 LLM 太大（2B 起、词表 embedding 是预算黑洞）；② 端侧算力/内存严苛；③ 流式同传与离线翻译目标不同（prefix-to-prefix、暴露偏差、延迟-质量权衡）。
- **贡献**：
  1. 一套面向端侧的**结构化压缩 pipeline**：剥视觉塔 + 减层 + 缩 FFN + **可逆词表裁剪**，把 2B 通用多模态 LLM 压到 0.685B 纯文本同传模型；
  2. **师生同词表的离线 KD**（top-k logits + 师生分词桥接），heal 剪枝损伤；
  3. **wait-k 流式改造 + on-policy OPD**，给出延迟-质量权衡曲线；
  4. **端侧落地**：量化 + MNN/llama.cpp 部署，移动端实测延迟/内存；落成语音→字幕同传 app。
- **强调**：贡献在"系统集成 + 端侧可行性证据"，各环节选择都由端侧约束驱动。

## 2. Related Work
（详见 `03-related-work.md`）
- 模型压缩：结构化剪枝（减层/缩宽/FFN 剪枝）、词表裁剪。
- 知识蒸馏：序列级 KD、logits/KL 蒸馏、on-policy distillation（OPD/MiniCPM）。
- 同声传译：wait-k 及 prefix-to-prefix、自适应策略、延迟指标 AL/LAAL。
- 端侧 LLM 推理：llama.cpp / MNN，线性注意力（Gated DeltaNet/Mamba 类）端侧支持。
- 与本文差异：多数工作只做其一；本文是**面向端侧同传的端到端集成**，且处理混合架构（线性+全注意力）的剪枝与端侧适配。

## 3. Method
### 3.1 Overview & 设计约束
端侧 + 低延迟 + 流式三约束如何贯穿各环节（一张 pipeline 图）。

### 3.2 结构化剪枝（混合架构友好）
- 只动通用维度：剥 `model.visual`、减层 24→12（保块结构 `[linear×3,full]`，不破 full_attention_interval）、缩 FFN 6144→3072。
- **不剪线性注意力内部**——规避混合架构的适配风险（端侧框架对线性注意力算子敏感）。
- 减层选择与 FFN 神经元选择准则（‖·‖₂）。

### 3.3 可逆词表裁剪
- 248320→110000：embedding 是预算主负担；裁剪 + `vocab_map` + 保留原始 embedding → **可逆**，扩语言只补数据不动结构。
- `VocabRemapper` 桥接，避开 BPE merge 图手术。

### 3.4 离线知识蒸馏
- `L = α·CE + β·forward-KL(teacher‖student)`，教师 4-bit 离线导出 top-k。
- **师生分词桥接**：教师 shard 内存学生 new-id input/labels，杜绝两次分词的位置漂移（工程关键，可作为一个 insight）。
- 批前向 + 仅 target 位过 lm_head（端侧训练资源约束下的工程优化）。

### 3.5 wait-k 流式改造（Phase 3，待做）
- prefix-to-prefix 训练，固定 k 跑通后上自适应。
- on-policy OPD（reverse KL + 师生 top-k 并集）并入此阶段——暴露偏差在流式逐前缀解码下最严重。

### 3.6 端侧部署（Phase 4，待做）
- 量化（int4/int8）+ MNN/llama.cpp 导出；混合架构线性注意力的端侧算子映射。

## 4. Experiments
### 4.1 Setup
数据（OpenSubtitles+OPUS-100 中英，7:3 口语/通用，train 391446、test 2000 去重）、教师/学生、硬件、指标。
### 4.2 质量
- 中英双向 BLEU（中文用 sacrebleu zh 分词）/ chrF；Flores-200 devtest 对外可比。
- 对照：未剪枝 2B、教师 9B、（可选）同规模 baseline。
- **质量-epoch 曲线**（3k/6k/9k/12k checkpoint）定位饱和点。
### 4.3 失败模式与训练充分性（已有素材）
- 过早 eos / 退化 / 混杂的定量分析 + min-new 判别实验（说明欠训练 vs 能力缺失）——可作为"训练充分性诊断"小节，方法论上有价值。
### 4.4 延迟-质量权衡（Phase 3）
- 不同 k 的 AL/LAAL vs BLEU 曲线。
### 4.5 端侧实测（Phase 4）
- 移动端首字延迟、吞吐、峰值内存、模型体积；端侧 vs 服务端质量一致性。
### 4.6 消融
- 减层策略 / FFN 剪枝量 / 词表大小 / α-β / top-k / 温度。

## 5. Discussion / Limitations
- 域局限（字幕域为主）、语言局限（首版中英）、流式延迟下限、端侧框架对线性注意力支持成熟度。

## 6. Conclusion
端侧流式同传可行；可逆词表裁剪 + 混合架构友好剪枝 + 离线KD 是一条可复制的端侧化路径。

---

## 投稿前必补实验清单（gap）
1. Phase 2 收尾：质量-epoch 曲线 + 最优 checkpoint 全集分方向分数 + 教师/未剪枝对照。
2. 消融（至少：减层、词表大小、α-β）。
3. Flores-200 对外可比数。
4. **Phase 3 流式**：wait-k 多 k 的 AL/LAAL-质量曲线（系统论文的核心卖点之一）。
5. **Phase 4 端侧**：移动端延迟/内存/体积实测（"端侧"论点的硬证据，缺则降级为"可压缩"而非"端侧系统"）。
