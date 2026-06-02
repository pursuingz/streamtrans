# StreamTrans 工作记录（论文原始素材）

> 本文件是 `docs/paper/` 的工作记录，按"事实层"整理已完成的工作、决策、实验与结果，
> 作为论文写作的原始素材，不掺写作叙事（叙事在 `02-paper-outline.md`）。
> `docs/paper/` 约定：`01-worklog.md` 记录事实，`02-paper-outline.md` 论文结构，
> `03-related-work.md` 相关工作，`figures/` 图表素材。结论性数字以实验日志为准，随实验更新。
>
> 最后更新：2026-06-02（12000-step 蒸馏进行中）

---

## 1. 研究目标

训练一个 **~0.7B 参数、可移动端部署、低延迟流式**的同声传译模型。
- 模态：**文本→文本流式翻译**（语音由外部 ASR 提供）。
- 路线：以 Qwen3.5-2B 为学生底座，**结构化剪枝 + 知识蒸馏 + 结构调整**压到端侧规模，
  再用 wait-k 改造为真正的流式（prefix-to-prefix）同传。
- 第一版聚焦中英双向；词表裁剪可逆，扩语言时只补数据不动结构。

## 2. 模型选型

| 角色 | 模型 | 关键属性 |
|---|---|---|
| 学生底座 | Qwen/Qwen3.5-2B | 混合架构（线性注意力 + 周期性全注意力），24 层，hidden 2048，FFN 6144，词表 248320，带 Vision Encoder |
| 教师 | Qwen/Qwen3.5-9B | 同系、**同词表 248320**，保证 logits 逐 token 对齐 |

- 起点与教师**同词表**是硬约束——否则 logits 蒸馏退化为弱蒸馏。
- 架构注：Qwen3.5 为混合模型，层模式 `[linear,linear,linear,full]×6`（full_attention_interval=4），
  社区描述为 "Gated DeltaNet 3:1"。剪枝**只动通用维度，不剪线性注意力内部**（规避适配风险）。

## 3. 方法

### 3.1 Phase 1：结构化剪枝 + 可逆词表裁剪（已完成）

以 state_dict 手术为中心，从 Qwen3.5-2B 产出纯文本学生：

1. **剥离 Vision Encoder**（`model.visual`，约 0.331B 参数；纯文本部分 1.882B）。
   - 实现：分支 A，用 `Qwen3_5ForCausalLM` + `Qwen3_5TextConfig` 构纯文本因果 LM。
2. **减层 24→12**：保留层索引 `[0,1,2,3,8,9,10,11,20,21,22,23]`（端点 + 中段块），
   保持 `[linear×3, full]` 的块结构完整，整块保留或整块删，不破坏 full_attention_interval。
3. **缩 FFN 中间维 6144→3072**：按 gate/up 权重行的 ‖·‖₂ 选 top 神经元。
4. **缩词表 248320→110000（可逆）**：按"中英为主 + 预留中英日韩 + 主要欧洲语言"全集选 keep id，
   存 `vocab_map.json`（old→new）+ `embed_original.pt`（原始 embedding，供扩语言还原）。
   - embed/lm_head 切片对应裁剪；eos/bos/pad token id 经 vocab_map 重映射到新 id（config + generation_config）。
5. **词表桥接层 `VocabRemapper`**：原 248320 tokenizer + vocab_map 做 old↔new id 互转，
   避免重建 BPE merge 图（byte-level BPE 的 256 基础字节 token 必在 keep 集，中英几乎无 OOV）。

**产出**：`checkpoints/pruned_20260530`，**0.685B 参数**（transformer ≈0.47B），
reload + forward 验证通过（logits.shape 正确）。词表裁剪可逆。

### 3.2 Phase 2：离线知识蒸馏（进行中）

借鉴 MiniCPM 的 OPD 思路，分两步；本阶段为**离线 KD**（on-policy OPD 推迟到 Phase 3 与流式一起做）：

- **目标函数**：`L = α·CE(ref) + β·forward-KL(teacher‖student)`，α=β=0.5，温度 T=2.0（KD 项 ×T²）。
- **教师离线导出**（`run_teacher_export.py`）：9B 以 **4-bit（nf4）** 加载，对 train 集逐条 forward，
  导出每个 target 位置的 **top-64 logits**；old-id top-k 经 vocab_map 映射到 new-id、丢集外、
  在保留子集上重新 log_softmax 归一化、补齐到固定 K（补 id=0、logp=-30，避免 0×(-inf)=nan）。
- **师生对齐（头号风险的处理）**：教师 shard **同时存入学生 new-id 的 input_ids/labels**，
  训练时直接读盘、不再分词，杜绝"教师导出"与"学生训练"两次分词的位置漂移。
- **训练**（`run_distill.py`）：从剪枝底座加载，梯度检查点 + 8bit AdamW；
  **批前向（右 padding + attention_mask），仅 target 位置过 lm_head**（避开整 [B,L,110000] logits 的显存大头），
  全 batch 的 target 摊平成大 N 算 CE+KD；按 shard 迭代 + shard 内打乱（防 I/O 抖动）。
- **数据**：中英平行语料，口语/字幕为主 + 通用混合（约 7:3），来源
  `sentence-transformers/parallel-sentences-opensubtitles`(en-zh_cn) + `Helsinki-NLP/opus-100`(en-zh)；
  `split_corpus.py` 按唯一句对去重切 train/test（test 2000 对，防评测泄漏）。train 集 391446 条（双向）。

## 4. 实验与结果

### 4.1 训练超参演进（踩坑记录）

| 问题 | 现象 | 修复 |
|---|---|---|
| 开局疑似发散 | 单点 loss 12→15→23 | 实为单样本 loss 噪声；改 step 内均值 + 近 50 步滑动均值 avg50 看趋势 |
| lr 过高无 warmup | 常数 lr=2e-4 | lr→5e-5 + warmup(5%) + cosine 衰减 |
| 训练太慢 | 逐条前向 10.9s/step | 批处理（batch=32）：GPU 利用率 31%→97%，吞吐≈4×，1.4s/step |

### 4.2 5000-step（≈1.6 epoch）结果

- loss：19.2 → 2.47（avg50≈2.58），收敛健康，1h59m。
- 全测试集（4000 行）质量：**BLEU 2.82 / chrF 16.25（all）**；分方向 chrF en2zh 9.57、zh2en 18.72。
- **测量陷阱**：中文目标的 BLEU 默认按空格切词 → 整句中文当一个词 → BLEU 虚低≈0.84；
  须用 sacrebleu `tokenize='zh'`。chrF 字符级、跨语言可比，作为主参考。

### 4.3 失败模式分析（5000-step checkpoint）

定量统计 4000 条译文，三种欠训练症状（按危害排序）：

| 症状 | zh2en | en2zh | 说明 |
|---|---|---|---|
| **过早 eos（空输出）** | 33.5% | 37.1% | 第 0 步贪心即吐 eos；强相关短句（字幕超短目标多） |
| 退化死循环（撞 256 token） | 238 条 | 107 条 | 如 "100% of the 100% of…" 无限重复 |
| 中英混杂 | 8.8% | 11.8% | 先抄一个源 token 再切语言 |

**关键判别实验**：`--min-new 1` 强制至少生成 1 token 后——
- zh2en 空输出 33.5%→4%（能自然写下去，gen 均值 51 token），
- en2zh 仍 22% 出一个字（如"你"）就在第 2 步停。

**结论**：翻译能力确已学到（正常长度样例质量好，如 `ls Daddy in trouble?`→`爸爸有麻烦吗？`、
`There is a guy who came from Australia to see me.`→`有一个来自澳大利亚的人来见我。`），
三种症状均为 **1.6 epoch 欠训练**所致（eos 校准差、LM 弱、语言未分干净）。
低 BLEU 主因是"35% 空输出清零 + 参考译文偏意译"，非模型不会翻。

### 4.4 12000-step（≈4 epoch）实验：质量-epoch 曲线（500 子集，rep-penalty 1.3 + no-repeat-3）

| step | ~epoch | en2zh BLEU/chrF/均长 | zh2en BLEU/chrF/均长 | chrF_all |
|---|---|---|---|---|
| 3000 | 1 | 3.07 / 5.51 / 8.9 | 5.50 / 18.92 / 23.1 | 15.21 |
| 6000 | 2 | 4.04 / 6.39 / 7.3 | 5.09 / 18.31 / 18.3 | 15.02 |
| 9000 | 3 | 5.28 / **7.89** / 7.1 | 5.38 / **20.35** / 19.4 | **16.93** |
| 12000 | 4 | 3.32 / 6.58 / 7.0 | 5.40 / 19.40 / 19.9 | 15.88 |

- **质量在 9k 见顶、12k 回落**，与 loss 单调下降（19.4→1.69）相悖——非饱和，是后续诊断出的词表 bug 在作祟。
- **关键不要误读为"训练充分性"问题**：见 §4.5。

### 4.5 根因诊断：词表裁剪 bug 导致过早 eos（推翻"欠训练"定性）

5000/12000-step 的低分主因不是欠训练，而是 **Phase 1 词表裁剪选错 keep 集 + OOV→eos 映射**：

1. **现象**：en2zh 空输出（≤1 token）高达 42–47%（12k 最糟），中位生成仅 2–3 token；zh2en 32–37%。
   但正常长度样本翻译质量良好（如 `As for the others...`→`至于其他人，他们完全不知道她曾经离开过…`），
   能力在、被 eos 卡死。
2. **数据排除**：`inspect_corpus.py` 查 train 集——en2zh 目标空 0%、纯标点 0%、超短 0.1%，语料干净。
3. **OOV 定位**：`check_oov.py`——**en2zh 目标 token OOV 率 12.35%、句首 token OOV 20.62%**（zh2en 仅 3.36%/7.31%）。
   句首 OOV 样例全是高频中文合并 token：`希望你`/`晚安`/`什么事`/`我们知道`/`快点`…
4. **机理**：`run_teacher_export.py` 把 OOV 映射成 eos（第 49/97 行）→ OOV 目标 token 在训练标签里变成假 eos →
   CE 直接训出"句首吐 eos"；句中 12% OOV 抬高整体 eos 概率。**en2zh 中文 OOV 远高于 zh2en 英文 → 方向不对称由此而来。**
5. **预算浪费**：`vocab_coverage.py`——训练语料仅用 **74271** unique token（110k 绰绰有余），
   旧 keep 集却 **53.9%（59318）名额给了语料用不到的预留语言/校准 token**，同时**裁掉 23589 个语料必需 token**。
   根因：keep 集频次统计用的是"多语言校准语料"而非训练语料，且预留语言挤占中英。

**结论**：9k/12k 谁优无意义——都建在坏词表上。修复见 §6（重建词表→重切 embedding→重导教师→重训）。
这段"诊断把表面的训练充分性问题归因到词表工程 bug"本身有方法论价值，可作为论文的一个 insight。

### 4.6（原 4.4 设置）训练配置

- 12000 step，batch=32（有效 batch 128），每 3000 step 存中间 checkpoint；推理侧 rep-penalty 1.3 + no-repeat-3 压退化。

## 5. 工程与环境

- 算力：远程单卡（实测 32GB，非最初假设的 24GB），conda env `streamtrans`，Python 3.11，transformers 5.10.0.dev0。
- 本地 Windows Python 3.9 仅写代码（无 torch）。三节点 git 拓扑：本地 commit → GitHub → 服务器只 pull。
- 显存：批处理 batch=32 峰值 ~12.6GB/32GB；模型 fp32 ~2.7GB，单 checkpoint ~2.7GB。
- 国内下载坑：HF Xet 后端（cas-bridge.xethub.hf.co）国内不可达；权重缓存后用 `HF_HUB_OFFLINE=1` 离线加载绕开。
- 端侧可行性（已核实）：llama.cpp（`ggml_gated_delta_net`）与 MNN 3.4.1（Linear Attention 算子）均支持 Qwen3.5 线性注意力。

## 6. 待办（论文需补的实验）

- **【最高优先】修词表 bug 重做 Phase 1–2**（§4.5 根因）：
  1. 频次改用真实训练语料统计，强制保全部 74271 语料 token（中英 OOV≈0），预留语言只填剩余 ~35k；
  2. `run_teacher_export.py` OOV 兜底从 eos 改为非破坏 token；
  3. 从 `embed_original.pt` 按新 vocab_map 重切 embed/lm_head（transformer 权重不动）→ 新 pruned ckpt；
  4. 重新导出教师 logits（shard 存 new-id，词表变则必须重来——最贵一步）；5. 重训蒸馏。
- Phase 2 收尾：质量-epoch 曲线、最优 checkpoint 全集 BLEU/chrF（中文用 zh 分词）、与教师 9B / 未剪枝 2B 的质量差。
- 消融：减层策略、FFN 剪枝量、词表大小、α/β、top-k、温度对质量的影响。
- 标准基准：Flores-200 devtest（对外可比，注意字幕训练数据的域差）。
- Phase 3：wait-k prefix-to-prefix 流式 + on-policy OPD（reverse KL，师生 top-k 并集）；
  延迟指标 AL/LAAL，质量-延迟权衡曲线。
- Phase 4：量化 + 端侧导出（llama.cpp/MNN），端侧实测延迟/吞吐/内存。

## 7. 局限

- 第一版仅中英、字幕域为主，域外（书面、术语）未验证。
- 5000-step 为欠训练中间态；最终质量数以 Phase 2 收尾为准。
- 流式（wait-k/OPD）与端侧部署尚未做，是论文完整性的关键缺口。
