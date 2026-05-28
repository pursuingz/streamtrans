# StreamTrans 设计文档：端侧多语言流式同传模型

- 日期：2026-05-28
- 状态：待 Will review
- 模态：文本→文本流式同声传译（语音由外部 ASR 处理）

---

## 1. 目标与约束

| 维度 | 取值 | 来源 |
|---|---|---|
| 任务 | 流式文本→文本同声传译（SimulMT） | 用户确认 |
| 参数量 | ~0.7B（中英为主，预留主要语言） | 用户确认 |
| 部署 | 移动端，低延迟流式（0.7B 4bit≈350MB） | 用户确认 |
| 语言 | 中英为主，词表按"中英日韩+主要欧洲语言"全集预留；第一版只训中英 | 用户确认 + §6 |
| 训练资源 | 单卡 24GB | 用户确认 |
| 压缩策略 | 剪枝 + 蒸馏 + 结构调整 三者结合 | 用户确认 |
| 读写策略 | 先 wait-k，后自适应 | 用户确认 |
| 数据 | 公开语料 + 教师生成混合 | 用户确认 |

**核心立场**：先用最朴素的 wait-k 把「剥视觉塔→剪枝→词表裁剪→蒸馏→流式→量化」整条链路打通，拿到端到端可部署的 0.5B 模型；再在流式策略上迭代研究。先有能跑的全链路，再谈高级。

---

## 2. 模型选型（已联网核实，2026-05-28）

| | Qwen3.5-2B（起点/学生底座） | Qwen3.5-9B（教师） |
|---|---|---|
| model_type | `qwen3_5`（`Qwen3_5ForConditionalGeneration`） | 同系 |
| transformers | **需 4.57.0.dev0+（dev 版，可能从 GitHub 装）** | 同 |
| 架构 | 混合：SSM/Mamba 风格线性注意力（含 conv kernel）+ 周期性 full attention | 同类混合架构 |
| 层结构 | 24 层，`layer_types`=[linear,linear,linear,**full**]×6（每 4 层 1 个 full attention） | 32 层，同节律 |
| hidden / FFN | 2048 / 6144 | 4096 / — |
| 线性注意力 | key/value head_dim 128，各 16 heads，conv_kernel 4，ssm_dtype fp32 | 同类 |
| full attention | 8 heads / 2 KV(GQA) / head_dim 256 | GQA |
| 额外结构 | **MTP 头**（`mtp_num_hidden_layers:1`，multi-token prediction） | 同 |
| 词表 / tie | 248,320 / **tie_word_embeddings=true（嵌入仅一份）** | 248,320（同词表） |
| 语言 | 201 | 201 |
| 多模态 | 带 `vision_config`（text 部分在 `text_config` 子树） | 带 vision_config |
| 版本 | Base + Instruct | Base + Instruct |

> 以上为 2026-05-29 直接拉取 `config.json` 核实的一手事实（修正了早期二手描述「Gated DeltaNet」——实为带 conv 的 SSM/Mamba 风格线性注意力）。

两点关键：
- **同词表 248,320** → 教师 logits 与学生逐 token 对齐，OPD 软标签蒸馏成立。
- **SSM/线性注意力是流式利好**：SSM/Mamba 风格线性注意力维护固定大小循环状态（RNN 式），流式增量推理时显存/计算近似恒定，端侧长时同传不爆内存。这是该底座相对标准 Transformer 的天然优势（且 RNN 式状态比 DeltaNet 论点更强）。每 4 层混入的 full attention 层仍有 KV cache，但占比仅 1/4。

---

## 3. 四阶段流水线

```
阶段0  起点         Qwen3.5-2B（Base 或 Instruct）
   │
阶段1  结构调整+剪枝  剥 Vision Encoder → 词表裁剪 → 结构化剪枝(减层/缩FFN/缩hidden) → ~0.5B
   │   只动通用维度，DeltaNet/GatedAttn 块整块保留或整块删
   ▼
阶段2  OPD 蒸馏      教师 Qwen3.5-9B 离线导出 top-k logits/生成数据；
   │   学生在自生成序列上做 reverse KL 蒸馏（on-policy），恢复质量
   ▼
阶段3  流式改造      wait-k prefix-to-prefix 微调；训练时随机采样 k；
   │   预留自适应策略扩展点
   ▼
阶段4  量化导出      4bit (GGUF/AWQ) → llama.cpp / MNN，端侧部署
```

每阶段产出 1 个 checkpoint + 1 份评测报告；阶段间解耦，可独立重跑。

### 阶段1 —— 结构调整 + 结构化剪枝
- **剥视觉塔**：移除 Vision Encoder 及相关投影层，只保留纯文本 LLM。最干净安全的一刀结构调整，白送一块参数。
- **词表裁剪（按目标语言全集，可逆，见 §6）**：按"中英为主 + 预留主要语言（中英日韩+主要欧洲语言）"的**最终全集**统计 token 频次裁词表，相应裁剪 embedding/lm_head 行。第一版只用中英数据训练，但词表已为未来语言预留位置——扩语言时只补数据继续训练，**不动模型结构**。裁剪做成可逆：保存「原 token id → 新 token id」映射，保留 Qwen3.5-2B 这些 token 的原始预训练 embedding，未来若需扩词表可取回预训练权重而非随机初始化。
- **结构化剪枝（只动通用维度）**：
  - 减层：按层重要性（如基于激活/梯度的得分）删整层。
  - 缩 FFN 中间维（6144→目标）。
  - 缩 hidden_size（连带调整各层投影）。
  - **DeltaNet / GatedAttn 块整块保留或整块删除，不剪其线性注意力内部维度。**
- 剪枝后立即接一轮短蒸馏恢复（剪枝-恢复循环），避免性能断崖。
- 24GB 适配：剪枝阶段以前向重要性评估为主，训练态恢复用梯度检查点 + 8bit 优化器。

### 阶段2 —— OPD 知识蒸馏（借鉴 MiniCPM5）
- **教师离线导出**（`offline`）：9B 教师 4bit 量化（权重≈5GB），离线只跑前向，对训练集导出每个位置 top-k (token, logit) 落盘。只存 top-k，省存储省显存。
- **OPD 训练**：
  - on-policy：在**学生自己生成**的序列上算损失（纠正暴露偏差，对同传残缺前缀场景关键）。
  - 损失：在师生 **top-k logits 并集**上算 **reverse KL**；可加 CE(硬标签) 作锚。
  - 只加载学生 + 离线教师 logits，24GB 可行。
- 产出：质量恢复后的 ~0.5B 离线翻译模型（尚非流式）。

### 阶段3 —— wait-k 流式改造
- **数据变换**（`waitk_dataset`）：把整句平行语料切成 (源前缀, 应输出译文前缀) 对。这是同传训练数据的本质，区别于普通整句对。
- **训练**（`waitk_trainer`）：prefix-to-prefix 微调，训练时随机采样 k（wait-1/3/5/7），让模型适应不同延迟档位。
- **流式解码器**（`decoder`，推理时）：维护源缓冲区，按 wait-k 策略决定 READ（等输入）/ WRITE（输出 token）。利用 DeltaNet 循环状态做高效增量推理。
- 预留自适应策略（单调注意力 / 策略网络）扩展点，作为「进阶」阶段。

### 阶段4 —— 量化与端侧导出
- 4bit 量化（AWQ 或 GGUF Q4），导出 llama.cpp / MNN 格式。
- 注意 DeltaNet 在端侧推理框架的算子支持——若框架不支持，需自定义算子或回退（实现阶段核实，列为风险）。

---

## 4. 目录结构

```
trany_1/
├── CLAUDE.md
├── configs/                       # 每阶段一个 yaml
│   ├── prune.yaml
│   ├── distill.yaml
│   ├── streaming.yaml
│   └── export.yaml
├── src/streamtrans/
│   ├── data/
│   │   ├── parallel_corpus.py     # WMT/OPUS/CCMatrix 接入、清洗、长度过滤
│   │   └── teacher_gen.py         # 教师把单语料翻译成平行数据
│   ├── prune/
│   │   ├── strip_vision.py        # 剥离 Vision Encoder
│   │   ├── vocab_prune.py         # 词表裁剪 + embedding/lm_head 裁剪
│   │   └── structured.py          # 减层/缩FFN/缩hidden（只动通用维度）
│   ├── distill/
│   │   ├── offline_logits.py      # 教师离线导出 top-k logits
│   │   └── opd_trainer.py         # on-policy + reverse KL + top-k 并集
│   ├── streaming/
│   │   ├── waitk_dataset.py       # prefix-to-prefix 样本构造
│   │   ├── waitk_trainer.py       # wait-k 微调
│   │   └── decoder.py             # 流式增量解码器（READ/WRITE 控制）
│   ├── eval/
│   │   ├── quality.py             # BLEU / COMET
│   │   └── latency.py             # AL / LAAL 等同传延迟指标
│   └── export/
│       └── quantize.py            # 4bit 量化 + GGUF/MNN 导出
├── scripts/
│   └── run_{prune,distill,streaming,export}.py
├── data/                          # 语料与中间产物（.gitignore）
├── checkpoints/                   # 各阶段产物（.gitignore）
└── docs/superpowers/specs/
```

---

## 5. 数据流

```
公开平行语料(WMT/OPUS) ──┐
                         ├─→ 清洗/对齐/长度过滤 ─→ 训练集 ──┬─→ 词表频次统计(阶段1词表裁剪)
单语语料 ─→ 教师翻译 ─────┘                                ├─→ 教师离线 top-k logits(阶段2)
                                                          ├─→ OPD on-policy 蒸馏(阶段2)
                                                          └─→ wait-k 前缀切分(阶段3)
```

同传训练数据特殊点：阶段3 的样本不是整句对，而是 (源前缀, 译文前缀) 流，由 `waitk_dataset` 从整句对动态生成。

---

## 6. 参数预算与可扩展词表（关键约束）

**总预算定为 ~0.7B**（用户确认，从 0.5B 放宽以换取语言可扩展性）。

全词表 248,320 × hidden 2048 ≈ **508M 参数仅嵌入层**，是小模型的主要负担。嵌入越大，留给 transformer 越少：

| 语言全集 | 词表 | 嵌入参数 | 0.5B 下留给 transformer | 0.7B 下留给 transformer |
|---|---|---|---|---|
| 中英 | ~50–60k | ~0.11B | ~0.39B | ~0.59B |
| 中英日韩 | ~80k | ~0.16B | ~0.34B | ~0.54B |
| **中英+主要欧洲语言（选定）** | **~110k** | **~0.23B** | ~0.27B | **~0.47B** |
| 全 201 | 248k | ~0.5B | ≈0 ❌ | ~0.2B |

**选定方案**：总预算 0.7B，词表按"中英为主 + 预留主要语言（中英日韩+主要欧洲语言）"≈110k 一次性裁定，transformer ≈0.47B。

**可扩展性设计（关键）**：词表按**最终全集**而非第一版语言裁定，第一版只用中英数据训练。这样：
- 扩语言 = 补该语言平行语料 + 继续训练，**永不改动模型结构**。
- 词表裁剪可逆（保存 id 映射 + 原始 embedding），即便将来真要超出预留全集扩词表，新 token 也能从 Qwen3.5-2B 取回预训练 embedding，质量起点远好于随机初始化。

**本质矛盾仍在**：语言全集越大，同预算下 transformer 越小、单语种质量越弱。0.7B + 110k 词表是当前选定的平衡点。

---

## 7. 评测

同传必须**同时报质量与延迟**：
- 质量：BLEU + COMET（在不同延迟档位下）。
- 延迟：AL（Average Lagging，同传标准指标）、LAAL。
- **核心产出图**：延迟-质量曲线——横轴 AL，纵轴 BLEU，不同 wait-k 一条曲线，与「离线全句翻译」上界对比。这是判断「是不是真同传」的依据。

研究对比实验：
1. 原生 Qwen3.5-0.8B（若存在最小档）vs 我们从 2B 剪出的 0.5B —— 验证剪枝路线优于原生小模型。
2. 蒸馏前 vs 蒸馏后 —— 验证 OPD 恢复有效。
3. OPD（on-policy）vs 普通 off-policy KD —— 验证 on-policy 对同传暴露偏差的增益。
4. 不同 wait-k 的延迟-质量曲线。

---

## 8. 风险与现实预期

1. **≤0.5B + 多语言质量吃紧**：物理限制，非代码问题。缓解：先收窄语言对跑通，再扩；语言范围做配置开关（见 §6）。
2. **SSM 线性注意力端侧算子支持未知**：架构实为带 conv 的 SSM/Mamba 风格 + 周期 full attention + MTP 头。llama.cpp 对 Mamba 类有部分支持，但 Qwen3.5 这个特定变体（conv_kernel、mrope、MTP）能否被端侧框架转换/运行待实测。若不支持需自定义算子或回退到标准架构起点。实现阶段第一批要验证的事。
3. **9B 教师离线生成慢**：4bit 量化教师或降到更小教师档；配置留开关。
4. **wait-k 在语序差异大的方向（如中↔英）固定 k 效果有限**：正是「进阶」阶段要解决的，结构预留自适应扩展点。
5. **2B 纯文本部分确切参数量待核实**：剥视觉塔后的实际规模决定剪枝幅度，实现第一步拿真实 config 确认。
6. **24GB 训练偏紧**：默认梯度检查点 + 8bit 优化器；不足时降 batch + 梯度累积，不改模型规模。

---

## 9. 实现顺序（每阶段 smoke test 先行）

1. 环境 + 拉 Qwen3.5-2B/9B，核实纯文本参数量、DeltaNet 端侧算子支持。
2. 数据管线（公开语料接入 + 教师生成）。
3. 阶段1：剥视觉塔 → 词表裁剪 → 结构化剪枝 → 剪枝恢复。
4. 阶段2：教师离线 logits → OPD 蒸馏。
5. 阶段3：wait-k 数据 + 微调 + 流式解码器。
6. 评测：延迟-质量曲线。
7. 阶段4：量化导出 + 端侧验证。
