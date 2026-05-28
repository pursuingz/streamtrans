# Qwen3.5 模型勘探结论

> 在服务器跑完 `python scripts/run_inspect.py --model Qwen/Qwen3.5-2B`（及 9B）后，
> 用 `*.json` 报告填写下列各项。这是 Phase 1（剥视觉塔/词表裁剪/结构化剪枝）的前置依据。
> **本文件填完前不要进入 Phase 1。**

- 勘探日期：____
- transformers 版本：____（`pip show transformers`）
- 运行环境：____（服务器 / Python 版本 / GPU）

---

## 1. 纯文本 LLM 参数量
- Qwen3.5-2B total = ____ B
- 其中 vision = ____ B
- **纯文本 = total − vision = ____ B**
- （从 config 已知 `tie_word_embeddings=true`，嵌入仅一份）

## 2. 嵌入层
- `embed_tokens` 参数量 = ____ M（预期 ≈248320×2048≈508M）
- lm_head 是否与 embed_tokens tie：____（应为 是）

## 3. Vision Encoder（剥离用）
- 视觉塔的 module 前缀名 = ____（如 `visual.` / `model.visual.` / `vision_tower.`，以 `module_names` 实测为准）
- 视觉塔参数量 = ____ B

## 4. text 主干 module 命名（剪枝按真名写）
- 层容器路径（如 `model.language_model.layers` / `model.layers`）= ____
- SSM/线性注意力块 module 路径 = ____
- full attention 块 module 路径 = ____
- FFN/MLP module 路径 = ____
- （config 已知 24 层，`layer_types`=[linear,linear,linear,full]×6）

## 5. MTP 头
- MTP module 命名路径 = ____（config: `mtp_num_hidden_layers=1`）
- MTP 参数量 = ____ M
- 剥离 / 保留决策 = ____（同传不需多 token 预测，倾向剥离；待定）

## 6. transformers 加载情况
- `loaded via:` 实际成功的 Auto 类 = ____（脚本打印 + json 的 `loader_class`）
- PyPI 稳定版是否够，还是需 GitHub dev 版 = ____
- → 解除 spec §8 风险2 的「能否加载」一半

## 7. 端侧可行性预判（spec §8 风险2 的另一半）
- llama.cpp 是否支持该 SSM 变体（`linear_conv_kernel_dim`、`mamba_ssm_dtype`）= ____
- mrope（多模态 RoPE）剥视觉塔后是否退化为普通 RoPE = ____
- MTP 头对端侧导出的影响 = ____
- **初步结论：用 Qwen3.5 做端侧是否可行 / 是否需回退标准架构 = ____**

---

## 9B 教师（蒸馏用，简要）
- total = ____ B，词表 = ____（应与 2B 同为 248320，确认可对齐 logits）
- 能否同样加载 = ____

## 给 Phase 1 的要点（一句话总结）
____
