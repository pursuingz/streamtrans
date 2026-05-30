# Qwen3.5 模型勘探结论

> 数据来源：`docs/model_inspection/Qwen_Qwen3.5-2B.json`、`Qwen_Qwen3.5-9B.json`
> meta device 实例化，参数量为精确 numel 计数（不依赖权重下载）。
> 这是 Phase 1（剥视觉塔/词表裁剪/结构化剪枝）的前置依据。

- 勘探日期：2026-05-30
- transformers 版本：5.10.0.dev0
- 运行环境：远程 24GB Linux 机，Python 3.11.15
- 实际加载类：**`AutoModelForImageTextToText`**（CausalLM 不认 `Qwen3_5ForConditionalGeneration`）

---

## 1. 纯文本 LLM 参数量（2B）
- total = **2.213B**
- 其中 vision (`model.visual`) = **0.331B**
- **纯文本 = total − vision = 1.882B**
- 构成：embed 0.509B + 24 层 1.373B + 末层 norm（~2k）
- `tie_word_embeddings=true`，嵌入仅一份，lm_head 复用

## 2. 嵌入层（2B）
- `embed_tokens` = **508.56M**（=248320×2048，与预测 ≈508M 吻合）
- lm_head 与 embed_tokens **tie = 是**（lm_head 额外参数 0）
- → 词表是预算第一大头，裁词表收益最直接

## 3. Vision Encoder（剥离用）
- 视觉塔前缀 = **`model.visual.`**（含 `patch_embed / blocks.{0-23} / merger`）
- 参数量 = **0.331B**
- 剥离即删 `model.visual` 整棵子树 + 配置里 `vision_config`、`image/video/vision_start/vision_end_token_id`

## 4. text 主干 module 命名（剪枝按真名写）
- 层容器路径 = **`model.language_model.layers`**（注意多一层 `language_model`，非 `model.layers`）
- 嵌入 = `model.language_model.embed_tokens`；末层 norm = `model.language_model.norm`；`model.language_model.rotary_emb`
- **linear attention 层**（18 层，idx 0,1,2,4,5,6,…非 4 的倍数-1）：`layers.{i}.linear_attn.{in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, conv1d, norm, out_proj, act}`
- **full attention 层**（6 层，idx 3,7,11,15,19,23，`full_attention_interval=4`）：`layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}`
- **每层 MLP**：`layers.{i}.mlp.{gate_proj, up_proj, down_proj}`（intermediate 6144）
- 每层两个 norm：`input_layernorm`、`post_attention_layernorm`
- 层节律：`[linear, linear, linear, full] × 6`
- 注意力规格：8 heads / 2 KV heads（GQA）/ head_dim 256；linear: 16 key+16 value heads, head_dim 128, conv kernel 4
- **层内参数分布**：MLP ≈906M（66%）、注意力（linear+full）≈467M → **缩 FFN 杠杆最高**

## 5. MTP 头
- **实例化图里没有 MTP 模块**（param=0，module_names 无 mtp）。config 虽有 `mtp_num_hidden_layers=1`，但 `Qwen3_5ForConditionalGeneration` 未构建。
- **决策：无需剥离**（本就不在前向图里），省一桩工作。

## 6. transformers 加载情况
- 成功类 = **`AutoModelForImageTextToText`**（`loader_class` 已记录）
- PyPI 稳定版不够，需 5.10.0.dev0（GitHub dev）；服务器已装好
- → spec §8 风险2「能否加载」**已解除**

## 7. 端侧可行性预判（spec §8 风险2 的另一半）—— 已联网核实（2026-05-30），风险大幅下降
- **llama.cpp：已支持。** 已实现 `ggml_gated_delta_net` + `build_mamba2_layer`（CUDA/Vulkan 优化），HF 已有 `Qwen3.5-2B-GGUF`，issue tracker 有人实跑 Qwen3.5（#20225）。需用最新版以含新算子。
- **MNN：已支持，且专门适配。** MNN **3.4.1** 核心主题之一即「Qwen3.5 Model Support with Linear Attention」：新增 Linear Attention 算子覆盖 CPU/Metal/OpenCL/Vulkan，支持线性注意力循环状态 disk load/store；另有 Arm KleidiAI × MNN × Qwen 边缘集成。**MNN 是更对口的移动端目标。**
- **架构吻合确认**：社区把 Qwen3.5 线性注意力描述为 **Gated DeltaNet，线性:全 = 3:1**，与我们 config 的 `full_attention_interval=4`（3 linear+1 full）一致 → 两引擎支持的正是本架构。
- **mrope**：剥视觉后纯文本 position_ids 退为 1D，理论等价普通 RoPE 时间段；Task1 验证 text-only 前向。
- **MTP**：不在前向图，端侧导出无影响。
- **残留约束（重要）**：剪枝后的 student **config 必须仍被转换器识别为 qwen3_5 架构**——保留 `model_type=qwen3_5_text`、`layer_types` 节律、`linear_attn` 维度字段，只改层数/FFN/词表这些标准维度。**不得重命名模块或改动 linear_attn 内部**，否则 GGUF/MNN 转换器会不认。
- **结论**：训练侧用 Qwen3.5 可行；端侧落地风险从「未知」降为「可控」——两大引擎均已支持。**Phase 4 仍须实测转换一遍**（剪枝后的非标准层数/FFN 走一遍 GGUF + MNN 转换并跑通），但不再是项目级阻断风险。

---

## 9B 教师（蒸馏用）
- total = **9.41B**，词表 = **248320**（**与 2B 同，logits 可逐 token 对齐 ✓**）
- eos=248044、image/video/vision token id 与 2B 完全一致 → tokenizer 对齐
- hidden 4096 / FFN 12288 / 32 层 / `[L,L,L,F]×8`；`tie_word_embeddings=false`（embed 1.017B + lm_head 1.017B）
- vision `model.visual` = 0.456B；同样 `AutoModelForImageTextToText` 加载成功
- 离线导出 top-k logits 无障碍（注意：9B 全精度装不下训练态，蒸馏走离线落盘，符合既定方案）

## 给 Phase 1 的要点（一句话总结）
保持 hidden=2048 不动（绕开 head_dim/mrope 风险），靠 **词表 248320→~110k（embed 509M→225M）+ 减层 + 缩 FFN** 三招把纯文本 1.882B 压到 ~0.7B（transformer ≈0.47B）；剥 `model.visual`，MTP 无需处理；剪枝按 `model.language_model.layers.*` 真实命名写，linear/full 层整块保留或整块删、不动其内部。
