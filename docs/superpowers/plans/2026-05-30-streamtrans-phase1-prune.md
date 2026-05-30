# Phase 1 实现计划 —— 结构化剪枝（2B → ~0.69B 学生底座）

> 前置：`docs/model_inspection/CONCLUSIONS.md` 已回填（2026-05-30）。
> 目标取向已定：**中庸 12 层 / FFN 3072 / hidden 2048 不变 / 词表 110k**。
> 产物：`checkpoints/pruned_<date>/`，一个可加载、可前向、结构有效的纯文本因果 LM。
> **质量恢复（OPD 蒸馏）属 Phase 2**；Phase 1 只做「外科手术 + 轻量 heal + 结构验证」。

## 0. 目标 student 配置（已锁定）

| 维度 | 现状 2B | Phase 1 student | 手段 |
|---|---|---|---|
| 词表 | 248320 | 110000 | 裁词表（可逆） |
| 视觉塔 | 0.331B | 删 | 剥 `model.visual` |
| hidden | 2048 | 2048 | **不动**（绕开 head_dim 256 / mrope_section 风险） |
| 层数 | 24 `[L,L,L,F]×6` | 12 `[L,L,L,F]×3` | 减层 |
| FFN | 6144 | 3072 | 缩 FFN 中间维 |
| embed(tied) | 0.509B | 0.225B | 随词表 |
| transformer | 1.373B | ≈0.46B | 减层+缩FFN |
| **总计** | 1.882B | **≈0.69B** | |

不变量：linear/full 注意力块**整块保留或整块删，不剪内部**（`linear_attn.*`、`self_attn.*` 子模块原样搬运）；MTP 不在前向图，无需处理。

---

## 前置探针 P0（服务器执行，决定后续代码形态）

本地无法确认，**Phase 1 编码前必须先在服务器跑一次**，把结果回报：

```python
import transformers as tf
print([n for n in dir(tf) if "Qwen3_5" in n or "Qwen3.5" in n])
# 关注是否存在 Qwen3_5ForCausalLM / Qwen3_5TextModel / Qwen3_5TextConfig
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
print(type(cfg.text_config))           # 纯文本子配置类名
```

- **分支 A（存在纯文本 CausalLM 类）**：student 直接 build 成纯文本因果 LM（`Qwen3_5ForCausalLM` from `text_config`），最干净。
- **分支 B（只有 ConditionalGeneration）**：student 仍用 `Qwen3_5ForConditionalGeneration` 的壳，但 `del model.model.visual`、清空 `vision_config` 与 image/video token，前向只走文本路径。次优但可行。

计划其余部分对两个分支通用，仅 §1 落地方式不同。

---

## 任务拆解（src/streamtrans/prune/，每个纯函数 + 可单测）

### Task 1 — 剥视觉塔 `strip_vision.py`
- 输入：完整 `Qwen/Qwen3.5-2B`（real weights，非 meta；需下载，HF_ENDPOINT 镜像）。
- 产出：纯文本 backbone（`model.language_model`）+ `lm_head` + 纯文本 config。
- 分支 A：用 `text_config` 实例化 `Qwen3_5ForCausalLM`，搬运 `embed_tokens / layers / norm / lm_head` 权重。
- 分支 B：原模型 `del visual`，改写 config 去掉 vision 字段。
- 验证：参数量 = 1.882B − 0.331B = 1.882B→1.551B（含 24 层、110k 前的全词表）；前向一句话 logits 非 NaN。

### Task 2 — 减层 `prune_layers.py`
- 策略 `block_uniform`：原 24 层 = 6 个 `[L,L,L,F]` 块，**端点+中间均匀保留第 0/2/5 块**（含首块与末块，保留靠近输入与输出的层），即原 idx `{0,1,2,3, 8,9,10,11, 20,21,22,23}`（3 个完整 `[L,L,L,F]` 块）。
- 新层重编号 0–11，full-attn 落在新 idx 3/7/11，`layer_types` 与 `full_attention_interval=4` 自洽。
- 权重：直接搬保留层（含 `linear_attn`/`self_attn` 内部，不改）。
- 纯函数 `select_layer_indices(num_layers=24, target=12, interval=4) -> list[int]`，可单测（不需 GPU）。
- 备选策略留接口：`importance`（按层输出扰动选层），Phase 1 先用 `block_uniform`。

### Task 3 — 缩 FFN `prune_ffn.py`
- 每层 MLP：`gate_proj [6144,2048]`、`up_proj [6144,2048]`、`down_proj [2048,6144]`。
- 选 3072 个中间神经元：`magnitude` = 按 `||gate_proj 行||₂ + ||up_proj 行||₂` 取 top-3072。
- 切片：`gate_proj`/`up_proj` 留对应行，`down_proj` 留对应列。SiLU 门控逐神经元独立，切片不破坏语义。
- 纯函数 `select_ffn_neurons(gate_w, up_w, keep=3072) -> idx`，可单测（小随机张量）。
- 备选 `activation`（校准集上按激活均值选）留接口，需校准数据，Phase 1 用 magnitude 起步。

### Task 4 — 裁词表 `prune_vocab.py`（可逆，复用已写的 `data/vocab_stats.py`）
- keep 集 = 全部 special/added token + 覆盖 `reserve_languages` 样本的 token 频次补满到 110000（`select_keep_tokens` 已实现）。
- 校准语料：中英 + 日韩法德西俄各若干（只为统计 token 命中，不需平行）。`scripts/build_calib_corpus.py` 流式拉 wikimedia/wikipedia 各语言单语文本 → `data/calib_multiling.txt`（gitignore）。
- 切 `embed_tokens` 行（tied → `lm_head` 自动跟随）。
- **可逆产物**：`vocab_map.json`（old_id → new_id）+ `embed_original.pt`（原始 248320×2048 行，扩语言时按 map 回填新 token）。
- 纯函数部分可单测；切权重部分用小词表 mock 验证 round-trip。

### Task 5 — 组装 + 落盘 `assemble_student.py` / `scripts/run_prune.py`
- 串起 Task1–4：strip → 减层 → 缩FFN → 裁词表，写出 `checkpoints/pruned_<date>/`（config.json + safetensors + vocab_map.json + embed_original.pt）。
- `--smoke`：用极小校准样本跑通全链路，断言最终参数量 ∈ [0.66B, 0.72B]、模型可 `from_pretrained` 再加载、前向 logits 形状 = [B,T,110000] 且非 NaN。
- config 不硬编码，全读 `configs/prune.yaml`。

### Task 6 — 轻量 heal（可选，Phase 1 收尾）
- 剪枝后结构断裂，做**极短**的 LM 续训（中英 clean 文本，几百 step）让权重重新自洽，避免把「坏初始化」带进 Phase 2 蒸馏。
- 不追质量，只追「不崩」；真正质量恢复交 Phase 2 OPD。
- 若时间紧可跳过，直接进 Phase 2（蒸馏本身会 heal）。**默认做最小 heal**。

---

## 验证（服务器，24GB）
```bash
pytest -v tests/test_prune_layers.py tests/test_prune_ffn.py tests/test_prune_vocab.py  # 纯函数，GPU-free
python scripts/run_prune.py --config configs/prune.yaml --smoke                          # 小数据全链路
python scripts/run_inspect.py --model checkpoints/pruned_<date>                          # 复用勘探脚本核参数量≈0.69B
python scripts/run_prune.py --config configs/prune.yaml                                  # 全量
```

## 风险与对策
1. **纯文本类是否存在**（P0 探针）→ 分支 A/B 都已设计，不阻塞。
2. **mrope text-only 退化**：剥视觉后 position_ids 应退为 1D，但实现走 mrope 分支；Task1 验证前向，若异常则在 config 固定 `mrope_section` 走纯时间段。
3. **tied 切词表**：切 `embed_tokens` 后须确认 `lm_head` 同步（tie 关系不被切片破坏）；Task4 round-trip 测试覆盖。
4. **减层/缩FFN 后质量塌陷**：Phase 1 不解决，靠 Phase 2 OPD；Task6 最小 heal 防止初始化过差。
5. **端侧框架支持**（CONCLUSIONS §7，已核实 2026-05-30）：llama.cpp（`ggml_gated_delta_net`）与 MNN 3.4.1（Linear Attention 算子全后端）**均已支持** Qwen3.5 线性注意力，风险已降级。**Phase 1 硬约束**：剪枝后 config 必须仍被转换器识别为 qwen3_5 架构——保留 `model_type=qwen3_5_text` / `layer_types` 节律 / `linear_attn` 维度字段，只改层数/FFN/词表，不重命名模块、不动 linear_attn 内部。Phase 4 须实测 GGUF+MNN 转换跑通。

## 产物与解耦
- 输出 `checkpoints/pruned_<date>/`（不入 git）+ `vocab_map.json`/`embed_original.pt`（可逆凭据）。
- Phase 2（OPD 蒸馏）以此 checkpoint + 9B 离线 logits 为输入，阶段间靠 checkpoint+config 解耦。
