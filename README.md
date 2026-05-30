# StreamTrans —— 端侧多语言流式同声传译模型

训练一个 **~0.7B 参数、可移动端部署、低延迟流式** 的同声传译模型。
模态为 **文本→文本流式翻译**（语音由外部 ASR 处理）。

技术主线：以 **Qwen3.5-2B** 为学生底座，**结构化剪枝 + 知识蒸馏 + 结构调整** 压成端侧模型，
并用 **wait-k** 改造为真正的流式同传（prefix-to-prefix）。

> 详细设计见 [`docs/superpowers/specs/2026-05-28-streamtrans-design.md`](docs/superpowers/specs/2026-05-28-streamtrans-design.md)，
> 工程规范与红线见 [`CLAUDE.md`](CLAUDE.md)。

## 模型选型（已核实，勿擅自更换）

| 角色 | 模型 | 说明 |
|------|------|------|
| 学生底座 | `Qwen/Qwen3.5-2B` | 混合架构（SSM/Mamba 线性注意力 + 周期性 full attention），24 层，hidden 2048，FFN 6144，词表 248320，含 Vision Encoder |
| 教师 | `Qwen/Qwen3.5-9B` | 同系、**同词表 248320**，保证 logits 逐 token 对齐 |

起点与教师**必须同词表**，否则 logits 蒸馏退化为弱蒸馏。换型号需先改 `CLAUDE.md` 再改代码。

## 核心技术决策

1. **剪枝只动通用维度**：剥离 Vision Encoder + 减层 + 缩 FFN 中间维 + 缩 hidden；线性注意力 / full-attn 块整块保留或整块删，不剪其内部。
2. **词表裁剪按目标全集、可逆**：全词表 248320 仅嵌入≈0.5B，是预算主负担。按"中英为主 + 预留主要语言（中英日韩 + 主要欧洲语言）"裁到 ~110k，第一版只训中英；裁剪可逆（存 id 映射 + 原始 embedding），扩语言只补数据不动结构。
3. **蒸馏用 OPD**（借鉴 MiniCPM5）：on-policy（学生自生成序列）+ reverse KL + 师生 top-k logits 并集，针对同传的暴露偏差。
4. **流式先 wait-k 后进阶**：固定 k 跑通全链路，再上自适应策略。
5. **教师离线**：24GB 单卡装不下 9B 在线教师 + 训练态学生，教师软标签/生成数据离线落盘。

## 目录结构

```
configs/                  每阶段一个 yaml，超参与路径全在此，代码不硬编码
src/streamtrans/
  data/                   数据加载（公开语料 + 教师生成）
  prune/                  剥视觉塔、结构化剪枝、词表裁剪、模型勘探
  distill/                教师离线导出、OPD 蒸馏
  streaming/              wait-k 数据构造、流式微调、流式解码器
  eval/                   质量(BLEU/COMET) + 延迟(AL/LAAL)
  export/                 量化 + 端侧导出
scripts/                  每阶段一个入口 run_{stage}.py
docs/superpowers/specs/   设计文档
docs/model_inspection/    模型勘探报告与结论
data/  checkpoints/       语料与产物（.gitignore，不入库）
```

## 环境与验证（Phase 0）

> 训练/勘探在远程 **24GB Linux 机**（Python 3.10+）执行；本地 Windows 为 Python 3.9，仅写代码。

```bash
# 建议 Python 3.10+ 虚拟环境（venv 或 conda）
pip install -e . && pip install -r requirements.txt
# 若 transformers 不识别 qwen3_5：pip install git+https://github.com/huggingface/transformers.git
# 中国网络可设镜像：export HF_ENDPOINT=https://hf-mirror.com

pytest -v                                             # 预期 10 passed

python scripts/run_inspect.py --model Qwen/Qwen3.5-2B # 模型勘探，产出 docs/model_inspection/*.json
python scripts/run_inspect.py --model Qwen/Qwen3.5-9B
# 回填 docs/model_inspection/CONCLUSIONS.md 后，方可进 Phase 1
```

## 路线图

- **Phase 0（当前）**：项目地基、配置系统、AL 延迟指标、模型勘探、词频统计、平行语料管线。
- **Phase 1**：剥视觉塔 + 词表裁剪 + 结构化剪枝 + 剪枝恢复（待勘探结论回填后启动）。
- **Phase 2**：教师离线导出 + OPD 蒸馏。
- **Phase 3**：wait-k 流式数据构造 + 流式微调 + 流式解码。
- **Phase 4**：量化 + 端侧导出 + 质量/延迟联合评测。
