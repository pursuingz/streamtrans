# StreamTrans —— 端侧多语言流式同传模型

## 项目目标
训练一个 **~0.7B 参数、可移动端部署、低延迟流式** 的同声传译模型。
模态：**文本→文本流式翻译**（语音交给外部 ASR）。
技术主线：以 Qwen3.5-2B 为起点，结构化剪枝 + 知识蒸馏 + 结构调整，做成端侧 0.5B 模型，并用 wait-k 改造为真正的流式同传。

## 模型选型（已核实，勿擅自更换）
- **起点（学生底座）**：`Qwen/Qwen3.5-2B`（混合架构：Gated DeltaNet 线性注意力 + Gated Attention；24 层，hidden 2048，FFN 6144，词表 248320，多模态带 Vision Encoder）
- **教师**：`Qwen/Qwen3.5-9B`（同系、**同词表 248320**，保证 logits 逐 token 对齐）
- 起点与教师**必须同词表**，否则 logits 蒸馏退化为弱蒸馏。换型号需先改本文件再改代码。

## 核心技术决策（改动需先改本文档）
1. **剪枝只动通用维度**：剥离 Vision Encoder + 减层 + 缩 FFN 中间维 + 缩 hidden。DeltaNet / GatedAttn 块整块保留或整块删，**不剪线性注意力内部**（规避适配风险）。
2. **词表裁剪按目标全集、可逆**：全词表 248320 仅嵌入≈0.5B，是预算主负担。按"中英为主+预留主要语言（中英日韩+主要欧洲语言）"全集裁到 ~110k，第一版只训中英数据。裁剪可逆（存 id 映射 + 保留原始 embedding），扩语言时只补数据不动结构。总预算 ~0.7B，transformer ≈0.47B。
3. **蒸馏分两步**（借鉴 MiniCPM5 的 OPD）：
   - **Phase 2 = 离线 KD**：9B 教师离线导出 top-k logits，`L = CE(ref) + forward-KL(teacher‖student)`，heal 剪枝损伤、做成强离线（全句）中英翻译器。教师软标签落盘，训练时只读盘。
   - **Phase 3 = on-policy OPD + wait-k**：on-policy（学生前缀自生成序列）+ reverse KL + 师生 top-k 并集，与流式同改造一起做——暴露偏差在流式逐前缀解码下才最严重，故 on-policy 并入此阶段。
4. **流式先 wait-k 后进阶**：prefix-to-prefix 训练，固定 k 跑通全链路后再上自适应策略；on-policy OPD 在此阶段叠加（见决策 3）。
5. **教师离线**：24GB 单卡装不下 9B 教师在线 + 训练态学生。教师软标签/生成数据**离线落盘**。

## 目录结构约定
```
configs/        每阶段一个 yaml，超参与路径全在此，代码不硬编码路径
src/streamtrans/
  data/         数据加载（公开语料 + 教师生成）
  prune/        剥视觉塔、结构化剪枝、词表裁剪
  distill/      教师离线导出、OPD 蒸馏
  streaming/    wait-k 数据构造、流式微调、流式解码器
  eval/         质量(BLEU/COMET) + 延迟(AL/LAAL)
  export/       量化 + 端侧导出
scripts/        每阶段一个入口脚本 run_{stage}.py
docs/superpowers/specs/   设计文档
data/           语料与中间产物（.gitignore，不入库）
checkpoints/    各阶段模型产物（.gitignore，不入库）
```
- 命名：模块 snake_case，配置键 snake_case，checkpoint 目录 `{stage}_{date}`。
- 中间产物（教师 logits、生成数据、checkpoint）一律不入 git。
- 每阶段独立可重跑，阶段间通过 checkpoint + config 解耦。

## 工程纪律
- 改完跑验证：每阶段先 `scripts/run_<stage>.py --smoke`（小数据跑通）再上全量。
- 不注释报错凑跑通，找根因。
- 24GB 显存约束：训练默认开梯度检查点 + 8bit 优化器；显存不足优先降 batch / 上梯度累积，不改模型规模。
- 密钥/token 不进代码、commit、日志。

## 验证命令（在远程 24GB Linux 机执行；本地 Windows 为 Python 3.9 仅写代码）
Phase 0：
```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -e . && pip install -r requirements.txt
# 若 transformers 稳定版不识别 qwen3_5：pip install git+https://github.com/huggingface/transformers.git
pytest -v                                            # 应 10 passed (Task1-3,5,6)
python scripts/run_inspect.py --model Qwen/Qwen3.5-2B   # Task4 勘探，产出 docs/model_inspection/
python scripts/run_inspect.py --model Qwen/Qwen3.5-9B
# 回填 docs/model_inspection/CONCLUSIONS.md 后，方可进 Phase 1
```
（Phase 1-4 验证命令各自计划补全）

## 红线（遵从全局 CLAUDE.md）
删文件/目录/git 历史、改密钥配置、git push/rebase/reset --hard、装全局依赖、公开发布——先问 Will。
