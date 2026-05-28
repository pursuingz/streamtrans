# StreamTrans Phase 0：地基与模型勘探 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭好 StreamTrans 项目地基，勘探 Qwen3.5-2B/9B 真实结构（参数量分解、DeltaNet/GatedAttn module 命名、端侧算子可行性），并实现后续所有阶段都要用的 GPU-free 纯函数模块（数据管线、词频统计、同传延迟指标）。

**Architecture:** Phase 0 不做任何训练，只产出：① 可安装的 Python 包骨架 ② 配置系统 ③ 一份真实模型结构勘探报告（驱动 Phase 1 剪枝代码）④ 经单测验证的数据/词频/评测纯函数。Phase 1-4 依赖本阶段的勘探报告才能写精确代码。

**Tech Stack:** Python 3.10+、PyTorch、transformers（需支持 Qwen3.5，`trust_remote_code=True`）、datasets、sacrebleu、pydantic v2、PyYAML、pytest。

**关于本计划的 TDD：** 纯函数（词频、延迟指标、数据清洗、配置校验）走标准 TDD（先写失败测试→实现→通过）。需要真实模型/网络的勘探任务走 smoke test（小规模跑通 + 断言关键不变量），不强求纯单测。

---

### Task 1: 项目骨架与依赖

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/streamtrans/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: 写 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "streamtrans"
version = "0.0.0"
requires-python = ">=3.10"
description = "End-device streaming simultaneous translation model"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: 写 `requirements.txt`**

```
torch>=2.2
transformers>=4.51
datasets>=2.18
sacrebleu>=2.4
pydantic>=2.6
PyYAML>=6.0
pytest>=8.0
```

> 注：transformers 版本需实际支持 Qwen3.5；Task 4 勘探时若加载失败，按报错升级版本或加 `trust_remote_code=True`，并把可用版本写回此文件。

- [ ] **Step 3: 写 `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    needs_model: 需要下载/加载真实 HF 模型的测试（默认可跳过）
    needs_gpu: 需要 GPU 的测试
```

- [ ] **Step 4: 建空包文件**

`src/streamtrans/__init__.py` 与 `tests/__init__.py` 写入单行：

```python
"""StreamTrans package."""
```

- [ ] **Step 5: 验证可安装**

Run: `pip install -e . && python -c "import streamtrans; print('ok')"`
Expected: 输出 `ok`，无 ImportError。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt pytest.ini src/streamtrans/__init__.py tests/__init__.py
git commit -m "chore: project skeleton and dependencies"
```

---

### Task 2: 配置系统（pydantic + YAML）

每阶段一个 YAML，代码不硬编码路径/超参。本任务建通用加载器 + Phase 1 剪枝配置 schema。

**Files:**
- Create: `src/streamtrans/config.py`
- Create: `configs/prune.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from streamtrans.config import PruneConfig, load_config


def test_load_prune_config(tmp_path: Path):
    yaml_text = """
starter_model: Qwen/Qwen3.5-2B
target_params_b: 0.7
target_vocab_size: 110000
keep_languages: [zh, en]
reserve_languages: [ja, ko, fr, de, es, ru]
drop_vision: true
"""
    p = tmp_path / "prune.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(p, PruneConfig)
    assert cfg.starter_model == "Qwen/Qwen3.5-2B"
    assert cfg.target_params_b == 0.7
    assert cfg.target_vocab_size == 110000
    assert cfg.drop_vision is True
    assert "ja" in cfg.reserve_languages


def test_invalid_config_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("target_params_b: not_a_number\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(p, PruneConfig)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'streamtrans.config'`

- [ ] **Step 3: 写实现**

```python
# src/streamtrans/config.py
from pathlib import Path
from typing import Type, TypeVar
import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class PruneConfig(BaseModel):
    starter_model: str
    target_params_b: float
    target_vocab_size: int
    keep_languages: list[str]
    reserve_languages: list[str] = []
    drop_vision: bool = True


def load_config(path: str | Path, schema: Type[T]) -> T:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return schema(**data)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: 写 `configs/prune.yaml`（真实默认值）**

```yaml
# Phase 1 剪枝配置
starter_model: Qwen/Qwen3.5-2B
target_params_b: 0.7
target_vocab_size: 110000        # 中英为主 + 预留主要语言
keep_languages: [zh, en]          # 第一版训练语言
reserve_languages: [ja, ko, fr, de, es, ru]  # 词表预留、暂不训练
drop_vision: true
```

- [ ] **Step 6: Commit**

```bash
git add src/streamtrans/config.py configs/prune.yaml tests/test_config.py
git commit -m "feat: config system with pydantic schemas"
```

---

### Task 3: 同传延迟指标 Average Lagging（核心评测，纯函数 TDD）

AL 是同传标准延迟指标。给定每个目标 token 输出时已读入的源 token 数序列 `g`，源长 `S`、目标长 `T`，算落后程度。wait-k 的理论 AL≈k。

**Files:**
- Create: `src/streamtrans/eval/__init__.py`
- Create: `src/streamtrans/eval/latency.py`
- Test: `tests/test_latency.py`

- [ ] **Step 1: 写失败测试（含手算验证的固定例子）**

```python
# tests/test_latency.py
from streamtrans.eval.latency import average_lagging, waitk_delays


def test_waitk_delays():
    # S=4, k=2: g(i)=min(k+i-1, S) -> [2,3,4,4]
    assert waitk_delays(src_len=4, tgt_len=4, k=2) == [2, 3, 4, 4]


def test_average_lagging_waitk_equals_k():
    # S=4, T=4, k=2, g=[2,3,4,4]; r=T/S=1
    # tau = 第一个 g(i)=S 的 i (1-based) = 3
    # AL = (1/3)[(2-0)+(3-1)+(4-2)] = 2.0
    g = [2, 3, 4, 4]
    al = average_lagging(g, src_len=4, tgt_len=4)
    assert abs(al - 2.0) < 1e-6


def test_average_lagging_handles_length_ratio():
    # S=4, T=2 (r=0.5), g=[2,4]; tau=2
    # AL = (1/2)[(2-0/0.5)+(4-1/0.5)] = (1/2)[2 + (4-2)] = 2.0
    g = [2, 4]
    al = average_lagging(g, src_len=4, tgt_len=2)
    assert abs(al - 2.0) < 1e-6
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_latency.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# src/streamtrans/eval/latency.py
"""同传延迟指标。AL: Ma et al. 2019, "STACL"。"""


def waitk_delays(src_len: int, tgt_len: int, k: int) -> list[int]:
    """wait-k 策略下每个目标 token 输出时已读入的源 token 数 g(i)。"""
    return [min(k + i, src_len) for i in range(tgt_len)]


def average_lagging(delays: list[int], src_len: int, tgt_len: int) -> float:
    """Average Lagging。

    delays[i] = 输出第 i 个目标 token 时已读入的源 token 数（1-based g(i)）。
    """
    if tgt_len == 0 or src_len == 0:
        return 0.0
    r = tgt_len / src_len  # 目标/源 长度比
    # tau = 第一个读完整句的目标位置（1-based）
    tau = tgt_len
    for i, g in enumerate(delays):
        if g >= src_len:
            tau = i + 1
            break
    total = 0.0
    for i in range(tau):
        total += delays[i] - i / r
    return total / tau
```

> 注：`waitk_delays` 用 `min(k + i, src_len)`（i 从 0 起），等价于 g(i)=min(k+i-1, S)（i 从 1 起）。test 已锁定期望值 `[2,3,4,4]`。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_latency.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/streamtrans/eval/__init__.py src/streamtrans/eval/latency.py tests/test_latency.py
git commit -m "feat: average lagging latency metric"
```

---

### Task 4: 模型结构勘探（meta device，GPU-free，驱动 Phase 1）

在 meta device 上实例化 Qwen3.5-2B（不下载全权重、不需 GPU），导出参数量分解（embedding / 各层 / vision）、DeltaNet 与 GatedAttn 的真实 module 命名，写成报告。**这是 Phase 1 剪枝代码的前置依赖。**

**Files:**
- Create: `src/streamtrans/prune/__init__.py`
- Create: `src/streamtrans/prune/inspect_model.py`
- Create: `scripts/run_inspect.py`
- Test: `tests/test_inspect_model.py`

- [ ] **Step 1: 写失败测试（纯函数部分：参数量分解）**

```python
# tests/test_inspect_model.py
import torch
import torch.nn as nn
from streamtrans.prune.inspect_model import param_breakdown


def test_param_breakdown_groups_by_prefix():
    model = nn.ModuleDict({
        "embed_tokens": nn.Embedding(100, 8),      # 800
        "layers": nn.ModuleList([nn.Linear(8, 8)]),  # 64 + 8 bias = 72
        "vision": nn.Linear(8, 4),                  # 32 + 4 = 36
    })
    bd = param_breakdown(model, group_prefixes=["embed_tokens", "layers", "vision"])
    assert bd["embed_tokens"] == 800
    assert bd["layers"] == 72
    assert bd["vision"] == 36
    assert bd["__total__"] == 800 + 72 + 36
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_inspect_model.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# src/streamtrans/prune/inspect_model.py
"""模型结构勘探：参数量分解 + module 命名导出。"""
from typing import Iterable
import torch.nn as nn


def param_breakdown(model: nn.Module, group_prefixes: Iterable[str]) -> dict[str, int]:
    """按 named_parameters 的前缀分组统计参数量（meta tensor 也可，numel 不依赖数据）。"""
    prefixes = list(group_prefixes)
    out: dict[str, int] = {p: 0 for p in prefixes}
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        for pref in prefixes:
            if pref in name:
                out[pref] += n
                break
    out["__total__"] = total
    return out


def load_meta_model(model_id: str):
    """在 meta device 上实例化模型（不下载权重，看结构/数参数用）。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return model, cfg


def dump_module_names(model: nn.Module) -> list[str]:
    """所有 module 的限定名（用于发现 DeltaNet/GatedAttn/vision 的真实命名）。"""
    return [name for name, _ in model.named_modules() if name]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_inspect_model.py -v`
Expected: 1 passed

- [ ] **Step 5: 写勘探脚本**

```python
# scripts/run_inspect.py
"""勘探 Qwen3.5-2B/9B 真实结构，输出报告到 docs/。
用法: python scripts/run_inspect.py --model Qwen/Qwen3.5-2B
"""
import argparse
import json
from pathlib import Path
from streamtrans.prune.inspect_model import load_meta_model, param_breakdown, dump_module_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--out", default="docs/model_inspection")
    args = ap.parse_args()

    model, cfg = load_meta_model(args.model)
    prefixes = ["embed_tokens", "lm_head", "layers", "visual", "vision", "mlp", "self_attn"]
    bd = param_breakdown(model, group_prefixes=prefixes)
    names = dump_module_names(model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_")
    (out_dir / f"{safe}.json").write_text(
        json.dumps({
            "model": args.model,
            "config": cfg.to_dict(),
            "param_breakdown": bd,
            "module_names": names,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"total params: {bd['__total__'] / 1e9:.3f}B")
    print(f"embed_tokens: {bd['embed_tokens'] / 1e6:.1f}M")
    print(f"report -> {out_dir / (safe + '.json')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑勘探（smoke，需网络拉 config，无需 GPU）**

Run: `python scripts/run_inspect.py --model Qwen/Qwen3.5-2B`
Expected: 打印 total params（应接近 2B 含 vision；纯文本=total−vision）、embed_tokens（应≈0.5B，248320×2048）；生成 `docs/model_inspection/Qwen_Qwen3.5-2B.json`。

> **若加载失败**（架构不被 transformers 识别）：这是 spec §8 风险2/5 的关键信号。记录确切报错，按提示升级 transformers 或确认 `trust_remote_code`，并把结果写入下方 Step 7 的勘探结论。**在结论拿到前不要开始 Phase 1。**

- [ ] **Step 7: 写勘探结论文档**

把关键发现落成 `docs/model_inspection/CONCLUSIONS.md`，必须回答：
1. 纯文本 LLM 参数量 = ?（total − vision）
2. embed_tokens 实际参数量、是否 tie lm_head
3. Vision Encoder 的 module 前缀名（剥离用）= ?
4. DeltaNet 块、GatedAttn 块、FFN 的真实 module 命名路径 = ?
5. transformers 能否原生加载，还是需 trust_remote_code/特定版本 = ?

- [ ] **Step 8: Commit**

```bash
git add src/streamtrans/prune/__init__.py src/streamtrans/prune/inspect_model.py scripts/run_inspect.py tests/test_inspect_model.py docs/model_inspection/
git commit -m "feat: model structure inspection (meta device)"
```

---

### Task 5: 词频统计（为词表裁剪准备，纯函数 TDD）

按目标语言全集统计语料 token 频次，输出保留 token 集合的依据。Phase 1 的可逆词表裁剪会消费它。

**Files:**
- Create: `src/streamtrans/data/__init__.py`
- Create: `src/streamtrans/data/vocab_stats.py`
- Test: `tests/test_vocab_stats.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_vocab_stats.py
from streamtrans.data.vocab_stats import count_token_freq, select_keep_tokens


def test_count_token_freq():
    # 假 tokenizer：按空格切，token->id 用长度模拟
    def fake_encode(text):
        return [len(w) for w in text.split()]
    freq = count_token_freq(["a bb ccc", "bb bb"], fake_encode)
    assert freq[1] == 1   # "a"
    assert freq[2] == 3   # "bb" x3
    assert freq[3] == 1   # "ccc"


def test_select_keep_tokens_respects_special_and_topk():
    freq = {10: 100, 11: 50, 12: 1, 13: 1}
    keep = select_keep_tokens(freq, special_ids=[0, 1], max_vocab=4)
    # 必保留 special，再按频次补到 max_vocab
    assert 0 in keep and 1 in keep
    assert 10 in keep and 11 in keep
    assert len(keep) == 4
    assert 12 not in keep or 13 not in keep  # 低频被裁
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_vocab_stats.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# src/streamtrans/data/vocab_stats.py
"""词表裁剪前的 token 频次统计与保留集选择。"""
from collections import Counter
from typing import Callable, Iterable


def count_token_freq(texts: Iterable[str], encode: Callable[[str], list[int]]) -> dict[int, int]:
    c: Counter[int] = Counter()
    for t in texts:
        c.update(encode(t))
    return dict(c)


def select_keep_tokens(freq: dict[int, int], special_ids: list[int], max_vocab: int) -> set[int]:
    """保留全部 special + 按频次补满到 max_vocab。"""
    keep = set(special_ids)
    remaining = max_vocab - len(keep)
    if remaining <= 0:
        return keep
    ranked = sorted((tid for tid in freq if tid not in keep),
                    key=lambda t: freq[t], reverse=True)
    keep.update(ranked[:remaining])
    return keep
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_vocab_stats.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/streamtrans/data/__init__.py src/streamtrans/data/vocab_stats.py tests/test_vocab_stats.py
git commit -m "feat: token frequency stats for vocab pruning"
```

---

### Task 6: 平行语料管线（清洗/长度过滤，纯函数 TDD）

公开语料 + 教师生成混合的统一入口。本任务做与数据源无关的清洗/过滤纯函数；具体数据集接入（WMT/OPUS）在 Phase 1 配置里指定。

**Files:**
- Create: `src/streamtrans/data/parallel_corpus.py`
- Test: `tests/test_parallel_corpus.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_parallel_corpus.py
from streamtrans.data.parallel_corpus import clean_pair, filter_pairs


def test_clean_pair_strips_and_normalizes_space():
    src, tgt = clean_pair("  Hello   world ", "你好  世界")
    assert src == "Hello world"
    assert tgt == "你好 世界"


def test_filter_pairs_drops_empty_and_ratio_outliers():
    pairs = [
        ("hello", "你好"),          # ok
        ("", "你好"),               # 空源 -> drop
        ("a", "这是一个非常非常长的句子超过比例上限了哦哦哦"),  # 长度比异常 -> drop
        ("a normal sentence here", "一个正常长度的句子"),  # ok
    ]
    kept = filter_pairs(pairs, min_len=1, max_len=200, max_ratio=4.0)
    assert ("hello", "你好") in kept
    assert ("", "你好") not in kept
    assert len(kept) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_parallel_corpus.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 写实现**

```python
# src/streamtrans/data/parallel_corpus.py
"""平行语料清洗与过滤（数据源无关）。"""
import re

_WS = re.compile(r"\s+")


def clean_pair(src: str, tgt: str) -> tuple[str, str]:
    return _WS.sub(" ", src).strip(), _WS.sub(" ", tgt).strip()


def filter_pairs(pairs, min_len: int, max_len: int, max_ratio: float):
    kept = []
    for src, tgt in pairs:
        src, tgt = clean_pair(src, tgt)
        if not src or not tgt:
            continue
        ls, lt = len(src), len(tgt)
        if ls < min_len or lt < min_len or ls > max_len or lt > max_len:
            continue
        ratio = max(ls, lt) / max(1, min(ls, lt))
        if ratio > max_ratio:
            continue
        kept.append((src, tgt))
    return kept
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_parallel_corpus.py -v`
Expected: 2 passed

- [ ] **Step 5: 全量回归**

Run: `pytest -v`
Expected: 全部 passed（Task 2/3/5/6 共 9 个用例；Task 4 纯函数 1 个）

- [ ] **Step 6: Commit**

```bash
git add src/streamtrans/data/parallel_corpus.py tests/test_parallel_corpus.py
git commit -m "feat: parallel corpus cleaning and filtering"
```

---

## 后续计划路线图（Phase 0 勘探完成后各自出独立计划）

下列阶段的精确代码依赖 Task 4 的 `docs/model_inspection/CONCLUSIONS.md`，**必须等勘探结论拿到后再写**：

- **Phase 1 — 结构调整与剪枝**：剥视觉塔（依赖结论#3 vision 前缀）→ 可逆词表裁剪（消费 Task 5，依赖结论#2 embedding 结构）→ 结构化剪枝减层/缩 FFN/缩 hidden（依赖结论#4 module 命名）→ 剪枝恢复。产出 ~0.7B 模型。
- **Phase 2 — OPD 蒸馏**：教师 Qwen3.5-9B 离线导出 top-k logits → on-policy + reverse KL 蒸馏。
- **Phase 3 — wait-k 流式**：prefix-to-prefix 数据构造（复用 Task 3 的 wait-k 工具）→ 流式微调 → 流式增量解码器。
- **Phase 4 — 量化导出**：4bit 量化 → GGUF/MNN → 端侧 DeltaNet 算子验证（spec §8 风险2）。

## 关键阻塞点提醒

Task 4 Step 6 若加载 Qwen3.5 失败，是 spec §8 风险2/5 的实锤信号——可能动摇"用 Qwen3.5 做端侧"的前提。**在 Task 4 勘探结论产出前，不要进入 Phase 1。**
