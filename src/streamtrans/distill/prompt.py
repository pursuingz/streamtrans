"""翻译任务的 prompt 模板与 target-段 mask。

因果 LM 形式：input = prompt(含 src + 方向标签) + target；loss 只在 target 段。
教师导出与学生训练**必须共用** build_example，保证 logits 逐位置对齐（Phase 2 头号风险）。

核心拼装 assemble_labels 是 torch-free 纯逻辑，可本地单测；build_example 叠加 tokenization。
"""
from typing import Tuple

DIRECTION_TAGS = {"zh2en": "[zh2en]", "en2zh": "[en2zh]"}


def direction_tag(direction: str) -> str:
    if direction not in DIRECTION_TAGS:
        raise ValueError(f"未知方向 {direction}，应为 {list(DIRECTION_TAGS)}")
    return DIRECTION_TAGS[direction]


def render_prompt(src: str, direction: str) -> str:
    """source + 方向标签构成的 prompt 前缀（target 接在其后）。"""
    return f"{src}\n{direction_tag(direction)}\n"


def assemble_labels(prompt_ids: list[int], target_ids: list[int]) -> Tuple[list[int], list[int]]:
    """拼 input_ids 与 labels；labels 的 prompt 段置 -100（不计 loss），仅 target 段计 loss。"""
    input_ids = list(prompt_ids) + list(target_ids)
    labels = [-100] * len(prompt_ids) + list(target_ids)
    return input_ids, labels


def build_example(
    remapper, tokenizer, src: str, tgt: str, direction: str, eos_new_id: int, max_len: int = 256
) -> Tuple[list[int], list[int]]:
    """编码一条训练样本，返回 (input_ids, labels)，均为 new-id 空间。

    remapper: VocabRemapper（原 tokenizer + vocab_map → new-id）。
    target 末尾补 eos_new_id；超长从尾部截断（保留 prompt 头部与 target 头部的对齐前缀）。
    """
    prompt_ids = remapper.encode(tokenizer, render_prompt(src, direction))
    target_ids = remapper.encode(tokenizer, tgt) + [eos_new_id]
    input_ids, labels = assemble_labels(prompt_ids, target_ids)
    return input_ids[:max_len], labels[:max_len]
