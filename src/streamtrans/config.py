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
    # 结构目标（勘探后确定）
    hidden_size: int = 2048
    target_layers: int = 12
    target_ffn: int = 3072
    full_attention_interval: int = 4
    # 策略
    layer_selection: str = "block_uniform"
    ffn_selection: str = "magnitude"
    vocab_selection: str = "freq_multilingual"
    reversible: bool = True


class DistillConfig(BaseModel):
    teacher_model: str
    student_ckpt: str
    vocab_map: str
    corpus_file: str                       # 平行语料 jsonl: {src, tgt, direction}
    teacher_logits_dir: str                # 教师 top-k shard 落盘目录
    topk: int = 64
    temperature: float = 2.0
    alpha_ce: float = 0.5
    beta_kd: float = 0.5
    directions: list[str] = ["zh2en", "en2zh"]
    max_len: int = 256
    shard_size: int = 10000
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 2.0e-4
    steps: int = 2000
    grad_checkpointing: bool = True
    teacher_4bit: bool = True


def load_config(path: str | Path, schema: Type[T]) -> T:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return schema(**data)
