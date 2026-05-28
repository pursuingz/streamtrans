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
