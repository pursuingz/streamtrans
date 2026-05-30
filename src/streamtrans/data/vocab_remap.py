"""裁剪词表的 id 重映射层（torch-free，可逆）。

学生模型词表裁到 110k 后，原始 248320 tokenizer 仍可用——只需在编/解码时
把 old_id↔new_id 互转（依据 prune 产出的 vocab_map.json）。比重建 BPE merge 图安全，
且天然可逆（扩语言时换更大的 map 即可，结构不动）。

注意：Qwen 是 byte-level BPE，256 个基础字节 token 频次极高、必在 keep 集，
故对训练数据(中英为主)几乎不会产生 OOV；极少数落集外的 old_id 按 unk_new_id 处理。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class VocabRemapper:
    def __init__(self, vocab_map: dict[int, int], unk_new_id: Optional[int] = None):
        self.old2new: dict[int, int] = dict(vocab_map)
        self.new2old: dict[int, int] = {n: o for o, n in vocab_map.items()}
        self.unk_new_id = unk_new_id

    @classmethod
    def from_file(cls, path, unk_new_id: Optional[int] = None) -> "VocabRemapper":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({int(k): int(v) for k, v in raw.items()}, unk_new_id=unk_new_id)

    def map_ids(self, old_ids) -> list[int]:
        """old_id 序列 → new_id 序列。集外 token：有 unk 用 unk，否则丢弃。"""
        out: list[int] = []
        for i in old_ids:
            n = self.old2new.get(int(i))
            if n is None:
                if self.unk_new_id is None:
                    continue
                n = self.unk_new_id
            out.append(n)
        return out

    def unmap_ids(self, new_ids) -> list[int]:
        """new_id 序列 → old_id 序列（喂回原 tokenizer.decode）。"""
        return [self.new2old[int(i)] for i in new_ids]

    def encode(self, tokenizer, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.map_ids(tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, tokenizer, new_ids, **kw) -> str:
        return tokenizer.decode(self.unmap_ids(new_ids), **kw)
