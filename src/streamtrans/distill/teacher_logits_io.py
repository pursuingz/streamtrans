"""教师 top-k logits 的落盘/读盘，及 old→new id 重映射 + 重归一化。

shard 格式：每个 shard 一个 .pt，存 list[dict{ids: LongTensor[L,K], logp: HalfTensor[L,K]}]，
按导出顺序与平行语料对齐；manifest.json 记录每 shard 的样本数，供全局索引定位。

torch 延迟导入：locate / remap_topk_ids 是 torch-free 纯逻辑，可本地单测。
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple


def locate(shard_sizes: List[int], global_idx: int) -> Tuple[int, int]:
    """全局样本下标 → (shard 序号, shard 内局部下标)。"""
    if global_idx < 0:
        raise IndexError(global_idx)
    acc = 0
    for si, n in enumerate(shard_sizes):
        if global_idx < acc + n:
            return si, global_idx - acc
        acc += n
    raise IndexError(f"global_idx={global_idx} 超出总样本数 {acc}")


def remap_topk_ids(top_ids_old: List[int], old2new: dict) -> Tuple[List[int], List[int]]:
    """教师 top-k 的 old-id → (保留位置下标, 对应 new-id)。落集外的丢弃。"""
    keep_pos: List[int] = []
    new_ids: List[int] = []
    for j, o in enumerate(top_ids_old):
        n = old2new.get(int(o))
        if n is not None:
            keep_pos.append(j)
            new_ids.append(n)
    return keep_pos, new_ids


def map_and_renormalize_topk(top_ids_old, top_logits, old2new):
    """单个位置：old top-k → 映射到 new-id、丢集外、对剩余 logits 重新 log_softmax 归一化。

    返回 (new_ids LongTensor[k'], logprobs HalfTensor[k']). k' ≤ K。
    """
    import torch
    import torch.nn.functional as F

    keep_pos, new_ids = remap_topk_ids(list(top_ids_old), old2new)
    if not keep_pos:  # 极端：top-k 全落集外（中英数据几乎不会）
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float16)
    # top_logits 在 cuda 上；落盘前一律搬回 cpu，保证后续 pad/stack/torch.save 设备一致
    sub = top_logits[torch.tensor(keep_pos, dtype=torch.long, device=top_logits.device)]
    logp = F.log_softmax(sub.float(), dim=-1)
    return torch.tensor(new_ids, dtype=torch.long), logp.to(torch.float16).cpu()


class ShardWriter:
    """按 shard_size 累积样本、分片落盘，最后写 manifest.json。"""

    def __init__(self, out_dir, shard_size: int = 10000):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._buf: list = []
        self._sizes: List[int] = []
        self._shard_i = 0

    def add(self, record: dict):
        self._buf.append(record)
        if len(self._buf) >= self.shard_size:
            self._flush()

    def _flush(self):
        import torch

        if not self._buf:
            return
        torch.save(self._buf, self.dir / f"shard_{self._shard_i:05d}.pt")
        self._sizes.append(len(self._buf))
        self._shard_i += 1
        self._buf = []

    def close(self):
        self._flush()
        (self.dir / "manifest.json").write_text(
            json.dumps({"shard_sizes": self._sizes, "shard_size": self.shard_size}),
            encoding="utf-8",
        )


class ShardReader:
    """按全局下标读取一条教师记录（带单 shard 缓存）。"""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        man = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.shard_sizes: List[int] = man["shard_sizes"]
        self.total = sum(self.shard_sizes)
        self._cache_i: Optional[int] = None
        self._cache = None

    def __len__(self):
        return self.total

    @property
    def num_shards(self) -> int:
        return len(self.shard_sizes)

    def load_shard(self, si: int) -> list:
        """整片读入（带单 shard 缓存）。批处理按 shard 内打乱，避免全局 shuffle 反复重载。"""
        import torch

        if si != self._cache_i:
            self._cache = torch.load(self.dir / f"shard_{si:05d}.pt")
            self._cache_i = si
        return self._cache

    def get(self, global_idx: int):
        si, li = locate(self.shard_sizes, global_idx)
        return self.load_shard(si)[li]
