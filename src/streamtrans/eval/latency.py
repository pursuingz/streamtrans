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
