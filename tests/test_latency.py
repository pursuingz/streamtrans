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
