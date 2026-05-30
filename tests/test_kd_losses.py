import torch

from streamtrans.distill.kd_losses import combined_loss, kd_forward_kl_topk


def test_kl_nonneg_and_zero_when_matched():
    # 教师质量全压在 token 0（top-1，重归一化后 logprob=0）
    teacher_ids = torch.tensor([[0]])
    teacher_logprobs = torch.tensor([[0.0]])  # exp=1
    # 学生也把几乎全部质量给 token 0 → KL≈0
    student_logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    kl = kd_forward_kl_topk(student_logits, teacher_ids, teacher_logprobs, temperature=1.0)
    assert kl.item() >= 0.0
    assert kl.item() < 1e-3

    # 学生把质量给别的 token → KL 明显 > 0
    student_bad = torch.tensor([[0.0, 20.0, 0.0, 0.0]])
    kl_bad = kd_forward_kl_topk(student_bad, teacher_ids, teacher_logprobs, temperature=1.0)
    assert kl_bad.item() > kl.item()


def test_combined_loss_finite():
    V = 8
    student_logits = torch.randn(5, V)
    target_labels = torch.tensor([1, 2, -100, 3, 0])  # 含被忽略位置
    teacher_ids = torch.randint(0, V, (5, 4))
    teacher_logprobs = torch.log_softmax(torch.randn(5, 4), dim=-1)
    total, ce, kd = combined_loss(
        student_logits, target_labels, teacher_ids, teacher_logprobs, 0.5, 0.5, 2.0
    )
    assert torch.isfinite(total) and torch.isfinite(ce) and torch.isfinite(kd)
