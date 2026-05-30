"""离线 KD 损失：教师 top-k 上的 forward-KL + 交叉熵。

约定无关：调用方（训练脚本）负责按模型 logits 偏移约定，把 target 位置的
student_logits、gold labels、对齐的 teacher top-k 摊平成 [N, ...] 再传入。
本模块只算损失，便于小张量单测。

teacher_logprobs: 教师在其 top-k 上、**已重归一化**（跨 K 求和≈1）的 log-prob。
"""
import torch
import torch.nn.functional as F


def kd_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """forward-KL( teacher || student )，仅在教师 top-k 子集上。

    student_logits [N,V]，teacher_ids [N,K] (new-id)，teacher_logprobs [N,K]。
    乘 T² 以平衡软标签梯度尺度（Hinton 蒸馏惯例）。
    """
    student_logp = F.log_softmax(student_logits / temperature, dim=-1)  # [N,V]
    s_at_k = torch.gather(student_logp, 1, teacher_ids)                 # [N,K]
    p_t = teacher_logprobs.exp()                                       # [N,K]
    kl = (p_t * (teacher_logprobs - s_at_k)).sum(dim=-1)               # [N]
    return kl.mean() * (temperature * temperature)


def ce_loss(student_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    """硬标签交叉熵；target_labels 中 -100 位置被忽略。"""
    return F.cross_entropy(student_logits, target_labels, ignore_index=-100)


def combined_loss(
    student_logits: torch.Tensor,
    target_labels: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    temperature: float = 2.0,
):
    """L = alpha*CE + beta*KD。返回 (total, ce, kd) 便于日志。"""
    ce = ce_loss(student_logits, target_labels)
    kd = kd_forward_kl_topk(student_logits, teacher_ids, teacher_logprobs, temperature)
    return alpha * ce + beta * kd, ce.detach(), kd.detach()
