"""DPO loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = True,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100

    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * valid_mask
    seq_log_probs = token_log_probs.sum(dim=-1)

    if average_log_prob:
        denom = valid_mask.sum(dim=-1).clamp_min(1)
        seq_log_probs = seq_log_probs / denom

    return seq_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    loss = -F.logsigmoid(beta * logits).mean()
    return loss, logits.detach()
