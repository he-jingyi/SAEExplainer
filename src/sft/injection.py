"""Norm-matched feature-vector injection hooks."""

from __future__ import annotations

import contextlib
from typing import Callable

import torch
import torch.nn.functional as F


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def make_norm_matched_injection_hook(
    feature_vectors_BD: torch.Tensor,
    act_positions_B: torch.Tensor,
    inject_mask_B: torch.Tensor,
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    normalized_vectors = F.normalize(feature_vectors_BD.to(device=device, dtype=dtype), dim=-1).detach()
    act_positions_B = act_positions_B.to(device=device)
    inject_mask_B = inject_mask_B.to(device=device)

    def hook_fn(module, _inputs, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            rest = []
            output_is_tuple = False

        if resid_BLD.shape[1] <= 1:
            return output

        resid_out_BLD = resid_BLD.clone()

        for batch_index in range(resid_BLD.shape[0]):
            if not bool(inject_mask_B[batch_index]):
                continue

            position = int(act_positions_B[batch_index].item())
            if position < 0 or position >= resid_BLD.shape[1]:
                raise IndexError(
                    f"Injection position {position} out of range for sequence length {resid_BLD.shape[1]}"
                )

            hidden_D = resid_BLD[batch_index, position, :]
            hidden_norm = hidden_D.norm(dim=-1, keepdim=True)
            steered_D = normalized_vectors[batch_index] * hidden_norm * steering_coefficient
            resid_out_BLD[batch_index, position, :] = hidden_D + steered_D

        if output_is_tuple:
            return (resid_out_BLD, *rest)
        return resid_out_BLD

    return hook_fn
