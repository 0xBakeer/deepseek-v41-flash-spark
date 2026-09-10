"""
moe_fallback.py -- a slow but exact expert arena + MoE forward used when the Triton kernel
(tools/fp4_moe.py) is unavailable: dequantizes each active expert from its packed FP4 slot and
runs bf16 matmuls. Same interface as the kernel module.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import v41_ref as R  # noqa: E402


class ExpertArena:
    def __init__(self, slots: int, device: str):
        self.slots = slots
        self.device = device
        self.w1 = torch.empty(slots, 2304, 2560, dtype=torch.uint8, device=device)
        self.s1 = torch.empty(slots, 2304, 160, dtype=torch.uint8, device=device)
        self.w3 = torch.empty(slots, 2304, 2560, dtype=torch.uint8, device=device)
        self.s3 = torch.empty(slots, 2304, 160, dtype=torch.uint8, device=device)
        self.w2 = torch.empty(slots, 5120, 1152, dtype=torch.uint8, device=device)
        self.s2 = torch.empty(slots, 5120, 72, dtype=torch.uint8, device=device)

    @property
    def bytes_per_slot(self):
        return 3 * (2304 * 2560 + 2304 * 160)

    def load_slot(self, slot, w1, s1, w2, s2, w3, s3):
        self.w1[slot].copy_(w1.view(torch.uint8), non_blocking=True)
        self.s1[slot].copy_(s1.view(torch.uint8), non_blocking=True)
        self.w2[slot].copy_(w2.view(torch.uint8), non_blocking=True)
        self.s2[slot].copy_(s2.view(torch.uint8), non_blocking=True)
        self.w3[slot].copy_(w3.view(torch.uint8), non_blocking=True)
        self.s3[slot].copy_(s3.view(torch.uint8), non_blocking=True)


def moe_forward(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: ExpertArena,
                swiglu_limit: float = 10.0) -> torch.Tensor:
    """Sum of the `slots.shape[1]` routed experts of each token.

    The per-expert results are parked in a [T, K, dim] buffer and only then summed over K, so the
    order in which the experts are added up is the token's own routing order and NOT the order the
    arena happens to hold them in. Adding them straight into an accumulator (`y.index_add_` while
    looping over `torch.unique(slots)`) makes the fp32 summation order depend on which arena slots
    the store handed out, which differs between a chunked and a single-chunk run and shows up as
    bf16 ulp flips in the residual stream -- which the next layer's router then amplifies.
    """
    T, D = x.shape
    K = slots.size(1)
    parts = torch.zeros(T * K, D, dtype=torch.float32, device=x.device)
    for s in torch.unique(slots).tolist():
        w1 = R.dequant_fp4_packed(arena.w1[s], arena.s1[s])
        w2 = R.dequant_fp4_packed(arena.w2[s], arena.s2[s])
        w3 = R.dequant_fp4_packed(arena.w3[s], arena.s3[s])
        t, k = torch.where(slots == s)
        parts.index_copy_(0, t * K + k, R.expert_ffn(x[t], w1, w2, w3, swiglu_limit, weights[t, k, None]).float())
    return parts.view(T, K, D).sum(dim=1).to(x.dtype)
