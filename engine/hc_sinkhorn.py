"""Fused Hyper-Connection coefficient kernel (Triton): one program per token row computes
pre = sigmoid(m[:hc]*s0 + b[:hc]) + eps, post = 2*sigmoid(m[hc:2hc]*s1 + b[hc:2hc]) and the
Sinkhorn-balanced 4x4 `comb` (softmax over rows, then `iters` alternating column/row
normalisations), exactly like tools/v41_ref.hc_split_sinkhorn / the reference
kernel.hc_split_sinkhorn_kernel. Replaces ~160 tiny torch ops per layer with one launch."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_kernel(M, S, B, PRE, POST, COMB, n_rows, iters: tl.constexpr, eps: tl.constexpr, HC: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    MIX: tl.constexpr = (2 + HC) * HC
    j = tl.arange(0, HC)
    s0 = tl.load(S + 0)
    s1 = tl.load(S + 1)
    s2 = tl.load(S + 2)
    m_pre = tl.load(M + row * MIX + j)
    b_pre = tl.load(B + j)
    pre = tl.sigmoid(m_pre * s0 + b_pre) + eps
    tl.store(PRE + row * HC + j, pre)
    m_post = tl.load(M + row * MIX + HC + j)
    b_post = tl.load(B + HC + j)
    post = 2.0 * tl.sigmoid(m_post * s1 + b_post)
    tl.store(POST + row * HC + j, post)
    r = tl.arange(0, HC)[:, None]
    c = tl.arange(0, HC)[None, :]
    off = 2 * HC + r * HC + c
    comb = tl.load(M + row * MIX + off) * s2 + tl.load(B + off)
    # softmax over the last dim (columns), + eps
    mx = tl.max(comb, axis=1)[:, None]
    e = tl.exp(comb - mx)
    comb = e / tl.sum(e, axis=1)[:, None] + eps
    # column normalisation, then (iters-1) x (row, column)
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    for _ in range(iters - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    tl.store(COMB + row * HC * HC + r * HC + c, comb)


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc: int = 4,
                      iters: int = 20, eps: float = 1e-6):
    """mixes [n, (2+hc)*hc] fp32 -> (pre [n,hc], post [n,hc], comb [n,hc,hc]) fp32."""
    n = mixes.size(0)
    mixes = mixes.contiguous()
    pre = torch.empty(n, hc, dtype=torch.float32, device=mixes.device)
    post = torch.empty(n, hc, dtype=torch.float32, device=mixes.device)
    comb = torch.empty(n, hc, hc, dtype=torch.float32, device=mixes.device)
    _hc_kernel[(max(n, 1),)](mixes, hc_scale.contiguous(), hc_base.contiguous(), pre, post, comb, n,
                             iters=iters, eps=eps, HC=hc)
    return pre, post, comb


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import v41_ref as R
    torch.manual_seed(0)
    m = torch.randn(6, 24, device="cuda") * 2
    s = torch.tensor([0.7, 1.3, 0.9], device="cuda"); b = torch.randn(24, device="cuda")
    a1 = R.hc_split_sinkhorn(m, s, b, 4, 20, 1e-6)
    a2 = hc_split_sinkhorn(m, s, b, 4, 20, 1e-6)
    for x, y, nm in zip(a1, a2, ("pre", "post", "comb")):
        print(nm, float((x - y).abs().max()))
