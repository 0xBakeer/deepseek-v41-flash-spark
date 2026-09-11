"""
fp32_skinny.py -- y = x @ W^T in fp32 for the very skinny shapes the decode path uses.

The Hyper-Connection mix projection is M=6, N=24, K=20480 with an fp32 weight (1.97 MB) and an fp32
accumulation that has to stay fp32: the Sinkhorn that consumes `mixes` is sensitive, and the router
top-k downstream flips on 1e-4 differences. cuBLAS picks `gemmSN_TN_kernel<float, 128, 16, ...>` for
it and takes 80-130 us to move 2.5 MB -- ~30 GB/s, i.e. latency-bound, not bandwidth-bound. Two of
these run per backbone layer, so they were 10.5 ms of a 152 ms decode step.

This kernel splits K across programs so the weight is streamed by many SMs at once (N=24 is a single
BLOCK_N tile, so there is nothing to parallelise over except K), accumulates each partial in fp32 and
reduces the partials in a second, single-program kernel (one wide load + `tl.sum` over the split
axis, a fixed order), so the result is bit-reproducible from replay to replay -- unlike an fp32
atomic_add split-K, which is why that is not used.

Every program covers exactly `ceil(ceil(K/BLOCK_K)/SPLIT_K)*BLOCK_K` columns of K whatever M and N
are, so a row's accumulation order does not depend on the number of rows in the call (the same
chunk-invariance property the fp8 kernels have).

X may be bf16: the upcast to fp32 is done on the loaded tile, which is exactly the value
`x.float()` would have produced, and halves the bytes read for the activation.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_kernel(X, W, Y, M, N, K,
                   stride_xm, stride_wn, stride_ys, stride_ym,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm < M
    n_mask = rn < N
    k_per = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        km = kk < k_hi
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :],
                    mask=m_mask[:, None] & km[None, :], other=0.0).to(tl.float32)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :],
                    mask=n_mask[:, None] & km[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w), input_precision="ieee", out_dtype=tl.float32)
    tl.store(Y + pid_k * stride_ys + rm[:, None] * stride_ym + rn[None, :], acc,
             mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _skinny_reduce(P, Y, M, N, stride_ps, stride_pm, stride_ym,
                   SPLIT_K, SPLIT_P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Sum the SPLIT_K partials. One wide [SPLIT_P, BLOCK_M, BLOCK_N] load and a tree reduction, not
    a loop over the splits: the loop version is latency-serialised (46 dependent loads = 11 us for
    94 kB). `tl.sum` over a fixed axis is a fixed order, so this stays bit-reproducible."""
    si = tl.arange(0, SPLIT_P)
    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    msk = (si[:, None, None] < SPLIT_K) & (rm[None, :, None] < M) & (rn[None, None, :] < N)
    v = tl.load(P + si[:, None, None] * stride_ps + rm[None, :, None] * stride_pm + rn[None, None, :],
                mask=msk, other=0.0)
    o = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], tl.sum(v, 0), mask=o)


BLOCK_M = 16     # tl.dot needs 16 rows; the extra rows are masked off, M is 5 or 6 here
BLOCK_MP = 8     # rows of the partial buffer (>= M); keeps the reduce's wide load small
BLOCK_K = 64
TARGET_CTAS = 48        # one per SM on the 48-SM GB10; 96 measured slower for the hc shape
MAX_N = int(os.environ.get("DSV41_SKINNY_MAX_N", 64))


def _plan(N: int, K: int):
    bn = 32
    while bn < N and bn < 128:
        bn *= 2
    n_blocks = triton.cdiv(N, bn)
    n_kblocks = triton.cdiv(K, BLOCK_K)
    kb = max(1, -(-n_kblocks // max(1, TARGET_CTAS // n_blocks)))
    return bn, -(-n_kblocks // kb)   # every program gets kb K-blocks, none idle


def wins(N: int) -> bool:
    """Whether this N is a shape cuBLAS handles badly. Measured on a GB10: N=24 K=20480 -> cuBLAS
    79 us (31 GB/s, latency-bound), this kernel 3x faster; N=512 K=5120 -> cuBLAS 27 us (397 GB/s),
    this kernel 2x SLOWER. So only the genuinely skinny N goes through the kernel."""
    return N <= MAX_N


def skinny_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [M, K] fp32 or bf16, w [N, K] fp32 -> y [M, N] fp32. Same math as F.linear(x.float(), w)
    up to the K-summation order (fp32 throughout, no tf32, no atomics)."""
    assert w.dtype == torch.float32 and w.dim() == 2 and x.dim() == 2
    assert x.stride(1) == 1 and w.stride(1) == 1
    M, K = x.shape
    N = w.size(0)
    assert w.size(1) == K and M <= BLOCK_MP
    bn, sk = _plan(N, K)
    y = torch.empty(M, N, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(N, bn), sk)
    if sk == 1:
        _skinny_kernel[grid](x, w, y, M, N, K, x.stride(0), w.stride(0), 0, y.stride(0),
                             BLOCK_M=BLOCK_M, BLOCK_N=bn, BLOCK_K=BLOCK_K, SPLIT_K=1,
                             num_warps=4, num_stages=3)
        return y
    nb = bn * triton.cdiv(N, bn)
    p = torch.empty(sk, BLOCK_MP, nb, dtype=torch.float32, device=x.device)
    _skinny_kernel[grid](x, w, p, M, N, K, x.stride(0), w.stride(0), p.stride(0), p.stride(1),
                         BLOCK_M=BLOCK_M, BLOCK_N=bn, BLOCK_K=BLOCK_K, SPLIT_K=sk,
                         num_warps=4, num_stages=3)
    _skinny_reduce[(1,)](p, y, M, N, p.stride(0), p.stride(1), y.stride(0),
                         sk, SPLIT_P=1 << (sk - 1).bit_length(), BLOCK_M=BLOCK_MP, BLOCK_N=nb,
                         num_warps=8, num_stages=1)
    return y
