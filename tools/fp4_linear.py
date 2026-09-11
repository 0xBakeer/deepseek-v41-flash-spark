"""
fp4_linear.py -- the dense projections in FP4, i.e. y = x @ W^T with W stored as E2M1 codes
(two per byte along K) plus one UE8M0 power-of-two scale per 32 consecutive K weights of a row --
exactly the format the routed experts already use (tools/fp4_moe.py), and exactly the kernel
machinery: the hardware `cvt.rn.f16x2.e2m1x2` decoder, the 64-byte-wide row tile, the K-permutation
trick that reads the even and the odd nibbles as two independent dot products.

The weights are NOT stored in FP4 in the checkpoint: the dense projections are fp8 e4m3 with UE8M0
32x32 block scales (tools/fp8_linear.py). `quantize_fp8_to_fp4` re-quantizes them once at load
time -- dequantize to fp32, per-32-along-K max-abs, UE8M0 (power-of-two, rounded up so nothing
clips) scale, round-to-nearest-even onto the E2M1 grid -- so the stored bytes per output element
go from 1 + 1/1024 to 1/2 + 1/32, i.e. 0.5307x.

Two tile shapes as in fp8_linear.py: BLOCK_M=16 for decode-sized M, BLOCK_M=64 for prefill.
Accumulation is fp32; the group scale is applied to the fp32 partial of each 32-wide K step.
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# same decoder / packing as the routed experts; imported from fp4_moe so there is exactly one
# copy of the PTX. (v41_ref is NOT imported here: it imports this module.)
from fp4_moe import _fp4_decode, _split4, _ue8m0, dequant_fp4_packed  # noqa: E402

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

# activations are fed to the tensor cores as fp16 (what the routed-expert kernel does) unless this
# is 0, in which case they -- and the decoded weights, which are exact in bf16 -- go in as bf16.
USE_F16_DEFAULT = os.environ.get("DSV41_FP4_DENSE_F16", "1") == "1"


@triton.jit
def _chunk_dot(x_base, xk, mask_m, packed, scale_u8, USE_F16: tl.constexpr):
    """One 32-wide K group: x[BM, 32] . w[BN, 32]^T, the packed FP4 chunk [BN, 16] decoded in
    registers. Returns the fp32 [BM, BN] partial, already multiplied by the group's UE8M0 scale."""
    we, wo = _fp4_decode(packed)
    if USE_F16:
        xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
        xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
        p = tl.dot(xe, tl.trans(we))
        p = tl.dot(xo, tl.trans(wo), acc=p)
    else:
        # every E2M1 value (0, +-0.5, 1, 1.5, 2, 3, 4, 6) is exact in bf16, so this conversion is
        # lossless; it only widens the activation's exponent range.
        xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.bfloat16)
        xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.bfloat16)
        p = tl.dot(xe, tl.trans(we.to(tl.bfloat16)))
        p = tl.dot(xo, tl.trans(wo.to(tl.bfloat16)), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _quad_dot(x_base, xk, mask_m, w_tile_ptr, s_ptr, BN: tl.constexpr, USE_F16: tl.constexpr):
    """Four consecutive K groups (128 logical K) = one 64-byte-wide packed tile [BN, 64] + [BN, 4]
    scales. 64-byte row tiles are what reaches this box's practical copy bandwidth (fp4_moe)."""
    packed = tl.load(w_tile_ptr)
    c0, c1, c2, c3 = _split4(packed, BN, 16)
    s = tl.load(s_ptr)  # [BN, 4], column = hi*2 + lo
    sa, sb = tl.split(tl.permute(tl.reshape(s, [BN, 2, 2]), [0, 2, 1]))
    s0, s1 = tl.split(sa)
    s2, s3 = tl.split(sb)
    acc = _chunk_dot(x_base, xk, mask_m, c0, s0, USE_F16)
    acc += _chunk_dot(x_base + 32, xk, mask_m, c1, s1, USE_F16)
    acc += _chunk_dot(x_base + 64, xk, mask_m, c2, s2, USE_F16)
    acc += _chunk_dot(x_base + 96, xk, mask_m, c3, s3, USE_F16)
    return acc


@triton.jit
def _fp4_linear_kernel(X, W, S, Y, M, N, KQ,
                       stride_xm, stride_wn, stride_sn, stride_ym,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, USE_F16: tl.constexpr):
    """Y[M, N] = X[M, K] @ W[N, K]^T; W packed [N, K/2] uint8, S [N, K/32] uint8. KQ = K/128.

    Row-count- and row-offset-invariant by construction (each output element is one fp32
    accumulation over K in fixed 128-wide steps, independent of M and of the row's position).
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    rn_ = tl.where(n_mask, rn, 0)          # clamp: the weight tile is loaded unmasked (64-byte rows)
    m_mask = (rm < M)[:, None]
    rm_ = tl.where(rm < M, rm, 0)
    xk = 2 * tl.arange(0, 16)[None, :]
    x_base = X + rm_[:, None] * stride_xm
    w_tile = W + rn_[:, None] * stride_wn + tl.arange(0, 64)[None, :]
    s_tile = S + rn_[:, None] * stride_sn + tl.arange(0, 4)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for q in range(0, KQ):
        acc += _quad_dot(x_base + q * 128, xk, m_mask, w_tile + q * 64, s_tile + q * 4,
                         BLOCK_N, USE_F16)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc.to(tl.bfloat16),
             mask=m_mask & n_mask[None, :])


class FP4Weight:
    """A dense weight as E2M1 codes [N, K/2] (low nibble = even K element) + UE8M0 scales [N, K/32].

    0.5307 bytes per weight against the stored fp8 format's 1.0010 (the scale table is 32x finer,
    one per 32 K elements of a row instead of one per 32x32 block, which is what makes the 4-bit
    grid usable at all).
    """

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor, N: int, K: int):
        assert codes.dtype == torch.uint8 and scales.dtype == torch.uint8
        assert K % 128 == 0, K
        assert tuple(codes.shape) == (N, K // 2), (codes.shape, N, K)
        assert tuple(scales.shape) == (N, K // 32), (scales.shape, N, K)
        self.w = codes.contiguous()
        self.s = scales.contiguous()
        self.N, self.K = N, K

    @property
    def shape(self):
        return (self.N, self.K)

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.s.numel()

    def dequant(self) -> torch.Tensor:
        return dequant_fp4_packed(self.w, self.s)


def _round_e2m1(a: torch.Tensor) -> torch.Tensor:
    """|a| in [0, 6] -> index into FP4_GRID, round-to-nearest with ties to even (= the code whose
    low bit is 0), which is what `cvt.rn.*.e2m1x2` does. Ties are common here, not measure-zero:
    the input is an e4m3 value divided by a power of two, so it lands exactly on 0.25/0.75/1.25/
    1.75/2.5/3.5/5.0 often."""
    grid = FP4_GRID.to(a.device)
    mid = (grid[1:] + grid[:-1]) / 2
    dn = torch.bucketize(a, mid, right=False, out_int32=True)   # a == mid[i] -> i      (round down)
    up = torch.bucketize(a, mid, right=True, out_int32=True)    # a == mid[i] -> i + 1  (round up)
    return torch.where((up != dn) & (dn % 2 == 1), up, dn).to(torch.uint8)


def quantize_to_fp4(ref: torch.Tensor, group: int = 32, rows: int = 2048) -> FP4Weight:
    """fp32/bf16 [N, K] -> FP4Weight. Per 32 consecutive K elements of a row: amax, UE8M0 scale
    rounded UP so nothing clips, then round-to-nearest-even onto the E2M1 grid."""
    N, K = ref.shape
    codes = torch.empty(N, K // 2, dtype=torch.uint8, device=ref.device)
    scales = torch.empty(N, K // group, dtype=torch.uint8, device=ref.device)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        x = ref[r0:r1].float().reshape(-1, group)
        amax = x.abs().amax(dim=1, keepdim=True).clamp_min(6 * 2.0 ** -126)
        e = torch.ceil(torch.log2(amax / 6.0))
        e = torch.where(amax > 6.0 * torch.exp2(e), e + 1, e)      # never clip
        b = (e + 127.0).clamp(0, 255)
        s = torch.exp2(b - 127.0)
        v = x / s
        idx = _round_e2m1(v.abs().clamp(0.0, 6.0))
        nib = (idx | ((v < 0).to(torch.uint8) * 8)).reshape(r1 - r0, K)
        codes[r0:r1] = nib[:, 0::2] | (nib[:, 1::2] << 4)
        scales[r0:r1] = b.to(torch.uint8).reshape(r1 - r0, K // group)
    return FP4Weight(codes, scales, N, K)


def quantize_fp8_to_fp4(w, group: int = 32, rows: int = 2048) -> FP4Weight:
    """`fp8_linear.FP8Weight` -> FP4Weight, dequantizing to fp32 a row block at a time."""
    N, K = w.shape
    s = torch.exp2(w.s.float() - 127.0)
    out_c = torch.empty(N, K // 2, dtype=torch.uint8, device=w.w.device)
    out_s = torch.empty(N, K // group, dtype=torch.uint8, device=w.w.device)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        sb = s[r0 // 32:(r1 + 31) // 32].repeat_interleave(32, 0)[: r1 - r0]
        sb = sb.repeat_interleave(32, 1)[:, :K]
        blk = quantize_to_fp4(w.w[r0:r1].float() * sb, group, rows)
        out_c[r0:r1] = blk.w
        out_s[r0:r1] = blk.s
        del sb, blk
    return FP4Weight(out_c, out_s, N, K)


SMS = 48  # GB10


def pick_block_n(N: int, M: int, BLOCK_M: int) -> int:
    """32 at decode M, 128 at prefill M (measured, M=6 and M=2048, on wkv/wq_a/wq_b/wo_b/w1/w2).

    At M = 6 the grid is one M-block wide, so an N tile of 128 gives 4 (wkv) to 40 (wo_b, w2)
    programs for 48 SMs and the DRAM latency is never hidden; 32 gives 16..160 and is 5-25 % faster
    on every shape measured, including the ones that were already wide enough. At M = 2048 the
    M axis fills the machine by itself and the wider tile's larger row loads win by 1.6-1.7x.
    """
    return 32 if BLOCK_M <= 16 else 128


def fp4_linear(x: torch.Tensor, W: FP4Weight, use_f16: bool | None = None,
               block_n: int | None = None, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    """x bf16 [..., K] -> bf16 [..., N]."""
    shape = x.shape
    x2 = x.reshape(-1, W.K)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    M = x2.size(0)
    y = torch.empty(M, W.N, dtype=torch.bfloat16, device=x.device)
    BLOCK_M = 16 if M <= 16 else 64
    BLOCK_N = pick_block_n(W.N, M, BLOCK_M) if block_n is None else block_n
    grid = (triton.cdiv(W.N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    _fp4_linear_kernel[grid](x2, W.w, W.s, y, M, W.N, W.K // 128,
                             x2.stride(0), W.w.stride(0), W.s.stride(0), y.stride(0),
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                             USE_F16=USE_F16_DEFAULT if use_f16 is None else use_f16,
                             num_warps=num_warps, num_stages=num_stages)
    return y.view(*shape[:-1], W.N)
