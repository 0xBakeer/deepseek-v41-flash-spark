"""
fp8_linear.py -- y = x @ W^T for the checkpoint's dense projections in their STORED format:
W fp8 (e4m3) [N, K] with UE8M0 scales [ceil(N/32), ceil(K/32)] (one power-of-two scale per 32x32
block), x bf16 [M, K]. The weights are never dequantized to bf16 in memory, which halves both the
bytes read per decode step and the resident footprint of the dense layers.

Two tile shapes: BLOCK_M=16 for decode-sized M (padded up to 16 rows) and BLOCK_M=64 for prefill.
Accumulation is fp32; the block scale is applied to the fp32 partial of each 32-wide K step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_linear_kernel(X, W, S, Y, M, N, K,
                       stride_xm, stride_wn, stride_sn, stride_ym,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rs = tl.arange(0, BLOCK_K // 32)
    m_mask = rm < M
    n_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_scale_row = rn // 32
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(X + rm[:, None] * stride_xm + kk[None, :], mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * stride_wn + kk[None, :], mask=n_mask[:, None] & (kk[None, :] < K), other=0.0)
        # one ue8m0 scale per 32-wide K group: exact in bf16 (3 mantissa bits, power-of-two scale)
        s = tl.load(S + n_scale_row[:, None] * stride_sn + (k0 // 32 + rs)[None, :],
                    mask=n_mask[:, None] & ((k0 // 32 + rs)[None, :] < (K + 31) // 32), other=127).to(tl.int32)
        scale = tl.exp2((s - 127).to(tl.float32)).to(tl.bfloat16)  # [BLOCK_N, BLOCK_K // 32]
        w3 = tl.reshape(w.to(tl.bfloat16), (BLOCK_N, BLOCK_K // 32, 32)) * scale[:, :, None]
        wb = tl.reshape(w3, (BLOCK_N, BLOCK_K))
        acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _fp8_grouped_kernel(X, W, S, Y, T, R, K,
                        stride_xt, stride_xg, stride_wn, stride_sn, stride_yt, stride_yg,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Per group g: Y[:, g, :] = X[:, g, :] @ W[g*R:(g+1)*R, :]^T, with W in the stored fp8 format.

    Same tiling as `_fp8_linear_kernel` plus a third grid axis for the group; a group is nothing
    but a row range of W (rows g*R .. (g+1)*R) and the matching row range of the block-scale table
    (rows g*R/32 .. (g+1)*R/32), so no data is copied or re-laid-out. R must be a multiple of 32.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rs = tl.arange(0, BLOCK_K // 32)
    m_mask = rm < T
    n_mask = rn < R
    wrow = g * R + rn
    n_scale_row = wrow // 32
    Xg = X + g * stride_xg
    n_sk = (K + 31) // 32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(Xg + rm[:, None] * stride_xt + kk[None, :],
                    mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
        w = tl.load(W + wrow[:, None] * stride_wn + kk[None, :],
                    mask=n_mask[:, None] & (kk[None, :] < K), other=0.0)
        s = tl.load(S + n_scale_row[:, None] * stride_sn + (k0 // 32 + rs)[None, :],
                    mask=n_mask[:, None] & ((k0 // 32 + rs)[None, :] < n_sk), other=127).to(tl.int32)
        scale = tl.exp2((s - 127).to(tl.float32)).to(tl.bfloat16)  # exact: e4m3 -> bf16 is exact, scale is a power of two
        w3 = tl.reshape(w.to(tl.bfloat16), (BLOCK_N, BLOCK_K // 32, 32)) * scale[:, :, None]
        wb = tl.reshape(w3, (BLOCK_N, BLOCK_K))
        acc += tl.dot(x, tl.trans(wb), out_dtype=tl.float32)
    tl.store(Y + rm[:, None] * stride_yt + g * stride_yg + rn[None, :], acc.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


class FP8Weight:
    """Holds a dense weight in the stored format (fp8 e4m3 + ue8m0 32x32 block scales)."""

    def __init__(self, weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor):
        assert weight_fp8.dtype == torch.float8_e4m3fn and weight_fp8.dim() == 2
        self.w = weight_fp8.contiguous()
        self.s = scale_e8m0.view(torch.uint8).contiguous()
        self.N, self.K = self.w.shape
        assert self.s.shape == ((self.N + 31) // 32, (self.K + 31) // 32), (self.s.shape, self.w.shape)
        assert self.K % 32 == 0

    @property
    def shape(self):
        return (self.N, self.K)

    def dequant(self) -> torch.Tensor:
        s = torch.exp2(self.s.float() - 127.0)
        s = s.repeat_interleave(32, 0)[: self.N].repeat_interleave(32, 1)[:, : self.K]
        return (self.w.float() * s).to(torch.bfloat16)


def quantize_to_fp8(ref: torch.Tensor, rows: int = 4096) -> FP8Weight:
    """fp32/bf16 [N, K] -> FP8Weight in the checkpoint's own dense format: e4m3 codes plus one
    UE8M0 power-of-two scale per 32x32 block.

    The dense projections arrive in this format from disk; the LM head does not (it is stored bf16),
    so this is the quantizer that puts it there. Per block: amax, scale = 2^ceil(log2(amax/448))
    corrected upward if that would still clip, then the hardware's round-to-nearest-even e4m3
    conversion. Nothing can overflow: a value <= 448 never rounds above 448, whose neighbour in the
    grid is 512 with the tie at 480.

    Unlike the FP4 quantizer this one does not need a finer scale table: e4m3 carries its own 4-bit
    exponent, so the block scale only has to place the block inside the representable range, and the
    3-bit mantissa fixes the relative error wherever it lands.
    """
    assert ref.dim() == 2
    N, K = ref.shape
    nb_k = (K + 31) // 32
    rows = max(32, rows - rows % 32)
    codes = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=ref.device)
    scales = torch.empty((N + 31) // 32, nb_k, dtype=torch.uint8, device=ref.device)
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        x = ref[r0:r1].float()
        n = r1 - r0
        pn, pk = (-n) % 32, (-K) % 32
        if pn or pk:
            x = torch.nn.functional.pad(x, (0, pk, 0, pn))
        xb = x.view((n + pn) // 32, 32, nb_k, 32)
        amax = xb.abs().amax(dim=(1, 3)).clamp_min(448.0 * 2.0 ** -126)     # [nb, nb_k]
        e = torch.ceil(torch.log2(amax / 448.0))
        e = torch.where(amax > 448.0 * torch.exp2(e), e + 1, e)             # never clip
        b = (e + 127.0).clamp(0, 255)
        q = (xb / torch.exp2(b - 127.0)[:, None, :, None]).to(torch.float8_e4m3fn)
        q = q.view((n + pn), K + pk)
        codes[r0:r1] = q[:n, :K]
        scales[r0 // 32:(r1 + 31) // 32] = b.to(torch.uint8)
        del x, xb, amax, e, b, q
    return FP8Weight(codes, scales)


def fp8_linear(x: torch.Tensor, W: FP8Weight) -> torch.Tensor:
    """x bf16 [..., K] -> bf16 [..., N]."""
    shape = x.shape
    x2 = x.reshape(-1, W.K)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    M = x2.size(0)
    y = torch.empty(M, W.N, dtype=torch.bfloat16, device=x.device)
    BLOCK_M = 16 if M <= 16 else 64
    BLOCK_N = 128
    BLOCK_K = 128 if W.K % 128 == 0 else 64
    grid = (triton.cdiv(W.N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    _fp8_linear_kernel[grid](x2, W.w, W.s, y, M, W.N, W.K, x2.stride(0), W.w.stride(0), W.s.stride(0), y.stride(0),
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=3)
    return y.view(*shape[:-1], W.N)


class FP8GroupedWeight:
    """The attention output LoRA `wo_a` in its stored format: fp8 e4m3 [G*R, K] with UE8M0 32x32
    block scales [G*R/32, K/32], addressed as G independent [R, K] matrices.

    `convert.py` dequantizes this one to bf16 and the engine kept it that way ([8, 1024, 4096] bf16
    = 67 MB per layer, 2.7 GB read per decode step over 40 layers). Kept fp8 it is 33.5 MB per layer
    and 1.34 GB less resident.
    """

    def __init__(self, weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor, groups: int, rank: int):
        assert weight_fp8.dtype == torch.float8_e4m3fn and weight_fp8.dim() == 2
        self.w = weight_fp8.contiguous()
        self.s = scale_e8m0.view(torch.uint8).contiguous()
        N, self.K = self.w.shape
        assert N == groups * rank, (N, groups, rank)
        assert rank % 32 == 0 and self.K % 32 == 0
        assert self.s.shape == ((N + 31) // 32, (self.K + 31) // 32), (self.s.shape, self.w.shape)
        self.G, self.R = groups, rank

    @property
    def shape(self):
        return (self.G, self.R, self.K)

    def dequant(self) -> torch.Tensor:
        """[G, R, K] bf16 -- the tensor the old path held resident."""
        s = torch.exp2(self.s.float() - 127.0)
        n = self.G * self.R
        s = s.repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, : self.K]
        return (self.w.float() * s).to(torch.bfloat16).view(self.G, self.R, self.K)


def fp8_grouped_linear(x: torch.Tensor, W: FP8GroupedWeight) -> torch.Tensor:
    """x bf16 [T, G, K] -> bf16 [T, G, R]; equals einsum("tgk,grk->tgr", x, W.dequant()).

    Row-count- and row-offset-invariant by construction (each output element is one fp32
    accumulation over K in fixed BLOCK_K steps, independent of T and of the row's position), so it
    needs no MM_TILE row tiling.
    """
    assert x.dim() == 3 and x.size(1) == W.G and x.size(2) == W.K, (x.shape, W.shape)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = x.contiguous()
    T = x.size(0)
    y = torch.empty(T, W.G, W.R, dtype=torch.bfloat16, device=x.device)
    BLOCK_M = 16 if T <= 16 else 64
    BLOCK_N = 128
    BLOCK_K = 128 if W.K % 128 == 0 else 64
    grid = (triton.cdiv(W.R, BLOCK_N), triton.cdiv(T, BLOCK_M), W.G)
    _fp8_grouped_kernel[grid](x, W.w, W.s, y, T, W.R, W.K,
                              x.stride(0), x.stride(1), W.w.stride(0), W.s.stride(0),
                              y.stride(0), y.stride(1),
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              num_warps=4, num_stages=3)
    return y


if __name__ == "__main__":
    import json, os, sys, time
    from safetensors import safe_open
    md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
    idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(f"{md}/{idx['layers.0.attn.wq_b.weight']}", "pt", device="cpu")
    torch.manual_seed(0)
    for name in ("layers.0.attn.wq_b", "layers.0.attn.wo_b", "layers.0.attn.wq_a", "layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w2"):
        W = FP8Weight(f.get_tensor(name + ".weight").cuda(), f.get_tensor(name + ".scale").cuda())
        ref_w = W.dequant()
        for M in (1, 6, 16, 64, 512):
            x = (torch.randn(M, W.K, device="cuda") * 0.5).to(torch.bfloat16)
            y = fp8_linear(x, W)
            r = torch.nn.functional.linear(x, ref_w)
            rel = float((y.float() - r.float()).norm() / r.float().norm())
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20): fp8_linear(x, W)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 20
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20): torch.nn.functional.linear(x, ref_w)
            torch.cuda.synchronize(); dt2 = (time.perf_counter() - t0) / 20
            print(f"{name:34s} N={W.N:6d} K={W.K:5d} M={M:3d}: rel {rel:.2e}  fp8 {dt*1e3:6.3f} ms ({W.N*W.K/dt/1e9:5.0f} GB/s of fp8)  bf16 {dt2*1e3:6.3f} ms ({W.N*W.K*2/dt2/1e9:5.0f} GB/s)")
