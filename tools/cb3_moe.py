"""
cb3_moe.py -- grouped MoE kernel for experts stored in the CB3 format (tools/cb3.py): 3-bit
per-row-codebook indices in a plane layout + the original UE8M0 group scales. The kernel rebuilds the
packed-FP4 byte tile of each 128-K quad in registers (index -> codebook nibble via one variable
shift of a per-row 32-bit codebook word) and then reuses the FP4 kernel's decode/dot path
(tools/fp4_moe.py: `_split4`, `_chunk_dot` with the hardware e2m1 -> f16 cvt). Routing, block layout
and the down/scatter scheme are shared with fp4_moe.

Per slot: w1/w3 lo [2304, 1280] + hi [2304, 640] + cb [2304, 8] + s [2304, 160];
          w2    lo [5120, 576]  + hi [5120, 288] + cb [5120, 8] + s [5120, 72]  -> 14.45 MB (3.07 bpw).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import fp4_moe as F4
from fp4_moe import DIM, INTER, _chunk_dot, _split4, _ue8m0, build_routing, _pick_bm  # noqa: F401
from cb3 import dequant_cb3, fp4_to_cb3

SG1, SG2 = DIM // 32, INTER // 32


class CB3Arena:
    def __init__(self, slots: int, device: torch.device | str = "cuda"):
        self.slots = slots
        self.device = torch.device(device)
        u8 = dict(dtype=torch.uint8, device=self.device)
        self.w1_lo = torch.empty((slots, INTER, DIM // 4), **u8)
        self.w1_hi = torch.empty((slots, INTER, DIM // 8), **u8)
        self.w1_cb = torch.empty((slots, INTER, 8), **u8)
        self.s1 = torch.empty((slots, INTER, SG1), **u8)
        self.w3_lo = torch.empty((slots, INTER, DIM // 4), **u8)
        self.w3_hi = torch.empty((slots, INTER, DIM // 8), **u8)
        self.w3_cb = torch.empty((slots, INTER, 8), **u8)
        self.s3 = torch.empty((slots, INTER, SG1), **u8)
        self.w2_lo = torch.empty((slots, DIM, INTER // 4), **u8)
        self.w2_hi = torch.empty((slots, DIM, INTER // 8), **u8)
        self.w2_cb = torch.empty((slots, DIM, 8), **u8)
        self.s2 = torch.empty((slots, DIM, SG2), **u8)
        self.sim = None  # engine.codebook_sim.CodebookSim(3), set by the caller

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1_lo, self.w1_hi, self.w1_cb, self.s1, self.w3_lo, self.w3_hi,
                                          self.w3_cb, self.s3, self.w2_lo, self.w2_hi, self.w2_cb, self.s2))

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False) -> None:
        """Takes packed-FP4 CPU tensors as read from the checkpoint (same signature as the FP4 arena)
        and converts them to CB3 on the GPU on the way in."""
        assert self.sim is not None, "CB3Arena.sim must be a CodebookSim(3)"
        dev = self.device
        for (w, s, lo_t, hi_t, cb_t, s_t) in ((w1, s1, self.w1_lo, self.w1_hi, self.w1_cb, self.s1),
                                               (w3, s3, self.w3_lo, self.w3_hi, self.w3_cb, self.s3),
                                               (w2, s2, self.w2_lo, self.w2_hi, self.w2_cb, self.s2)):
            wg = w.view(torch.uint8).to(dev, non_blocking=non_blocking)
            sg = s.view(torch.uint8).to(dev, non_blocking=non_blocking)
            lo, hi, cb = fp4_to_cb3(wg, sg, self.sim)
            lo_t[slot].copy_(lo); hi_t[slot].copy_(hi); cb_t[slot].copy_(cb); s_t[slot].copy_(sg)

    def dequant_slot(self, slot: int):
        w1 = dequant_cb3(self.w1_lo[slot], self.w1_hi[slot], self.w1_cb[slot], self.s1[slot])
        w2 = dequant_cb3(self.w2_lo[slot], self.w2_hi[slot], self.w2_cb[slot], self.s2[slot])
        w3 = dequant_cb3(self.w3_lo[slot], self.w3_hi[slot], self.w3_cb[slot], self.s3[slot])
        return w1, w2, w3


@triton.jit
def _cb3_pack_quad(lo_row, hi_row, cbword, BN: tl.constexpr):
    """Rebuild the packed-FP4 byte tile [BN, 64] of one 128-K quad from the plane layout.
    lo_row / hi_row: [BN, 1] pointers at the quad's first lo byte (32 per quad) / hi byte (16 per quad);
    cbword: [BN, 1] int32 with the row's 8 codebook nibbles. Byte j of the tile holds codes 2j, 2j+1:
    their low bits sit in lo[j // 2] (at 4*(j%2) and +2), their high bits in hi[j // 4] (bit 2*(j%4), +1).
    Gather loads (L1-served duplicates) instead of register reshapes."""
    j = tl.arange(0, 64)[None, :]
    lo_e = tl.load(lo_row + j // 2).to(tl.int32)  # [BN, 64]
    hi_e = tl.load(hi_row + j // 4).to(tl.int32)  # [BN, 64]
    sh0 = (j % 2) * 4
    bh = (j % 4) * 2
    idx0 = ((lo_e >> sh0) & 3) | (((hi_e >> bh) & 1) << 2)
    idx1 = ((lo_e >> (sh0 + 2)) & 3) | (((hi_e >> (bh + 1)) & 1) << 2)
    nib0 = (cbword >> (idx0 * 4)) & 15
    nib1 = (cbword >> (idx1 * 4)) & 15
    return (nib0 | (nib1 << 4)).to(tl.uint8)


@triton.jit
def _cb3_quad_dot(x_base, xk, mask_m, lo_row, hi_row, cbword, s_ptr, BN: tl.constexpr):
    packed = _cb3_pack_quad(lo_row, hi_row, cbword, BN)
    c0, c1, c2, c3 = _split4(packed, BN, 16)
    s = tl.load(s_ptr)  # [BN, 4]
    sa, sb = tl.split(tl.permute(tl.reshape(s, [BN, 2, 2]), [0, 2, 1]))
    s0, s1 = tl.split(sa)
    s2, s3 = tl.split(sb)
    acc = _chunk_dot(x_base, xk, mask_m, c0, s0)
    acc += _chunk_dot(x_base + 32, xk, mask_m, c1, s1)
    acc += _chunk_dot(x_base + 64, xk, mask_m, c2, s2)
    acc += _chunk_dot(x_base + 96, xk, mask_m, c3, s3)
    return acc


@triton.jit
def _cbword(cb_ptr, BN: tl.constexpr):
    cb = tl.load(cb_ptr).to(tl.int32)  # [BN, 8]
    return tl.sum(cb << (tl.arange(0, 8) * 4)[None, :], axis=1)[:, None]  # [BN, 1]


@triton.jit
def _cb3_up_kernel(
    x_ptr, lo1_ptr, hi1_ptr, cb1_ptr, s1_ptr, lo3_ptr, hi3_ptr, cb3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_q = tl.arange(0, 4)
    offs_c = tl.arange(0, 8)
    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    lo1 = lo1_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi1 = hi1_ptr + slot * (N * KH) + offs_n[:, None] * KH
    lo3 = lo3_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi3 = hi3_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s1t = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]
    s3t = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]
    cw1 = _cbword(cb1_ptr + slot * (N * 8) + offs_n[:, None] * 8 + offs_c[None, :], BN)
    cw3 = _cbword(cb3_ptr + slot * (N * 8) + offs_n[:, None] * 8 + offs_c[None, :], BN)
    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc_g += _cb3_quad_dot(x_base + q * 128, xk, mask_m[:, None], lo1 + q * 32, hi1 + q * 16, cw1, s1t + q * 4, BN)
        acc_u += _cb3_quad_dot(x_base + q * 128, xk, mask_m[:, None], lo3 + q * 32, hi3 + q * 16, cw3, s3t + q * 4, BN)
    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _cb3_down_kernel(
    h_ptr, lo2_ptr, hi2_ptr, cb2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)
    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_q = tl.arange(0, 4)
    offs_c = tl.arange(0, 8)
    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h
    xk = 2 * tl.arange(0, 16)[None, :]
    lo2 = lo2_ptr + slot * (N * KL) + offs_n[:, None] * KL
    hi2 = hi2_ptr + slot * (N * KH) + offs_n[:, None] * KH
    s2t = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]
    cw2 = _cbword(cb2_ptr + slot * (N * 8) + offs_n[:, None] * 8 + offs_c[None, :], BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc += _cb3_quad_dot(h_base + q * 128, xk, mask_m[:, None], lo2 + q * 32, hi2 + q * 16, cw2, s2t + q * 4, BN)
    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


_UP_CFG = {16: (128, 4, 1), 32: (128, 4, 1), 64: (64, 4, 1)}
_DOWN_CFG = {16: (128, 8, 2), 32: (128, 4, 2), 64: (128, 8, 2)}


def moe_forward(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: CB3Arena,
                swiglu_limit: float = 10.0, block_m: int | None = None) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    P = T * K
    dev = x.device
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = _UP_CFG[BM]
    bn2, nw2, ns2 = _DOWN_CFG[BM]
    block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()
    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
    _cb3_up_kernel[(NB, INTER // bn1)](
        x, arena.w1_lo, arena.w1_hi, arena.w1_cb, arena.s1, arena.w3_lo, arena.w3_hi, arena.w3_cb, arena.s3, h,
        wgt, block_slot, block_pair, x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
    _cb3_down_kernel[(NB, DIM // bn2)](
        h, arena.w2_lo, arena.w2_hi, arena.w2_cb, arena.s2, parts, block_slot, block_pair,
        h.stride(0), parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2)
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


if __name__ == "__main__":
    import json, os, sys, time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
    from codebook_sim import CodebookSim
    from safetensors import safe_open
    md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
    idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(f"{md}/{idx['layers.0.ffn.experts.0.w1.weight']}", "pt", device="cpu")
    torch.manual_seed(0)
    S = 32
    arena = CB3Arena(S, "cuda"); arena.sim = CodebookSim(3, "cuda")
    t0 = time.time()
    for s in range(S):
        p = f"layers.0.ffn.experts.{s}."
        arena.load_slot(s, f.get_tensor(p + "w1.weight"), f.get_tensor(p + "w1.scale"), f.get_tensor(p + "w2.weight"),
                        f.get_tensor(p + "w2.scale"), f.get_tensor(p + "w3.weight"), f.get_tensor(p + "w3.scale"))
    torch.cuda.synchronize(); print(f"loaded+converted {S} experts in {time.time() - t0:.1f}s ({arena.bytes_per_slot / 1e6:.2f} MB/slot)")
    for T in (1, 6, 64, 512):
        x = (torch.randn(T, DIM, device="cuda") * 0.5).to(torch.bfloat16)
        slots = torch.stack([torch.randperm(S, device="cuda")[:6] for _ in range(T)]).to(torch.int32)
        w = torch.rand(T, 6, device="cuda")
        y = moe_forward(x, slots, w, arena)
        r = F4.moe_forward_reference(x, slots, w, arena)  # uses arena.dequant_slot -> CB3 dequant
        rel = float((y.float() - r.float()).norm() / r.float().norm())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10): moe_forward(x, slots, w, arena)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 10
        n_exp = len(torch.unique(slots))
        print(f"T={T:3d} experts={n_exp:2d}: rel err {rel:.2e}  {dt * 1e3:6.2f} ms  {n_exp * arena.bytes_per_slot / dt / 1e9:5.0f} GB/s of CB3 bytes")
