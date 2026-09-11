"""fp4_moe.py -- Triton grouped-MoE forward for DeepSeek-V4.1-Flash's FP4 routed experts.

Target: NVIDIA GB10 (sm_121a, 48 SMs, unified memory, ~273 GB/s). Decode is bandwidth-bound on the
18.8 MB every active expert weighs, so the design goal is "read each active expert's bytes once per
launch, decode FP4 in registers, feed the tensor cores".

Weight format (per expert, as stored in the HF safetensors):
  w1, w3 : int8  [2304, 2560]  packed FP4 E2M1, two values per byte along K (logical [2304, 5120]);
                               low nibble = even K element, high nibble = odd K element
  s1, s3 : uint8 [2304, 160]   UE8M0 block scale, one per 32 consecutive K elements, value 2^(b-127)
  w2     : int8  [5120, 1152]  (logical [5120, 2304]),  s2 : uint8 [5120, 72]

Pipeline for one call (T tokens, K experts per token), three Triton launches and no host sync:
  1. _route_kernel: groups the (token, k) pairs by expert slot, each slot's run padded to a multiple of
     BM pairs -> block_slot[b], block_pair[b*BM + i].
  2. _moe_up_kernel: grid (pair block, n-block). For its expert it streams the packed w1/w3 rows ONCE,
     decodes them with the hardware `cvt.rn.f16x2.e2m1x2` instruction (one instruction per byte -> two
     fp16), does tl.dot against the (up to BM) pairs of the block and applies the UE8M0 scale on the fp32
     partial sum of each 32-wide K group. Epilogue = clamps + SiLU + routing weight; writes h[pair, :]
     (bf16 [T*K, 2304]). Its n-block-0 programs also zero the fp32 output rows for step 3.
  3. _moe_down_kernel: same structure over w2, atomically scatter-adds the fp32 tile into y32[token, :];
     y32 is cast to bf16 at the end.

K-permutation trick: because a dot product is order-invariant along K, the even/odd nibbles never have to
be interleaved. The decoder returns the even K elements and the odd K elements as two [BN, 16] fp16 tiles
(via `prmt`), and the activation tile is loaded with stride 2 to match. No layout shuffles for the weights.

For decode-size calls every expert has <= BM pairs, so its weight bytes are read exactly once per launch.
For prefill-size calls (more than BM pairs on one expert) the extra pair blocks re-read the expert (L2).

Measured on GB10 (tools/test_fp4_moe.py, 32 real layer-0 experts): ~190-200 GB/s effective for decode
sizes (T=1..6) against a ~210 GB/s practical copy ceiling on this box (273 GB/s nominal).
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

DIM = 5120  # model hidden size (K of w1/w3, N of w2)
INTER = 2304  # expert intermediate size (N of w1/w3, K of w2)
GROUP = 32  # K elements per UE8M0 scale
KB1 = DIM // 2  # packed bytes per w1/w3 row
SG1 = DIM // GROUP  # scale groups per w1/w3 row
KB2 = INTER // 2  # packed bytes per w2 row
SG2 = INTER // GROUP  # scale groups per w2 row

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# --------------------------------------------------------------------------- reference dequant
def _local_dequant_fp4_packed(weight_i8: torch.Tensor, scale: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Same contract as tools/v41_ref.py::dequant_fp4_packed (used only if that module is not importable)."""
    x = weight_i8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    tab = FP4_TABLE.to(x.device)
    vals = torch.stack([tab[low.long()], tab[high.long()]], dim=-1).flatten(1)
    s = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0).repeat_interleave(group, 1)
    return (vals * s).to(torch.bfloat16)


try:  # prefer the verified reference implementation next to this file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from v41_ref import dequant_fp4_packed  # type: ignore
except Exception:  # pragma: no cover
    dequant_fp4_packed = _local_dequant_fp4_packed


# --------------------------------------------------------------------------- arena
class ExpertArena:
    """GPU-resident slots of packed FP4 expert weights (18.8 MB per slot)."""

    def __init__(self, slots: int, device: torch.device | str = "cuda"):
        self.slots = slots
        self.device = torch.device(device)
        u8 = dict(dtype=torch.uint8, device=self.device)
        self.w1 = torch.empty((slots, INTER, KB1), **u8)
        self.s1 = torch.empty((slots, INTER, SG1), **u8)
        self.w3 = torch.empty((slots, INTER, KB1), **u8)
        self.s3 = torch.empty((slots, INTER, SG1), **u8)
        self.w2 = torch.empty((slots, DIM, KB2), **u8)
        self.s2 = torch.empty((slots, DIM, SG2), **u8)

    @property
    def bytes_per_slot(self) -> int:
        return sum(t[0].numel() for t in (self.w1, self.s1, self.w3, self.s3, self.w2, self.s2))

    @staticmethod
    def _u8(t: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        t = t.view(torch.uint8)
        if tuple(t.shape) != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(t.shape)}")
        return t.contiguous()

    def load_slot(self, slot: int, w1, s1, w2, s2, w3, s3, non_blocking: bool = False) -> None:
        """Copy one expert (CPU int8/uint8/e8m0 tensors as read from safetensors) into `slot`.

        `non_blocking=True` is for the serving path only (`engine.experts.ExpertStore`), where the
        source is a *pinned* staging buffer and the caller synchronises its own copy stream before
        handing that buffer to the next expert. Blocking is the default because a plain `.copy_()`
        from pinned memory synchronises the calling stream once per tensor -- six GPU syncs per
        expert, on the stream the model is computing on, for every miss.
        """
        self.w1[slot].copy_(self._u8(w1, (INTER, KB1)), non_blocking=non_blocking)
        self.s1[slot].copy_(self._u8(s1, (INTER, SG1)), non_blocking=non_blocking)
        self.w3[slot].copy_(self._u8(w3, (INTER, KB1)), non_blocking=non_blocking)
        self.s3[slot].copy_(self._u8(s3, (INTER, SG1)), non_blocking=non_blocking)
        self.w2[slot].copy_(self._u8(w2, (DIM, KB2)), non_blocking=non_blocking)
        self.s2[slot].copy_(self._u8(s2, (DIM, SG2)), non_blocking=non_blocking)

    def dequant_slot(self, slot: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(w1, w2, w3) as bf16 [N, K] logical matrices -- for the reference path / tests."""
        w1 = dequant_fp4_packed(self.w1[slot], self.s1[slot])
        w2 = dequant_fp4_packed(self.w2[slot], self.s2[slot])
        w3 = dequant_fp4_packed(self.w3[slot], self.s3[slot])
        return w1, w2, w3


# --------------------------------------------------------------------------- device helpers
# One 32-bit register holds 4 packed bytes = 8 FP4 values. `cvt.rn.f16x2.e2m1x2` (sm_100a/120a/121a)
# turns one byte into {f16(high nibble), f16(low nibble)}; the prmt shuffles regroup the four results into
# "even elements" (low nibbles) and "odd elements" (high nibbles), 4 fp16 each. The .b8 source operand is
# mandatory -- ptxas rejects a .b16/.b32 source with "Arguments mismatch for instruction 'cvt'".
_FP4_DECODE_ASM = tl.constexpr("""
{
.reg .b8 b0, b1, b2, b3;
.reg .b32 t0, t1, t2, t3;
mov.b32 {b0, b1, b2, b3}, $4;
cvt.rn.f16x2.e2m1x2 t0, b0;
cvt.rn.f16x2.e2m1x2 t1, b1;
cvt.rn.f16x2.e2m1x2 t2, b2;
cvt.rn.f16x2.e2m1x2 t3, b3;
prmt.b32 $0, t0, t1, 0x5410;
prmt.b32 $1, t2, t3, 0x5410;
prmt.b32 $2, t0, t1, 0x7632;
prmt.b32 $3, t2, t3, 0x7632;
}
""")


@triton.jit
def _fp4_decode(packed):
    """uint8 tile [..., 16] -> (even fp16 tile, odd fp16 tile), same shape; values in the E2M1 grid."""
    return tl.inline_asm_elementwise(
        _FP4_DECODE_ASM, "=r,=r,=r,=r,r", [packed], dtype=(tl.float16, tl.float16), is_pure=True, pack=4
    )


@triton.jit
def _ue8m0(scale_u8):
    """UE8M0 byte -> fp32 2^(b-127), exactly, by building the float bit pattern."""
    return (scale_u8.to(tl.int32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _split4(t, BN: tl.constexpr, W: tl.constexpr):
    """[BN, 4*W] register tile -> four [BN, W] column chunks, in order, without touching shared memory."""
    t = tl.reshape(t, [BN, 2, 2, W])  # (hi, lo, i)  with column = hi*2W + lo*W + i
    t = tl.permute(t, [0, 3, 2, 1])  # [BN, i, lo, hi]
    a, b = tl.split(t)  # hi = 0 -> chunks 0,1 ; hi = 1 -> chunks 2,3
    c0, c1 = tl.split(a)
    c2, c3 = tl.split(b)
    return c0, c1, c2, c3


@triton.jit
def _chunk_dot(x_base, xk, mask_m, packed, scale_u8):
    """One 32-wide K group: x[BM, 32] . w[BN, 32]^T with the packed FP4 chunk [BN, 16] decoded in registers.

    x_base: [BM, 1] pointers at the group's first element; xk: [1, 16] even offsets inside the group.
    Returns the fp32 [BM, BN] partial, already multiplied by the group's UE8M0 scale ([BN] uint8).
    """
    xe = tl.load(x_base + xk, mask=mask_m, other=0.0).to(tl.float16)
    xo = tl.load(x_base + xk + 1, mask=mask_m, other=0.0).to(tl.float16)
    we, wo = _fp4_decode(packed)
    p = tl.dot(xe, tl.trans(we))
    p = tl.dot(xo, tl.trans(wo), acc=p)
    return p * _ue8m0(scale_u8)[None, :]


@triton.jit
def _quad_dot(x_base, xk, mask_m, w_tile_ptr, s_ptr, BN: tl.constexpr):
    """Four consecutive K groups (128 logical K) = one 64-byte-wide packed tile [BN, 64] + scales [BN, 4].

    Loading 64 contiguous bytes per row matters on GB10: 16-byte-wide row tiles cap at ~130 GB/s, 64-byte
    tiles reach ~210 GB/s (the box's practical copy bandwidth). The tile is split in registers.
    """
    packed = tl.load(w_tile_ptr)
    c0, c1, c2, c3 = _split4(packed, BN, 16)
    s = tl.load(s_ptr)  # [BN, 4], column = hi*2 + lo
    sa, sb = tl.split(tl.permute(tl.reshape(s, [BN, 2, 2]), [0, 2, 1]))  # split on hi: (0,1) / (2,3)
    s0, s1 = tl.split(sa)
    s2, s3 = tl.split(sb)
    acc = _chunk_dot(x_base, xk, mask_m, c0, s0)
    acc += _chunk_dot(x_base + 32, xk, mask_m, c1, s1)
    acc += _chunk_dot(x_base + 64, xk, mask_m, c2, s2)
    acc += _chunk_dot(x_base + 96, xk, mask_m, c3, s3)
    return acc


# --------------------------------------------------------------------------- kernel 1: gate/up + SwiGLU
@triton.jit
def _moe_up_kernel(
    x_ptr, w1_ptr, s1_ptr, w3_ptr, s3_ptr, h_ptr,
    wgt_ptr, block_slot_ptr, block_pair_ptr,
    stride_x, stride_h, limit,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    KB: tl.constexpr = K // 2
    SG: tl.constexpr = K // 32
    mb = tl.program_id(0)  # padded pair block (all pairs of one expert)
    nb = tl.program_id(1)
    slot = tl.load(block_slot_ptr + mb)
    if slot < 0:
        return
    slot = slot.to(tl.int64)

    offs_m = tl.load(block_pair_ptr + mb * BM + tl.arange(0, BM))  # (token, k) pair ids, -1 = padding
    mask_m = offs_m >= 0
    offs_m = tl.where(mask_m, offs_m, 0)
    tok = (offs_m // TOPK).to(tl.int64)
    offs_n = nb * BN + tl.arange(0, BN)
    offs_j = tl.arange(0, 64)
    offs_q = tl.arange(0, 4)

    x_base = x_ptr + tok[:, None] * stride_x
    xk = 2 * tl.arange(0, 16)[None, :]
    w1_tile = w1_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    w3_tile = w3_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    s1_tile = s1_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]
    s3_tile = s3_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]

    acc_g = tl.zeros([BM, BN], dtype=tl.float32)
    acc_u = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc_g += _quad_dot(x_base + q * 128, xk, mask_m[:, None], w1_tile + q * 64, s1_tile + q * 4, BN)
        acc_u += _quad_dot(x_base + q * 128, xk, mask_m[:, None], w3_tile + q * 64, s3_tile + q * 4, BN)

    gate = tl.minimum(acc_g, limit)
    up = tl.minimum(tl.maximum(acc_u, -limit), limit)
    wgt = tl.load(wgt_ptr + offs_m, mask=mask_m, other=0.0)
    h = gate * tl.sigmoid(gate) * up * wgt[:, None]
    tl.store(h_ptr + offs_m[:, None] * stride_h + offs_n[None, :], h.to(tl.bfloat16), mask=mask_m[:, None])


# --------------------------------------------------------------------------- kernel 2: down + scatter
# `y_ptr` is a [TOPK, T, DIM] fp32 buffer: every (k, token) pair owns its own row and is written
# exactly once, so there are no atomics and no cross-block accumulation. The TOPK experts of a
# token are summed afterwards in torch, over the outermost axis, always in the same order.
# Accumulating here with tl.atomic_add(..., sem="relaxed") instead made the fp32 summation order
# depend on block scheduling -- CUDA guarantees nothing about it -- so the same token could come
# out different in a short and in a long prefill chunk, which the next layer's router amplifies.
# k-major and not pair-major: with [T*TOPK, DIM] the scattered fp32 stores cost ~50% at T=512,
# k-major costs ~6% (the reduction reads one contiguous [T, DIM] plane per k).
@triton.jit
def _moe_down_kernel(
    h_ptr, w2_ptr, s2_ptr, y_ptr,
    block_slot_ptr, block_pair_ptr,
    stride_h, stride_y,
    TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, NTOK: tl.constexpr,
):
    KB: tl.constexpr = K // 2
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
    offs_j = tl.arange(0, 64)
    offs_q = tl.arange(0, 4)

    h_base = h_ptr + offs_m[:, None].to(tl.int64) * stride_h  # h row = pair id
    xk = 2 * tl.arange(0, 16)[None, :]
    w2_tile = w2_ptr + slot * (N * KB) + offs_n[:, None] * KB + offs_j[None, :]
    s2_tile = s2_ptr + slot * (N * SG) + offs_n[:, None] * SG + offs_q[None, :]

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for q in range(0, SG // 4):
        acc += _quad_dot(h_base + q * 128, xk, mask_m[:, None], w2_tile + q * 64, s2_tile + q * 4, BN)

    row = ((offs_m % TOPK) * NTOK + offs_m // TOPK).to(tl.int64)
    tl.store(y_ptr + row[:, None] * stride_y + offs_n[None, :], acc, mask=mask_m[:, None])


# --------------------------------------------------------------------------- routing
@triton.jit
def _route_kernel(
    slots_ptr, block_slot_ptr, block_pair_ptr, P, NB,
    S: tl.constexpr, PB: tl.constexpr, PC: tl.constexpr, BM: tl.constexpr, NBB: tl.constexpr,
):
    """Program s owns arena slot s: finds its pairs and assigns them BM-padded block positions.

    block_pair[pos] = pair id (pos = block_start[s]*BM + rank of the pair inside slot s), -1 = padding
    block_slot[b]   = s for the blocks of slot s, -1 for the unused tail [total_blocks, NB)
    Block starts come from a per-slot histogram that every program recomputes in PC-sized chunks
    (O(P*S) per program: fine for P <= 4096 and S <= a few hundred slots).
    """
    s = tl.program_id(0)
    slot_ids = tl.arange(0, S)
    counts = tl.zeros([S], dtype=tl.int32)
    for c in range(0, PB, PC):
        offs_c = c + tl.arange(0, PC)
        v = tl.load(slots_ptr + offs_c, mask=offs_c < P, other=-1)
        counts += tl.sum((v[:, None] == slot_ids[None, :]).to(tl.int32), 0)
    nblk = (counts + BM - 1) // BM
    blk_start = tl.cumsum(nblk, 0) - nblk
    my_start = tl.sum(tl.where(slot_ids == s, blk_start, 0))
    my_nblk = tl.sum(tl.where(slot_ids == s, nblk, 0))
    total = tl.sum(nblk)

    # 1) padding: my blocks' pair rows <- -1, and (program 0) the unused block tail <- -1
    offs_pad = tl.arange(0, NBB * BM)
    tl.store(block_pair_ptr + my_start * BM + offs_pad, tl.full([NBB * BM], -1, tl.int32), mask=offs_pad < my_nblk * BM)
    bo = tl.arange(0, NBB)
    tl.store(block_slot_ptr + my_start + bo, tl.full([NBB], 0, tl.int32) + s, mask=bo < my_nblk)
    if s == 0:  # the unused tail [total, NB) can be longer than NBB -> loop
        for c in range(0, NB, NBB):
            idx = total + c + bo
            tl.store(block_slot_ptr + idx, tl.full([NBB], -1, tl.int32), mask=idx < NB)
    tl.debug_barrier()  # the -1 fill must land before this program's own pair ids below
    # 2) my pairs, in increasing pair order
    offs = tl.arange(0, PB)
    flat = tl.load(slots_ptr + offs, mask=offs < P, other=-1)
    mine = flat == s
    rank = tl.cumsum(mine.to(tl.int32), 0) - mine.to(tl.int32)
    tl.store(block_pair_ptr + my_start * BM + rank, offs.to(tl.int32), mask=mine)


def _next_pow2(v: int) -> int:
    return 1 << max(0, (v - 1).bit_length())


def build_routing_small(slots: torch.Tensor, BM: int):
    """Decode-sized routing (P <= 64 pairs) with plain torch ops: sort the pairs by slot, one BM-block per
    distinct slot (a slot never has more than BM pairs at this size). Static shapes, graph-capturable,
    ~10 tiny kernels instead of one program per ARENA slot (the Triton router costs ~29 ms per step
    with a 4,800-slot arena). Same output contract as build_routing; NB = P."""
    T, K = slots.shape
    P = T * K
    flat = slots.reshape(-1).to(torch.int32)
    order = torch.argsort(flat, stable=True)
    ss = flat[order]
    first = torch.ones(P, dtype=torch.bool, device=flat.device)
    first[1:] = ss[1:] != ss[:-1]
    blk = torch.cumsum(first.to(torch.int32), 0) - 1            # block id per sorted pair
    start = torch.cummax(torch.where(first, torch.arange(P, device=flat.device, dtype=torch.int32), torch.zeros_like(blk)), 0).values
    rank = torch.arange(P, device=flat.device, dtype=torch.int32) - start
    block_pair = torch.full((P * BM,), -1, dtype=torch.int32, device=flat.device)
    block_pair[blk * BM + rank] = order.to(torch.int32)
    block_slot = torch.full((P,), -1, dtype=torch.int32, device=flat.device)
    block_slot[blk] = ss
    return block_slot, block_pair, P


def build_routing(slots: torch.Tensor, n_slots: int, BM: int):
    """Group the (token, k) pairs by arena slot, each slot's run padded to a multiple of BM, so that
    one program == one expert x BM pairs. One small Triton launch,
    no host sync, so the CPU launch overhead of the main kernels overlaps with GPU work.

    Returns (block_slot i32[NB], block_pair i32[NB*BM], NB) with -1 marking unused blocks / padding rows.
    NB is an upper bound: ceil(P/BM) + min(n_slots, P).
    """
    T, K = slots.shape
    P = T * K
    dev = slots.device
    flat = slots.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    NB = (P + BM - 1) // BM + min(n_slots, P)
    block_slot = torch.empty((NB,), dtype=torch.int32, device=dev)
    block_pair = torch.empty((NB * BM,), dtype=torch.int32, device=dev)
    S = _next_pow2(max(n_slots, 2))
    PB = _next_pow2(max(P, 16))
    _route_kernel[(n_slots,)](
        flat, block_slot, block_pair, P, NB,
        S=S, PB=PB, PC=max(16, min(PB, 8192 // S)), BM=BM, NBB=_next_pow2((P + BM - 1) // BM + 1), num_warps=4,
    )
    return block_slot, block_pair, NB


def _pick_bm(P: int) -> int:
    if P <= 64:
        return 16
    if P <= 1024:
        return 32
    return 64


# (BN, num_warps, num_stages) per kernel and BM, from the sweep on GB10 (tools/test_fp4_moe.py)
_UP_CFG = {16: (256, 4, 1), 32: (128, 4, 1), 64: (64, 4, 1)}
_DOWN_CFG = {16: (128, 8, 3), 32: (128, 4, 3), 64: (128, 8, 3)}


def moe_forward(
    x: torch.Tensor,
    slots: torch.Tensor,
    weights: torch.Tensor,
    arena: ExpertArena,
    swiglu_limit: float = 10.0,
    block_m: int | None = None,
    up_cfg: tuple[int, int, int] | None = None,
    down_cfg: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """x bf16 [T, 5120], slots int32 [T, K] (arena slot per routed expert), weights fp32 [T, K] -> bf16 [T, 5120]."""
    assert x.dtype == torch.bfloat16 and x.shape[1] == DIM and x.is_contiguous()
    T, K = slots.shape
    assert weights.shape == (T, K)
    P = T * K
    dev = x.device
    BM = block_m or _pick_bm(P)
    bn1, nw1, ns1 = up_cfg or _UP_CFG[BM]
    bn2, nw2, ns2 = down_cfg or _DOWN_CFG[BM]
    if P <= 64:  # decode-sized call (<= 64 pairs, <= 6 per slot < BM): pure-torch routing, see build_routing_small
        block_slot, block_pair, NB = build_routing_small(slots, BM)
    else:
        block_slot, block_pair, NB = build_routing(slots, arena.slots, BM)
    wgt = weights.reshape(-1)
    if wgt.dtype != torch.float32 or not wgt.is_contiguous():
        wgt = wgt.float().contiguous()

    h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
    parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)  # one row per (token, k) pair
    _moe_up_kernel[(NB, INTER // bn1)](
        x, arena.w1, arena.s1, arena.w3, arena.s3, h,
        wgt, block_slot, block_pair,
        x.stride(0), h.stride(0), float(swiglu_limit),
        TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1,
    )
    _moe_down_kernel[(NB, DIM // bn2)](
        h, arena.w2, arena.s2, parts,
        block_slot, block_pair,
        h.stride(0), parts.stride(0),
        TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T, num_warps=nw2, num_stages=ns2,
    )
    return parts.view(K, T, DIM).sum(dim=0).to(torch.bfloat16)


@torch.no_grad()
def moe_forward_reference(
    x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena: ExpertArena, swiglu_limit: float = 10.0
) -> torch.Tensor:
    """Dequant + torch matmul per expert, mirroring inference/model.py::Expert.forward (bf16 GEMMs, fp32 epilogue)."""
    T, K = slots.shape
    y = torch.zeros((T, DIM), dtype=torch.float32, device=x.device)
    for s in torch.unique(slots).tolist():
        w1, w2, w3 = arena.dequant_slot(s)
        t_idx, k_idx = torch.nonzero(slots == s, as_tuple=True)
        xs = x[t_idx]
        gate = (xs @ w1.T).float()
        up = (xs @ w3.T).float()
        if swiglu_limit > 0:
            up = up.clamp(-swiglu_limit, swiglu_limit)
            gate = gate.clamp(max=swiglu_limit)
        hs = torch.nn.functional.silu(gate) * up * weights[t_idx, k_idx].float()[:, None]
        y.index_add_(0, t_idx, (hs.to(torch.bfloat16) @ w2.T).float())
    return y.to(torch.bfloat16)
