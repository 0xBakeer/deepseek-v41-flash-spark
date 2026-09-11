"""
cb3.py -- "CB3": a 3-bit expert format derived from the checkpoint's FP4 experts.

Per matrix row: a codebook of 8 FP4 grid codes (the 8-of-16 subset that best represents the row,
see engine/codebook_sim.py) and one 3-bit index per weight. The UE8M0 scale per 32 weights is kept
as is. Bytes per row of K weights: K*3/8 (codes) + 8 (codebook, one FP4 code per byte) + K/32
(scales) -> 3.0 + 0.25 bit/weight + a negligible codebook: an expert is 14.5 MB instead of 18.8.

Plane layout (Triton-friendly, no 3-byte unpacking): the two low bits of all K codes come first,
4 per byte (K/4 bytes), then the high bit of all K codes, 8 per byte (K/8 bytes). Weight k has
low bits (lo[k // 4] >> (2 * (k % 4))) & 3 and high bit (hi[k // 8] >> (k % 8)) & 1.
The dequantized value is FP4_TABLE[codebook[row, code]] * scale[row, k // 32].
"""

from __future__ import annotations

import torch

FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def pack_cb3(codes: torch.Tensor, codebook: torch.Tensor):
    """codes: long [N, K] FP4 codes already restricted to the row codebook; codebook: long [N, 8].
    Returns (lo uint8 [N, K/4], hi uint8 [N, K/8], cb uint8 [N, 8]) where each weight is the 3-bit
    index of its code inside its row's codebook."""
    N, K = codes.shape
    assert K % 8 == 0
    # index of each code within the row codebook
    eq = codes[:, :, None] == codebook[:, None, :]  # [N, K, 8]
    assert bool(eq.any(-1).all()), "a code is not in its row codebook"
    idx = eq.float().argmax(-1)  # [N, K] in 0..7
    lo2 = (idx & 3).view(N, K // 4, 4)
    lo = (lo2[..., 0] | (lo2[..., 1] << 2) | (lo2[..., 2] << 4) | (lo2[..., 3] << 6)).to(torch.uint8)
    hb = ((idx >> 2) & 1).view(N, K // 8, 8)
    hi = torch.zeros(N, K // 8, dtype=torch.long, device=codes.device)
    for j in range(8):
        hi |= hb[..., j] << j
    return lo, hi.to(torch.uint8), codebook.to(torch.uint8)


def unpack_cb3(lo: torch.Tensor, hi: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """-> long [N, K] FP4 codes."""
    N = lo.size(0)
    K = lo.size(1) * 4
    lo = lo.long(); hi = hi.long()
    idx = torch.zeros(N, K, dtype=torch.long, device=lo.device)
    for j in range(4):
        idx[:, j::4] |= (lo >> (2 * j)) & 3
    for j in range(8):
        idx[:, j::8] |= ((hi >> j) & 1) << 2
    return cb.long().gather(1, idx)


def dequant_cb3(lo, hi, cb, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """-> bf16 [N, K] (same math as v41_ref.dequant_fp4_packed on the re-quantized codes)."""
    codes = unpack_cb3(lo, hi, cb)
    vals = FP4_TABLE.to(codes.device)[codes]
    s = torch.exp2(scale_e8m0.view(torch.uint8).float() - 127.0).repeat_interleave(32, 1)
    return (vals * s).to(torch.bfloat16)


def fp4_to_cb3(w_packed: torch.Tensor, scale: torch.Tensor, sim) -> tuple:
    """Convert one packed-FP4 matrix (uint8 [N, K/2] + scale [N, K/32]) to CB3 using a
    engine.codebook_sim.CodebookSim(3) instance for the per-row codebook choice."""
    N, K2 = w_packed.shape
    x = w_packed.view(torch.uint8)
    codes = torch.stack([(x & 0x0F).long(), ((x >> 4) & 0x0F).long()], dim=-1).reshape(N, K2 * 2)
    scale2 = torch.exp2(2.0 * (scale.view(torch.uint8).float() - 127.0)).repeat_interleave(32, dim=1)
    hist = torch.zeros(N, 16, device=x.device, dtype=torch.float32).scatter_add_(1, codes, scale2)
    best = (hist @ sim.cost.T).argmin(dim=1)
    new_codes = sim.near[best][torch.arange(N, device=x.device)[:, None], codes]
    codebook = torch.tensor(sim.subsets, device=x.device)[best]  # [N, 8]
    return pack_cb3(new_codes, codebook)


# --------------------------------------------------------------------------- v2 layout
# The v1 plane layout is correct but no Triton kernel can consume it at bandwidth: byte j of the
# rebuilt FP4 tile needs lo[j // 2] and hi[j // 4], i.e. cross-element movement inside a register
# tile. Doing that with gather loads costs 4-8 GB/s, with register reshapes 39-54 GB/s.
#
# v2 keeps exactly the same bytes per weight (2 bits in the lo plane + 1 bit in the hi plane, 8
# codebook bytes and the UE8M0 scales per row) and only changes WHICH weight goes in which bit, so
# that a kernel can build the packed-FP4 byte tile of a scale group with elementwise ops on tiles of
# the same width. A dot product is order-invariant along K and the UE8M0 scale is per 32 consecutive
# K, so any permutation inside a 32-group is free as long as the activation is loaded to match --
# and the permutation below is chosen so the existing `_chunk_dot` x-load pattern (even offsets,
# then odd offsets) still matches.
#
# One block = 256 weights = 8 scale groups = 64 lo bytes + 32 hi bytes. Per block:
#   group g (0..7) -> lo sub-tile k = g // 2 (16 bytes), 2-bit shift 4 * (g % 2) [+2 for the odd half]
#                  -> hi sub-tile m = k // 2 (16 bytes), bit (k % 2) * 4 + (g % 2) * 2 [+1 for odd]
#   weight K-offset 32 * g + 2 * e + r  lives at lo byte 16 * k + e and hi byte 16 * m + e.
# So a [BN, 64] lo load and a [BN, 32] hi load, split in registers into four and two [BN, 16] tiles,
# give every group its even and its odd nibble tile with nothing but shifts and masks.
# Block size in weights. The row tile a kernel loads is BLOCK_W/4 lo bytes + BLOCK_W/8 hi bytes, and
# on GB10 the achievable read bandwidth of a row-strided tile is a cliff in its width: 16 B 101 GB/s,
# 32 B 101, 64 B 185, 128 B 218 (measured). 256 gives a 64 B lo tile but only a 32 B hi tile, which
# caps the whole kernel; 512 gives 128 B + 64 B and is what w1/w3 (K = 5120) use. K = 2304 (w2) is
# 2^8 * 9, so 256 is the largest power-of-two block it admits and its hi tile stays 32 B wide.
BLOCK_W = 512


def block_plan(K: int) -> tuple[int, int]:
    """(number of 512-weight blocks, number of trailing 256-weight blocks) for a row of K weights.
    K = 5120 (w1/w3) -> (10, 0); K = 2304 = 4*512 + 256 (w2) -> (4, 1). The 256 tail exists only
    because 2304 = 2^8 * 9 has no larger power-of-two factor; without it w2's hi tile would be 32 B
    wide for the whole row and cap the kernel at ~100 GB/s."""
    assert K % 256 == 0, K
    return K // 512, (K % 512) // 256


def block_w_for(K: int) -> int:
    return 512 if K % 512 == 0 else 256


def _v2_fields(block_w: int = BLOCK_W):
    """(g, r) -> (k, lo_shift, m, hi_bit). Same formulas for any block size; only the counts grow."""
    out = []
    for g in range(block_w // 32):
        k, gg = g // 2, g % 2
        for r in range(2):
            out.append((g, r, k, 4 * gg + 2 * r, k // 2, (k % 2) * 4 + gg * 2 + r))
    return out


def pack_idx_v2(idx: torch.Tensor, codebook: torch.Tensor, block_w: int | None = None):
    """Pack codebook INDICES (uint8/int16 [N, K], values 0..7) into the v2 planes. This is the hot
    path of the arena packer: `pack_cb3_v2` has to find each code's position in its row codebook with
    an [N, K, 8] comparison (94 MB of temporaries per matrix), which `CodebookSim.pos` makes
    unnecessary."""
    N, K = idx.shape
    if block_w is None:
        n512, n256 = block_plan(K)
        if n512 and n256:
            cut = n512 * 512
            a = pack_idx_v2(idx[:, :cut], codebook, 512)
            b = pack_idx_v2(idx[:, cut:], codebook, 256)
            return (torch.cat([a[0], b[0]], 1), torch.cat([a[1], b[1]], 1), a[2])
        block_w = 512 if n512 else 256
    NB, G = K // block_w, block_w // 32
    v5 = idx.view(N, NB, G, 16, 2).to(torch.int16)
    lo = torch.zeros(N, NB, G // 2, 16, dtype=torch.int16, device=idx.device)
    hi = torch.zeros(N, NB, G // 4, 16, dtype=torch.int16, device=idx.device)
    for g, r, k, sh, m, hb in _v2_fields(block_w):
        v = v5[:, :, g, :, r]
        lo[:, :, k, :] |= (v & 3) << sh
        hi[:, :, m, :] |= ((v >> 2) & 1) << hb
    return (lo.reshape(N, K // 4).to(torch.uint8), hi.reshape(N, K // 8).to(torch.uint8),
            codebook.to(torch.uint8))


def pack_cb3_v2(codes: torch.Tensor, codebook: torch.Tensor, block_w: int | None = None):
    """Same contract as `pack_cb3`, v2 bit layout. codes/codebook: long.

    A row is packed as `block_plan(K)`: n512 blocks of 512 weights followed by n256 blocks of 256.
    The two parts are simply concatenated in both planes, so the 512 part owns lo[:n512*128] /
    hi[:n512*64] and the tail follows it."""
    N, K = codes.shape
    if block_w is None:
        n512, n256 = block_plan(K)
        if n512 and n256:
            cut = n512 * 512
            a = pack_cb3_v2(codes[:, :cut], codebook, 512)
            b = pack_cb3_v2(codes[:, cut:], codebook, 256)
            return (torch.cat([a[0], b[0]], 1), torch.cat([a[1], b[1]], 1), a[2])
        block_w = 512 if n512 else 256
    bw = block_w
    assert K % bw == 0, (N, K, bw)
    NB, G = K // bw, bw // 32
    eq = codes[:, :, None] == codebook[:, None, :]
    assert bool(eq.any(-1).all()), "a code is not in its row codebook"
    idx = eq.to(torch.uint8).argmax(-1).long().view(N, NB, G, 16, 2)  # [N, b, g, e, r]
    lo = torch.zeros(N, NB, G // 2, 16, dtype=torch.long, device=codes.device)
    hi = torch.zeros(N, NB, G // 4, 16, dtype=torch.long, device=codes.device)
    for g, r, k, sh, m, hb in _v2_fields(bw):
        v = idx[:, :, g, :, r]
        lo[:, :, k, :] |= (v & 3) << sh
        hi[:, :, m, :] |= ((v >> 2) & 1) << hb
    return (lo.reshape(N, K // 4).to(torch.uint8), hi.reshape(N, K // 8).to(torch.uint8),
            codebook.to(torch.uint8))


def unpack_cb3_v2(lo: torch.Tensor, hi: torch.Tensor, cb: torch.Tensor,
                  block_w: int | None = None) -> torch.Tensor:
    """-> long [N, K] FP4 codes in natural K order."""
    N = lo.size(0)
    K = lo.size(1) * 4
    if block_w is None:
        n512, n256 = block_plan(K)
        if n512 and n256:
            a = unpack_cb3_v2(lo[:, :n512 * 128], hi[:, :n512 * 64], cb, 512)
            b = unpack_cb3_v2(lo[:, n512 * 128:], hi[:, n512 * 64:], cb, 256)
            return torch.cat([a, b], 1)
        block_w = 512 if n512 else 256
    bw = block_w
    NB, G = K // bw, bw // 32
    lo = lo.long().view(N, NB, G // 2, 16)
    hi = hi.long().view(N, NB, G // 4, 16)
    idx = torch.zeros(N, NB, G, 16, 2, dtype=torch.long, device=lo.device)
    for g, r, k, sh, m, hb in _v2_fields(bw):
        idx[:, :, g, :, r] = ((lo[:, :, k, :] >> sh) & 3) | (((hi[:, :, m, :] >> hb) & 1) << 2)
    return cb.long().gather(1, idx.reshape(N, K))


def dequant_cb3_v2(lo, hi, cb, scale_e8m0: torch.Tensor, block_w: int | None = None) -> torch.Tensor:
    codes = unpack_cb3_v2(lo, hi, cb, block_w)
    vals = FP4_TABLE.to(codes.device)[codes]
    s = torch.exp2(scale_e8m0.view(torch.uint8).float() - 127.0).repeat_interleave(32, 1)
    return (vals * s).to(torch.bfloat16)


def fp4_to_cb3_v2(w_packed: torch.Tensor, scale: torch.Tensor, sim) -> tuple:
    """One packed-FP4 matrix -> the v2 CB3 planes, via `sim`'s per-row subset choice.

    Everything stays in the smallest dtype that fits: the codes are uint8, the histogram scatter is
    the only fp32 step, and the codebook index comes straight out of `sim.pos` instead of an
    [N, K, 8] equality test. Bit-identical to `pack_cb3_v2(sim.near[best][codes], subsets[best])`.
    """
    N, K2 = w_packed.shape
    x = w_packed.view(torch.uint8)
    codes = torch.stack([x & 0x0F, (x >> 4) & 0x0F], dim=-1).reshape(N, K2 * 2)  # uint8 [N, K]
    scale2 = torch.exp2(2.0 * (scale.view(torch.uint8).float() - 127.0)).repeat_interleave(32, dim=1)
    hist = torch.zeros(N, 16, device=x.device, dtype=torch.float32).scatter_add_(1, codes.long(), scale2)
    best = (hist @ sim.cost.T).argmin(dim=1)  # [N]
    rows = torch.arange(N, device=x.device)[:, None]
    idx = sim.pos[best][rows, codes.long()]           # [N, K] uint8, 0..7
    codebook = sim.subsets_t[best].long()             # [N, 8]
    return pack_idx_v2(idx, codebook)


if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v41_ref as R
    from codebook_sim import CodebookSim
    from safetensors import safe_open
    md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
    idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(f"{md}/{idx['layers.0.ffn.experts.0.w1.weight']}", "pt", device="cpu")
    sim = CodebookSim(3, "cuda")
    for name in ("layers.0.ffn.experts.3.w1", "layers.0.ffn.experts.3.w2"):
        w = f.get_tensor(name + ".weight").cuda().view(torch.uint8); s = f.get_tensor(name + ".scale").cuda().view(torch.uint8)
        lo, hi, cb = fp4_to_cb3(w, s, sim)
        deq = dequant_cb3(lo, hi, cb, s)
        ref_sim = R.dequant_fp4_packed(sim.requant_packed(w, s), s)
        ref = R.dequant_fp4_packed(w, s)
        print(f"{name}: cb3 == simulated-codebook dequant: {bool((deq == ref_sim).all())}; rel err vs FP4 {(deq.float() - ref.float()).norm() / ref.float().norm():.4f}; "
              f"bytes {lo.numel() + hi.numel() + cb.numel() + s.numel()} vs fp4 {w.numel() + s.numel()}")
        lo2, hi2, cb2 = fp4_to_cb3_v2(w, s, sim)
        deq2 = dequant_cb3_v2(lo2, hi2, cb2, s)
        print(f"{name}: v2 == v1 dequant: {bool((deq2 == deq).all())}; v2 == simulated-codebook dequant: {bool((deq2 == ref_sim).all())}; "
              f"bytes {lo2.numel() + hi2.numel() + cb2.numel() + s.numel()}")
