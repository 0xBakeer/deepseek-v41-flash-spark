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
