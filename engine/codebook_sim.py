"""
codebook_sim.py -- simulate a lower-bit expert format WITHOUT a new kernel or packing.

The FP4 experts are codes 0..15 on the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, ...} times a
UE8M0 scale per 32 weights. A B-bit "codebook" format keeps, per matrix ROW, the 2^B grid levels
that best represent that row (weighted by how often each level occurs and by the group scale), and
stores every weight as an index into that codebook. Because the codebook is a SUBSET of the FP4
grid, the quantized matrix can be written back into the FP4 arena as ordinary FP4 codes, so the
existing Triton kernel computes with it unchanged. Memory is not saved in this simulation; the
teacher-forced loss it produces is exactly what the packed format would give.

Selection: for each row, hist[level] = sum over 32-groups of scale_g^2 * count(level in group);
cost(subset) = sum_level hist[level] * min_{m in subset} (v_level - v_m)^2; pick the best subset
among all C(16, 2^B) (12,870 for 3 bits, 1,820 for 2 bits) with one small GEMM per matrix.
"""

from __future__ import annotations

import itertools

import torch

FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


class CodebookSim:
    def __init__(self, bits: int, device: str):
        self.bits = bits
        k = 2 ** bits
        subsets = list(itertools.combinations(range(16), k))
        vals = FP4_VALS
        S = len(subsets)
        # nearest member and its squared error, per (subset, level)
        near = torch.zeros(S, 16, dtype=torch.long)
        cost = torch.zeros(S, 16, dtype=torch.float32)
        for i, sub in enumerate(subsets):
            members = torch.tensor(sub)
            d = (vals[None, :] - vals[members][:, None]).abs()  # [k, 16]
            j = d.argmin(dim=0)
            near[i] = members[j]
            cost[i] = (d.min(dim=0).values ** 2)
        # position of each level's nearest member INSIDE the subset (0..k-1). `near` gives the FP4
        # code, `pos` gives the codebook index, which is what a packed low-bit format stores; having
        # it here means the packer never has to search for a code in its row's codebook.
        pos = torch.zeros(S, 16, dtype=torch.uint8)
        for i, sub in enumerate(subsets):
            lut = {c: j for j, c in enumerate(sub)}
            pos[i] = torch.tensor([lut[int(c)] for c in near[i]], dtype=torch.uint8)
        self.subsets = subsets
        self.subsets_t = torch.tensor(subsets, dtype=torch.uint8, device=device)  # [S, k]
        self.near = near.to(device)
        self.pos = pos.to(device)
        self.cost = cost.to(device)
        self.device = device

    @torch.no_grad()
    def requant_packed(self, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """w: uint8 [N, K/2] packed FP4 codes (low nibble = even element); s: uint8 [N, K/32] ue8m0.
        Returns the re-quantized packed codes (same layout)."""
        N, K2 = w.shape
        K = K2 * 2
        lo = (w & 0x0F).long()
        hi = ((w >> 4) & 0x0F).long()
        codes = torch.stack([lo, hi], dim=-1).reshape(N, K)  # [N, K]
        scale2 = torch.exp2(2.0 * (s.float() - 127.0))  # [N, K/32]
        wgt = scale2.repeat_interleave(32, dim=1)  # [N, K]
        hist = torch.zeros(N, 16, device=w.device, dtype=torch.float32)
        hist.scatter_add_(1, codes, wgt)
        # best subset per row: [N, S]
        total = hist @ self.cost.T
        best = total.argmin(dim=1)  # [N]
        new_codes = self.near[best][torch.arange(N, device=w.device)[:, None], codes]  # [N, K]
        lo2 = new_codes[:, 0::2]
        hi2 = new_codes[:, 1::2]
        return (lo2 | (hi2 << 4)).to(torch.uint8)

    @torch.no_grad()
    def requant_slot(self, arena, slot: int):
        for wn, sn in (("w1", "s1"), ("w2", "s2"), ("w3", "s3")):
            w = getattr(arena, wn)[slot]
            s = getattr(arena, sn)[slot]
            w.copy_(self.requant_packed(w, s))


if __name__ == "__main__":
    # sanity: relative error of the simulated format on a real expert
    import json, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import v41_ref as R
    from safetensors import safe_open
    md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
    idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
    f = safe_open(f"{md}/{idx['layers.0.ffn.experts.0.w1.weight']}", "pt", device="cpu")
    for bits in (3, 2):
        sim = CodebookSim(bits, "cuda")
        for e in (0, 7):
            w = f.get_tensor(f"layers.0.ffn.experts.{e}.w1.weight").cuda().view(torch.uint8)
            s = f.get_tensor(f"layers.0.ffn.experts.{e}.w1.scale").cuda().view(torch.uint8)
            ref = R.dequant_fp4_packed(w, s).float()
            q = R.dequant_fp4_packed(sim.requant_packed(w, s), s).float()
            print(f"bits={bits} expert {e} w1: rel err {(q - ref).norm() / ref.norm():.4f}  unchanged codes {(q == ref).float().mean():.3f}")
