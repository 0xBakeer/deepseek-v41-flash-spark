"""Unit test for tools/fp4_linear.py against real checkpoint weights.

1. round trip: FP4Weight.dequant() vs the fp32 the quantizer saw -- relative weight error, and the
   worst 32-wide group's error, per weight.
2. kernel vs `F.linear(x, W.dequant())` (the dequantized reference) at M = 1, 6, 16, 64, 2048.
3. row-count / row-offset invariance (the graphed decode path replays the same call at any M).
4. effective bandwidth of the weight bytes at M = 6, against the fp8 kernel on the same weight.

    python tools/test_fp4_linear.py [name ...]
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fp4_linear import FP4Weight, fp4_linear, quantize_fp8_to_fp4  # noqa: E402
from fp8_linear import FP8Weight, fp8_linear  # noqa: E402

MD = os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/DeepSeek-V4.1-Flash"))
DEFAULT = [
    "layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w2", "layers.7.ffn.shared_experts.w3",
    "layers.0.attn.wq_a", "layers.0.attn.wq_b", "layers.7.attn.wkv", "layers.7.attn.wo_b",
    "mtp.0.ffn.shared_experts.w1", "mtp.0.attn.wq_b",
]


def main(names):
    idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
    handles = {}

    def get(n):
        f = idx[n]
        if f not in handles:
            handles[f] = safe_open(os.path.join(MD, f), "pt", device="cpu")
        return handles[f].get_tensor(n)

    torch.manual_seed(0)
    use_f16 = os.environ.get("DSV41_FP4_DENSE_F16", "1") == "1"
    print(f"activation dtype into tl.dot: {'fp16' if use_f16 else 'bf16'}")
    ok = True
    for name in names:
        W8 = FP8Weight(get(name + ".weight").cuda(), get(name + ".scale").cuda())
        ref32 = W8.dequant().float()                      # what the quantizer sees
        W4 = quantize_fp8_to_fp4(W8)
        deq = W4.dequant().float()
        rel_w = float((deq - ref32).norm() / ref32.norm())
        g = (deq - ref32).reshape(-1, 32).norm(dim=1) / ref32.reshape(-1, 32).norm(dim=1).clamp_min(1e-20)
        b8 = W8.w.numel() + W8.s.numel()
        b4 = W4.nbytes
        print(f"\n{name}  [{W4.N}, {W4.K}]  fp8 {b8/2**20:.1f} MiB -> fp4 {b4/2**20:.1f} MiB "
              f"({b4/b8:.4f}x)   weight rel err {rel_w:.4f}   per-32-group rel err "
              f"mean {float(g.mean()):.4f} p99 {float(g.quantile(0.99)):.4f} max {float(g.max()):.4f}")
        ref_bf = W4.dequant()
        for M in (1, 6, 16, 64, 2048):
            x = (torch.randn(M, W4.K, device="cuda") * 0.5).to(torch.bfloat16)
            y = fp4_linear(x, W4)
            r = F.linear(x, ref_bf)
            rel = float((y.float() - r.float()).norm() / r.float().norm())
            mx = float((y.float() - r.float()).abs().max())
            # fp32 reference on the same fp4 weights: says which side of the bf16 comparison moved
            r32 = x.float() @ deq.T
            rel32 = float((y.float() - r32).norm() / r32.norm())
            relc = float((r.float() - r32).norm() / r32.norm())
            # against the fp8 weight, i.e. the error the model actually sees
            r8 = F.linear(x, W8.dequant())
            rel8 = float((y.float() - r8.float()).norm() / r8.float().norm())
            bad = rel32 > 5e-3
            ok &= not bad
            print(f"  M={M:5d}: rel vs fp4 ref {rel:.2e}  max abs {mx:.2e}   vs fp32(fp4) "
                  f"kernel {rel32:.2e} / cuBLAS-bf16 {relc:.2e}   rel vs fp8 ref {rel8:.4f}"
                  + ("   <-- FAIL" if bad else ""))
        # row-count / row-offset invariance
        x = (torch.randn(2048, W4.K, device="cuda") * 0.5).to(torch.bfloat16)
        full = fp4_linear(x, W4)
        for lo, hi in ((0, 6), (7, 13), (1000, 1006), (0, 1), (0, 16)):
            sub = fp4_linear(x[lo:hi].contiguous(), W4)
            same = bool((sub == full[lo:hi]).all())
            ok &= same
            print(f"  rows[{lo}:{hi}] identical to the 2048-row call: {same}")
        # Bandwidth at M=6, against the fp8 kernel on the same weight. Steady state: the call is
        # cycled over ~300 MB worth of copies of the weight, so nothing is L2-resident and -- unlike
        # an explicit L2 flush -- no flush write competes for DRAM bandwidth. `hot` (one copy,
        # repeated) is the L2-resident upper bound, which flatters small weights by 2-3x.
        x6 = (torch.randn(6, W4.K, device="cuda") * 0.5).to(torch.bfloat16)
        nc = max(2, min(12, int(300e6 // b4)))
        c4 = [W4] + [FP4Weight(W4.w.clone(), W4.s.clone(), W4.N, W4.K) for _ in range(nc - 1)]
        c8 = [W8] + [FP8Weight(W8.w.clone(), W8.s.clone()) for _ in range(nc - 1)]
        for lbl, fn, nb in (("fp4", lambda c: fp4_linear(x6, c4[c]), b4),
                            ("fp8", lambda c: fp8_linear(x6, c8[c]), b8)):
            for c in range(nc):
                fn(c)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(4):
                for c in range(nc):
                    fn(c)
            torch.cuda.synchronize()
            cold = (time.perf_counter() - t0) / (4 * nc)
            t0 = time.perf_counter()
            for _ in range(50):
                fn(0)
            torch.cuda.synchronize()
            hot = (time.perf_counter() - t0) / 50
            print(f"  M=6 {lbl}: steady {cold*1e6:7.1f} us = {nb/cold/1e9:6.1f} GB/s   "
                  f"L2-hot {hot*1e6:7.1f} us = {nb/hot/1e9:6.1f} GB/s   ({nb/2**20:.1f} MiB of weights)")
        del c4, c8
        del W8, W4, ref32, deq, ref_bf
        torch.cuda.empty_cache()
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT))
