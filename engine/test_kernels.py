"""Unit tests for the two decode kernels added for the wo_a projection and the sinked softmax
attention: tools/fp8_linear.fp8_grouped_linear and tools/decode_attn.decode_attention.

wo_a is checked against the real checkpoint weight (layer 0 and one DSpark block), dequantized the
way convert.py/v41_ref did it, at decode (T=6) and prefill (T=2048) row counts. The attention kernel
is checked against the fp32 torch path it replaces at the shapes the engine actually runs, including
a row whose keys are all masked, and once more inside a CUDA graph replay.

Run:  python engine/test_kernels.py
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from safetensors import safe_open  # noqa: E402

import v41_ref as R  # noqa: E402
from fp8_linear import FP8GroupedWeight, fp8_grouped_linear  # noqa: E402
from fp4_linear import (FP4GroupedWeight, fp4_grouped_linear,  # noqa: E402
                        quantize_fp8_grouped_to_fp4)
from decode_attn import decode_attention, decode_attention_ref  # noqa: E402
from fp32_skinny import skinny_linear  # noqa: E402

MD = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
fails = []


def check(name, got, ref, tol_rel, tol_abs=None):
    g, r = got.float(), ref.float()
    rel = float((g - r).norm() / r.norm())
    mx = float((g - r).abs().max())
    scale = float(r.abs().max())
    ok = rel <= tol_rel and (tol_abs is None or mx <= tol_abs)
    print(f"  {'PASS' if ok else 'FAIL'} {name:52s} rel {rel:.3e}  max|d| {mx:.3e}  (max|ref| {scale:.3e})")
    if not ok:
        fails.append(name)


# ----------------------------------------------------------------- 1. grouped fp8 wo_a
print("== wo_a grouped fp8 GEMM vs the bf16 einsum it replaces ==")
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
args = R.Args.from_json(f"{MD}/config.json") if os.path.exists(f"{MD}/config.json") else R.Args()
torch.manual_seed(0)
for name in ("layers.0.attn.wo_a", "layers.7.attn.wo_a", "mtp.0.attn.wo_a"):
    if name + ".weight" not in idx:
        print(f"  SKIP {name} (not in the checkpoint index)")
        continue
    f = safe_open(f"{MD}/{idx[name + '.weight']}", "pt", device="cpu")
    wq = f.get_tensor(name + ".weight").cuda()
    sc = f.get_tensor(name + ".scale").cuda()
    W = FP8GroupedWeight(wq, sc, args.o_groups, args.o_lora_rank)
    ref_w = W.dequant()  # [G, R, K] bf16 -- exactly what dequant_fp8_block(...).view(...) produced
    old = R.dequant_fp8_block(wq, sc).view(args.o_groups, args.o_lora_rank, -1)
    assert torch.equal(ref_w, old), "dequant of the grouped weight must match dequant_fp8_block"
    for T in (1, 5, 6, 16, 64, 2048):
        x = (torch.randn(T, W.G, W.K, device="cuda") * 0.25).to(torch.bfloat16)
        y = fp8_grouped_linear(x, W)
        r = torch.einsum("sgd,grd->sgr", x, ref_w)
        check(f"{name} T={T}", y, r, 6e-3, None)
    # row-count / row-offset invariance: the same row must come out bit-identical in any call
    x = (torch.randn(2048, W.G, W.K, device="cuda") * 0.25).to(torch.bfloat16)
    full = fp8_grouped_linear(x, W)
    for lo, hi in ((0, 6), (7, 13), (1000, 1006)):
        sub = fp8_grouped_linear(x[lo:hi].contiguous(), W)
        same = bool(torch.equal(sub, full[lo:hi]))
        print(f"  {'PASS' if same else 'FAIL'} {name} rows [{lo}:{hi}] bit-identical to the full call")
        if not same:
            fails.append(f"{name} invariance {lo}:{hi}")
    torch.cuda.synchronize()

    # ---- the same weight in the packed FP4 format, on the grouped FP4 kernel
    W4 = quantize_fp8_grouped_to_fp4(W)
    deq4 = W4.dequant()                     # [G, R, K] bf16 -- the dequantized reference
    b8, b4 = W.w.numel() + W.s.numel(), W4.nbytes
    g_err = (deq4.float() - ref_w.float()).reshape(-1, 32).norm(dim=1) / \
        ref_w.float().reshape(-1, 32).norm(dim=1).clamp_min(1e-20)
    print(f"  {name} fp4: {b8 / 2**20:.1f} MiB -> {b4 / 2**20:.1f} MiB ({b4 / b8:.4f}x)  weight rel err "
          f"{float((deq4.float() - ref_w.float()).norm() / ref_w.float().norm()):.4f}  per-32-group "
          f"mean {float(g_err.mean()):.4f} p99 {float(g_err.quantile(0.99)):.4f} max {float(g_err.max()):.4f}")
    for T in (1, 6, 16, 64, 2048):
        x = (torch.randn(T, W.G, W.K, device="cuda") * 0.25).to(torch.bfloat16)
        y4 = fp4_grouped_linear(x, W4)
        r4 = torch.einsum("sgd,grd->sgr", x, deq4)          # bf16 einsum on the SAME fp4 weights
        r32 = torch.einsum("sgd,grd->sgr", x.float(), deq4.float())
        rel32 = float((y4.float() - r32).norm() / r32.norm())
        relc = float((r4.float() - r32).norm() / r32.norm())
        check(f"{name} fp4 T={T} (vs bf16 einsum on fp4 weights)", y4, r4, 6e-3, None)
        print(f"       vs fp32(fp4): kernel {rel32:.2e} / einsum-bf16 {relc:.2e}")
    x = (torch.randn(2048, W.G, W.K, device="cuda") * 0.25).to(torch.bfloat16)
    full4 = fp4_grouped_linear(x, W4)
    for lo, hi in ((0, 6), (7, 13), (1000, 1006)):
        sub = fp4_grouped_linear(x[lo:hi].contiguous(), W4)
        same = bool(torch.equal(sub, full4[lo:hi]))
        print(f"  {'PASS' if same else 'FAIL'} {name} fp4 rows [{lo}:{hi}] bit-identical to the full call")
        if not same:
            fails.append(f"{name} fp4 invariance {lo}:{hi}")
    del full4
    torch.cuda.synchronize()

    # ---- bandwidth at T=6. Steady state: cycle the call over ~300 MB of copies of the weight so
    # nothing is L2-resident (an explicit L2 flush would put a flush write on the same DRAM).
    x6 = (torch.randn(6, W.G, W.K, device="cuda") * 0.25).to(torch.bfloat16)
    nc = max(2, min(8, int(300e6 // b4)))
    c8 = [W] + [FP8GroupedWeight(W.w.clone(), W.s.clone(), W.G, W.R) for _ in range(nc - 1)]
    c4 = [W4] + [FP4GroupedWeight(W4.w.clone(), W4.s.clone(), W4.G, W4.R, W4.K) for _ in range(nc - 1)]
    cases = [("fp8 grouped kernel", lambda c: fp8_grouped_linear(x6, c8[c]), b8),
             ("fp4 grouped kernel", lambda c: fp4_grouped_linear(x6, c4[c]), b4),
             ("fp4 grouped BLOCK_N=32", lambda c: fp4_grouped_linear(x6, c4[c], block_n=32), b4),
             ("fp4 grouped BLOCK_N=128", lambda c: fp4_grouped_linear(x6, c4[c], block_n=128), b4),
             ("bf16 einsum (original)", lambda c: torch.einsum("sgd,grd->sgr", x6, ref_w), 2 * b8)]
    for lbl, fn, nb in cases:
        for c in range(nc):
            fn(c)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(6):
            for c in range(nc):
                fn(c)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / (6 * nc)
        print(f"       T=6 {lbl:24s} {dt * 1e6:7.1f} us  ({nb / dt / 1e9:5.0f} GB/s of weights, "
              f"{nb / 2**20:.1f} MiB)")
    del c8, c4, W, W4, deq4, ref_w, old, x, full
    torch.cuda.empty_cache()

# ----------------------------------------------------------------- 2. fused decode attention
print("== fused decode attention vs the fp32 torch path ==")
D = args.head_dim   # 512; the last rope_head_dim dims carry RoPE, they are not extra width
H = args.n_heads
scale = args.head_dim ** -0.5


def case(T, N, masked_rows=(), all_masked_col=False, seed=0, d=None):
    d = D if d is None else d
    torch.manual_seed(seed)
    q = (torch.randn(T, H, d, device="cuda") * 0.5).to(torch.bfloat16)
    kv = (torch.randn(T, N, d, device="cuda") * 0.5).to(torch.bfloat16)
    mask = torch.ones(T, N, dtype=torch.bool, device="cuda")
    mask[:, N // 3: N // 3 + 7] = False            # scattered holes
    for t in masked_rows:
        mask[t] = False                            # a query whose whole key set is masked
    if all_masked_col:
        mask[:, :] = False
        mask[:, 0] = True
        mask[0, 0] = False                         # row 0 fully masked, the rest see one key
    sink = (torch.randn(H, device="cuda") * 0.5).float()
    return q, kv, mask, sink


for T, N, lbl in ((6, 640, f"verify 128 window + 512 csa2, d={D}"), (6, 128, "verify, no compression"),
                  (5, 133, "dspark draft 128 + 5"), (1, 640, "single token")):
    q, kv, mask, sink = case(T, N)
    ref = decode_attention_ref(q, kv, mask, sink, scale)
    for sp in (1, 2, 4):
        got = decode_attention(q, kv, None, mask, sink, scale, split=sp)
        check(f"T={T} n={N} split={sp}  ({lbl})", got, ref, 2e-3, 6e-3)
    got = decode_attention(q, kv, None, mask, sink, scale, split=2, pv_split=0)
    check(f"T={T} n={N} split=2 PV in plain bf16", got, ref, 8e-3, 3e-2)

print("  -- non-power-of-two head dim (exercises the DA+DB split) --")
q, kv, mask, sink = case(6, 640, d=576)
ref = decode_attention_ref(q, kv, mask, sink, scale)
for sp in (1, 2):
    check(f"T=6 n=640 d=576 split={sp}", decode_attention(q, kv, None, mask, sink, scale, split=sp), ref, 2e-3, 6e-3)

print("  -- two key segments (no torch.cat) --")
q, kv, mask, sink = case(6, 640, seed=5)
ref = decode_attention_ref(q, kv, mask, sink, scale)
one = decode_attention(q, kv, None, mask, sink, scale)
for n1 in (128, 512, 32):
    a_, b_ = kv[:, :n1].contiguous(), kv[:, n1:].contiguous()
    got = decode_attention(q, a_, b_, mask, sink, scale)
    same = bool(torch.equal(got, one))
    print(f"  {'PASS' if same else 'FAIL'} split at n1={n1:4d}: bit-identical to the single-tensor call")
    if not same:
        fails.append(f"two-segment n1={n1}")
    check(f"two segments n1={n1} vs fp32 torch", got, ref, 2e-3, 6e-3)

print("  -- draft shape: stride-0 broadcast window + T draft keys --")
torch.manual_seed(11)
Td = 5
win = (torch.randn(128, D, device="cuda") * 0.5).to(torch.bfloat16)
dkv = (torch.randn(Td, D, device="cuda") * 0.5).to(torch.bfloat16)
kv1b = win[None].expand(Td, -1, -1)
kv2b = dkv[None].expand(Td, -1, -1)
mk = torch.ones(Td, 128 + Td, dtype=torch.bool, device="cuda")
mk[:, 3:9] = False
sk_ = (torch.randn(H, device="cuda") * 0.5).float()
qd = (torch.randn(Td, H, D, device="cuda") * 0.5).to(torch.bfloat16)
print(f"       kv1 strides {tuple(kv1b.stride())}  kv2 strides {tuple(kv2b.stride())} (0 on the token axis)")
refd = decode_attention_ref(qd, torch.cat([kv1b, kv2b], dim=1), mk, sk_, scale)
check("draft, broadcast segments", decode_attention(qd, kv1b, kv2b, mk, sk_, scale), refd, 2e-3, 6e-3)

print("  -- all-masked rows --")
q, kv, mask, sink = case(6, 640, masked_rows=(2, 5))
ref = decode_attention_ref(q, kv, mask, sink, scale)
got = decode_attention(q, kv, None, mask, sink, scale)
check("T=6 n=640 rows 2 and 5 fully masked", got, ref, 2e-3, 6e-3)
print(f"       masked rows are exactly zero: ref {bool((ref[[2, 5]] == 0).all())}  kernel {bool((got[[2, 5]] == 0).all())}")
if not bool((got[[2, 5]] == 0).all()):
    fails.append("all-masked rows not zero")
q, kv, mask, sink = case(6, 640, all_masked_col=True)
ref = decode_attention_ref(q, kv, mask, sink, scale)
got = decode_attention(q, kv, None, mask, sink, scale)
check("T=6 n=640 one key visible, row 0 none", got, ref, 2e-3, 6e-3)
print(f"       finite: {bool(torch.isfinite(got.float()).all())}")

print("  -- CUDA graph replay --")
q, kv, mask, sink = case(6, 640, seed=3)
ref = decode_attention_ref(q, kv, mask, sink, scale)
decode_attention(q, kv[:, :128].contiguous(), kv[:, 128:].contiguous(), mask, sink, scale)  # JIT before capture
torch.cuda.synchronize()
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    decode_attention(q, kv[:, :128].contiguous(), kv[:, 128:].contiguous(), mask, sink, scale)
torch.cuda.current_stream().wait_stream(st)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = decode_attention(q, kv[:, :128].contiguous(), kv[:, 128:].contiguous(), mask, sink, scale)
g.replay(); torch.cuda.synchronize()
check("graph replay", out, ref, 2e-3, 6e-3)
q2, kv2, mask2, sink2 = case(6, 640, seed=9)
q.copy_(q2); kv.copy_(kv2); mask.copy_(mask2); sink.copy_(sink2)
g.replay(); torch.cuda.synchronize()
check("graph replay, new inputs", out, decode_attention_ref(q, kv, mask, sink, scale), 2e-3, 6e-3)

print("  -- one-layer timing (T=6, n=640) --")
q, kv, mask, sink = case(6, 640)
for lbl, fn in (("torch fp32 path (old)", lambda: decode_attention_ref(q, kv, mask, sink, scale)),
                ("fused kernel split=1", lambda: decode_attention(q, kv, None, mask, sink, scale, split=1)),
                ("fused kernel split=2", lambda: decode_attention(q, kv, None, mask, sink, scale, split=2)),
                ("fused kernel split=4", lambda: decode_attention(q, kv, None, mask, sink, scale, split=4))):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 50
    print(f"       {lbl:24s} {dt * 1e6:7.1f} us/layer   -> {dt * 40 * 1e3:5.2f} ms/step over 40 layers")

# ----------------------------------------------------------------- 3. skinny fp32 GEMM
print("== skinny fp32 GEMM (HC mix projection) vs F.linear ==")
torch.backends.cuda.matmul.allow_tf32 = False
hc = args.hc_mult
for M, N, K, lbl in ((6, (2 + hc) * hc, hc * args.dim, "verify hc_fn"),
                     (5, (2 + hc) * hc, hc * args.dim, "draft hc_fn"),
                     (6, args.head_dim, args.dim, "ratio-2 compressor")):
    torch.manual_seed(1)
    w = (torch.randn(N, K, device="cuda") * 0.02).float()
    for xdt in (torch.bfloat16, torch.float32):
        x = (torch.randn(M, K, device="cuda") * 0.3).to(xdt)
        ref = torch.nn.functional.linear(x.float(), w)
        y = skinny_linear(x, w)
        rel = float((y - ref).norm() / ref.norm())
        mx = float((y - ref).abs().max() / ref.abs().max())
        det = bool(torch.equal(y, skinny_linear(x, w)))
        ok = rel <= 1e-6 and mx <= 1e-6 and det
        print(f"  {'PASS' if ok else 'FAIL'} {lbl:20s} M={M} N={N:4d} K={K:6d} x={str(xdt)[6:]:9s}"
              f" rel {rel:.2e}  max rel {mx:.2e}  reproducible {det}")
        if not ok:
            fails.append(f"skinny {lbl} {xdt}")
    x = (torch.randn(M, K, device="cuda") * 0.3).to(torch.bfloat16)
    for lb, fn in (("F.linear fp32", lambda: torch.nn.functional.linear(x.float(), w)),
                   ("skinny split-K", lambda: skinny_linear(x, w))):
        fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 100
        print(f"       {lb:16s} {dt * 1e6:7.1f} us  ({(N * K + M * K) * 4 / 1e9 / dt:6.1f} GB/s)")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
