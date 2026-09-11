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
from decode_attn import decode_attention, decode_attention_ref  # noqa: E402

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
    x6 = (torch.randn(6, W.G, W.K, device="cuda") * 0.25).to(torch.bfloat16)
    for lbl, fn in (("fp8 grouped kernel", lambda: fp8_grouped_linear(x6, W)),
                    ("bf16 einsum (old)", lambda: torch.einsum("sgd,grd->sgr", x6, ref_w))):
        fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 50
        gb = (W.G * W.R * W.K * (1 if "fp8" in lbl else 2)) / 1e9
        print(f"       T=6 {lbl:22s} {dt * 1e6:7.1f} us  ({gb / dt:5.0f} GB/s of weights)")
    del W, ref_w, old, x, full
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
        got = decode_attention(q, kv, mask, sink, scale, split=sp)
        check(f"T={T} n={N} split={sp}  ({lbl})", got, ref, 2e-3, 6e-3)
    got = decode_attention(q, kv, mask, sink, scale, split=2, pv_split=0)
    check(f"T={T} n={N} split=2 PV in plain bf16", got, ref, 8e-3, 3e-2)

print("  -- non-power-of-two head dim (exercises the DA+DB split) --")
q, kv, mask, sink = case(6, 640, d=576)
ref = decode_attention_ref(q, kv, mask, sink, scale)
for sp in (1, 2):
    check(f"T=6 n=640 d=576 split={sp}", decode_attention(q, kv, mask, sink, scale, split=sp), ref, 2e-3, 6e-3)

print("  -- all-masked rows --")
q, kv, mask, sink = case(6, 640, masked_rows=(2, 5))
ref = decode_attention_ref(q, kv, mask, sink, scale)
got = decode_attention(q, kv, mask, sink, scale)
check("T=6 n=640 rows 2 and 5 fully masked", got, ref, 2e-3, 6e-3)
print(f"       masked rows are exactly zero: ref {bool((ref[[2, 5]] == 0).all())}  kernel {bool((got[[2, 5]] == 0).all())}")
if not bool((got[[2, 5]] == 0).all()):
    fails.append("all-masked rows not zero")
q, kv, mask, sink = case(6, 640, all_masked_col=True)
ref = decode_attention_ref(q, kv, mask, sink, scale)
got = decode_attention(q, kv, mask, sink, scale)
check("T=6 n=640 one key visible, row 0 none", got, ref, 2e-3, 6e-3)
print(f"       finite: {bool(torch.isfinite(got.float()).all())}")

print("  -- CUDA graph replay --")
q, kv, mask, sink = case(6, 640, seed=3)
ref = decode_attention_ref(q, kv, mask, sink, scale)
decode_attention(q, kv, mask, sink, scale)  # warm up / JIT before capture
torch.cuda.synchronize()
st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    decode_attention(q, kv, mask, sink, scale)
torch.cuda.current_stream().wait_stream(st)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = decode_attention(q, kv, mask, sink, scale)
g.replay(); torch.cuda.synchronize()
check("graph replay", out, ref, 2e-3, 6e-3)
q2, kv2, mask2, sink2 = case(6, 640, seed=9)
q.copy_(q2); kv.copy_(kv2); mask.copy_(mask2); sink.copy_(sink2)
g.replay(); torch.cuda.synchronize()
check("graph replay, new inputs", out, decode_attention_ref(q, kv, mask, sink, scale), 2e-3, 6e-3)

print("  -- one-layer timing (T=6, n=640) --")
q, kv, mask, sink = case(6, 640)
for lbl, fn in (("torch fp32 path (old)", lambda: decode_attention_ref(q, kv, mask, sink, scale)),
                ("fused kernel split=1", lambda: decode_attention(q, kv, mask, sink, scale, split=1)),
                ("fused kernel split=2", lambda: decode_attention(q, kv, mask, sink, scale, split=2)),
                ("fused kernel split=4", lambda: decode_attention(q, kv, mask, sink, scale, split=4))):
    fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 50
    print(f"       {lbl:24s} {dt * 1e6:7.1f} us/layer   -> {dt * 40 * 1e3:5.2f} ms/step over 40 layers")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
