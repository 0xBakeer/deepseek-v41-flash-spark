"""Unit test for the CB3 3-bit expert format and its kernels (tools/cb3.py, tools/cb3_moe.py).

Checks, on real layer-0 experts:
  1. the v2/v3 bit layout is bit-exact with the v1 layout, with `engine/codebook_sim.py`'s
     simulated 8-level requantization, and with the packer's own unpack;
  2. the CB3 kernels agree with the dequantized reference and with the FP4 kernel run on the SAME
     re-quantized weights;
  3. the bytes per expert and the achieved GB/s of expert bytes at decode shapes.

Run:  python tools/test_cb3_moe.py
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))

import cb3 as CB3  # noqa: E402
import cb3_moe as C3  # noqa: E402
import fp4_moe as F4  # noqa: E402
import v41_ref as R  # noqa: E402
from codebook_sim import CodebookSim  # noqa: E402
from safetensors import safe_open  # noqa: E402

MD = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
S = int(os.environ.get("EXPERTS", "24"))
fails = []


def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
f = safe_open(f"{MD}/{idx['layers.0.ffn.experts.0.w1.weight']}", "pt", device="cpu")
sim = CodebookSim(3, "cuda")

print("== format: v2/v3 layout vs v1 and vs the simulation ==")
print(f"   block plan: K=5120 -> {CB3.block_plan(5120)} (n512, n256);  K=2304 -> {CB3.block_plan(2304)}")
for name in ("layers.0.ffn.experts.3.w1", "layers.0.ffn.experts.3.w2"):
    w = f.get_tensor(name + ".weight").cuda().view(torch.uint8)
    sc = f.get_tensor(name + ".scale").cuda().view(torch.uint8)
    lo1, hi1, cb1 = CB3.fp4_to_cb3(w, sc, sim)
    lo2, hi2, cb2 = CB3.fp4_to_cb3_v2(w, sc, sim)
    d1 = CB3.dequant_cb3(lo1, hi1, cb1, sc)
    d2 = CB3.dequant_cb3_v2(lo2, hi2, cb2, sc)
    ref_sim = R.dequant_fp4_packed(sim.requant_packed(w, sc), sc)
    ref = R.dequant_fp4_packed(w, sc)
    ok(bool((d2 == d1).all()), f"{name}: v2 dequant bit-identical to v1")
    ok(bool((d2 == ref_sim).all()), f"{name}: v2 dequant bit-identical to codebook_sim")
    ok(lo2.numel() == lo1.numel() and hi2.numel() == hi1.numel(), f"{name}: v2 byte count unchanged")
    bits = (lo2.numel() + hi2.numel() + cb2.numel() + sc.numel()) * 8 / w.numel() / 2
    print(f"       {name}: {bits:.3f} bit/weight, rel err vs FP4 {float((d2.float()-ref.float()).norm()/ref.float().norm()):.4f}")

print("== a narrower codebook inside the same slot (the 2-bit tier's quality simulation) ==")
sim2 = CodebookSim(2, "cuda")
for name in ("layers.0.ffn.experts.3.w1", "layers.0.ffn.experts.3.w2"):
    w = f.get_tensor(name + ".weight").cuda().view(torch.uint8)
    sc = f.get_tensor(name + ".scale").cuda().view(torch.uint8)
    lo, hi, cb = CB3.fp4_to_cb3_v2(w, sc, sim2)
    d = CB3.dequant_cb3_v2(lo, hi, cb, sc)
    ref_sim2 = R.dequant_fp4_packed(sim2.requant_packed(w, sc), sc)
    ref = R.dequant_fp4_packed(w, sc)
    idx_used = CB3.unpack_cb3_v2(lo, hi, torch.arange(8, device=cb.device, dtype=torch.uint8).expand(cb.shape).contiguous())
    ok(bool((d == ref_sim2).all()), f"{name}: 2-bit codebook in a CB3 slot == codebook_sim(2)")
    ok(int(idx_used.max()) <= 3, f"{name}: only the first 4 codebook entries are ever indexed")
    ok(bool((cb[:, :4] == cb[:, 4:]).all()), f"{name}: the 8-entry codebook is the 4-entry one repeated")
    print(f"       {name}: rel err vs FP4 {float((d.float()-ref.float()).norm()/ref.float().norm()):.4f} "
          f"(3-bit was printed above); bytes unchanged")

print("== kernels ==")
a3 = C3.CB3ArenaV2(S, "cuda"); a3.sim = sim
fp4q = F4.ExpertArena(S, "cuda")
t0 = time.time()
for s in range(S):
    p = f"layers.0.ffn.experts.{s}."
    g = [f.get_tensor(p + n) for n in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")]
    a3.load_slot(s, *g)
    fp4q.load_slot(s, *g)
    for wn, sn in (("w1", "s1"), ("w2", "s2"), ("w3", "s3")):
        wt = getattr(fp4q, wn)[s]; st = getattr(fp4q, sn)[s]
        wt.copy_(sim.requant_packed(wt, st))   # the SAME 8-level weights, in the FP4 arena
torch.cuda.synchronize()
print(f"   {S} experts packed in {time.time()-t0:.0f}s: CB3 {a3.bytes_per_slot/1e6:.2f} MB vs FP4 "
      f"{fp4q.bytes_per_slot/1e6:.2f} MB (ratio {a3.bytes_per_slot/fp4q.bytes_per_slot:.3f})")
torch.manual_seed(0)
for T in (1, 6, 64):
    x = (torch.randn(T, F4.DIM, device="cuda") * 0.5).to(torch.bfloat16)
    slots = torch.stack([torch.randperm(S, device="cuda")[:6] for _ in range(T)]).to(torch.int32)
    wgt = torch.rand(T, 6, device="cuda")
    deq = F4.moe_forward_reference(x, slots, wgt, a3)
    fp4_out = F4.moe_forward(x, slots, wgt, fp4q)
    y = C3.moe_forward_v3(x, slots, wgt, a3)
    r_deq = float((y.float() - deq.float()).norm() / deq.float().norm())
    r_fp4 = float((y.float() - fp4_out.float()).norm() / fp4_out.float().norm())
    base = float((fp4_out.float() - deq.float()).norm() / deq.float().norm())
    ok(r_deq <= 5e-3, f"T={T}: CB3 v3 vs dequant reference rel {r_deq:.2e} (FP4 kernel's own: {base:.2e})")
    ok(r_fp4 <= 5e-3, f"T={T}: CB3 v3 vs FP4 kernel on the same weights rel {r_fp4:.2e}")
    if T == 6:
        n_exp = len(torch.unique(slots))
        for lbl, fn, bps in (("FP4", lambda: F4.moe_forward(x, slots, wgt, fp4q), fp4q.bytes_per_slot),
                             ("CB3 v2 (Triton byte ops)", lambda: C3.moe_forward_v2(x, slots, wgt, a3, block_m=16, cfg_up=(64, 4, 2), cfg_down=(64, 4, 2)), a3.bytes_per_slot),
                             ("CB3 v3 (PTX + wide tiles)", lambda: C3.moe_forward_v3(x, slots, wgt, a3), a3.bytes_per_slot)):
            fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 20
            print(f"       T=6 top-6, {n_exp} experts: {lbl:26s} {dt*1e3:5.2f} ms  {n_exp*bps/dt/1e9:6.1f} GB/s of expert bytes")

print("== CB2 (2-bit tier): format, kernel, bandwidth ==")
a2 = C3.CB2ArenaV2(S, "cuda"); a2.sim = sim2
fp4q2 = F4.ExpertArena(S, "cuda")
t0 = time.time()
for s_ in range(S):
    p = f"layers.0.ffn.experts.{s_}."
    g = [f.get_tensor(p + n) for n in ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")]
    a2.load_slot(s_, *g)
    fp4q2.load_slot(s_, *g)
    for wn, sn in (("w1", "s1"), ("w2", "s2"), ("w3", "s3")):
        wt = getattr(fp4q2, wn)[s_]; st = getattr(fp4q2, sn)[s_]
        wt.copy_(sim2.requant_packed(wt, st))   # the SAME 4-level weights, in the FP4 arena
torch.cuda.synchronize()
print(f"   {S} experts packed in {time.time()-t0:.0f}s: CB2 {a2.bytes_per_slot/1e6:.2f} MB vs CB3 "
      f"{a3.bytes_per_slot/1e6:.2f} MB ({a2.bytes_per_slot/a3.bytes_per_slot:.3f}x) vs FP4 "
      f"{fp4q2.bytes_per_slot/1e6:.2f} MB ({a2.bytes_per_slot/fp4q2.bytes_per_slot:.3f}x); "
      f"{a2.bytes_per_slot*8/(3*F4.INTER*F4.DIM):.3f} bit/weight")
for name, which in (("layers.0.ffn.experts.3.w1", "w1"), ("layers.0.ffn.experts.3.w2", "w2")):
    w = f.get_tensor(name + ".weight").cuda().view(torch.uint8)
    sc = f.get_tensor(name + ".scale").cuda().view(torch.uint8)
    lo, cb = CB3.fp4_to_cb2(w, sc, sim2)
    lo3, hi3, cb3_ = CB3.fp4_to_cb3_v2(w, sc, sim2)
    ok(bool((lo == lo3).all()) and int(hi3.max()) == 0,
       f"{name}: the CB2 lo plane is the CB3 lo plane and the CB3 hi plane is all zeros")
    d = CB3.dequant_cb2(lo, cb, sc)
    ok(bool((d == R.dequant_fp4_packed(sim2.requant_packed(w, sc), sc)).all()),
       f"{name}: CB2 dequant bit-identical to codebook_sim(2)")
torch.manual_seed(0)
for T in (1, 6, 64):
    x = (torch.randn(T, F4.DIM, device="cuda") * 0.5).to(torch.bfloat16)
    slots = torch.stack([torch.randperm(S, device="cuda")[:6] for _ in range(T)]).to(torch.int32)
    wgt = torch.rand(T, 6, device="cuda")
    fp4_out = F4.moe_forward(x, slots, wgt, fp4q2)
    y = C3.moe_forward_cb2(x, slots, wgt, a2)
    r_fp4 = float((y.float() - fp4_out.float()).norm() / fp4_out.float().norm())
    ok(r_fp4 <= 5e-3, f"T={T}: CB2 kernel vs FP4 kernel on the same 4-level weights rel {r_fp4:.2e}")
    if T == 6:
        n_exp = len(torch.unique(slots))
        for lbl, fn, bps in (("FP4 (4-level weights)", lambda: F4.moe_forward(x, slots, wgt, fp4q2), fp4q2.bytes_per_slot),
                             ("CB3 v3", lambda: C3.moe_forward_v3(x, slots, wgt, a3), a3.bytes_per_slot),
                             ("CB2", lambda: C3.moe_forward_cb2(x, slots, wgt, a2), a2.bytes_per_slot)):
            fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 20
            print(f"       T=6 top-6, {n_exp} experts: {lbl:26s} {dt*1e3:5.2f} ms  {n_exp*bps/dt/1e9:6.1f} GB/s of expert bytes")
# prefill path: the unpack fallback must be bit-exact against the FP4 kernel on the same weights
xp = (torch.randn(128, F4.DIM, device="cuda") * 0.5).to(torch.bfloat16)
sp = torch.stack([torch.randperm(S, device="cuda")[:6] for _ in range(128)]).to(torch.int32)
wp = torch.rand(128, 6, device="cuda")
yp = C3.moe_forward_cb2(xp, sp, wp, a2)
rp = F4.moe_forward(xp, sp, wp, fp4q2)
ok(float((yp.float() - rp.float()).abs().max()) == 0.0,
   f"T=128 (prefill): CB2 unpack fallback bit-exact vs the FP4 kernel (max |delta| {float((yp.float()-rp.float()).abs().max()):.1e})")

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
