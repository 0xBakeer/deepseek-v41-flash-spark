"""Tier 0 for CB3 — the repo's shipped expert format, with RANDOM weights and no checkpoint.

Tier 0 covered `fp4_moe` only; CB3 is what v0.4.0 actually serves (`EXPERT_FORMAT=cb3`, keep 0.44),
so it is the kernel that matters. Mirrors tools/test_cb3_moe.py's kernel section: build a CB3ArenaV2
and an FP4 arena holding *the same* 8-level requantized weights, so the comparison isolates the CB3
kernel from the 3-bit quantization loss rather than conflating them.

Random packed FP4 in, exactly as in test_fp4_moe_random.py: uniform bytes are two valid e2m1 nibbles,
UE8M0 scale bytes drawn from a narrow band so the dynamic range resembles trained weights.

Three questions, the same ones Tier 0 asked of fp4:
  1. CB3 kernel vs the FP4 kernel on identical 8-level weights -- pure kernel error
  2. determinism across repeated identical calls
  3. chunk invariance
"""
import sys, torch
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE); sys.path.insert(0, HERE.rsplit("/", 1)[0] + "/engine")
import cb3 as CB3, cb3_moe as C3, fp4_moe as F4
from codebook_sim import CodebookSim

SLOTS, TOPK, DEV = 8, 6, "cuda"
gen = torch.Generator().manual_seed(20260912)
ru8 = lambda *s: torch.randint(0, 256, s, generator=gen, dtype=torch.uint8)
rsc = lambda *s: torch.randint(119, 127, s, generator=gen, dtype=torch.uint8)

print(f"== CB3 (the shipped format), random weights, {SLOTS} slots on {torch.cuda.get_device_name(0)} ==",
      flush=True)
sim = CodebookSim(3, DEV)
a3 = C3.CB3ArenaV2(SLOTS, DEV); a3.sim = sim
fp4q = F4.ExpertArena(SLOTS, DEV)
for s in range(SLOTS):
    g = [ru8(F4.INTER, F4.KB1), rsc(F4.INTER, F4.SG1),
         ru8(F4.DIM,   F4.KB2), rsc(F4.DIM,   F4.SG2),
         ru8(F4.INTER, F4.KB1), rsc(F4.INTER, F4.SG1)]
    a3.load_slot(s, *g)
    fp4q.load_slot(s, *g)
    for wn, sn in (("w1", "s1"), ("w2", "s2"), ("w3", "s3")):
        wt = getattr(fp4q, wn)[s]; st = getattr(fp4q, sn)[s]
        wt.copy_(sim.requant_packed(wt, st))      # SAME 8-level weights in the FP4 arena
torch.cuda.synchronize()
print(f"   CB3 {a3.bytes_per_slot/1e6:.2f} MB/slot vs FP4 {fp4q.bytes_per_slot/1e6:.2f} MB "
      f"(ratio {a3.bytes_per_slot/fp4q.bytes_per_slot:.3f})", flush=True)

def routing(T):
    sl = torch.stack([torch.randperm(SLOTS, device=DEV)[:TOPK] for _ in range(T)]).to(torch.int32)
    w = torch.rand(T, TOPK, device=DEV)
    return sl, w / w.sum(1, keepdim=True)

torch.manual_seed(0)
ok = True
print("\n-- 1. CB3 kernel vs FP4 kernel on identical 8-level weights")
print(f"{'T':>4} {'rel vs fp4':>12} {'rel vs deq':>12}")
for T in (1, 6, 64):
    x = (torch.randn(T, F4.DIM, device=DEV) * 0.5).to(torch.bfloat16)
    sl, w = routing(T)
    deq = F4.moe_forward_reference(x, sl, w, a3)
    f4  = F4.moe_forward(x, sl, w, fp4q)
    y   = C3.moe_forward_v3(x, sl, w, a3)
    r1 = ((y.float()-f4.float()).norm()/f4.float().norm()).item()
    r2 = ((y.float()-deq.float()).norm()/deq.float().norm()).item()
    ok &= r1 < 2e-2
    print(f"{T:>4} {r1:>12.4e} {r2:>12.4e}", flush=True)

print("\n-- 2. determinism: 5 identical calls bit-exact")
det = True
for T in (1, 64):
    x = (torch.randn(T, F4.DIM, device=DEV) * 0.5).to(torch.bfloat16); sl, w = routing(T)
    base = C3.moe_forward_v3(x, sl, w, a3)
    same = sum(torch.equal(C3.moe_forward_v3(x, sl, w, a3), base) for _ in range(4))
    det &= same == 4
    print(f"  T={T:<3} {same}/4 bit-exact  {'ok' if same==4 else 'NOT DETERMINISTIC'}", flush=True)

print("\n-- 3. chunk invariance")
T = 64
x = (torch.randn(T, F4.DIM, device=DEV) * 0.5).to(torch.bfloat16); sl, w = routing(T)
full = C3.moe_forward_v3(x, sl, w, a3)
bad = [m for m in (1, 2, 4, 8, 16, 32)
       if not torch.equal(C3.moe_forward_v3(x[:m], sl[:m], w[:m], a3), full[:m])]
print(f"  prefixes differing from the full batch: {bad or 'none'}", flush=True)

print(f"\n== {'ALL PASS' if (ok and det and not bad) else 'FAILURES ABOVE'} ==", flush=True)
print("== ALL DONE ==" if (ok and det and not bad) else "== VOID ==", flush=True)
