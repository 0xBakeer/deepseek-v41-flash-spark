"""Synthetic check of tools/fp4_moe.py (whichever FP4 decoder is active) against the dequant reference.
Needs no checkpoint: random packed FP4 bytes and scales near 2^0."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_moe as K

dev = "cuda"
g = torch.Generator().manual_seed(0)
n = 16
a = K.ExpertArena(n, dev)
for s in range(n):
    r = lambda *sh: torch.randint(0, 256, sh, generator=g, dtype=torch.uint8)
    sc = lambda *sh: torch.randint(120, 124, sh, generator=g, dtype=torch.uint8)
    a.load_slot(s, r(2304, 2560), sc(2304, 160), r(5120, 1152), sc(5120, 72), r(2304, 2560), sc(2304, 160))
torch.cuda.synchronize()
print("portable decode:", getattr(K._FP4_PORTABLE, "value", K._FP4_PORTABLE))
for T in (1, 6, 64):
    x = (torch.randn(T, K.DIM, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    slots = torch.stack([torch.randperm(n, generator=g)[:6] for _ in range(T)]).to(torch.int32).to(dev)
    w = torch.rand(T, 6, generator=g); w = (w / w.sum(1, keepdim=True)).to(dev)
    y = K.moe_forward(x, slots, w, a, 10.0)
    ref = K.moe_forward_reference(x, slots, w, a, 10.0)
    rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): K.moe_forward(x, slots, w, a, 10.0)
    torch.cuda.synchronize(); dt = (time.time() - t0) / 10
    nd = len(set(slots.flatten().tolist()))
    print(f"T={T:3d} rel_err={rel:.2e} {dt*1e3:.2f} ms  ~{nd*a.bytes_per_slot/dt/1e9:.0f} GB/s")
    assert rel < 2e-2, rel
print("OK")
