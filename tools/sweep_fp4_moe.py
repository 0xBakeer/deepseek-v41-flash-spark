"""Sweep (BN, num_warps, num_stages) of the two FP4 MoE kernels at decode size on the current GPU."""
import os, sys, time, itertools
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_moe as K
dev = "cuda"; g = torch.Generator().manual_seed(0); n = 48
a = K.ExpertArena(n, dev)
for s in range(n):
    r = lambda *sh: torch.randint(0, 256, sh, generator=g, dtype=torch.uint8)
    sc = lambda *sh: torch.randint(120, 124, sh, generator=g, dtype=torch.uint8)
    a.load_slot(s, r(2304, 2560), sc(2304, 160), r(5120, 1152), sc(5120, 72), r(2304, 2560), sc(2304, 160))
T = int(os.environ.get("T", "6"))
x = torch.randn(T, K.DIM, generator=g).to(torch.bfloat16).to(dev)
slots = torch.stack([torch.randperm(n, generator=g)[:6] for _ in range(T)]).to(torch.int32).to(dev)
w = torch.full((T, 6), 1 / 6, device=dev)
nd = len(set(slots.flatten().tolist())); BM = K._pick_bm(T * 6)
def bench(up, down):
    try:
        K.moe_forward(x, slots, w, a, 10.0, up_cfg=up, down_cfg=down); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20): K.moe_forward(x, slots, w, a, 10.0, up_cfg=up, down_cfg=down)
        torch.cuda.synchronize(); return (time.time() - t0) / 20
    except Exception as e:
        return float("inf")
base_up, base_down = K._UP_CFG[BM], K._DOWN_CFG[BM]
print(f"T={T} BM={BM} distinct={nd} default {bench(base_up, base_down)*1e3:.2f} ms")
grid = list(itertools.product((32, 64, 128, 256), (2, 4, 8), (1, 2)))
best_up = min(grid, key=lambda c: bench(c, base_down)); tu = bench(best_up, base_down)
best_dn = min(grid, key=lambda c: bench(best_up, c)); td = bench(best_up, best_dn)
print(f"best up {best_up} -> {tu*1e3:.2f} ms; best down {best_dn} -> {td*1e3:.2f} ms = {nd*a.bytes_per_slot/td/1e9:.0f} GB/s")
