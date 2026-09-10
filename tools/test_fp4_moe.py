"""test_fp4_moe.py -- accuracy + bandwidth test for tools/fp4_moe.py on real DeepSeek-V4.1-Flash experts."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_moe  # noqa: E402
from fp4_moe import DIM, ExpertArena, moe_forward, moe_forward_reference  # noqa: E402

SHARD = os.path.expanduser("~/models/DeepSeek-V4.1-Flash/model-00003-of-00048.safetensors")
N_EXPERTS = 32
TOPK = 6


def load_arena(n: int, device) -> ExpertArena:
    arena = ExpertArena(n, device)
    with safe_open(SHARD, "pt", device="cpu") as f:
        for e in range(n):
            p = f"layers.0.ffn.experts.{e}."
            arena.load_slot(
                e,
                f.get_tensor(p + "w1.weight"), f.get_tensor(p + "w1.scale"),
                f.get_tensor(p + "w2.weight"), f.get_tensor(p + "w2.scale"),
                f.get_tensor(p + "w3.weight"), f.get_tensor(p + "w3.scale"),
            )
    torch.cuda.synchronize()
    return arena


def random_routing(T: int, n_slots: int, gen: torch.Generator, device):
    slots = torch.stack([torch.randperm(n_slots, generator=gen)[:TOPK] for _ in range(T)]).to(torch.int32)
    w = torch.rand((T, TOPK), generator=gen)
    w = w / w.sum(1, keepdim=True)
    return slots.to(device), w.to(device)


def routing_with_distinct(T: int, n_distinct: int, gen: torch.Generator, device):
    """T tokens x TOPK slots using exactly `n_distinct` different experts (each token's slots distinct)."""
    assert n_distinct <= T * TOPK and n_distinct >= TOPK
    pool = torch.randperm(N_EXPERTS, generator=gen)[:n_distinct]
    flat = torch.cat([pool, pool[torch.randint(n_distinct, (T * TOPK - n_distinct,), generator=gen)]])
    flat = flat[torch.randperm(T * TOPK, generator=gen)].view(T, TOPK)
    for t in range(T):  # make each token's K slots distinct (mirrors top-k), keeping the distinct set
        used = set()
        for k in range(TOPK):
            v = int(flat[t, k])
            while v in used:
                v = int(pool[torch.randint(n_distinct, (1,), generator=gen)])
            flat[t, k] = v
            used.add(v)
    assert len(set(flat.flatten().tolist())) == n_distinct
    w = torch.rand((T, TOPK), generator=gen)
    w = w / w.sum(1, keepdim=True)
    return flat.to(torch.int32).to(device), w.to(device)


def bench(fn, iters: int = 20, warm: int = 3) -> tuple[float, float]:
    """Returns (pipelined ms, synced ms): CUDA-event time per call with `iters` calls enqueued back to back
    (the CPU runs ahead, like inside a model forward), and wall time per call with a sync after each call
    (includes the eager-mode CPU launch overhead)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    piped = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))[iters // 2]
    walls = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        walls.append(1e3 * (time.perf_counter() - t0))
    return piped, sorted(walls)[iters // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(1234)

    t0 = time.time()
    arena = load_arena(N_EXPERTS, device)
    mb = arena.bytes_per_slot / 1e6
    print(f"arena: {N_EXPERTS} slots x {mb:.2f} MB loaded in {time.time() - t0:.1f}s "
          f"(GPU alloc {torch.cuda.memory_allocated() / 1e9:.2f} GB)")

    # ------------------------------------------------------------------ accuracy
    print("\n== accuracy (kernel vs dequant+torch reference), K=6")
    print(f"{'T':>5} {'max|diff|':>12} {'rel err':>10} {'max|ref|':>10}")
    ok = True
    for T in (1, 6, 64, 512):
        x = (torch.randn((T, DIM), generator=gen) * 1.0).to(torch.bfloat16).to(device)
        slots, w = random_routing(T, N_EXPERTS, gen, device)
        y = moe_forward(x, slots, w, arena)
        y_ref = moe_forward_reference(x, slots, w, arena)
        diff = (y.float() - y_ref.float())
        rel = diff.norm() / y_ref.float().norm()
        print(f"{T:>5} {diff.abs().max().item():>12.4e} {rel.item():>10.4e} {y_ref.float().abs().max().item():>10.3f}")
        ok &= bool(rel.item() < args.tol)
    assert ok, f"relative error exceeded {args.tol}"
    print("accuracy: PASS")

    # ------------------------------------------------------------------ timing
    print(f"\n== timing (CUDA events, median of {args.iters}); slot = {mb:.2f} MB; GB10 peak ~273 GB/s")
    print("pipelined = calls enqueued back to back; synced = wall per call incl. eager CPU launch overhead")
    print(f"{'T':>5} {'experts':>8} {'pipelined ms':>13} {'GB/s':>7} {'% peak':>7} {'synced ms':>10} {'GB/s':>7} {'ref ms':>9} {'ref GB/s':>9}")
    for T, n_distinct in ((1, 6), (6, 30), (64, 32), (512, 32)):
        x = torch.randn((T, DIM), generator=gen).to(torch.bfloat16).to(device)
        slots, w = routing_with_distinct(T, n_distinct, gen, device)
        ms, wall = bench(lambda: moe_forward(x, slots, w, arena), args.iters)
        ref_ms, _ = bench(lambda: moe_forward_reference(x, slots, w, arena), max(5, args.iters // 4))
        y1 = moe_forward(x, slots, w, arena); y2 = moe_forward_reference(x, slots, w, arena)
        assert (y1.float() - y2.float()).norm() / y2.float().norm() < args.tol
        nbytes = n_distinct * arena.bytes_per_slot
        gbs = nbytes / ms / 1e6
        print(f"{T:>5} {n_distinct:>8} {ms:>13.3f} {gbs:>7.1f} {100 * gbs / 273:>6.1f}% {wall:>10.3f} {nbytes / wall / 1e6:>7.1f} {ref_ms:>9.3f} {nbytes / ref_ms / 1e6:>9.1f}")
    print(f"\npeak GPU memory during test: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
