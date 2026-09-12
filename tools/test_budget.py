"""Checks on the cost model -- the point is that it agrees with what the box did.

Run: python3 tools/test_budget.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (want {want}{f' +-{tol}' if tol else ''})")
    if not ok:
        fails.append(name)


# --- the slot sizes are the arena's, not a rounded quote of it ---------------
check("fp4 bytes/expert", B.EXPERT_BYTES["fp4"], 18_800_640)
check("cb3 bytes/expert", B.EXPERT_BYTES["cb3"], 14_454_784)
try:
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from cb3_moe import CB3_BYTES_PER_SLOT  # needs torch; skipped where it is absent
    check("cb3 matches the kernel", B.EXPERT_BYTES["cb3"], CB3_BYTES_PER_SLOT)
except Exception as e:  # noqa: BLE001
    print(f"skip cb3_moe cross-check ({type(e).__name__})")

# --- against a real load: 2026-09-12, ARENA_GB=88, cb3 ----------------------
# "arena 88.0 GB = 6087 cb3 expert slots of 14.45 MB (40% of all routed experts)"
check("88 GB is 6087 cb3 slots", int(88e9 // B.EXPERT_BYTES["cb3"]), 6087)

# --- the KV formula against the two lengths that were measured --------------
check("KV at 32k", round(B.kv_bytes(32768) / 1e9 - B.WINDOW_BYTES / 1e9, 2), 0.10, 0.005)
check("KV at 128k", round(B.kv_bytes(131072) / 1e9 - B.WINDOW_BYTES / 1e9, 2), 0.42, 0.005)

# --- the launch gate is the engine's own ------------------------------------
# engine/v41_engine.py: arena + pack_scratch + keep_free <= MemAvailable, checked
# once the dense weights are resident. The box had MemAvailable 111.1 GB there
# and accepted an 88 GB arena; it would not have accepted 103.
host = B.Host("test", 130.6e9, 111.1e9 + 7.61e9, True)
for arena, want in ((88.0, True), (98.0, True), (103.0, False)):
    p = B.plan(host, None, (), 0.39, 32768, arena_gb=arena)
    check(f"arena {arena:.0f} GB accepted", p.launch_slack >= 0, want)

# --- the two configurations this box actually ran, 2026-09-12 ---------------
# Same box, same .env except the keep fraction. MemAvailable was 111.0 GB with
# the dense weights already resident, so 118.6 GB before them.
#   arena 87 GB -> served two 2,400-token generations, 0.0 GB read from NVMe
#   arena 98 GB -> "FATAL: host MemAvailable 0.4 GB stayed below the 2.5 GB
#                   floor for 3.0 s" on the FIRST request
# The engine's own pre-flight accepts both; it does not know about the drafter
# experts, the cache or a prefill chunk. The model here has to reject the second.
box = B.Host("gb10-20260912", 130.6e9, 111.0e9 + 7.61e9, True)
served = B.plan(box, None, (), 0.39, 32768)
killed = B.plan(box, None, (), 0.44, 32768)
check("keep 0.39 is offered", served.verdict != "over", True)
check("keep 0.44 is refused", killed.verdict, "over")
check("  the engine would have started it anyway", killed.launch_slack >= 0, True)
check("  free at 0.39 clears the prefill reserve", served.free_after_load > served.prefill, True)
check("  free at 0.44 does not", killed.free_after_load < killed.prefill, True)
check("max keep on that box", round(served.max_keep(), 2), 0.42, 0.01)

# --- keep fraction -> slots -------------------------------------------------
p = B.plan(host, None, (), 0.39, 32768)
check("keep 39% kept experts", p.kept, 150 * 40)          # ceil(.39*384)=150 per layer
check("arena holds kept + transient ring", p.slots, 6008)
# the engine's OWN default ring is 400, not 8: sizing for 8 and running with
# 400 leaves 392 kept experts outside the LRU, streaming from NVMe every step
big = B.plan(host, None, (), 0.39, 32768, transient_slots=400)
check("a 400-slot ring is sized for", big.slots, 6400)
check("and costs more arena", round(big.arena - p.arena, 1), 5.7, 0.1)
check("keep 39% arena GB", round(p.arena, 1), 86.8, 0.05)
# max_keep is the largest fraction that both starts and survives a prefill
# chunk, so it is bounded by `fits`, not by the engine's own launch gate alone
mk = p.max_keep()
check("max keep fits", B.plan(host, None, (), mk - 0.003, 32768).fits, True)
check("just past max keep does not", B.plan(host, None, (), mk + 0.01, 32768).fits, False)

# --- the ARENA_GB the tool writes must actually hold the kept set -----------
# It is written as a whole number of GB, and the engine turns that back into
# slots by flooring. If the rounding ever went the other way the kept tail
# would stream from NVMe with nothing but the tok/s to say so.
import math as _math  # noqa: E402
bad = []
for _k in [round(0.02 * i, 2) for i in range(3, 31)]:
    for _ring in (8, 16, 400):
        _p = B.plan(host, None, (), _k, 32768, transient_slots=_ring)
        written = _math.ceil(_p.arena)                      # what env_for writes
        slots = int(written * B.GB // B.EXPERT_BYTES["cb3"])
        if slots - _ring < _p.kept:
            bad.append((_k, _ring, slots - _ring, _p.kept))
check("written ARENA_GB holds the kept set at every keep and ring", bad, [])

# --- coverage: a narrower selection is better served at the same budget -----
cov = os.path.join(ROOT, "results/keepsets/general/coverage.json")
if os.path.exists(cov):
    idx = B.TopicIndex(cov)
    if len(idx.topics) >= 2:
        a, b = idx.topics[0], idx.topics[1]
        alone = idx.coverage((a,), 0.39)[a]
        both = idx.coverage((a, b), 0.39)[a]
        check(f"{a} alone beats {a}+{b} at the same keep", alone > both, True)
        # and reaches a given coverage at a smaller budget
        k1 = idx.keep_for((a,), 0.85)
        k2 = idx.keep_for((a, b), 0.85)
        check("one topic needs a smaller keep than two", k1 < k2, True)
        print(f"     {a} alone {k1:.0%} vs {a}+{b} {k2:.0%} for 0.85 coverage")
        # coverage is monotone in the budget
        c = idx.curves((a, b))[0][a]
        check("coverage is monotone in keep", all(c[i] <= c[i + 1] + 1e-12 for i in range(384)), True)
        check("coverage reaches 1.0 at keep 100%", round(c[384], 3), 1.0, 0.001)
else:
    print("skip coverage checks (no keep-set in the checkout)")

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
