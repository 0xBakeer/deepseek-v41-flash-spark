"""What a routing pick does when its expert is not resident: substitute vs drop.

Run: python3 tools/test_route_modes.py

The two routing blocks (engine/model.py `moe`, engine/fastdecode.py `_layer_a`) cannot be lifted
out the way tools/test_maxmin.py lifts `_maxmin_counts`: they are methods that reach into a Model,
a weight object, a CUDA graph's static buffers and a Triton MoE kernel. So this file does two
things instead.

1. A pure-numpy REFERENCE of both modes, and the properties that define them. The reference is
   short enough to read against the engine line by line, which is the point -- it says what
   `substitute` and `drop` MEAN, independently of how torch spells it.

2. Regex pins on the engine source, the way tools/test_budget_rank.py pins the rank expressions.
   The risk this addresses is not that the arithmetic is wrong today; it is that one of the two
   paths gets edited and the other does not. Prefill and decode disagreeing pick for pick is not a
   crash -- the verify step just rejects its own drafts and the speedup quietly disappears -- so
   the check has to be mechanical. The pins also assert that the default is still `substitute`,
   because `drop` is an experiment and the shipped behaviour must not move under it.
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "engine/model.py")
FAST = os.path.join(ROOT, "engine/fastdecode.py")
ENGINE = os.path.join(ROOT, "engine/v41_engine.py")
ENV = os.path.join(ROOT, "env.example")
fails = []

ROUTE_SCALE = 2.5  # any positive constant; the engine's comes from the checkpoint


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


# ============================================================ the reference
def topk(v, k):
    """Indices of the k largest entries, largest first. `np.argsort` is stable, so equal logits
    break toward the lower expert id -- which is also what torch.topk does for this shape."""
    return np.argsort(-v, kind="stable")[:k]


def route(scores, bias, resident, k, mode, fallback):
    """One token's routing. Mirrors engine/model.py `moe` and engine/fastdecode.py `_layer_a`.

    scores    [E] fp32, the gate's sqrt(softplus(.)) scores -- strictly positive
    bias      [E] fp32, the gate bias, added for the ARGMAX only and never to the weights
    resident  [E] bool, True = the expert is in the keep-set (the engine's `prune_mask[L]`)
    fallback  [k] int, k distinct resident expert ids, one per routing column

    Returns (slot_idx [k], true_idx [k], weights [k]): the ids whose arena slots are gathered, the
    ids the router actually named, and the weights those slots are multiplied by.
    """
    logits = scores + bias
    if mode == "substitute":
        # the evicted experts are hidden from the argmax, so every pick is resident by construction
        idx = topk(np.where(resident, logits, -np.inf), k)
        w = scores[idx]
        return idx, idx, w / (w.sum() + 1e-20) * ROUTE_SCALE
    if mode == "drop":
        true_idx = topk(logits, k)                  # the router's real choice, mask or no mask
        w = scores[true_idx]
        live = resident[true_idx]
        w = np.where(live, w, 0.0)                  # exactly 0: the term leaves the sum
        slot_idx = np.where(live, true_idx, fallback)
        return slot_idx, true_idx, w / (w.sum() + 1e-20) * ROUTE_SCALE
    raise ValueError(f"unknown mode {mode!r}")


# ============================================================ the properties
rng = np.random.default_rng(20260913)
E, K = 384, 6


def a_case(n_resident, forced=None):
    """scores/bias/resident for one token. `forced` fixes which experts are resident."""
    scores = np.sqrt(np.log1p(np.exp(rng.normal(size=E).astype(np.float64))))  # sqrt(softplus(.)) > 0
    bias = rng.normal(scale=0.3, size=E)
    resident = np.zeros(E, dtype=bool)
    resident[forced if forced is not None else rng.choice(E, n_resident, replace=False)] = True
    fallback = np.flatnonzero(resident)[:K]         # build_prune_fallback: the k lowest resident ids
    return scores, bias, resident, fallback


# --- (a) substitute picks only resident experts and its weights sum to route_scale
bad_res, bad_sum = 0, 0.0
for _ in range(400):
    scores, bias, resident, fallback = a_case(139)   # ~139 of 384, the shipped budget
    idx, true_idx, w = route(scores, bias, resident, K, "substitute", fallback)
    bad_res += int(not resident[idx].all())
    bad_res += int(not (idx == true_idx).all())      # substitute never remaps: the two are one
    bad_sum = max(bad_sum, abs(w.sum() - ROUTE_SCALE))
check("substitute picks only resident experts", bad_res == 0, f"{bad_res} violations")
check("substitute weights sum to route_scale", bad_sum < 1e-12, f"max error {bad_sum:.2e}")

# --- (b) drop: the unmasked top-k, non-resident remapped, non-resident weight exactly 0
bad_idx = bad_zero = bad_live = 0
bad_sum = 0.0
seen_displaced = seen_all_dropped = 0
for _ in range(400):
    scores, bias, resident, fallback = a_case(139)
    idx, true_idx, w = route(scores, bias, resident, K, "drop", fallback)
    want_true = topk(scores + bias, K)               # the router's choice, with NO mask applied
    live = resident[true_idx]
    seen_displaced += int((~live).any())
    bad_idx += int(not (true_idx == want_true).all())
    bad_idx += int(not (idx == np.where(live, true_idx, fallback)).all())
    bad_zero += int(not (w[~live] == 0.0).all())     # exactly 0, not merely small
    bad_live += int(not resident[idx].all())         # every gathered id has an arena slot
    if live.any():
        bad_sum = max(bad_sum, abs(w.sum() - ROUTE_SCALE))
    else:
        # at a keep-set of 139/384 a token loses its whole top-6 about 7 % of the time, so this
        # is not a corner case the harness has to construct -- it happens in the loop above
        seen_all_dropped += 1
        bad_sum = max(bad_sum, abs(w.sum()))
check("drop's true indices are the UNMASKED top-k", bad_idx == 0, f"{bad_idx} violations")
check("drop remaps only the non-resident picks, to that column's fallback", bad_idx == 0)
check("drop weights of non-resident picks are exactly 0", bad_zero == 0, f"{bad_zero} violations")
check("every id drop hands to the slot lookup is resident", bad_live == 0, f"{bad_live} violations")
check("drop weights sum to route_scale, or to 0 when no pick survived", bad_sum < 1e-12,
      f"max error {bad_sum:.2e}")
check("the cases above actually contained displaced picks", seen_displaced > 200,
      f"only {seen_displaced}/400")
check("... and tokens that lost their whole top-k", seen_all_dropped > 0,
      f"{seen_all_dropped}/400")

# --- (b, continued) no resident pick at all: all-zero weights, only the shared expert contributes
scores, bias, _, _ = a_case(139)
top = topk(scores + bias, K)
resident = np.ones(E, dtype=bool)
resident[top] = False                                # evict exactly the router's whole top-6
fallback = np.flatnonzero(resident)[:K]
idx, true_idx, w = route(scores, bias, resident, K, "drop", fallback)
check("drop: a token with no resident pick gets all-zero weights", float(np.abs(w).max()) == 0.0)
check("drop: ... and its slot lookup is still all resident", bool(resident[idx].all()))
check("drop: ... and it still names the router's real top-k", bool((true_idx == top).all()))

# --- (c) with every top-k pick resident the two modes are identical
same = 0
for _ in range(200):
    scores, bias, _, _ = a_case(139)
    top = topk(scores + bias, K)
    resident = np.zeros(E, dtype=bool)
    # the whole top-k survives, plus enough others that the keep-set is a realistic size
    resident[top] = True
    resident[rng.choice(E, 139, replace=False)] = True
    fallback = np.flatnonzero(resident)[:K]
    s_idx, _, s_w = route(scores, bias, resident, K, "substitute", fallback)
    d_idx, d_true, d_w = route(scores, bias, resident, K, "drop", fallback)
    same += int((s_idx == d_idx).all() and (d_idx == d_true).all() and np.array_equal(s_w, d_w))
check("the modes are identical when the whole top-k is resident", same == 200, f"{same}/200")


# ============================================================ pins on the engine source
model_src, fast_src = open(MODEL).read(), open(FAST).read()
engine_src, env_src = open(ENGINE).read(), open(ENV).read()

# the default, and the only two legal values
check("the engine reads DSV41_PRUNE_MODE with 'substitute' as the default",
      bool(re.search(r'os\.environ\.get\("DSV41_PRUNE_MODE"\)\s*or\s*"substitute"', engine_src)))
check("an unknown DSV41_PRUNE_MODE is refused, not quietly served as the default",
      bool(re.search(r'if self\.prune_mode not in \("substitute", "drop"\):\s*\n\s*raise ValueError',
                     engine_src)))
check("the mode is reported in the engine's config dict",
      bool(re.search(r'"prune_mode": self\.prune_mode', engine_src)))
check("the fallback table is built from the same mask the router is masked with",
      bool(re.search(r"def build_prune_fallback\(masks: dict, topk: int\)", engine_src)))

# both paths still have the substitute branch, and it is still a hard mask of the logits
for name, src in (("model.py", model_src), ("fastdecode.py", fast_src)):
    check(f"{name} still hard-masks the logits when substituting",
          bool(re.search(r"not (?:self\.)?drop(?:_mode)?:\s*\n(?:\s*#[^\n]*\n)*"
                         r"\s*logits = logits\.masked_fill\(~pm\[L\], float\(\"-inf\"\)\)", src)))

# both paths have the drop branch, spelled the same way: zero the displaced weights, remap the
# displaced indices, then renormalise over what is left
for name, src, idx_v, w_v, fb_v in (("model.py", model_src, "indices", "weights", "self.prune_fallback"),
                                    ("fastdecode.py", fast_src, "idx", "wts", "self.prune_fb")):
    check(f"{name} takes the unmasked top-k and gathers its scores",
          bool(re.search(rf"{re.escape(w_v)} = scores\.gather\(1, {idx_v}\)", src)))
    check(f"{name} zeroes a displaced pick's weight exactly",
          bool(re.search(rf"live = pm\[L\]\[{idx_v}\][^\n]*\n\s*{re.escape(w_v)} = "
                         rf"{re.escape(w_v)}\.masked_fill\(~live, 0\.0\)", src)))
    check(f"{name} remaps a displaced pick to its column's resident fallback",
          bool(re.search(rf"slot_idx = torch\.where\(live, {idx_v}, {re.escape(fb_v)}\[L\]\)", src)))
    check(f"{name} renormalises over what is left, with the same epsilon",
          bool(re.search(rf"{re.escape(w_v)} = {re.escape(w_v)} / \({re.escape(w_v)}"
                         rf"\.sum\(dim=-1, keepdim=True\) \+ 1e-20\) \* a\.route_scale", src)))

# the slot lookups must read the remapped ids, never the router's true ones -- a non-resident id
# reaches -1 in the LUT and NVMe in ExpertStore.resolve, which all-resident mode must never do
check("model.py resolves slots from the remapped ids",
      bool(re.search(r"slots = lut\[L\]\[slot_idx\]", model_src))
      and bool(re.search(r"slots = store\.resolve\(L, slot_idx, prefill\)", model_src)))
check("fastdecode.py resolves slots from the remapped ids",
      bool(re.search(r"self\.slots\.copy_\(self\.lut\[L\]\[self\.route_slot\]\)", fast_src))
      and bool(re.search(r"idx = self\.route_slot\s*\n\s*slots = self\.m\.store\.resolve", fast_src)))
check("fastdecode.py's taps and route_idx still carry the router's TRUE picks",
      bool(re.search(r"self\.route_idx\.copy_\(idx\); self\.route_w\.copy_\(wts\)", fast_src))
      and bool(re.search(r"self\._tap\('route_idx', L, idx\)", fast_src)))
check("model.py's taps still carry the router's TRUE picks",
      bool(re.search(r'self\._tap\("route_idx", L, indices\); self\._tap\("route_w", L, weights\)',
                     model_src)))
check("substitute shares one buffer, so its captured graphs are unchanged",
      bool(re.search(r"self\.route_slot = torch\.zeros_like\(self\.route_idx\) if self\.drop_mode "
                     r"else self\.route_idx", fast_src)))

# the shipped default must not move: env.example may describe the switch but must not set it
check("env.example documents DSV41_PRUNE_MODE", "DSV41_PRUNE_MODE" in env_src)
check("env.example leaves it unset, so the shipped behaviour is unchanged",
      not re.search(r"^\s*DSV41_PRUNE_MODE\s*=", env_src, re.M))

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
