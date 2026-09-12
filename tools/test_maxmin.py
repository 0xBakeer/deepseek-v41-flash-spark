"""The maxmin keep-set rule must raise the worst-served topic, not the total.

DSV41_PRUNE_RANK=maxmin exists because one request is not one topic: a coding
request with reasoning on writes prose, deliberation and three code registers in
a single generation, and it degenerates at whichever of them the keep-set serves
least. The sum rule optimises total routing mass kept, which lets an already
well-covered topic go on taking slots while another starves.

The engine imports torch, so this lifts the function out with `ast` and runs it
against numpy alone, the way tools/test_engine_kwargs.py reads the signature.
"""
import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name: str):
    """The named top-level function from the engine, without importing torch."""
    tree = ast.parse(open(os.path.join(ROOT, "engine/v41_engine.py")).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<engine>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in engine/v41_engine.py")


maxmin_counts = load("_maxmin_counts")

N_EXPERTS, N_LAYERS = 384, 2
FRAC = 0.10                      # 39 experts per layer, so the budget really binds
N_KEEP = max(6, int(np.ceil(FRAC * N_EXPERTS)))


def topic(hot, mass=0.9, rng=None):
    """A histogram putting `mass` on the `hot` expert ids and the rest everywhere."""
    c = rng.random(N_EXPERTS) * 0.01
    c[hot] += mass / len(hot)
    return c


def coverage(keep, counts):
    return float(np.mean([counts[L][keep[L]].sum() / counts[L].sum() for L in range(N_LAYERS)]))


def top_n(scores):
    return {L: np.argsort(scores[L])[::-1][:N_KEEP] for L in range(N_LAYERS)}


rng = np.random.default_rng(0)
# "big" routes to many experts and carries most of the corpus; "small" is a narrow
# specialist that shares nothing with it. Under the sum rule big wins every slot.
big = {L: topic(np.arange(0, 120), mass=40.0, rng=rng) for L in range(N_LAYERS)}
small = {L: topic(np.arange(200, 230), mass=1.0, rng=rng) for L in range(N_LAYERS)}
per = {"big": big, "small": small}

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


scores = maxmin_counts(per, FRAC, n_layers=N_LAYERS, n_experts=N_EXPERTS)
keep_mm = top_n(scores)

# 1. the score vector is a selection: exactly N_KEEP experts sit above every rejected one
for L in range(N_LAYERS):
    admitted = np.where(scores[L] > 1.0)[0]
    check(f"layer {L}: {N_KEEP} experts admitted", len(admitted) == N_KEEP, f"got {len(admitted)}")
    check(f"layer {L}: admitted rank above rejected",
          scores[L][admitted].min() > scores[L][scores[L] <= 1.0].max())

# 2. the same slots, allocated by the shipped sum rule
def norm(c, L):
    s = c[L].sum()
    return c[L] / s if s > 0 else c[L]


keep_sum = top_n({L: sum(norm(c, L) for c in per.values()) for L in range(N_LAYERS)})

mm = {t: coverage(keep_mm, per[t]) for t in per}
sm = {t: coverage(keep_sum, per[t]) for t in per}
print(f"     sum    big {sm['big']:.3f}  small {sm['small']:.3f}  min {min(sm.values()):.3f}")
print(f"     maxmin big {mm['big']:.3f}  small {mm['small']:.3f}  min {min(mm.values()):.3f}")

check("maxmin raises the worst-served topic", min(mm.values()) > min(sm.values()),
      f"{min(sm.values()):.3f} -> {min(mm.values()):.3f}")
# Which topic the sum rule starves is not the obvious one. Normalising per layer divides the
# corpus size out, so a broad topic that spreads its mass over many experts scores low on every
# one of them and loses every slot to a peaky specialist. On the box that is exactly what happens
# to `english` (0.556) next to `css` and `javascript` (0.803, 0.814).
loser = min(sm, key=sm.get)
check("the sum rule leaves one topic far behind", max(sm.values()) - min(sm.values()) > 0.2,
      f"{loser} at {sm[loser]:.3f} vs {max(sm.values()):.3f}")
check("maxmin narrows the spread",
      (max(mm.values()) - min(mm.values())) < (max(sm.values()) - min(sm.values())))

# 3. a single topic must be indistinguishable from ranking on that topic alone
one = maxmin_counts({"big": big}, FRAC, n_layers=N_LAYERS, n_experts=N_EXPERTS)
check("one topic == that topic's own top-N",
      all(set(top_n(one)[L].tolist()) == set(top_n({L2: norm(big, L2) for L2 in range(N_LAYERS)})[L].tolist())
          for L in range(N_LAYERS)))

# 4. deterministic: the warm start and the mask must agree across processes
again = maxmin_counts(per, FRAC, n_layers=N_LAYERS, n_experts=N_EXPERTS)
check("deterministic", all(np.array_equal(scores[L], again[L]) for L in range(N_LAYERS)))

# 5. a topic with no routing mass in a layer must not take that layer
silent = {L: (np.zeros(N_EXPERTS) if L == 0 else topic(np.arange(300, 330), mass=1.0, rng=rng))
          for L in range(N_LAYERS)}
mixed = maxmin_counts({"big": big, "silent": silent}, FRAC, n_layers=N_LAYERS, n_experts=N_EXPERTS)
layer0 = set(np.where(mixed[0] > 1.0)[0].tolist())
check("a zero-mass topic does not vote in its empty layer",
      layer0 == set(top_n({L: norm(big, L) for L in range(N_LAYERS)})[0].tolist()))
k1 = top_n(mixed)[1]
cov_silent = silent[1][k1].sum() / silent[1].sum()
cov_big = big[1][k1].sum() / big[1].sum()
check("it still votes where it has mass: parity with the other topic",
      cov_silent > 0.1 and abs(cov_silent - cov_big) < 0.1, f"silent {cov_silent:.3f} big {cov_big:.3f}")

# 6. a budget wider than the histogram must not loop forever or over-admit
wide = maxmin_counts(per, 1.0, n_layers=N_LAYERS, n_experts=N_EXPERTS)
check("full budget admits every expert",
      all(int((wide[L] > 1.0).sum()) == N_EXPERTS for L in range(N_LAYERS)))

print()
print(f"{len(fails)} failed: {', '.join(fails)}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
