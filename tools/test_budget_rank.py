"""The screen's keep-set must be the engine's keep-set, under every rank rule.

Run: python3 tools/test_budget_rank.py

tools/budget.py reimplements the ranking because the engine imports torch and a
screen cannot afford to. A reimplementation that drifts is worse than none: the
coverage bar then promises what the server will not deliver, and nothing on the
box contradicts it until a generation degenerates. Over
{english, html, python, reasoning, css, javascript, typescript} at keep 0.36 the
two rules disagree by 0.15 on english, so drift of that size hides in plain
sight.

So this is the contract: the engine's own `_maxmin_counts` is lifted out with
`ast` (the pattern tools/test_maxmin.py uses) and run against numpy alone, and
its keep-set is compared with budget.py's, per layer, as sets, at several keep
fractions and on several shapes of histogram.
"""
import ast
import json
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engine/v41_engine.py")
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


def load(name: str):
    """The named top-level function from the engine, without importing torch."""
    tree = ast.parse(open(ENGINE).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<engine>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in {os.path.basename(ENGINE)}")


maxmin_counts = load("_maxmin_counts")

# How the engine turns a score vector into a layer's keep-set (build_keep_masks,
# select="uniform"). It needs torch for the masks, so the one line that decides
# WHICH experts is reproduced here -- and checked against the source, because a
# change to it would break this comparison silently.
src = open(ENGINE).read()
check("the engine still cuts a score vector at the top N",
      bool(re.search(r"keep\[int\(L\)\] = np\.argsort\(np\.asarray\(c\)\)\[::-1\]\[:n_keep\]", src)))


def engine_keep(scores: dict, frac: float) -> dict:
    n = max(6, int(np.ceil(frac * B.N_EXPERTS)))
    return {L: set(int(e) for e in np.argsort(np.asarray(c))[::-1][:n]) for L, c in scores.items()}


def tool_keep(index: B.TopicIndex, topics, frac: float, rank: str) -> dict:
    _curves, order, _combined = index.curves(tuple(topics), rank=rank)
    n = B.keep_n(frac)
    return {L: set(int(e) for e in order[L][:n]) for L in range(B.N_LAYERS)}


# --- the histograms ---------------------------------------------------------
# Four shapes, because the rules differ on shape and not on size: one broad
# topic whose mass is spread thin over a third of the experts (the shape `sum`
# starves -- normalising per layer divides the corpus out, so every one of its
# experts scores low), two narrow specialists that overlap each other but not
# it, and one that overlaps the broad topic. Every third layer is flat and the
# rest are peaky, so both ends of the router's own spread are in.
#
# Each layer of a topic is scaled to the same total, because a real trace has
# that property and one of the checks below depends on it: every token routes to
# six experts in EVERY layer, so a topic's layers all carry 6 x its tokens. The
# totals differ BETWEEN topics -- 20k tokens against 3k -- which is exactly what
# the per-layer normalisation exists to divide out.
rng = np.random.default_rng(7)


def histogram(hot, mass, layer, tokens):
    c = rng.random(B.N_EXPERTS) * (0.5 if layer % 3 else 0.02)
    c[hot] += mass / len(hot)
    c = c / c.sum() * (tokens * 6)
    return [float(x) for x in c]


SHAPES = {
    "broad":   (np.arange(0, 128), 30.0, 20_000),
    "narrow":  (np.arange(200, 224), 1.0, 3_000),
    "sharing": (np.arange(210, 250), 4.0, 12_000),
    "overlap": (np.arange(100, 160), 12.0, 7_500),
}
per = {t: {L: histogram(hot, mass, L, tokens) for L in range(B.N_LAYERS)}
       for t, (hot, mass, tokens) in SHAPES.items()}
# And one topic that is silent in two layers, because a topic with no mass in a
# layer is the least-covered topic in it forever and would otherwise be handed
# every slot in exchange for nothing. The engine takes it out of the running
# there; this has to take it out of the running in the same layers.
per["silent"] = {L: ([0.0] * B.N_EXPERTS if L in (0, 7)
                     else histogram(np.arange(300, 330), 2.0, L, 5_000))
                 for L in range(B.N_LAYERS)}

TMP = tempfile.mkdtemp(prefix="budget-rank-")
path = os.path.join(TMP, "coverage.json")
json.dump({"per_layer": {str(L): {f"counts_{t}": per[t][L] for t in per}
                         for L in range(B.N_LAYERS)}}, open(path, "w"))
index = B.TopicIndex(path)
check("the synthetic keep-set loads", sorted(index.topics) == sorted(list(SHAPES) + ["silent"]))

# The tool ranks the topics in the order tune.py writes EXPERT_TOPICS, sorted;
# the engine ranks them in the order it reads that variable. Same order, same
# tie-break, so the engine is handed the same dict here.
SELECTIONS = [
    sorted(SHAPES),                       # all four
    sorted(("broad", "narrow")),          # no overlap at all
    sorted(("narrow", "sharing")),        # overlapping specialists
    ["broad"],                            # one topic is its own top-N
    sorted(("broad", "silent")),          # one of them has nothing to ask for in two layers
    ["silent"],                           # and in those two layers nobody does
]
FRACS = (0.02, 0.06, 0.10, 0.36, 0.5, 0.9, 1.0)

for sel in SELECTIONS:
    label = ",".join(sel)
    for frac in FRACS:
        want = engine_keep(maxmin_counts({t: per[t] for t in sel}, frac), frac)
        got = tool_keep(index, sel, frac, "maxmin")
        bad = [L for L in range(B.N_LAYERS) if got[L] != want[L]]
        detail = ""
        if bad:
            L = bad[0]
            detail = (f"{len(bad)} layers differ; layer {L} by "
                      f"{len(got[L] ^ want[L])} of {len(want[L])} experts")
        check(f"maxmin keep-set == the engine's: {label} at keep {frac:.2f}", not bad, detail)

# --- one order per selection is enough, and this is why ---------------------
# budget.py computes the admission order ONCE per selection and reads every keep
# fraction off a prefix of it. That is only legitimate because the engine's own
# water-filling never reads the budget while it allocates -- `n_keep` appears in
# the loop condition and nowhere else -- so its keep-sets nest: the set at 139
# experts is the first 139 of the set at 200. Checked here against the engine
# itself, at the fractions the tool is actually used at, because the whole
# coverage curve rests on it.
nested = []
for sel in SELECTIONS:
    sets = [engine_keep(maxmin_counts({t: per[t] for t in sel}, f), f) for f in FRACS]
    for (fa, a), (fb, b) in zip(zip(FRACS, sets), zip(FRACS[1:], sets[1:])):
        for L in range(B.N_LAYERS):
            if not a[L] <= b[L]:
                nested.append(f"{','.join(sel)} layer {L}: keep {fa} is not inside keep {fb}")
check("the engine's own maxmin keep-sets nest as the budget grows", not nested,
      f"{len(nested)} do not: {nested[0] if nested else ''}")

# --- the rule has to matter, or the comparison above proves nothing ----------
sel = sorted(SHAPES)
mm = tool_keep(index, sel, 0.36, "maxmin")
sm = tool_keep(index, sel, 0.36, "sum")
check("the two rules really do choose different experts",
      any(mm[L] != sm[L] for L in range(B.N_LAYERS)))


def coverage_at(keep_sets, topic):
    """The routing of `topic` that those per-layer sets keep, as a fraction of
    all of it -- the definition behind the bar, counted straight off the
    histograms instead of read off the curve."""
    kept = sum(per[topic][L][e] for L in range(B.N_LAYERS) for e in keep_sets[L])
    return kept / sum(sum(per[topic][L]) for L in range(B.N_LAYERS))


cov_mm = {t: coverage_at(mm, t) for t in sel}
cov_sm = {t: coverage_at(sm, t) for t in sel}
print("     sum    " + "  ".join(f"{t} {cov_sm[t]:.3f}" for t in sel))
print("     maxmin " + "  ".join(f"{t} {cov_mm[t]:.3f}" for t in sel))
check("maxmin raises the worst-served topic", min(cov_mm.values()) > min(cov_sm.values()),
      f"{min(cov_sm.values()):.3f} -> {min(cov_mm.values()):.3f}")
check("  and narrows the spread",
      (max(cov_mm.values()) - min(cov_mm.values())) < (max(cov_sm.values()) - min(cov_sm.values())))

# --- and the coverage the screen reads off the curve is that keep-set's ------
# The bar is curve[ceil(keep * 384)], not a second computation, so it has to
# agree with the set the same call returned.
curves = index.curves(tuple(sel), rank="maxmin")[0]
n = B.keep_n(0.36)
check("the curve agrees with the keep-set it came from",
      all(abs(curves[t][n] - coverage_at(mm, t)) < 1e-12 for t in sel))
check("  and is still monotone in the keep fraction",
      all(curves[t][i] <= curves[t][i + 1] + 1e-12 for t in sel for i in range(B.N_EXPERTS)))
check("  reaching 1.0 with every expert kept",
      all(abs(curves[t][B.N_EXPERTS] - 1.0) < 1e-9 for t in sel))

# --- the other two rules ----------------------------------------------------
# `sum` and `max` are three lines inside the engine's constructor rather than a
# function, so they cannot be lifted; they are reproduced here from the source
# the same way and checked against it.
check("the engine still sums per-layer-normalised counts",
      bool(re.search(r'rank == "sum":\s*\n\s*counts = \{L: sum\(_norm\(c, L\) for c in per\.values\(\)\)', src)))
check("  and still takes the elementwise max for max",
      bool(re.search(r'rank == "max":\s*\n\s*counts = \{L: np\.maximum\.reduce', src)))


def norm(c, L):
    a = np.asarray(c[L], dtype=np.float64)
    s = a.sum()
    return a / s if s > 0 else a


for rank, combine in (("sum", lambda parts: sum(parts)),
                      ("max", lambda parts: np.maximum.reduce(parts))):
    for sel in SELECTIONS:
        for frac in (0.06, 0.36, 0.9):
            scores = {L: combine([norm(per[t], L) for t in sel]) for L in range(B.N_LAYERS)}
            want = engine_keep(scores, frac)
            got = tool_keep(index, sel, frac, rank)
            bad = [L for L in range(B.N_LAYERS) if got[L] != want[L]]
            check(f"{rank} keep-set == the engine's: {','.join(sel)} at keep {frac:.2f}", not bad,
                  f"{len(bad)} layers differ")

try:
    index.curves(tuple(sorted(SHAPES)), rank="mxmn")
    check("an unknown rank is refused, not quietly served as the default", False, "it ranked anyway")
except ValueError as e:
    check("an unknown rank is refused, not quietly served as the default",
          "maxmin" in str(e), str(e))

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
