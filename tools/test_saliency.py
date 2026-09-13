"""Frequency is not contribution: the saliency histograms, end to end, without torch.

Run: python3 tools/test_saliency.py

Every keep-set this engine has ever built ranked experts by how OFTEN the router
picked them. REAP (Lasby et al., Cerebras, ICLR 2026, arXiv 2510.13999) measured
what that costs on Kimi-K2 -- 384 routed experts, one shared, auxiliary-loss-free
routing, the same shape as this model -- and found frequency-based pruning
collapses (LiveCodeBench 0.434 -> 0.082 at 75 % of experts kept, 0.000 at 50 %)
where a saliency criterion holds (0.440, 0.429). Saliency is the magnitude the
expert actually contributed: over a calibration set, `gate_weight(t, e)` times
`||expert_e(x_t)||` for the tokens routed to it.

So `DSV41_PRUNE_SOURCE` picks which of the two families ranks a keep-set, and
that only means anything if four things hold, which is what this checks:

  1. the aggregation is the definition -- tools/expert_stats.py's histogram
     equals the sum written out token by token, on a synthetic trace, and the
     whole tool runs over one and writes `saliency_<topic>` next to
     `counts_<topic>`;
  2. the two families really do disagree: an expert picked constantly at a tiny
     magnitude ranks above one picked rarely at a large one under `counts` and
     below it under `saliency`, which is exactly the failure mode REAP names;
  3. the ranking rules are untouched -- the engine's own `_maxmin_counts`, lifted
     out with `ast` the way tools/test_maxmin.py does, water-fills saliency
     histograms the same way it water-fills counts, and the screen's keep-set
     (tools/budget.py) is the engine's under saliency too;
  4. a coverage file that predates the tracer fails LOUDLY: `category_counts`
     returns nothing for a saliency source it does not carry, which is what makes
     the engine raise by name instead of quietly serving a frequency keep-set.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE = os.path.join(ROOT, "engine/v41_engine.py")
EXPERTS = os.path.join(ROOT, "engine/experts.py")
sys.path.insert(0, HERE)
import budget as B          # noqa: E402
import expert_stats as S    # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        fails.append(name)


def lift(path: str, *names):
    """The named top-level functions/assignments from a module that imports torch,
    run against numpy alone. The pattern tools/test_maxmin.py uses."""
    tree = ast.parse(open(path).read())
    ns = {"np": np, "json": json, "os": os}
    body = []
    for node in tree.body:
        hit = (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names) or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
        if hit:
            body.append(node)
    got = [n for n in names if n in {getattr(b, "name", None) for b in body}
           or any(isinstance(b, ast.Assign) and any(getattr(t, "id", None) == n for t in b.targets)
                  for b in body)]
    missing = [n for n in names if n not in got]
    assert not missing, f"{missing} not found in {os.path.basename(path)}"
    exec(compile(ast.Module(body=body, type_ignores=[]), f"<{os.path.basename(path)}>", "exec"), ns)
    return [ns[n] for n in names]


maxmin_counts, = lift(ENGINE, "_maxmin_counts")
category_counts, COUNT_SOURCES = lift(EXPERTS, "category_counts", "COUNT_SOURCES")

N_EXP, TOPK, N_LAYERS = 384, 6, 4

# --- a synthetic trace ------------------------------------------------------
# Three topics, built so that frequency and magnitude give OPPOSITE orders.
# `quiet` is the pathological case REAP describes: an expert the router reaches
# for constantly and whose output barely moves the residual stream. `loud` is its
# mirror. `middle` is there so the corpus has three categories -- expert_stats
# has a two-category branch that is not what is under test here.
#
#   name      experts   tokens   weight per pick   ||expert out||
TOPIC_SPEC = {
    "quiet":  (range(10, 16), 60, 0.25, 0.01),
    "loud":   (range(20, 26),  6, 0.25, 50.0),
    "middle": (range(30, 36), 20, 0.25, 1.0),
}
# 6 picks x 0.25 = 1.5, which is this checkpoint's `route_scale`: the weights a
# trace records are post-renormalisation and post-route_scale, because that is
# the number actually multiplied into the expert's output.


def synth_layer(L: int):
    """One layer's npz arrays. Experts shift by layer so the layers are not copies."""
    idx, w, nrm, cat = [], [], [], []
    for name, (experts, n_tok, weight, norm) in TOPIC_SPEC.items():
        ids = [(e + 4 * L) % N_EXP for e in experts]
        for t in range(n_tok):
            idx.append(ids)
            # a little spread, so nothing below can pass by treating the rows as identical
            w.append([weight * (1.0 + 0.01 * k) for k in range(TOPK)])
            nrm.append([norm * (1.0 + 0.02 * ((t + k) % 3)) for k in range(TOPK)])
            cat.append(name)
    return (np.array(idx, dtype=np.int16), np.array(w, dtype=np.float16),
            np.array(nrm, dtype=np.float16), np.array(cat))


def write_trace(root: str, with_norms: bool = True) -> str:
    os.makedirs(os.path.join(root, "trace"), exist_ok=True)
    n_tok = sum(s[1] for s in TOPIC_SPEC.values())
    for L in range(N_LAYERS):
        idx, w, nrm, cat = synth_layer(L)
        arrays = {"indices": idx, "weights": w, "category": cat,
                  "token": np.arange(len(cat), dtype=np.int32),
                  "scores": np.zeros(0, np.float16)}
        if with_norms:
            arrays["out_norms"] = nrm
        np.savez_compressed(os.path.join(root, "trace", f"layer{L}.npz"), **arrays)
    json.dump({"n_tokens": n_tok * N_LAYERS, "n_seqs": len(TOPIC_SPEC)},
              open(os.path.join(root, "meta.json"), "w"))
    return root


TMP = tempfile.mkdtemp(prefix="saliency-")
TRACE = write_trace(os.path.join(TMP, "run"))
OLD = write_trace(os.path.join(TMP, "old"), with_norms=False)

# --- 1. the aggregation is the definition -----------------------------------
idx, w, nrm, cat = synth_layer(0)
want = np.zeros(N_EXP)
for t in range(idx.shape[0]):
    for k in range(TOPK):
        want[int(idx[t, k])] += float(w[t, k]) * float(nrm[t, k])
got = S.saliency_hist(idx.astype(np.int64), w.astype(np.float32) * nrm.astype(np.float32))
check("saliency_hist is the token-by-token sum of weight x output norm",
      np.allclose(got, want, rtol=1e-6, atol=1e-9), f"max |diff| {np.abs(got - want).max():.3e}")
check("  and puts nothing anywhere else",
      set(np.nonzero(got)[0].tolist()) == set(int(e) for e in idx.reshape(-1).tolist()))

hot_q = set(int((e + 0) % N_EXP) for e in TOPIC_SPEC["quiet"][0])
hot_l = set(int((e + 0) % N_EXP) for e in TOPIC_SPEC["loud"][0])
check("the loud experts carry more saliency than the quiet ones",
      min(got[list(hot_l)]) > max(got[list(hot_q)]),
      f"loud {min(got[list(hot_l)]):.2f} vs quiet {max(got[list(hot_q)]):.2f}")

# --- expert_stats writes both families next to each other -------------------
run = subprocess.run([sys.executable, os.path.join(HERE, "expert_stats.py"),
                      "--trace", TRACE, "--out", os.path.join(TMP, "stats"),
                      "--budgets", "64"], capture_output=True, text=True)
check("expert_stats runs over a trace with out_norms", run.returncode == 0,
      (run.stderr or run.stdout)[-400:])
COV = os.path.join(TMP, "stats", "coverage.json")
cov = json.load(open(COV))
L0 = cov["per_layer"]["0"]
check("  writing saliency_<topic> beside counts_<topic>",
      all(f"saliency_{t}" in L0 and f"counts_{t}" in L0 for t in TOPIC_SPEC),
      ", ".join(sorted(k for k in L0 if "_" in k)))
check("  and the mixed saliency histogram", "saliency" in L0 and "counts" in L0)
check("  the per-topic histogram is the aggregation above",
      np.allclose(np.asarray(L0["saliency_quiet"]),
                  S.saliency_hist(idx[cat == "quiet"].astype(np.int64), w[cat == "quiet"].astype(np.float32) * nrm[cat == "quiet"].astype(np.float32)), rtol=1e-6, atol=1e-9))
check("  and the mixed one is the topics added up",
      np.allclose(np.asarray(L0["saliency"]),
                  sum(np.asarray(L0[f"saliency_{t}"]) for t in TOPIC_SPEC), rtol=1e-6, atol=1e-9))

# --- an old trace still processes -------------------------------------------
old = subprocess.run([sys.executable, os.path.join(HERE, "expert_stats.py"),
                      "--trace", OLD, "--out", os.path.join(TMP, "stats-old"),
                      "--budgets", "64"], capture_output=True, text=True)
check("a trace from before out_norms still processes", old.returncode == 0,
      (old.stderr or old.stdout)[-400:])
old_cov = json.load(open(os.path.join(TMP, "stats-old", "coverage.json")))
check("  with every counts_<topic> and no saliency_<topic>",
      all(f"counts_{t}" in old_cov["per_layer"]["0"] for t in TOPIC_SPEC)
      and not any(k.startswith("saliency") for k in old_cov["per_layer"]["0"]))
notes = [ln for ln in old.stdout.splitlines() if ln.startswith("note:")]
check("  and says so once -- not once per layer -- naming the fix",
      len(notes) == 1 and "out_norms" in notes[0] and "expert_trace" in notes[0],
      f"{len(notes)} notes: {notes[:1]}")

# --- 2. the two families disagree, and in the direction REAP describes ------
# One topic's own histogram, both ways. `quiet` is picked ten times as often as
# `loud` and contributes a two-thousandth as much, so the top of the layer under
# one measurement is near the bottom under the other.
mixed_counts = np.asarray(L0["counts"])
mixed_sal = np.asarray(L0["saliency"])
top6_counts = set(int(e) for e in np.argsort(mixed_counts)[::-1][:6])
top6_sal = set(int(e) for e in np.argsort(mixed_sal)[::-1][:6])
check("under counts the constantly-picked, barely-contributing experts win",
      top6_counts == hot_q, f"{sorted(top6_counts)} != {sorted(hot_q)}")
check("under saliency the rarely-picked, large-magnitude experts win",
      top6_sal == hot_l, f"{sorted(top6_sal)} != {sorted(hot_l)}")
check("  which is the whole point: the two keep-sets are disjoint",
      not (top6_counts & top6_sal))

# --- 3. the ranking rules are untouched -------------------------------------
# Two topics whose saliency ladders are identical in shape and 1000x apart in
# total. maxmin normalises per topic before it water-fills, so the budget splits
# evenly; an un-normalised rule would hand all six slots to the larger one. Run
# on saliency histograms, which is the point -- the rule does not know or care
# which measurement it is ranking.
LADDER = (0.6, 0.3, 0.1)
per = {}
for name, base, scale in (("small", 10, 1.0), ("large", 20, 1000.0)):
    per[name] = {L: np.zeros(N_EXP) for L in range(N_LAYERS)}
    for L in range(N_LAYERS):
        for i, share in enumerate(LADDER):
            per[name][L][base + i] = share * scale
scores = maxmin_counts(per, 0.01, n_layers=N_LAYERS)      # floors at 6 experts per layer
admitted = {L: set(int(e) for e in np.argsort(scores[L])[::-1][:6]) for L in range(N_LAYERS)}
check("maxmin water-fills saliency histograms the same way it does counts",
      all(admitted[L] == {10, 11, 12, 20, 21, 22} for L in range(N_LAYERS)),
      f"layer 0 admitted {sorted(admitted[0])}")
check("  spending the budget evenly despite a 1000x difference in total magnitude",
      all(len(admitted[L] & {10, 11, 12}) == 3 for L in range(N_LAYERS)))

# --- the screen's keep-set is the engine's, under saliency too --------------
# tools/budget.py reimplements the ranking because a screen cannot import torch.
# tools/test_budget_rank.py holds the two to the same keep-set on counts; this
# does it on saliency, because TopicIndex now reads either family and a bar drawn
# off the wrong one promises routing the server will not keep.
check("the engine still cuts a score vector at the top N",
      bool(re.search(r"keep\[int\(L\)\] = np\.argsort\(np\.asarray\(c\)\)\[::-1\]\[:n_keep\]",
                     open(ENGINE).read())))
rng = np.random.default_rng(11)
SEL = ("alpha", "beta", "gamma")
sal_per = {t: {L: (rng.random(N_EXP) * (0.02 if L % 2 else 1.0) + (i + 1) * 3.0 *
                   (np.arange(N_EXP) // 64 == i))
               for L in range(N_LAYERS)}
          for i, t in enumerate(SEL)}
cnt_per = {t: {L: np.floor(rng.random(N_EXP) * 50) + 1 for L in range(N_LAYERS)} for t in SEL}
synth_cov = os.path.join(TMP, "synth.json")
json.dump({"per_layer": {str(L): {**{f"counts_{t}": cnt_per[t][L].tolist() for t in SEL},
                                  **{f"saliency_{t}": sal_per[t][L].tolist() for t in SEL}}
                         for L in range(N_LAYERS)}}, open(synth_cov, "w"))


def engine_keep(sc: dict, frac: float) -> dict:
    n = max(6, int(np.ceil(frac * N_EXP)))
    return {L: set(int(e) for e in np.argsort(np.asarray(c))[::-1][:n]) for L, c in sc.items()}


def norm(c, L):
    a = np.asarray(c[L], dtype=np.float64)
    s = a.sum()
    return a / s if s > 0 else a


saved_layers = B.N_LAYERS
B.N_LAYERS = N_LAYERS
try:
    index = B.TopicIndex(synth_cov, "saliency")
    check("TopicIndex reads the saliency family", sorted(index.topics) == sorted(SEL)
          and index.source == "saliency", f"{index.topics} / {index.source}")
    check("  and takes its sample size from the counts of the same trace, not from magnitudes",
          all(index.tokens[t] == int(round(sum(cnt_per[t][L].sum() for L in range(N_LAYERS))
                                           / (N_LAYERS * TOPK))) for t in SEL))
    for frac in (0.02, 0.10, 0.36, 0.9):
        for rank, combine in (("sum", lambda parts: sum(parts)),
                              ("max", lambda parts: np.maximum.reduce(parts)),
                              ("maxmin", None)):
            if rank == "maxmin":
                want_sets = engine_keep(maxmin_counts(sal_per, frac, n_layers=N_LAYERS), frac)
            else:
                want_sets = engine_keep(
                    {L: combine([norm(sal_per[t], L) for t in SEL]) for L in range(N_LAYERS)}, frac)
            _c, order, _s = index.curves(tuple(sorted(SEL)), rank=rank)
            n = B.keep_n(frac)
            got_sets = {L: set(int(e) for e in order[L][:n]) for L in range(N_LAYERS)}
            bad = [L for L in range(N_LAYERS) if got_sets[L] != want_sets[L]]
            check(f"the screen's saliency keep-set == the engine's: {rank} at keep {frac:.2f}",
                  not bad, f"{len(bad)} layers differ")
    # and the two families really would have chosen differently on this file, or
    # the comparison above would pass whatever TopicIndex read
    counts_index = B.TopicIndex(synth_cov, "counts")
    o_sal = index.curves(tuple(sorted(SEL)), rank="maxmin")[1]
    o_cnt = counts_index.curves(tuple(sorted(SEL)), rank="maxmin")[1]
    n = B.keep_n(0.10)
    check("  reading the other family really would give another keep-set",
          any(set(o_sal[L][:n]) != set(o_cnt[L][:n]) for L in range(N_LAYERS)))
finally:
    B.N_LAYERS = saved_layers

# --- 4. a file without the histograms fails loudly --------------------------
check("the engine names both families", tuple(COUNT_SOURCES) == ("counts", "saliency"),
      str(COUNT_SOURCES))
bare = os.path.join(TMP, "bare")
os.makedirs(bare, exist_ok=True)
bare_cov = os.path.join(bare, "coverage.json")
shutil.copyfile(os.path.join(TMP, "stats-old", "coverage.json"), bare_cov)
check("category_counts reads a topic's counts as it always did",
      len(category_counts(bare_cov, "quiet", n_layers=N_LAYERS)) == N_LAYERS)
check("  and returns NOTHING for a saliency source the file does not carry",
      category_counts(bare_cov, "quiet", n_layers=N_LAYERS, source="saliency") == {},
      "it returned a histogram, so the engine would build a keep-set from it")
try:
    category_counts(bare_cov, "quiet", source="salience")
    check("an unknown source is refused, not quietly served as counts", False, "it read anyway")
except ValueError as e:
    check("an unknown source is refused, not quietly served as counts", "saliency" in str(e), str(e))

# The raw-trace fallback: a stats file with no saliency in it, but the trace
# arrays still next to it, must recompute the same histogram rather than give up.
raw = os.path.join(TMP, "raw")
os.makedirs(os.path.join(raw, "stats"), exist_ok=True)
shutil.copytree(os.path.join(TRACE, "trace"), os.path.join(raw, "trace"))
json.dump({"per_layer": {}}, open(os.path.join(raw, "stats", "coverage.json"), "w"))
back = category_counts(os.path.join(raw, "stats", "coverage.json"), "quiet",
                       n_layers=N_LAYERS, source="saliency")
check("saliency is recomputed from the raw trace when the stats file has none",
      len(back) == N_LAYERS and np.allclose(back[0], np.asarray(L0["saliency_quiet"]),
                                            rtol=1e-6, atol=1e-9),
      f"{len(back)} layers")

# --- the engine actually threads the source through -------------------------
src = open(ENGINE).read()
check("the engine reads DSV41_PRUNE_SOURCE with `or`, so an empty .env line means the default",
      bool(re.search(r'os\.environ\.get\("DSV41_PRUNE_SOURCE"\)\s*or\s*"counts"', src)))
check("  refuses an unknown value", "unknown DSV41_PRUNE_SOURCE" in src)
check("  builds the keep-set from it",
      bool(re.search(r"EX\.category_counts\(trace_stats, t, source=self\.prune_source\)", src)))
check("  reports it in config() and on the startup line",
      '"prune_source": self.prune_source' in src and "on {self.prune_source}" in src)
check("  and the prune sweep ranks on it too",
      src.count("source=src") == 3, f"{src.count('source=src')} call sites")

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
