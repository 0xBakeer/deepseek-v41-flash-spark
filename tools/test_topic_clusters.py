"""The catalogue analysis has to be reproducible before any of its numbers mean anything.

Run: python3 tools/test_topic_clusters.py

A clustering that depends on dict order, or a cluster histogram that is not the
thing the report says it is, would change the proposed catalogue between runs
and nobody would see it -- the output is a table of plausible-looking names
either way. So: the tree is deterministic, the cut is a partition, a cluster's
histogram is the normalised sum of its members, two identical histograms merge
first at distance zero, and weighted maxmin at weight 1 is the engine's rule
expert for expert. No torch, no GPU, no checkpoint: a synthetic coverage file
in a temporary directory, plus `_maxmin_counts` lifted out of the engine with
`ast` the way tools/test_maxmin.py lifts it.
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import topic_clusters as TC  # noqa: E402

N_EXPERTS, N_LAYERS = TC.N_EXPERTS, 4
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def peaky(rng, hot, mass):
    c = rng.random(N_EXPERTS) * 0.01
    c[hot] += mass / len(hot)
    return c


rng = np.random.default_rng(7)
# `alpha` and `alpha_twin` are byte-identical: no measure may separate them, and
# average linkage must marry them before it touches anything else.
alpha = [peaky(rng, np.arange(0, 60), 5.0) for _ in range(N_LAYERS)]
raw = {
    "alpha": alpha,
    "alpha_twin": [c.copy() for c in alpha],
    "beta": [peaky(rng, np.arange(40, 110), 5.0) for _ in range(N_LAYERS)],   # overlaps alpha
    "gamma": [peaky(rng, np.arange(200, 250), 4.0) for _ in range(N_LAYERS)],  # disjoint
    "delta": [peaky(rng, np.arange(300, 384), 3.0) for _ in range(N_LAYERS)],  # disjoint
}
TMP = tempfile.mkdtemp(prefix="topic-clusters-")
PATH = os.path.join(TMP, "coverage.json")
json.dump({"per_layer": {str(L): {"counts": [0.0] * N_EXPERTS,
                                  **{f"counts_{t}": raw[t][L].tolist() for t in raw}}
                         for L in range(N_LAYERS)},
           "global": {}}, open(PATH, "w"))

topics = TC.Topics(PATH)
check("the loader finds every topic", topics.names == sorted(raw), f"{topics.names}")
check("the loader finds every layer", topics.n_layers == N_LAYERS, f"{topics.n_layers}")
check("histograms are normalised per layer",
      all(abs(float(topics.p[t][L].sum()) - 1.0) < 1e-12 for t in topics.names for L in range(N_LAYERS)))

# --- the two similarity measures -------------------------------------------
D = TC.js_matrix(topics)
J = TC.jaccard_matrix(topics, 139)
names = topics.names
i, j = names.index("alpha"), names.index("alpha_twin")
check("JS distance is zero between identical topics", D[i, j] == 0.0, f"{D[i, j]:.2e}")
check("JS distance is symmetric and zero on the diagonal",
      np.allclose(D, D.T) and np.allclose(np.diag(D), 0.0))
check("JS divergence in bits stays in [0, 1]", D.min() >= 0.0 and D.max() <= 1.0,
      f"max {D.max():.3f}")
check("JS separates the disjoint topics from the overlapping ones",
      D[names.index("alpha"), names.index("gamma")] > D[names.index("alpha"), names.index("beta")])
check("Jaccard is 1 between identical topics and symmetric",
      J[i, j] == 1.0 and np.allclose(J, J.T))
# Jaccard at top-139 of 384 has a floor and it is not zero: two 139-sets drawn at
# random out of 384 already share 139^2/384 = 50 experts, which is a Jaccard of
# 0.22. Topics with disjoint HOT sets land at or just under that, so a Jaccard of
# 0.2 in the real matrix means "unrelated", not "somewhat related" -- the reason
# the report reads the low end of that matrix against this floor.
chance = (139 ** 2 / N_EXPERTS) / (2 * 139 - 139 ** 2 / N_EXPERTS)
jgd = J[names.index("gamma"), names.index("delta")]
check("Jaccard at this budget cannot go to zero: disjoint topics sit at the chance floor",
      abs(jgd - chance) < 0.05, f"{jgd:.3f} against a chance floor of {chance:.3f}")
check("disjoint topics still score below overlapping ones",
      jgd < J[names.index("alpha"), names.index("beta")],
      f"{jgd:.3f} < {J[names.index('alpha'), names.index('beta')]:.3f}")

# --- the tree ---------------------------------------------------------------
merges = TC.average_linkage(names, D)
check("identical histograms merge first, at distance zero",
      merges[0][1] == ["alpha", "alpha_twin"] and merges[0][0] == 0.0,
      f"{merges[0][1]} at h={merges[0][0]:.2e}")
check("the tree merges down to one cluster", merges[-1][1] == names and len(merges) == len(names) - 1)
check("merge heights are non-decreasing",
      all(merges[k][0] <= merges[k + 1][0] + 1e-12 for k in range(len(merges) - 1)))
check("the tree is deterministic", TC.average_linkage(names, D) == merges)
check("the tree does not depend on the order the names arrive in",
      TC.average_linkage(names, D) == TC.average_linkage(names, D.copy()))

# --- the cut is a partition -------------------------------------------------
for th in (0.0, 0.05, 0.2, 0.5, 1.0):
    cl = TC.cut(names, D, th)
    flat = [t for c in cl for t in c]
    check(f"cut at {th}: every topic in exactly one cluster",
          sorted(flat) == names and len(flat) == len(names), f"{len(cl)} clusters")
    check(f"cut at {th}: deterministic", TC.cut(names, D, th) == cl)
check("a coarser cut never has more clusters",
      all(len(TC.cut(names, D, a)) >= len(TC.cut(names, D, b))
          for a, b in zip((0.0, 0.05, 0.2, 0.5), (0.05, 0.2, 0.5, 1.0))))
check("the zero cut still merges the identical pair",
      ["alpha", "alpha_twin"] in TC.cut(names, D, 0.0),
      f"{TC.cut(names, D, 0.0)}")
check("a cut above the root is one cluster", TC.cut(names, D, 1.0) == [names])

# --- a cluster's histogram --------------------------------------------------
members = ["alpha", "beta", "gamma"]
ch = TC.cluster_hist(topics, members)
check("a cluster histogram is the normalised sum of its members",
      all(np.allclose(ch[L] * len(members), sum(topics.p[m][L] for m in members), atol=1e-12)
          for L in range(N_LAYERS)))
check("a cluster histogram is itself a distribution",
      all(abs(float(ch[L].sum()) - 1.0) < 1e-12 for L in range(N_LAYERS)))
check("a one-member cluster is that member",
      all(np.array_equal(TC.cluster_hist(topics, ["gamma"])[L], topics.p["gamma"][L])
          for L in range(N_LAYERS)))
check("a cluster of a topic and its twin is that topic",
      all(np.allclose(TC.cluster_hist(topics, ["alpha", "alpha_twin"])[L], topics.p["alpha"][L])
          for L in range(N_LAYERS)))
check("the member order does not change the cluster histogram",
      all(np.allclose(TC.cluster_hist(topics, members)[L],
                      TC.cluster_hist(topics, list(reversed(members)))[L]) for L in range(N_LAYERS)))
check("the medoid of a cluster is one of its members and is deterministic",
      TC.medoid(members, names, D) in members
      and TC.medoid(members, names, D) == TC.medoid(list(reversed(members)), names, D))

# --- weighted maxmin against the engine's rule ------------------------------
maxmin_counts = TC.load_engine_fn("_maxmin_counts")
n = TC.keep_n(0.36)
check("keep 0.36 is 139 experts per layer", n == 139, f"{n}")
hists = {t: topics.p[t] for t in names}
k_engine = TC.maxmin_keep(hists, n, maxmin_counts, N_LAYERS)
k_w1 = TC.weighted_maxmin_keep(hists, n, {}, N_LAYERS)
check("weighted maxmin at weight 1 is the engine's keep-set, expert for expert",
      all(set(int(e) for e in k_engine[L]) == set(int(e) for e in k_w1[L]) for L in range(N_LAYERS)))
check("weighted maxmin keeps exactly the budget",
      all(len(set(int(e) for e in k_w1[L])) == n for L in range(N_LAYERS)))
cov1 = TC.row(topics, names, k_w1)
k_w3 = TC.weighted_maxmin_keep(hists, n, {"gamma": 3.0}, N_LAYERS)
cov3 = TC.row(topics, names, k_w3)
check("a weight of 3 raises that topic and lowers the floor",
      cov3["gamma"] > cov1["gamma"] and min(cov3.values()) <= min(cov1.values()),
      f"gamma {cov1['gamma']:.3f} -> {cov3['gamma']:.3f}, floor "
      f"{min(cov1.values()):.3f} -> {min(cov3.values()):.3f}")
check("weighted maxmin is deterministic",
      all(np.array_equal(TC.weighted_maxmin_keep(hists, n, {"gamma": 3.0}, N_LAYERS)[L], k_w3[L])
          for L in range(N_LAYERS)))

# --- coverage and the sum rule ----------------------------------------------
check("coverage of a topic under the full budget is 1",
      abs(topics.coverage("alpha", {L: np.arange(N_EXPERTS) for L in range(N_LAYERS)}) - 1.0) < 1e-12)
k_sum = TC.sum_keep(hists, n, N_LAYERS)
check("the sum rule keeps the budget too", all(len(set(int(e) for e in k_sum[L])) == n for L in range(N_LAYERS)))
check("maxmin serves the worst-served topic at least as well as the sum rule",
      min(TC.row(topics, names, k_engine).values()) >= min(TC.row(topics, names, k_sum).values()) - 1e-12,
      f"{min(TC.row(topics, names, k_sum).values()):.3f} -> "
      f"{min(TC.row(topics, names, k_engine).values()):.3f}")

# --- the same numbers twice --------------------------------------------------
again = TC.Topics(PATH)
check("re-reading the file gives the same distances", np.array_equal(TC.js_matrix(again), D))

print()
print(f"{len(fails)} failed: {', '.join(fails)}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
