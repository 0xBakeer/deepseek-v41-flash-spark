"""How coarse should the topic catalogue be?

The keep-set is composed from per-topic expert histograms, and under
DSV41_PRUNE_RANK=maxmin every SELECTED topic gets one vote for every slot: the
next expert of a layer goes to whichever selected topic currently has the least
of its routing mass covered. Granularity is therefore the weighting. Two topics
that route to nearly the same experts are two votes for one register and pull
slots away from everything else; one coarse topic covering several registers
votes once for their SUM, which is a vote for whichever register inside it
carries the most mass -- the minority register inside it is served only by
accident. That is the shape of the 2026-09-12 fault: `css` sat at 0.518 while
the bundle that was supposed to contain it read healthy.

So the catalogue question -- more topics, fewer topics -- is answerable from the
histograms alone, without a generation run:

  1. how similar are the 37 topics, measured two ways that fail differently:
     Jaccard of the per-layer top-N expert SETS (what the keep-set actually
     cares about, blind to mass) and Jensen-Shannon divergence between the
     per-layer normalised histograms (what the maxmin bookkeeping cares about,
     blind to the budget);
  2. what catalogue the distances themselves propose, by average-linkage
     agglomerative clustering cut at a threshold chosen by measurement rather
     than by eye: the COARSEST cut that costs no profile more than `--tol` of
     coverage on its worst-served fine topic;
  3. whether composing a profile from those clusters serves its worst FINE
     topic better than composing it from the fine topics -- measured against
     the fine topics in every case, so a register hidden inside a cluster is
     still measured on its own and cannot be flattered by its cluster-mates;
  4. (experiment) whether weighting a fine topic -- w votes instead of one --
     buys what re-clustering buys, without changing the catalogue at all.

`--drop-cost` answers the other half of the catalogue question. Clustering can
only make the catalogue coarser; the finer direction cannot be measured by
splitting topics that were never traced apart. What CAN be measured is the
mirror of a split: what a register loses when the selection stops naming it and
it has to live inside its neighbours. That is the 2026-09-12 fault with a number
on it.

Nothing here imports the engine: `_maxmin_counts` is lifted out of
engine/v41_engine.py with `ast` the way tools/test_maxmin.py lifts it, so this
runs on numpy alone with no torch and no GPU.

Run:
    python3 tools/topic_clusters.py --stats results/keepsets/topics/coverage.json
    python3 tools/topic_clusters.py --stats ... --keep 0.36 --profiles --weights
"""
import argparse
import ast
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_EXPERTS = 384
N_LAYERS = 40

try:                                   # only to say so; the clustering below never needs it
    import scipy.cluster.hierarchy as _sch   # noqa: F401
    HAVE_SCIPY = True
except Exception:                      # noqa: BLE001
    HAVE_SCIPY = False


# --- the engine's rule, without the engine ----------------------------------

def load_engine_fn(name: str):
    """The named top-level function from the engine, without importing torch."""
    src = open(os.path.join(ROOT, "engine/v41_engine.py")).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<engine>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in engine/v41_engine.py")


def keep_n(keep: float) -> int:
    return max(6, math.ceil(keep * N_EXPERTS))


def top_n(score, n: int):
    return np.argsort(np.asarray(score, dtype=np.float64))[::-1][:n]


# --- the histograms ---------------------------------------------------------

class Topics:
    """The per-layer expert histograms of a coverage.json, normalised per layer.

    `p[t]` is a list of 40 arrays summing to 1 (or to 0, for a topic with no
    mass in a layer). Normalising per layer is what the engine does before it
    ranks, and it is also what makes two topics comparable: the corpus sizes
    behind them differ and the layers differ in how peaked their routing is.
    """

    def __init__(self, path: str):
        d = json.load(open(path))
        per_layer = d["per_layer"]
        self.n_layers = len(per_layer)
        any_layer = per_layer[next(iter(per_layer))]
        self.names = sorted(k[len("counts_"):] for k in any_layer if k.startswith("counts_"))
        self.p = {}
        for t in self.names:
            rows = []
            for L in range(self.n_layers):
                c = np.asarray(per_layer[str(L)][f"counts_{t}"], dtype=np.float64)
                tot = c.sum()
                rows.append(c / tot if tot > 0 else c)
            self.p[t] = rows

    def hist(self, t):
        return self.p[t]

    def coverage(self, t, keep) -> float:
        """Mean over layers of the topic's routing mass on the kept experts."""
        return float(np.mean([self.p[t][L][keep[L]].sum() for L in range(self.n_layers)]))


def cluster_hist(topics: Topics, members) -> list:
    """A cluster's histogram: the mean of its members' per-layer-normalised
    histograms, which is the sum divided by the member count, so it is itself a
    distribution and one member cannot dominate by corpus size. maxmin
    re-normalises what it is handed, so the division changes no selection --
    it only makes the object comparable to a fine topic's histogram."""
    members = list(members)
    out = []
    for L in range(topics.n_layers):
        s = np.zeros(N_EXPERTS)
        for m in members:
            s += topics.p[m][L]
        out.append(s / len(members))
    return out


# --- section 1: two similarity matrices -------------------------------------

def jaccard_matrix(topics: Topics, n: int):
    """Mean over layers of |top-n(a) & top-n(b)| / |top-n(a) | top-n(b)|.

    This is the measure the keep-set is made of: it asks whether two topics
    WANT the same experts at this budget, and it cannot see how much mass they
    put on them. Two topics can share their whole top-139 and still disagree
    violently about which of those experts matter."""
    names = topics.names
    tops = {t: [set(int(e) for e in np.argsort(topics.p[t][L])[::-1][:n])
                for L in range(topics.n_layers)] for t in names}
    m = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            acc = 0.0
            for L in range(topics.n_layers):
                A, B = tops[a][L], tops[b][L]
                inter = len(A & B)
                acc += inter / (len(A) + len(B) - inter)
            v = acc / topics.n_layers
            m[i, j] = m[j, i] = v
    np.fill_diagonal(m, 1.0)
    return m


def _js(p, q) -> float:
    """Jensen-Shannon divergence in bits: 0 for identical, 1 for disjoint."""
    mix = 0.5 * (p + q)
    out = 0.0
    for x in (p, q):
        nz = x > 0
        out += 0.5 * float(np.sum(x[nz] * np.log2(x[nz] / mix[nz])))
    return out


def js_matrix(topics: Topics):
    """Mean over layers of the JS divergence between the normalised histograms.

    This one sees mass and is blind to the budget: it counts a disagreement
    about the 200th expert of a layer, which no keep-set at 139 will ever
    admit. Where it disagrees with Jaccard, that is the reason."""
    names = topics.names
    m = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            v = float(np.mean([_js(topics.p[a][L], topics.p[b][L]) for L in range(topics.n_layers)]))
            m[i, j] = m[j, i] = v
    return m


def pairs_sorted(names, m, reverse: bool):
    out = [(m[i, j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    out.sort(reverse=reverse)
    return out


def spearman(x, y) -> float:
    """Rank correlation, so the two measures can be compared on their orderings
    rather than on scales that have nothing to do with each other."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(list(x)), rank(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def section_similarity(topics: Topics, n_keep: int, top_pairs: int = 25, bottom_pairs: int = 10):
    names = topics.names
    J = jaccard_matrix(topics, n_keep)
    D = js_matrix(topics)
    inter = n_keep ** 2 / N_EXPERTS
    chance = inter / (2 * n_keep - inter)
    print(f"=== 1. similarity over {len(names)} topics "
          f"(Jaccard of per-layer top-{n_keep}; JS divergence in bits, both averaged over "
          f"{topics.n_layers} layers) ===")
    print(f"Jaccard at this budget has a floor: two {n_keep}-sets drawn at random out of {N_EXPERTS} "
          f"already share {inter:.0f} experts,")
    print(f"a Jaccard of {chance:.3f}. Read the low end of the Jaccard column against that, not "
          f"against zero.\n")

    jp = pairs_sorted(names, J, reverse=True)
    dp = pairs_sorted(names, D, reverse=False)
    jrank = {(a, b): i for i, (_, a, b) in enumerate(jp)}
    drank = {(a, b): i for i, (_, a, b) in enumerate(dp)}

    print(f"-- {top_pairs} most similar pairs --")
    print(f"{'#':>3}  {'by Jaccard':<34}{'jacc':>6}{'js':>7}{'jsRk':>6}   "
          f"{'by JS distance':<34}{'js':>7}{'jacc':>6}{'jcRk':>6}")
    for i in range(top_pairs):
        v1, a1, b1 = jp[i]
        v2, a2, b2 = dp[i]
        ia, ja = names.index(a1), names.index(b1)
        ib, jb = names.index(a2), names.index(b2)
        print(f"{i+1:>3}  {a1 + ' ~ ' + b1:<34}{v1:>6.3f}{D[ia, ja]:>7.4f}{drank[(a1, b1)]+1:>6}   "
              f"{a2 + ' ~ ' + b2:<34}{v2:>7.4f}{J[ib, jb]:>6.3f}{jrank[(a2, b2)]+1:>6}")

    print(f"\n-- {bottom_pairs} least similar pairs --")
    print(f"{'#':>3}  {'by Jaccard':<34}{'jacc':>6}{'js':>7}{'jsRk':>6}   "
          f"{'by JS distance':<34}{'js':>7}{'jacc':>6}{'jcRk':>6}")
    for i in range(bottom_pairs):
        v1, a1, b1 = jp[-1 - i]
        v2, a2, b2 = dp[-1 - i]
        ia, ja = names.index(a1), names.index(b1)
        ib, jb = names.index(a2), names.index(b2)
        print(f"{i+1:>3}  {a1 + ' ~ ' + b1:<34}{v1:>6.3f}{D[ia, ja]:>7.4f}{drank[(a1, b1)]+1:>6}   "
              f"{a2 + ' ~ ' + b2:<34}{v2:>7.4f}{J[ib, jb]:>6.3f}{jrank[(a2, b2)]+1:>6}")

    xs = [v for v, _, _ in jp]
    ys = [D[names.index(a), names.index(b)] for _, a, b in jp]
    rho = spearman(xs, ys)
    print(f"\nSpearman(Jaccard, -JS) over all {len(jp)} pairs: {-rho:+.3f}")
    dis = sorted(((abs(jrank[(a, b)] - drank[(a, b)]), a, b) for _, a, b in jp), reverse=True)
    print("largest rank disagreements between the two measures:")
    for d, a, b in dis[:10]:
        print(f"   {a + ' ~ ' + b:<34} jaccard rank {jrank[(a, b)]+1:>4} ({J[names.index(a), names.index(b)]:.3f})"
              f"   js rank {drank[(a, b)]+1:>4} ({D[names.index(a), names.index(b)]:.4f})   |d|={d}")
    return J, D


# --- section 2: hierarchical clustering -------------------------------------

def average_linkage(names, D):
    """Agglomerative average linkage, returning the merge sequence.

    scipy is used when it is installed and this is used when it is not; at 37
    leaves the difference is microseconds and the result is the same tree. Ties
    are broken by the lexicographic order of the cluster members, so two
    equidistant merges always happen in the same order and the catalogue this
    proposes is reproducible.
    """
    clusters = {i: [names[i]] for i in range(len(names))}
    dist = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist[(i, j)] = D[i, j]
    merges = []
    nxt = len(names)
    while len(clusters) > 1:
        best = None
        for (i, j), d in dist.items():
            if i not in clusters or j not in clusters:
                continue
            key = (round(d, 12), sorted(clusters[i])[0], sorted(clusters[j])[0])
            if best is None or key < best[0]:
                best = (key, i, j, d)
        _, i, j, d = best
        merged = clusters[i] + clusters[j]
        ni, nj = len(clusters[i]), len(clusters[j])
        for k in list(clusters):
            if k in (i, j):
                continue
            di = dist.get((min(i, k), max(i, k)))
            dj = dist.get((min(j, k), max(j, k)))
            dist[(min(nxt, k), max(nxt, k))] = (ni * di + nj * dj) / (ni + nj)
        del clusters[i], clusters[j]
        clusters[nxt] = merged
        merges.append((d, sorted(merged)))
        nxt += 1
    return merges


def cut(names, D, threshold: float):
    """The partition of `names` obtained by cutting the average-linkage tree at
    `threshold`: two topics share a cluster when they were merged below it."""
    clusters = {i: [names[i]] for i in range(len(names))}
    dist = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist[(i, j)] = D[i, j]
    nxt = len(names)
    while len(clusters) > 1:
        best = None
        for (i, j), d in dist.items():
            if i not in clusters or j not in clusters:
                continue
            key = (round(d, 12), sorted(clusters[i])[0], sorted(clusters[j])[0])
            if best is None or key < best[0]:
                best = (key, i, j, d)
        _, i, j, d = best
        if d > threshold:
            break
        ni, nj = len(clusters[i]), len(clusters[j])
        for k in list(clusters):
            if k in (i, j):
                continue
            di = dist.get((min(i, k), max(i, k)))
            dj = dist.get((min(j, k), max(j, k)))
            dist[(min(nxt, k), max(nxt, k))] = (ni * di + nj * dj) / (ni + nj)
        merged = clusters[i] + clusters[j]
        del clusters[i], clusters[j]
        clusters[nxt] = merged
        nxt += 1
    out = [sorted(v) for v in clusters.values()]
    out.sort(key=lambda c: (-len(c), c[0]))
    return out


def medoid(members, names, D) -> str:
    """The member with the least mean distance to the others -- the name that
    describes the cluster best out of the names there are."""
    if len(members) == 1:
        return members[0]
    idx = {n: i for i, n in enumerate(names)}
    return min(members, key=lambda m: (sum(D[idx[m], idx[o]] for o in members if o != m), m))


def cluster_label(members, names, D) -> str:
    m = medoid(members, names, D)
    return m if len(members) == 1 else f"{m}+{len(members) - 1}"


def section_clusters(topics: Topics, D, thresholds):
    names = topics.names
    print("\n\n=== 2. average-linkage hierarchical clustering on the JS distance ===")
    print(f"(scipy {'present, but the same loop runs either way' if HAVE_SCIPY else 'not installed'}; "
          f"{len(names)} leaves, pure-numpy agglomerative loop)\n")
    merges = average_linkage(names, D)
    print("-- merge heights, in order --")
    for k, (d, members) in enumerate(merges):
        print(f"{k+1:>3}  h={d:.4f}  ({len(members):>2}) {', '.join(members)}")
    hs = [d for d, _ in merges]
    gaps = sorted(((hs[i + 1] - hs[i], hs[i], i) for i in range(len(hs) - 1)), reverse=True)
    print("\nlargest gaps between consecutive merge heights (a gap is where a cut is safe):")
    for g, h, i in gaps[:6]:
        print(f"   after merge {i+1:>2} at h={h:.4f}: next merge at h={hs[i+1]:.4f}, gap {g:.4f}"
              f"  -> {len(names) - (i + 1)} clusters")
    print()
    for th in thresholds:
        cl = cut(names, D, th)
        print(f"-- cut at h <= {th:.3f}: {len(cl)} clusters "
              f"({sum(1 for c in cl if len(c) > 1)} multi-topic, {sum(1 for c in cl if len(c) == 1)} singletons) --")
        for c in cl:
            if len(c) > 1:
                print(f"   {cluster_label(c, names, D):<18} {', '.join(c)}")
        singles = [c[0] for c in cl if len(c) == 1]
        if singles:
            print(f"   {'singletons':<18} {', '.join(singles)}")
        print()
    return merges


# --- section 3: the decisive comparison -------------------------------------

def maxmin_keep(topics_hists, n, maxmin_counts, n_layers: int):
    """The engine's maxmin keep-set for a bundle of histograms, {L: expert ids}."""
    per = {name: h for name, h in topics_hists.items()}
    scores = maxmin_counts(per, n / N_EXPERTS, n_layers=n_layers, n_experts=N_EXPERTS)
    return {L: top_n(scores[L], n) for L in range(n_layers)}


def sum_keep(topics_hists, n, n_layers: int):
    """The old default: rank on the sum of the normalised histograms."""
    keep = {}
    for L in range(n_layers):
        s = np.zeros(N_EXPERTS)
        for h in topics_hists.values():
            s += h[L]
        keep[L] = top_n(s, n)
    return keep


def weighted_maxmin_keep(topics_hists, n, weights, n_layers: int):
    """maxmin where topic t counts as `weights[t]` votes.

    The engine hands the next slot to the topic with the smallest `got`; here it
    goes to the smallest `got / w`, so a topic with w=2 is served to twice the
    coverage of a w=1 topic before it stops asking. w=1 everywhere is exactly
    the engine's rule, and the test asserts that expert for expert.
    """
    names = list(topics_hists)
    w = [float(weights.get(t, 1.0)) for t in names]
    keep = {}
    for L in range(n_layers):
        p = [topics_hists[t][L] for t in names]
        order = [np.argsort(x)[::-1] for x in p]
        ptr = [0] * len(names)
        got = [0.0 if float(x.sum()) > 0 else float("inf") for x in p]
        if all(g == float("inf") for g in got):
            keep[L] = np.arange(n)
            continue
        admitted, seen = [], set()
        while len(admitted) < n:
            t = min(range(len(names)), key=lambda i: got[i] / w[i])
            while ptr[t] < N_EXPERTS and int(order[t][ptr[t]]) in seen:
                ptr[t] += 1
            if ptr[t] >= N_EXPERTS:
                got[t] = float("inf")
                if all(g == float("inf") for g in got):
                    break
                continue
            e = int(order[t][ptr[t]]); ptr[t] += 1
            seen.add(e); admitted.append(e)
            for u in range(len(names)):
                got[u] += float(p[u][e])
        keep[L] = np.asarray(admitted, dtype=np.int64)
    return keep


def propose_cut(topics: Topics, D, merges, profiles, keep_frac: float, maxmin_counts,
                tol: float, n_candidates: int = 12):
    """The coarsest cut that no profile pays for, and the table behind it.

    A threshold picked off the dendrogram by eye is a guess about a quantity
    that can simply be measured: for each candidate cut, compose every profile
    from the clusters its fine topics fall into and see how far the worst-served
    FINE topic drops against composing it from the fine topics. Merging two
    registers the router cannot tell apart costs nothing and is worth doing for
    the smaller catalogue; merging two it can tell apart shows up here as a real
    loss on whichever of them is the minority.
    """
    n = keep_n(keep_frac)
    hs = sorted({round(d, 6) for d, _ in merges})
    step = max(1, len(hs) // n_candidates)
    cands = hs[::step] + [hs[-1]]
    base = {}
    for (pname, _b, sel, _g, _r) in profiles:
        fine = sorted(sel) if sel else list(topics.names)
        base[pname] = (fine, min(row(topics, fine, maxmin_keep(
            {t: topics.p[t] for t in fine}, n, maxmin_counts, topics.n_layers)).values()))
    table = []
    for th in cands:
        clustering = cut(topics.names, D, th)
        of = {t: tuple(c) for c in clustering for t in c}
        losses = []
        for (pname, _b, sel, _g, _r) in profiles:
            fine, b = base[pname]
            cl_used = sorted({of[t] for t in fine})
            cl_h = {cluster_label(list(c), topics.names, D): cluster_hist(topics, c) for c in cl_used}
            w = min(row(topics, fine, maxmin_keep(cl_h, n, maxmin_counts, topics.n_layers)).values())
            losses.append((b - w, pname))
        table.append((th, len(clustering), max(losses), sum(l for l, _ in losses) / len(losses)))
    ok = [r for r in table if r[2][0] <= tol]
    chosen = min(ok, key=lambda r: r[1])[0] if ok else table[0][0]
    return chosen, table


def row(topics: Topics, fine, keep):
    return {t: topics.coverage(t, keep) for t in fine}


def fmt_row(cov, width: int = 4):
    items = sorted(cov.items(), key=lambda kv: kv[1])
    return "  ".join(f"{t} {v:.3f}" for t, v in items)


def section_profiles(topics: Topics, D, clustering, keep_frac: float, maxmin_counts, profiles):
    n = keep_n(keep_frac)
    names = topics.names
    of_cluster = {}
    for c in clustering:
        for t in c:
            of_cluster[t] = tuple(c)
    print("\n\n=== 3. profiles at keep {:.2f} ({} experts/layer): fine maxmin vs cluster maxmin vs sum ==="
          .format(keep_frac, n))
    print("coverage is always measured against the profile's ORIGINAL FINE topics.\n")
    summary = []
    for (pname, blurb, sel, _gated, _rank) in profiles:
        fine = sorted(sel) if sel else list(names)
        fine_h = {t: topics.p[t] for t in fine}
        cl_used = sorted({of_cluster[t] for t in fine})
        cl_h = {cluster_label(list(c), names, D): cluster_hist(topics, c) for c in cl_used}
        k_fine = maxmin_keep(fine_h, n, maxmin_counts, topics.n_layers)
        k_cl = maxmin_keep(cl_h, n, maxmin_counts, topics.n_layers)
        k_sum = sum_keep(fine_h, n, topics.n_layers)
        r_fine, r_cl, r_sum = row(topics, fine, k_fine), row(topics, fine, k_cl), row(topics, fine, k_sum)
        print(f"--- {pname} ({len(fine)} fine topics -> {len(cl_used)} clusters) : {blurb}")
        print(f"    clusters: {'; '.join(cluster_label(list(c), names, D) + '[' + ','.join(c) + ']' for c in cl_used)}")
        for tag, r in (("(i)   fine maxmin ", r_fine), ("(ii)  clust maxmin", r_cl), ("(iii) fine sum   ", r_sum)):
            worst = min(r, key=r.get)
            print(f"    {tag}  worst {worst} {r[worst]:.3f}   spread {max(r.values()) - min(r.values()):.3f}"
                  f"   mean {sum(r.values())/len(r):.3f}")
            print(f"          {fmt_row(r)}")
        win = "(i) fine" if min(r_fine.values()) >= min(r_cl.values()) else "(ii) clusters"
        print(f"    winner on the worst-served fine topic: {win} "
              f"({min(r_fine.values()):.3f} fine vs {min(r_cl.values()):.3f} clusters vs "
              f"{min(r_sum.values()):.3f} sum)\n")
        summary.append((pname, min(r_fine.values()), min(r_cl.values()), min(r_sum.values())))
    print("-- summary: worst-served fine topic per profile --")
    print(f"{'profile':<24}{'fine mm':>9}{'clust mm':>10}{'sum':>8}{'winner':>16}{'delta':>9}")
    for pname, a, b, c in summary:
        win = "fine" if a >= b else "clusters"
        print(f"{pname:<24}{a:>9.3f}{b:>10.3f}{c:>8.3f}{win:>16}{abs(a - b):>9.3f}")
    return summary


# --- section 4: weighting instead of granularity ----------------------------

def section_weights(topics: Topics, keep_frac: float, profiles, which, grid_topics, grid):
    n = keep_n(keep_frac)
    print("\n\n=== 4. EXPERIMENT: weighted maxmin instead of re-clustering ===")
    print("A topic with weight w is picked when got[t]/w is the minimum, i.e. it gets w votes.")
    print("This is a measurement, NOT a proposal to ship: nothing in the engine reads a weight,")
    print("and the grid is searched on the same histograms it is scored on.\n")
    for (pname, _blurb, sel, _g, _r) in profiles:
        if pname not in which:
            continue
        fine = sorted(sel)
        fine_h = {t: topics.p[t] for t in fine}
        base = row(topics, fine, maxmin_keep_cached[0](fine_h, n, topics.n_layers))
        print(f"--- {pname}: baseline (all weights 1) worst {min(base, key=base.get)} {min(base.values()):.3f}")
        rows = []
        for combo in grid:
            weights = dict(zip(grid_topics, combo))
            k = weighted_maxmin_keep(fine_h, n, weights, topics.n_layers)
            r = row(topics, fine, k)
            rows.append((min(r.values()), combo, r))
        rows.sort(reverse=True)
        print(f"    {'weights':<28}{'worst topic':<22}{'worst':>8}{'spread':>9}{'mean':>8}")
        for worst, combo, r in rows:
            wt = min(r, key=r.get)
            print(f"    {', '.join(f'{t}={w}' for t, w in zip(grid_topics, combo)):<28}{wt:<22}"
                  f"{worst:>8.3f}{max(r.values()) - min(r.values()):>9.3f}{sum(r.values())/len(r):>8.3f}")
        best = rows[0]
        print(f"    best: {', '.join(f'{t}={w}' for t, w in zip(grid_topics, best[1]))} "
              f"-> worst {best[0]:.3f} (baseline {min(base.values()):.3f}, "
              f"{best[0] - min(base.values()):+.3f})\n")


def section_drop_cost(topics: Topics, D, keep_frac: float, maxmin_counts, profiles):
    """What a register loses when the selection stops naming it.

    The coarse direction is measurable by clustering; the fine direction is not,
    because no sub-topic of `python` was ever traced separately. This is its
    mirror image: drop one topic from a profile's SELECTION and measure that
    topic anyway. If it keeps its coverage, the catalogue did not need it as a
    separate entry -- its neighbours were already asking for its experts. If it
    collapses, that is precisely the cost of a catalogue too coarse to name it,
    and the JS distance to its nearest still-selected neighbour predicts which.
    """
    n = keep_n(keep_frac)
    idx = {t: i for i, t in enumerate(topics.names)}
    print("\n\n=== 3b. what naming a register is worth: drop it from the selection, measure it anyway ===\n")
    print(f"{'profile':<24}{'dropped':<16}{'minJS':>7}{'selected':>10}{'dropped':>9}{'cost':>8}{'floor+':>8}")
    xs, ys = [], []
    for (pname, _b, sel, _g, _r) in profiles:
        if not sel:
            continue
        fine = sorted(sel)
        r_all = row(topics, fine, maxmin_keep({t: topics.p[t] for t in fine}, n, maxmin_counts, topics.n_layers))
        for d in fine:
            rest = [t for t in fine if t != d]
            r = row(topics, fine, maxmin_keep({t: topics.p[t] for t in rest}, n, maxmin_counts, topics.n_layers))
            mind = min(D[idx[d], idx[o]] for o in rest)
            floor_plus = min(v for k, v in r.items() if k != d) - min(r_all.values())
            xs.append(mind); ys.append(r_all[d] - r[d])
            print(f"{pname:<24}{d:<16}{mind:>7.3f}{r_all[d]:>10.3f}{r[d]:>9.3f}"
                  f"{r_all[d] - r[d]:>8.3f}{floor_plus:>+8.3f}")
    r_p = float(np.corrcoef(xs, ys)[0, 1])
    print(f"\nmean cost of dropping a topic from the selection: {sum(ys)/len(ys):.3f}")
    print(f"Pearson(JS distance to nearest still-selected topic, cost) = {r_p:+.3f}"
          f"   Spearman = {spearman(xs, ys):+.3f}")
    print("i.e. how much a separate catalogue entry is worth is a function of how far the register")
    print("sits from the ones already selected -- the same distance the clustering is cut on.")


maxmin_keep_cached = [None]     # filled in main(): a weight-free maxmin over fine histograms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stats", default=os.path.join(ROOT, "results/keepsets/topics/coverage.json"),
                    help="a coverage.json carrying counts_<topic> per layer")
    ap.add_argument("--keep", type=float, default=0.36, help="keep fraction (default 0.36 -> 139/384)")
    ap.add_argument("--thresholds", default="", help="comma-separated JS cut heights; default is data-chosen")
    ap.add_argument("--propose", type=float, default=None,
                    help="cut to propose as the catalogue; default is the coarsest one costing <= --tol")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="coverage a cut may cost the worst-served fine topic of any profile (default 0.01)")
    ap.add_argument("--profiles", action="store_true", help="also run section 3 over tools/tune.py PROFILES")
    ap.add_argument("--drop-cost", action="store_true", help="also run section 3b, the drop-one measurement")
    ap.add_argument("--weights", action="store_true", help="also run section 4, the weighting experiment")
    args = ap.parse_args(argv)

    if not os.path.exists(args.stats):
        print(f"no such stats file: {args.stats}", file=sys.stderr)
        return 2
    topics = Topics(args.stats)
    n = keep_n(args.keep)
    print(f"{len(topics.names)} topics x {topics.n_layers} layers x {N_EXPERTS} experts "
          f"from {os.path.relpath(args.stats, ROOT)}")
    print(f"keep {args.keep:.2f} -> {n} experts per layer\n")

    J, D = section_similarity(topics, n)

    merges = average_linkage(topics.names, D)
    hs = [d for d, _ in merges]
    if args.thresholds:
        ths = [float(x) for x in args.thresholds.split(",")]
    else:
        # four cuts spanning the tree: the quartiles of the merge heights, so the
        # thresholds come from this file's distances rather than from a guess.
        ths = [round(hs[int(q * (len(hs) - 1))], 3) for q in (0.25, 0.45, 0.62, 0.78)]
    merges = section_clusters(topics, D, ths)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tune as T                                            # noqa: E402  torch-free
    maxmin_counts = load_engine_fn("_maxmin_counts")
    maxmin_keep_cached[0] = lambda h, k, nl: maxmin_keep(h, k, maxmin_counts, nl)

    if args.propose is not None:
        th = args.propose
        print(f"-- catalogue cut given on the command line: h <= {th:.3f} --\n")
    else:
        th, table = propose_cut(topics, D, merges, T.PROFILES, args.keep, maxmin_counts, args.tol)
        print(f"-- choosing the cut by measurement: what each candidate costs the worst-served fine "
              f"topic of a profile (keep {args.keep:.2f}) --")
        print(f"{'cut h<=':>9}{'clusters':>10}{'worst loss':>12}{'  at profile':<26}{'mean loss':>11}")
        for h, ncl, (loss, pname), mean in table:
            mark = "  <= tol" if loss <= args.tol else ""
            print(f"{h:>9.4f}{ncl:>10}{loss:>12.3f}  {pname:<24}{mean:>11.3f}{mark}")
        print(f"the coarsest cut costing no profile more than {args.tol:.3f}: h <= {th:.4f}\n")
    clustering = cut(topics.names, D, th)
    print(f"-- PROPOSED catalogue: cut at h <= {th:.4f}, {len(clustering)} clusters --")
    for c in clustering:
        print(f"   {cluster_label(c, topics.names, D):<18} {', '.join(c)}")

    if args.profiles or args.drop_cost or args.weights:
        if args.profiles:
            section_profiles(topics, D, clustering, args.keep, maxmin_counts, T.PROFILES)
        if args.drop_cost:
            section_drop_cost(topics, D, args.keep, maxmin_counts, T.PROFILES)
        if args.weights:
            grid_topics = ["english", "reasoning"]
            grid = [(a, b) for a in (1, 2, 3) for b in (1, 2, 3)]
            section_weights(topics, args.keep, T.PROFILES,
                            {"Frontend", "Programming, broadly"}, grid_topics, grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
