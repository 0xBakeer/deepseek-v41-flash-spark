"""Why saliency+maxmin fixed some languages and not French, German, Chinese, Japanese.

Offline, torch-free, read-only. Every number here comes from two places and nothing
else:

  * results/keepsets/topics/coverage.json -- per_layer[L]["saliency_<topic>"] and
    per_layer[L]["counts_<topic>"], 384 floats per layer, 40 layers, 37 topics.
  * engine/v41_engine.py `_maxmin_counts`, lifted with `ast` exactly as
    tools/test_maxmin.py does, so the keep-set built here is the keep-set the engine
    builds for DSV41_PRUNE_RANK=maxmin + DSV41_PRUNE_SELECT=uniform.

The recipe under test: DSV41_PRUNE_SOURCE=saliency, DSV41_PRUNE_RANK=maxmin,
--prune-keep 0.40 -> n_keep = max(6, ceil(0.40*384)) = 154 experts per layer; 0.36
-> 139. The profile topic lists are tools/tune.py PROFILES ("European languages",
"World languages"), used verbatim.

Sections mirror the six questions:
  1  coverage of each language under its profile's recipe keep-set, by counts and by
     saliency, at 0.40 and 0.36
  2  concentration of each language's saliency (entropy in bits, n experts holding
     50%/90% of the mass) and how much of the top-50% set the keep-set rejects
  3  the saliency top-20 experts a failing language does not get, and whether the
     four failing languages are asking for the SAME experts
  4  EXPERIMENT ONLY: weighted maxmin (tools/topic_clusters.py's variant), a grid
  5  fewer topics: drop the two worst, and split a profile into two of five
  6  printed conclusions are in the report, not here

Run:  python3 tools/language_gap.py
"""
import ast
import itertools
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COV = os.path.join(ROOT, "results/keepsets/topics/coverage.json")
N_EXPERTS = 384


# --- the engine's rule, without the engine (same lift as tools/test_maxmin.py) ---
def load_engine_fn(name: str):
    tree = ast.parse(open(os.path.join(ROOT, "engine/v41_engine.py")).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<engine>", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in engine/v41_engine.py")


_maxmin_counts = load_engine_fn("_maxmin_counts")


def keep_n(frac: float) -> int:
    return max(6, math.ceil(frac * N_EXPERTS))


def top_n(score, n):
    return np.argsort(np.asarray(score, dtype=np.float64))[::-1][:n]


# --- histograms -------------------------------------------------------------
class Hists:
    """per-layer histograms, normalised per layer, for one source family."""

    def __init__(self, path, source):
        d = json.load(open(path))
        pl = d["per_layer"]
        self.n_layers = len(pl)
        any_layer = pl[next(iter(pl))]
        pre = f"{source}_"
        self.names = sorted(k[len(pre):] for k in any_layer if k.startswith(pre))
        self.raw, self.p = {}, {}
        for t in self.names:
            rows, nrows = [], []
            for L in range(self.n_layers):
                c = np.asarray(pl[str(L)][f"{pre}{t}"], dtype=np.float64)
                rows.append(c)
                s = c.sum()
                nrows.append(c / s if s > 0 else c)
            self.raw[t] = rows
            self.p[t] = nrows

    def coverage(self, t, keep):
        return float(np.mean([self.p[t][L][keep[L]].sum() for L in range(self.n_layers)]))


SAL = Hists(COV, "saliency")
CNT = Hists(COV, "counts")
NL = SAL.n_layers

PROFILES = {
    "European languages": ["english", "german", "french", "spanish", "italian",
                           "portuguese", "translation", "reasoning", "reasoning_code"],
    "World languages": ["english", "arabic", "chinese", "japanese", "russian",
                        "turkish", "translation", "reasoning", "reasoning_code"],
}
EU_LANGS = ["english", "german", "french", "spanish", "italian", "portuguese"]
WL_LANGS = ["english", "arabic", "chinese", "japanese", "russian", "turkish"]


def keepset(topics, frac, hists=SAL):
    """The engine's keep-set: maxmin over these topics' histograms of `hists`."""
    per = {t: {L: hists.raw[t][L] for L in range(NL)} for t in topics}
    scores = _maxmin_counts(per, frac, n_layers=NL, n_experts=N_EXPERTS)
    n = keep_n(frac)
    return {L: top_n(scores[L], n) for L in range(NL)}


def weighted_maxmin(topics, frac, weights, hists=SAL):
    """tools/topic_clusters.py's weighted variant: topic t is picked when got/w is
    the minimum, so w=1 everywhere reproduces the engine exactly."""
    n = keep_n(frac)
    names = list(topics)
    w = [float(weights.get(t, 1.0)) for t in names]
    keep = {}
    for L in range(NL):
        p = [hists.p[t][L] for t in names]
        order = [np.argsort(x)[::-1] for x in p]
        ptr = [0] * len(names)
        got = [0.0 if float(x.sum()) > 0 else float("inf") for x in p]
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
            e = int(order[t][ptr[t]])
            ptr[t] += 1
            seen.add(e)
            admitted.append(e)
            for u in range(len(names)):
                got[u] += float(p[u][e])
        keep[L] = np.asarray(admitted, dtype=np.int64)
    return keep


def entropy_bits(t, hists=SAL):
    out = []
    for L in range(NL):
        x = hists.p[t][L]
        nz = x[x > 0]
        out.append(-float(np.sum(nz * np.log2(nz))))
    return float(np.mean(out))


def n_for_mass(t, mass, hists=SAL):
    out = []
    for L in range(NL):
        x = np.sort(hists.p[t][L])[::-1]
        out.append(int(np.searchsorted(np.cumsum(x), mass) + 1))
    return float(np.mean(out))


def top_mass_set(t, L, mass, hists=SAL):
    o = np.argsort(hists.p[t][L])[::-1]
    c = np.cumsum(hists.p[t][L][o])
    k = int(np.searchsorted(c, mass) + 1)
    return set(int(e) for e in o[:k])


def missing_from_top_mass(t, keep, mass, hists=SAL):
    out = []
    for L in range(NL):
        s = top_mass_set(t, L, mass, hists)
        out.append(len(s - set(int(e) for e in keep[L])))
    return float(np.mean(out)), out


def missing_from_top_n(t, keep, n, hists=SAL):
    """(mean count missing, {L: set of missing expert ids})"""
    per, tot = {}, []
    for L in range(NL):
        s = set(int(e) for e in np.argsort(hists.p[t][L])[::-1][:n])
        m = s - set(int(e) for e in keep[L])
        per[L] = m
        tot.append(len(m))
    return float(np.mean(tot)), per


def fmt(x):
    return f"{x:.3f}"


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


keeps = {}      # {(profile, frac): {L: expert ids}} -- filled by main(), read by extra()


def main():
    frac_list = [0.40, 0.36]
    for pname, topics in PROFILES.items():
        for f in frac_list:
            keeps[(pname, f)] = keepset(topics, f)

    # ---------------- 1 ----------------
    hr("1. coverage of each language under its profile's recipe keep-set")
    print(f"   keep-set = _maxmin_counts(saliency histograms of the 9 profile topics, frac)")
    print(f"   n_keep: 0.40 -> {keep_n(0.40)}   0.36 -> {keep_n(0.36)}")
    print(f"\n{'profile':<20}{'topic':<16}"
          f"{'sal@.40':>9}{'cnt@.40':>9}{'sal@.36':>9}{'cnt@.36':>9}")
    rows = {}
    for pname, topics in PROFILES.items():
        for t in topics:
            r = []
            for f in frac_list:
                k = keeps[(pname, f)]
                r += [SAL.coverage(t, k), CNT.coverage(t, k)]
            rows[(pname, t)] = r
            print(f"{pname:<20}{t:<16}" + "".join(f"{v:>9.3f}" for v in
                                                  [r[0], r[1], r[2], r[3]]))

    # ---------------- 2 ----------------
    hr("2. concentration of each language's saliency")
    print(f"{'topic':<14}{'H bits':>8}{'n50':>7}{'n90':>7}"
          f"{'miss50@.40':>12}{'miss50@.36':>12}{'miss90@.40':>12}")
    langs = [("European languages", t) for t in EU_LANGS if t != "english"]
    langs = [("European languages", "english")] + langs + \
            [("World languages", t) for t in WL_LANGS if t != "english"]
    for pname, t in langs:
        h = entropy_bits(t)
        n50, n90 = n_for_mass(t, 0.5), n_for_mass(t, 0.9)
        m40, _ = missing_from_top_mass(t, keeps[(pname, 0.40)], 0.5)
        m36, _ = missing_from_top_mass(t, keeps[(pname, 0.36)], 0.5)
        m90, _ = missing_from_top_mass(t, keeps[(pname, 0.40)], 0.9)
        print(f"{t:<14}{h:>8.2f}{n50:>7.1f}{n90:>7.1f}{m40:>12.1f}{m36:>12.1f}{m90:>12.1f}")
    print("\n(english appears once, under European languages; its World-languages row:)")
    t = "english"
    m40, _ = missing_from_top_mass(t, keeps[("World languages", 0.40)], 0.5)
    print(f"{'english(WL)':<14}{'':>8}{'':>7}{'':>7}{m40:>12.1f}")

    # ---------------- 3 ----------------
    hr("3. saliency top-20 experts absent from the keep-set")
    TOPN = 20
    miss = {}
    for pname, t in langs:
        mean, per = missing_from_top_n(t, keeps[(pname, 0.40)], TOPN)
        miss[t] = per
        print(f"{t:<14} mean missing of top-20 @0.40: {mean:>5.2f}")
    print()
    print("overlap of the MISSING sets, per layer (mean size of intersection):")
    four = ["french", "german", "chinese", "japanese"]
    ctrl = ["portuguese", "spanish", "italian", "russian", "turkish", "arabic"]
    allnames = four + ctrl
    print(f"{'':<12}" + "".join(f"{n[:9]:>11}" for n in allnames))
    for a in allnames:
        line = f"{a:<12}"
        for b in allnames:
            v = float(np.mean([len(miss[a][L] & miss[b][L]) for L in range(NL)]))
            line += f"{v:>11.2f}"
        print(line)
    print()
    inter4 = float(np.mean([len(miss["french"][L] & miss["german"][L] &
                                miss["chinese"][L] & miss["japanese"][L]) for L in range(NL)]))
    union4 = float(np.mean([len(miss["french"][L] | miss["german"][L] |
                                miss["chinese"][L] | miss["japanese"][L]) for L in range(NL)]))
    print(f"all four failing languages, per layer: intersection {inter4:.2f}, union {union4:.2f}")
    # shared by >= 2 of the four
    sh2 = []
    for L in range(NL):
        cnt = {}
        for a in four:
            for e in miss[a][L]:
                cnt[e] = cnt.get(e, 0) + 1
        sh2.append(sum(1 for v in cnt.values() if v >= 2))
    print(f"experts missing for >= 2 of the four, per layer: {np.mean(sh2):.2f}")
    # and the same for the passing controls, as a baseline
    inter_ctrl = float(np.mean([len(miss["portuguese"][L] & miss["spanish"][L] &
                                    miss["italian"][L]) for L in range(NL)]))
    print(f"pt & es & it intersection, per layer: {inter_ctrl:.2f}")
    # do the failing languages' missing experts sit in the keep-set of the OTHER profile?
    print()
    print("are the missing experts admitted by the other profile's keep-set?")
    for t, own, other in [("french", "European languages", "World languages"),
                          ("german", "European languages", "World languages"),
                          ("chinese", "World languages", "European languages"),
                          ("japanese", "World languages", "European languages")]:
        k_other = keeps[(other, 0.40)]
        v = float(np.mean([len(miss[t][L] & set(int(e) for e in k_other[L])) for L in range(NL)]))
        print(f"  {t:<10} of its missing top-20, {v:.2f}/layer are in the {other} keep-set")

    # ---------------- 4 ----------------
    hr("4. EXPERIMENT (not a proposal): weighted maxmin")
    grid = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    for pname, pair, watch in [
        ("European languages", ["french", "german"],
         ["french", "german", "portuguese", "english", "reasoning_code", "reasoning",
          "translation", "spanish", "italian"]),
        ("World languages", ["chinese", "japanese"],
         ["chinese", "japanese", "russian", "english", "reasoning_code", "reasoning",
          "translation", "arabic", "turkish"])]:
        topics = PROFILES[pname]
        for frac in (0.40, 0.36):
            print(f"\n--- {pname}, weight w on {pair}, saliency coverage @ keep {frac} "
                  f"({keep_n(frac)} experts) ---")
            print(f"{'w':>6}" + "".join(f"{t[:10]:>11}" for t in watch) + f"{'worstlang':>11}")
            for w in grid:
                k = weighted_maxmin(topics, frac, {t: w for t in pair})
                cov = {t: SAL.coverage(t, k) for t in topics}
                wl = min((cov[t] for t in topics
                          if t not in ("reasoning", "reasoning_code", "translation")))
                print(f"{w:>6.2f}" + "".join(f"{cov[t]:>11.3f}" for t in watch)
                      + f"{wl:>11.3f}")

    # ---------------- 5 ----------------
    hr("5. fewer topics")
    for pname, drop, langs_p in [("European languages", ["french", "german"], EU_LANGS),
                                 ("World languages", ["chinese", "japanese"], WL_LANGS)]:
        topics = PROFILES[pname]
        seven = [t for t in topics if t not in drop]
        print(f"\n--- {pname} minus {drop} -> {len(seven)} topics: {seven}")
        for frac in (0.40, 0.36):
            k = keepset(seven, frac)
            print(f"  keep {frac} ({keep_n(frac)}): " +
                  "  ".join(f"{t}={SAL.coverage(t, k):.3f}" for t in seven))
            print(f"      dropped topics still get: " +
                  "  ".join(f"{t}={SAL.coverage(t, k):.3f}" for t in drop))

    print("\n--- splits into two profiles of five ---")
    print("each half = its languages + reasoning + reasoning_code (the two topics every")
    print("profile must carry, tools/tune.py PROFILES comment); 5 topics per half.")
    for pname, langs_p in [("European languages", EU_LANGS), ("World languages", WL_LANGS)]:
        print(f"\n{pname}: languages {langs_p}")
        best = []
        for a in itertools.combinations(langs_p, 3):
            b = tuple(t for t in langs_p if t not in a)
            if a > b:
                continue
            ka = keepset(list(a) + ["reasoning", "reasoning_code"], 0.36)
            kb = keepset(list(b) + ["reasoning", "reasoning_code"], 0.36)
            ca = {t: SAL.coverage(t, ka) for t in a}
            cb = {t: SAL.coverage(t, kb) for t in b}
            worst = min(min(ca.values()), min(cb.values()))
            best.append((worst, a, b, ca, cb))
        best.sort(reverse=True)
        for worst, a, b, ca, cb in best:
            mark = "OK " if worst >= 0.75 else "   "
            print(f" {mark}worst {worst:.3f}  [{', '.join(a)}] "
                  f"({', '.join(f'{v:.3f}' for v in ca.values())})  |  "
                  f"[{', '.join(b)}] ({', '.join(f'{v:.3f}' for v in cb.values())})")

    print("\n--- which language sets clear 0.75 saliency coverage at keep 0.36 ---")
    print("(set + reasoning + reasoning_code; a set is listed if EVERY language in it >= 0.75)")
    for pname, langs_p in [("European languages", EU_LANGS), ("World languages", WL_LANGS)]:
        print(f"\n{pname}:")
        for size in range(1, len(langs_p) + 1):
            ok = []
            for c in itertools.combinations(langs_p, size):
                k = keepset(list(c) + ["reasoning", "reasoning_code"], 0.36)
                cov = {t: SAL.coverage(t, k) for t in c}
                if min(cov.values()) >= 0.75:
                    ok.append((min(cov.values()), c, cov))
            ok.sort(reverse=True)
            if ok:
                print(f"  size {size}: " + " | ".join(
                    f"{'+'.join(c)} (min {m:.3f})" for m, c, _ in ok[:6])
                    + ("" if len(ok) <= 6 else f"  [+{len(ok)-6} more]"))
            else:
                print(f"  size {size}: none")

    # ---------------- extra: single-topic ceilings ----------------
    hr("reference: a language ALONE at these budgets (the ceiling maxmin is dividing)")
    print(f"{'topic':<14}{'alone@.40':>11}{'alone@.36':>11}{'+r+rc@.36':>11}")
    for _, t in langs:
        k40 = keepset([t], 0.40)
        k36 = keepset([t], 0.36)
        k3 = keepset([t, "reasoning", "reasoning_code"], 0.36)
        print(f"{t:<14}{SAL.coverage(t, k40):>11.3f}{SAL.coverage(t, k36):>11.3f}"
              f"{SAL.coverage(t, k3):>11.3f}")


def _run():
    main()
    extra()
    predictive_power()
    controls()
    divergence()
    repetition_abort()
    gate_stats()


# ---------------------------------------------------------------------------
# The sections above answer the six questions as asked. These answer the
# question the numbers then raise: if coverage does not discriminate, what does?
# ---------------------------------------------------------------------------

GATED = {
    "Frontend": ["html", "css", "javascript", "typescript", "english", "technical",
                 "config", "reasoning", "reasoning_code"],
    "Chat and explanation": ["english", "technical", "academic", "journalism",
                             "translation", "reasoning", "reasoning_code"],
    "European languages": PROFILES["European languages"],
    "World languages": PROFILES["World languages"],
}


def extra():
    hr("A. saliency is far more concentrated than counts -- so its coverage saturates")
    print(f"{'topic':<16}{'H_sal':>8}{'H_cnt':>8}{'n50_sal':>9}{'n50_cnt':>9}"
          f"{'n90_sal':>9}{'n90_cnt':>9}")
    for t in ["english", "german", "french", "spanish", "italian", "portuguese",
              "arabic", "chinese", "japanese", "russian", "turkish",
              "reasoning", "reasoning_code", "translation"]:
        print(f"{t:<16}{entropy_bits(t, SAL):>8.2f}{entropy_bits(t, CNT):>8.2f}"
              f"{n_for_mass(t, 0.5, SAL):>9.1f}{n_for_mass(t, 0.5, CNT):>9.1f}"
              f"{n_for_mass(t, 0.9, SAL):>9.1f}{n_for_mass(t, 0.9, CNT):>9.1f}")

    hr("B. per-layer floor: does the mean hide a starved layer?")
    print(f"{'profile':<20}{'topic':<14}{'mean':>7}{'min':>7}{'argmin L':>10}"
          f"{'#L<0.75':>9}{'#L<0.50':>9}   worst 5 layers (L:cov)")
    for pname, topics in PROFILES.items():
        k = keeps[(pname, 0.40)]
        for t in topics:
            per = np.array([SAL.p[t][L][k[L]].sum() for L in range(NL)])
            o = np.argsort(per)[:5]
            print(f"{pname:<20}{t:<14}{per.mean():>7.3f}{per.min():>7.3f}"
                  f"{int(np.argmin(per)):>10}{int((per < 0.75).sum()):>9}"
                  f"{int((per < 0.50).sum()):>9}   "
                  + " ".join(f"{int(L)}:{per[L]:.2f}" for L in o))

    hr("C. what the 2026-09-13 switch to saliency actually changed")
    print("keep-set built from counts (the old recipe) vs from saliency (tonight's),")
    print("same maxmin, same 9 topics, keep 0.40. Jaccard = overlap of the two keep-sets.")
    print(f"\n{'profile':<20}{'topic':<14}{'sal@sal':>9}{'sal@cnt':>9}"
          f"{'cnt@sal':>9}{'cnt@cnt':>9}")
    for pname, topics in PROFILES.items():
        k_sal = keeps[(pname, 0.40)]
        per = {t: {L: CNT.raw[t][L] for L in range(NL)} for t in topics}
        sc = _maxmin_counts(per, 0.40, n_layers=NL, n_experts=N_EXPERTS)
        k_cnt = {L: top_n(sc[L], keep_n(0.40)) for L in range(NL)}
        j = float(np.mean([len(set(k_sal[L].tolist()) & set(k_cnt[L].tolist())) /
                           len(set(k_sal[L].tolist()) | set(k_cnt[L].tolist()))
                           for L in range(NL)]))
        for t in topics:
            print(f"{pname:<20}{t:<14}{SAL.coverage(t, k_sal):>9.3f}"
                  f"{SAL.coverage(t, k_cnt):>9.3f}{CNT.coverage(t, k_sal):>9.3f}"
                  f"{CNT.coverage(t, k_cnt):>9.3f}")
        print(f"{'':<20}{'-> keep-set Jaccard saliency vs counts:':<14} {j:.3f}\n")

    hr("D. the four gated profiles, side by side (keep 0.40, saliency-maxmin)")
    print("p_clean = c**6: the chance all six of a token's routed picks are resident,")
    print("with c the profile's WORST counts coverage (6 picks/token, engine/experts.py")
    print("category_counts docstring: indices [tokens, 6]).")
    print(f"\n{'profile':<22}{'#topics':>8}{'min sal':>9}{'min cnt':>9}{'p_clean':>9}")
    for pname, topics in GATED.items():
        k = keepset(topics, 0.40)
        ms = min(SAL.coverage(t, k) for t in topics)
        mc = min(CNT.coverage(t, k) for t in topics)
        print(f"{pname:<22}{len(topics):>8}{ms:>9.3f}{mc:>9.3f}{mc**6:>9.3f}")


#: Which topic each gated prompt exercises, read off the two GATE.md tables.
#: (prompt, profile, topic, passed)
GATE_PROMPTS = [
    ("en-explain", "European languages", "english", True),
    ("en-note", "European languages", "english", True),
    ("fr-essay", "European languages", "french", False),
    ("de-essay", "European languages", "german", False),
    ("it-essay", "European languages", "italian", False),
    ("pt-essay", "European languages", "portuguese", True),
    ("es-essay", "European languages", "spanish", False),
    ("xl-en-fr", "European languages", "translation", True),
    ("ar-essay", "World languages", "arabic", False),
    ("zh-essay", "World languages", "chinese", False),
    ("en-explain", "World languages", "english", False),
    ("en-note", "World languages", "english", True),
    ("ja-essay", "World languages", "japanese", False),
    ("ru-essay", "World languages", "russian", False),
    ("xl-en-fr", "World languages", "translation", False),
    ("tr-essay", "World languages", "turkish", False),
]


def predictive_power():
    hr("G. does coverage predict the gate? one row per language-bearing prompt")
    print(f"{'prompt':<11}{'profile':<8}{'topic':<13}{'gate':<6}"
          f"{'sal@.40':>9}{'cnt@.40':>9}{'miss_top20':>12}")
    xs, xc, ys = [], [], []
    for prompt, pname, t, ok in GATE_PROMPTS:
        k = keeps[(pname, 0.40)]
        cs, cc = SAL.coverage(t, k), CNT.coverage(t, k)
        mt, _ = missing_from_top_n(t, k, 20)
        xs.append(cs); xc.append(cc); ys.append(1.0 if ok else 0.0)
        print(f"{prompt:<11}{pname[:3]:<8}{t:<13}{'PASS' if ok else 'FAIL':<6}"
              f"{cs:>9.3f}{cc:>9.3f}{mt:>12.2f}")

    def pb(x, y):
        x, y = np.asarray(x), np.asarray(y)
        sx = x.std()
        return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * y.std())) if sx else 0.0

    print(f"\npoint-biserial r(coverage, PASS): saliency {pb(xs, ys):+.3f}   "
          f"counts {pb(xc, ys):+.3f}")
    print(f"mean saliency coverage  PASS {np.mean([a for a, b in zip(xs, ys) if b]):.3f} "
          f"vs FAIL {np.mean([a for a, b in zip(xs, ys) if not b]):.3f}")
    print(f"mean counts   coverage  PASS {np.mean([a for a, b in zip(xc, ys) if b]):.3f} "
          f"vs FAIL {np.mean([a for a, b in zip(xc, ys) if not b]):.3f}")


def controls():
    """The two restarts section 6 proposes, costed in advance."""
    hr("F. the two cheapest restarts, costed on the catalogue")
    print("F1 -- one language alone (+reasoning+reasoning_code) at the SAME keep 0.40:")
    print(f"{'topic':<12}{'sal':>8}{'cnt':>8}{'p_clean':>9}   (3-topic profile)")
    for t in ["french", "german", "chinese", "japanese", "portuguese", "english"]:
        k = keepset([t, "reasoning", "reasoning_code"], 0.40)
        c = CNT.coverage(t, k)
        print(f"{t:<12}{SAL.coverage(t, k):>8.3f}{c:>8.3f}{c**6:>9.3f}")
    print("\nF2 -- the same 9-topic profile at a bigger budget:")
    print(f"{'keep':>6}{'n':>5}   " + "  ".join(f"{p[:3]}:{'sal':>5}/{'cnt':>5}/{'p6':>5}"
                                                for p in PROFILES))
    for frac in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
        cells = []
        for pname, topics in PROFILES.items():
            k = keepset(topics, frac)
            ms = min(SAL.coverage(t, k) for t in topics)
            mc = min(CNT.coverage(t, k) for t in topics)
            cells.append(f"{pname[:3]}:{ms:>5.3f}/{mc:>5.3f}/{mc**6:>5.3f}")
        print(f"{frac:>6.2f}{keep_n(frac):>5}   " + "  ".join(cells))


def divergence():
    """tools/topic_clusters.py's `_js`, on saliency instead of counts."""
    def js(a, b):
        m = 0.5 * (a + b)
        out = 0.0
        for x in (a, b):
            nz = x > 0
            out += 0.5 * float(np.sum(x[nz] * np.log2(x[nz] / m[nz])))
        return out

    def JS(a, b, h=SAL):
        return float(np.mean([js(h.p[a][L], h.p[b][L]) for L in range(NL)]))

    hr("H. how far apart are these topics really? (Jensen-Shannon, bits, saliency)")
    langs = ["english", "german", "french", "spanish", "italian", "portuguese",
             "arabic", "chinese", "japanese", "russian", "turkish"]
    print(f"{'topic':<13}{'->reasoning':>13}{'->reason_code':>15}{'->translation':>15}")
    for t in langs:
        print(f"{t:<13}{JS(t, 'reasoning'):>13.3f}{JS(t, 'reasoning_code'):>15.3f}"
              f"{JS(t, 'translation'):>15.3f}")
    m = [JS(a, b) for i, a in enumerate(langs) for b in langs[i + 1:]]
    print(f"\nmean JS among the 11 language topics: {np.mean(m):.3f} "
          f"(min {min(m):.3f}, max {max(m):.3f})")
    print(f"mean JS language -> reasoning: "
          f"{np.mean([JS(t, 'reasoning') for t in langs]):.3f}; "
          f"-> reasoning_code: {np.mean([JS(t, 'reasoning_code') for t in langs]):.3f}")
    print(f"scale: english-french {JS('english', 'french'):.3f}, "
          f"english-python {JS('english', 'python'):.3f}, "
          f"english-css {JS('english', 'css'):.3f}, "
          f"reasoning-reasoning_code {JS('reasoning', 'reasoning_code'):.3f}")


def repetition_abort():
    """`answer = 139` is a constant, not a language phenomenon."""
    import glob
    import re
    hr("I. the repetition-abort signature across EVERY gate in the repo")
    tot = hits = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "results/keepsets/*/GATE.md"))):
        rows = re.findall(r"\|\s*`([^`]+)`\s*\|\s*(?:on|off)\s*\|\s*(\w+)\s*\|"
                          r"\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|", open(path).read())
        n = sum(1 for _, _, _, a in rows if a.replace(",", "") == "139")
        tot += len(rows)
        hits += n
        if n:
            print(f"  {os.path.relpath(path, ROOT):<48} {n:>3} of {len(rows):>3} rows answer=139")
    print(f"  {'TOTAL':<48} {hits:>3} of {tot:>3}")


def gate_stats():
    """Reasoning-token length of PASS vs FAIL runs, read out of the gate files."""
    import re
    hr("E. the gate files themselves: PASS vs FAIL reasoning length")
    files = {
        "Frontend": "results/keepsets/frontend/GATE.md",
        "Chat and explanation": "results/keepsets/chat_and_explanation/GATE.md",
        "European languages": "results/keepsets/european_languages/GATE.md",
        "World languages": "results/keepsets/world_languages/GATE.md",
    }
    allp, allf = [], []
    for name, rel in files.items():
        text = open(os.path.join(ROOT, rel)).read()
        # last dated section only
        secs = text.split("\n# Generation gate")
        last = secs[-1]
        pa, fa = [], []
        for line in last.splitlines():
            m = re.match(r"\|\s*`([^`]+)`\s*\|\s*(on|off)\s*\|\s*(\w+)\s*\|\s*([\d,]+)\s*\|"
                         r"\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*(\*\*FAIL\*\*|PASS)\s*\|", line)
            if not m:
                continue
            r = int(m.group(4).replace(",", ""))
            (fa if "FAIL" in m.group(7) else pa).append((m.group(1), r, m.group(3)))
        allp += pa
        allf += fa
        mp = np.mean([r for _, r, _ in pa]) if pa else float("nan")
        mf = np.mean([r for _, r, _ in fa]) if fa else float("nan")
        print(f"{name:<22} pass {len(pa):>2} (mean reasoning {mp:>8,.0f})   "
              f"fail {len(fa):>2} (mean reasoning {mf:>8,.0f})")
    print(f"\nall four profiles, last run each: {len(allp)} PASS mean "
          f"{np.mean([r for _, r, _ in allp]):,.0f} reasoning tokens; "
          f"{len(allf)} FAIL mean {np.mean([r for _, r, _ in allf]):,.0f}")
    pr = sorted(r for _, r, _ in allp)
    fr = sorted(r for _, r, _ in allf)
    print(f"PASS  max {pr[-1]:,}   median {pr[len(pr)//2]:,}")
    print(f"FAIL  min {fr[0]:,}   median {fr[len(fr)//2]:,}")
    thr = 10000
    print(f"runs over {thr:,} reasoning tokens: "
          f"{sum(1 for r in pr if r > thr)} PASS / {sum(1 for r in fr if r > thr)} FAIL")
    print(f"runs under {thr:,}: {sum(1 for r in pr if r <= thr)} PASS / "
          f"{sum(1 for r in fr if r <= thr)} FAIL")


if __name__ == "__main__":
    _run()
