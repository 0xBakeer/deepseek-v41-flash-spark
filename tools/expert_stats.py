#!/usr/bin/env python3
"""
expert_stats.py -- turn the per-layer router traces from expert_trace.py into the numbers
that decide the expert strategy (Phase 1):

  * per-layer expert usage histogram (share of routed slots per expert)
  * per-layer and global cumulative coverage curves: fraction of routed slots covered by
    the top-N% (or top-K) experts; global = best allocation across layers, i.e. experts
    ranked by frequency over all layers
  * block-level unique experts: DSpark verifies blocks of up to 6 tokens (1 + 5 drafts)
    at once, so what matters for a streaming cache is the union of experts touched by
    consecutive tokens, per layer
  * cache simulation: LRU over (layer, expert) with a given number of resident experts
    (FP4, 18.8 MB each), hit rate per token and per 6-token block, split by category
  * memory projection for strategies A (hot cache + stream) and C (hot FP4 + cold low-bit)

Writes results/<name>/{coverage.md, coverage.json, coverage.png, layer_hist.png}.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

EXPERT_BYTES_FP4 = 3 * (5_898_240 + 368_640)  # w1/w2/w3 packed FP4 + UE8M0 scales, from the safetensors header
N_EXP = 384


def load(trace_dir: str):
    layers = {}
    for p in sorted(glob.glob(os.path.join(trace_dir, "trace", "layer*.npz")),
                    key=lambda p: int(re.search(r"layer(\d+)", p).group(1))):
        L = int(re.search(r"layer(\d+)", p).group(1))
        z = np.load(p)
        layers[L] = {"idx": z["indices"].astype(np.int64), "w": z["weights"].astype(np.float32), "cat": z["category"]}
    meta = json.load(open(os.path.join(trace_dir, "meta.json")))
    return layers, meta


def coverage_curve(counts: np.ndarray):
    c = np.sort(counts)[::-1]
    return np.cumsum(c) / max(c.sum(), 1)


def lru_sim(layers: dict, budget: int, block: int = 6, order: list[int] | None = None):
    """Token-by-token LRU over (layer, expert) keys with `budget` resident experts.
    Returns per-token hit rate and per-block (union of `block` consecutive tokens) hit rate."""
    from collections import OrderedDict
    Ls = sorted(layers)
    n_tok = layers[Ls[0]]["idx"].shape[0]

    def run(step: int):
        """One LRU simulation where the unit of work is `step` consecutive tokens (1 = per token,
        6 = a DSpark verify block). Hit = expert already resident when the unit needs it."""
        cache: OrderedDict = OrderedDict()
        hits = misses = 0
        for t0 in range(0, n_tok, step):
            t1 = min(n_tok, t0 + step)
            for L in Ls:
                for e in np.unique(layers[L]["idx"][t0:t1]):
                    k = (L, int(e))
                    if k in cache:
                        cache.move_to_end(k)
                        hits += 1
                    else:
                        misses += 1
                        cache[k] = 1
                        if len(cache) > budget:
                            cache.popitem(last=False)
        return hits / (hits + misses)

    return run(1), run(block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="1000,1500,2000,3000,4000,5000,6000,8000")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layers, meta = load(a.trace)
    Ls = sorted(layers)
    budgets = [int(x) for x in a.budgets.split(",")]
    n_tok = layers[Ls[0]]["idx"].shape[0]
    cats = sorted(set(layers[Ls[0]]["cat"].tolist()))

    per_layer = {}
    glob_counts = {}
    for L in Ls:
        idx = layers[L]["idx"]
        counts = np.bincount(idx.reshape(-1), minlength=N_EXP)
        cov = coverage_curve(counts)
        used = int((counts > 0).sum())
        per_layer[L] = {"used": used, "counts": counts, "cov": cov,
                        "top50": float(cov[N_EXP // 2 - 1]), "top25": float(cov[N_EXP // 4 - 1]),
                        "top10": float(cov[int(N_EXP * 0.10) - 1]),
                        "entropy_bits": float(-(counts / counts.sum() * np.log2(np.where(counts > 0, counts / counts.sum(), 1))).sum())}
        # block uniqueness (union over 6 consecutive tokens)
        bl = [len(np.unique(idx[t:t + 6])) for t in range(0, n_tok - 5, 6)]
        per_layer[L]["block6_unique_mean"] = float(np.mean(bl))
        for c in cats:
            m = layers[L]["cat"] == c
            per_layer[L][f"cov_{c}"] = coverage_curve(np.bincount(idx[m].reshape(-1), minlength=N_EXP))
        for e in range(N_EXP):
            glob_counts[(L, e)] = int(counts[e])

    # global coverage by budget (rank all (layer, expert) by frequency)
    gc = np.array(sorted(glob_counts.values(), reverse=True), dtype=np.float64)
    gcov = np.cumsum(gc) / gc.sum()
    n_keys = len(gc)
    # category overlap: experts in the top-K set of coding vs general
    lines = []
    lines.append(f"# Expert coverage -- {meta['n_tokens']} tokens, {meta['n_seqs']} sequences, layers {Ls[0]}-{Ls[-1]}\n")
    lines.append("## Per layer\n")
    lines.append("| layer | experts used | top-10% covers | top-25% covers | top-50% covers | entropy (bits, max 8.58) | unique experts / 6-token block (max 36) |")
    lines.append("|---|---|---|---|---|---|---|")
    for L in Ls:
        p = per_layer[L]
        lines.append(f"| {L} | {p['used']} | {p['top10']:.3f} | {p['top25']:.3f} | {p['top50']:.3f} | {p['entropy_bits']:.2f} | {p['block6_unique_mean']:.1f} |")
    lines.append("")
    lines.append(f"## Global coverage vs resident-expert budget (layers traced: {len(Ls)} of 40, {n_keys} (layer,expert) keys)\n")
    lines.append("Static resident set = the most frequent (layer, expert) pairs overall. 'covers' = share of routed slots that hit the resident set. Memory = FP4 experts only (18.8 MB each).\n")
    lines.append("| budget (experts) | share of all keys | resident GB (FP4) | static coverage | LRU hit/token | LRU hit/6-token block |")
    lines.append("|---|---|---|---|---|---|")
    res = {"per_layer": {}, "global": []}
    for b in budgets:
        b_eff = min(b, n_keys)
        cov = float(gcov[b_eff - 1])
        hr_tok, hr_blk = lru_sim(layers, b_eff)
        gb = b * EXPERT_BYTES_FP4 / 1e9
        lines.append(f"| {b} | {b_eff / n_keys:.2f} | {gb:.1f} | {cov:.3f} | {hr_tok:.3f} | {hr_blk:.3f} |")
        res["global"].append({"budget": b, "static_coverage": cov, "lru_hit_token": hr_tok, "lru_hit_block6": hr_blk, "resident_gb_fp4": gb})
    lines.append("")
    if len(cats) == 2:
        lines.append("## Coding vs general: overlap of the per-layer top-25% sets\n")
        lines.append("| layer | Jaccard(top25 coding, top25 general) | coding slots covered by general's top25 |")
        lines.append("|---|---|---|")
        for L in Ls:
            idx, cat = layers[L]["idx"], layers[L]["cat"]
            sets = {}
            for c in cats:
                cnt = np.bincount(idx[cat == c].reshape(-1), minlength=N_EXP)
                sets[c] = set(np.argsort(cnt)[::-1][: N_EXP // 4].tolist())
            j = len(sets[cats[0]] & sets[cats[1]]) / len(sets[cats[0]] | sets[cats[1]])
            cod = idx[cat == "coding"].reshape(-1)
            cross = float(np.isin(cod, list(sets["general"])).mean()) if cod.size else float("nan")
            lines.append(f"| {L} | {j:.2f} | {cross:.3f} |")
        lines.append("")
    for L in Ls:
        res["per_layer"][L] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in per_layer[L].items()
                               if k not in ("counts",)}
        res["per_layer"][L]["counts"] = per_layer[L]["counts"].tolist()
    json.dump(res, open(os.path.join(a.out, "coverage.json"), "w"))
    open(os.path.join(a.out, "coverage.md"), "w").write("\n".join(lines))
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(1, N_EXP + 1) / N_EXP * 100
        for L in Ls:
            ax[0].plot(x, per_layer[L]["cov"], lw=0.8, alpha=0.7, label=f"L{L}" if len(Ls) <= 8 else None)
        ax[0].set_xlabel("top N% of experts in the layer"); ax[0].set_ylabel("share of routed slots covered")
        ax[0].set_title("per-layer cumulative coverage"); ax[0].grid(alpha=0.3)
        if len(Ls) <= 8: ax[0].legend()
        ax[1].plot(np.arange(1, n_keys + 1), gcov)
        for b in budgets:
            if b <= n_keys: ax[1].axvline(b, color="orange", lw=0.6, alpha=0.6)
        ax[1].set_xlabel("resident (layer, expert) budget"); ax[1].set_ylabel("static coverage")
        ax[1].set_title(f"global coverage, {len(Ls)} layers traced"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "coverage.png"), dpi=130)
        fig, ax = plt.subplots(figsize=(13, 4))
        for L in Ls:
            ax.plot(np.sort(per_layer[L]["counts"])[::-1] / per_layer[L]["counts"].sum(), lw=0.8, alpha=0.7)
        ax.set_yscale("log"); ax.set_xlabel("expert rank within layer"); ax.set_ylabel("share of slots")
        ax.set_title("expert usage, sorted, per layer (log)"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "layer_hist.png"), dpi=130)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
