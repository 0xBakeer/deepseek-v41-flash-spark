"""test_atlas_export.py -- the Weight Atlas export is the keep-sets, not a picture of them.

`tools/atlas_export.py` writes ~700k numbers into a file a web page reads, and a web page
will happily draw a wrong grid without complaining. Everything checkable about that file is
checked here against the module the engine budgets with: the ten outlines must be the same
expert ids `tools/budget.py` hands the engine at the same keep fraction, every matrix must be
40 x 384, and the routing shares must be shares.

No torch, no GPU, no checkpoint: this reads `results/keepsets/` and writes to a temporary
directory. It takes a couple of seconds, which is one export.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_export as AX  # noqa: E402
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail and not ok else ''}")
    if not ok:
        fails.append(name)


stats = AX.STATS_DEFAULT
gates = AX.GATES_DEFAULT
if not (os.path.exists(stats) and os.path.exists(gates)):
    print(f"no keep-set to export: {stats}")
    sys.exit(0)

out = tempfile.mkdtemp(prefix="atlas-export-")
try:
    r = AX.export(stats=stats, gates=gates, out=out)
    payload = r["payload"]
    R = payload["routing"]
    meta = payload["meta"]

    N, E = B.N_LAYERS, B.N_EXPERTS
    check("the grid is the model", R["experts"] == E and R["layers"] == list(range(N)),
          f"{R['experts']} experts, {len(R['layers'])} layers")

    # --- every matrix is one row per layer and one column per expert ---------
    mats = {"reap": R["reap"], "route_share": R["route_share"], "contribution": R["contribution"]}
    for name, rows in R["reap_domains"].items():
        mats[f"reap_domains[{name}]"] = rows
    bad = [k for k, m in mats.items() if len(m) != N or any(len(row) != E for row in m)]
    check(f"all {len(mats)} matrices are {N} x {E}", not bad, ", ".join(bad[:3]))

    # --- a share is a share --------------------------------------------------
    off = [(L, s) for L, row in enumerate(R["route_share"])
           for s in [sum(row)] if abs(s - 1.0) > 2e-3]
    check("every route_share row sums to 1", not off, str(off[:3]))
    check("no negative route share",
          not any(v < 0 for row in R["route_share"] for v in row))

    # --- every topic in the trace is a slice on the page ---------------------
    idx = B.TopicIndex(stats, source="saliency")
    slices = [d for d in R["domains"] if not d.startswith("profile/")]
    missing = [t for t in idx.topics if t not in R["reap_domains"]]
    check(f"every one of the {len(idx.topics)} topic slices is there", not missing,
          ", ".join(missing[:5]))
    check("  and the domain list names them all", sorted(slices) == sorted(idx.topics),
          f"{len(slices)} slices against {len(idx.topics)} topics")
    check("  and each profile is a domain of its own",
          all(f"profile/{p['record']}" in R["reap_domains"] for p in meta["profiles"]))

    # --- the outlines are the engine's own keep-sets -------------------------
    # This is the whole point of the exporter: the ten sets drawn on the grid have to be the
    # experts the engine would actually hold, at the keep fraction each profile's generation
    # gate was measured at -- not a re-ranking that happens to look similar.
    check(f"{len(meta['profiles'])} profiles carry a gate record",
          len(meta["profiles"]) == len(R["prune_sets"]) == 10,
          f"{len(meta['profiles'])} profiles, {len(R['prune_sets'])} sets")
    wrong = []
    for pr in meta["profiles"]:
        sel = tuple(sorted(pr["topics"]))
        order = idx.curves(sel, only=sel, rank=pr["rank"])[1]
        n = B.keep_n(pr["keep"])
        got = R["prune_sets"][pr["prune_set_key"]]
        if n != pr["experts_per_layer"]:
            wrong.append(f"{pr['name']}: keep_n {n} != {pr['experts_per_layer']}")
            continue
        for L in range(N):
            if got[L] != sorted(order[L][:n]):
                wrong.append(f"{pr['name']} layer {L}")
                break
    check("every outline is TopicIndex.curves(...)[1][L][:keep_n(keep)]", not wrong,
          "; ".join(wrong[:3]))
    check("  and holds keep_n(keep) experts in every layer",
          all(len(row) == B.keep_n(pr["keep"])
              for pr in meta["profiles"] for row in R["prune_sets"][pr["prune_set_key"]]))

    # --- the manifest opens the page on this model, and on nothing else ------
    man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
    check("the manifest holds exactly one model", isinstance(man, list) and len(man) == 1,
          f"{len(man) if isinstance(man, list) else type(man).__name__}")
    if isinstance(man, list) and man:
        e = man[0]
        check("  and it is ours", e["slug"] == AX.SLUG and e["kind"] == "glm-live",
              f"{e.get('slug')} / {e.get('kind')}")
        shell = json.dumps(e["shell"], ensure_ascii=False)
        check("  and it carries its own shell text, not the page's",
              "288" not in shell and f"{E} → {B.TOPK}" in shell, shell[:120])

    # --- the weight inventory is honest about being unscanned ----------------
    with open(os.path.join(out, AX.SLUG, "atlas.jsonl"), encoding="utf-8") as f:
        inv = [json.loads(ln) for ln in f if ln.strip()]
    check("the weight inventory is one line per tensor", len(inv) == meta["tensor_count"],
          f"{len(inv)} != {meta['tensor_count']}")
    check("  with every measured statistic left at 0",
          all(t["absmax"] == 0 and t["dtype"] == "unscanned" for t in inv))

    # --- freshness -----------------------------------------------------------
    check("a fresh export is not stale", not AX.is_stale(out, stats, gates))
    os.utime(os.path.join(out, AX.SLUG, "insights.json"), (0, 0))
    check("  an export older than the trace is", AX.is_stale(out, stats, gates))
finally:
    shutil.rmtree(out, ignore_errors=True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
