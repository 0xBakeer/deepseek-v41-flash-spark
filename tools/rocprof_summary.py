"""Summarise a rocprofv3 --kernel-trace --marker-trace CSV directory: per-kernel time inside the STEPS
marker that engine/profile_rocprof.py sets (5 graphed verify steps). Usage: rocprof_summary.py <dir>"""
import csv, glob, collections, sys
d = sys.argv[1]
kt = glob.glob(f"{d}/**/*kernel_trace.csv", recursive=True)[0]
mk = glob.glob(f"{d}/**/*marker_api_trace.csv", recursive=True)
rows = list(csv.DictReader(open(kt)))
lo, hi = None, None
for f in mk:
    for r in csv.DictReader(open(f)):
        if "STEPS" in r.get("Function", "") + r.get("Operation", "") + str(r):
            lo, hi = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
print("window ms:", (hi - lo) / 1e6 if lo else None)
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
    if lo and not (lo <= s <= hi): continue
    k = (r["Kernel_Name"][:90], r.get("Grid_Size_X", r.get("Grid_Size", "")), r.get("Workgroup_Size_X", r.get("Workgroup_Size", "")))
    agg[k][0] += 1; agg[k][1] += e - s
tot = sum(v[1] for v in agg.values())
print(f"kernel time in window: {tot/1e6:.1f} ms over 5 steps = {tot/5e6:.1f} ms/step")
for (n, g, w), (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:30]:
    print(f"{t/5e6:7.2f} ms/step  n/step={c/5:6.1f}  avg={t/c/1e3:8.1f}us  grid={g:>8} wg={w:>4}  {n}")
