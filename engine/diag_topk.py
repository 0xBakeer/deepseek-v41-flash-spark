"""How many DISTINCT routed experts one verify block touches, per backbone layer.

The routed experts are read once per distinct (layer, expert) pair of a verify block, so this
count -- not the k*T routed pairs -- is what sets the expert bytes a decode step has to move.
Run with DSV41_ROUTE_STATS=1 and, optionally, DSV41_TOPK=<k>:

    DSV41_ROUTE_STATS=1 DSV41_TOPK=5 python engine/diag_topk.py

Env: PK / AG / TRANSIENT_SLOTS / KEEP_FREE_GB / EXPERT_FORMAT as in engine/profile_fast.py,
TOKENS (default 60) is the length of the measured generation (a first, unmeasured generation
captures the graphs -- capture's own warm-up runs would otherwise be counted). TF=<corpus.jsonl>
additionally scores that corpus teacher-forced in the same process, so the routing count and the
loss of one top-k setting come from one weight load.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine.v41_engine import V41Engine  # noqa: E402

md = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
eng = V41Engine(md, max_seq=8192, trace_stats="results/trace-full-20260910/stats/coverage.json", spec=True,
                prune_keep=float(os.environ.get("PK", "0.40")), arena_gb=float(os.environ.get("AG", "90.5")),
                transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "8")),
                keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "10")),
                expert_format=os.environ.get("EXPERT_FORMAT", "cb3"))
cfg = eng.config()
print("config:", {k: cfg[k] for k in ("expert_format", "expert_mb", "prune_keep", "arena_slots",
                                      "dense_fp4", "head_fmt", "routed_topk")})

sys.path.insert(0, os.path.join(md, "encoding"))
from encoding import encode_messages  # noqa: E402

PROMPT = ("Write a complete, production-quality Python module implementing an LRU cache with a TTL "
          "per entry, thread safety, and an eviction callback. Include docstrings and type hints.")
pr = encode_messages([{"role": "user", "content": PROMPT}], thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)

for _ in eng.generate(ids, max_tokens=24, temperature=0.0):
    pass                                   # captures the graphs; not measured
fd = eng.fast
if fd.rs_uniq is None:
    raise SystemExit("set DSV41_ROUTE_STATS=1")
fd.route_stats_reset()
n = int(os.environ.get("TOKENS", "60"))
for _ in eng.generate(ids, max_tokens=n, temperature=0.0):
    pass
rep = fd.route_stats_report()

a = eng.args
T = 6                                      # the verify block: 1 accepted token + 5 DSpark drafts
mb = eng.expert_bytes / 1e6
rep["routed_topk"] = a.n_activated_experts
rep["block_tokens"] = T
rep["routed_pairs_per_layer"] = T * a.n_activated_experts
rep["expert_mb"] = round(mb, 2)
rep["expert_mb_per_step"] = round(rep["total"] * mb, 1)
print(json.dumps({k: rep[k] for k in ("routed_topk", "steps", "block_tokens", "routed_pairs_per_layer",
                                      "mean", "total", "expert_mb", "expert_mb_per_step")}, indent=2))
print("per layer:", " ".join(f"{v:.2f}" for v in rep["per_layer"]))
tf = os.environ.get("TF")
if tf:
    res = eng.teacher_forced(tf, max_len=512)
    rep["teacher_forced"] = res["summary"]
    print("teacher-forced:", json.dumps({k: {"mean_nll": round(v["mean_nll"], 4),
                                             "top1_acc": round(v["top1_acc"], 4), "n": v["n"]}
                                         for k, v in res["summary"].items()}))
out = os.path.join("results", f"route_stats_topk{a.n_activated_experts}.json")
os.makedirs("results", exist_ok=True)
json.dump(rep, open(out, "w"), indent=1)
print("wrote", out)
