"""Kernel-level profile of one graphed verify step (all experts resident, pruned keep=0.25)."""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from engine.v41_engine import V41Engine, log
md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, trace_stats="results/trace-full-20260910/stats/coverage.json", spec=True, prune_keep=float(os.environ.get("PK", "0.31")), arena_gb=float(os.environ.get("AG", "90.5")), transient_slots=8, keep_free_gb=10)
sys.path.insert(0, os.path.join(md, "encoding")); from encoding import encode_messages
pr = encode_messages([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number, with tests."}], thinking_mode="chat")
ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
for _ in eng.generate(ids, max_tokens=40, temperature=0.0): pass
m, fd = eng.model, eng.fast
pos = m.c.len; tok = 128799
d, q = fd.draft(tok, pos - 1, 0.0)
block = torch.cat([torch.tensor([tok], device="cuda"), d.clone()])
hashes = m.hash_state(block[None], pos)[0]
rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}
def one():
    m.c.len = pos; fd.step(block, pos, rows)
one(); torch.cuda.synchronize()
t = {}
for name, fn in (("step", one), ("draft", lambda: fd.draft(tok, pos - 1, 0.0))):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5): fn()
    torch.cuda.synchronize(); t[name] = (time.perf_counter() - t0) / 5 * 1000
print("wall ms:", {k: round(v, 1) for k, v in t.items()}, "resolve_s/step:", round(fd.stats.get("resolve_s", 0) / max(1, fd.stats["steps"]) * 1000, 1))
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(3): one()
    torch.cuda.synchronize()
tab = prof.key_averages().table(sort_by="cuda_time_total", row_limit=28)
open("results/profile_fast.txt", "w").write(tab); print(tab)
# the same grouped by input shapes: which fp32 GEMMs / einsums are these
tab2 = prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=60, max_shapes_column_width=120)
open("results/profile_fast_shapes.txt", "w").write(tab2)
