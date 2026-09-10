"""Profile where a decode step's time goes, with every expert resident (pruned keep=0.25), so NVMe is out
of the picture and only the engine's compute/launch overhead remains.

    python engine/profile_decode.py --model-dir ... --trace-stats results/trace-full-20260910/stats/coverage.json
"""
import argparse, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from engine.v41_engine import V41Engine, log

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash")
ap.add_argument("--trace-stats", required=True)
ap.add_argument("--prune-keep", type=float, default=0.25)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--out", default="results/profile_decode.txt")
a = ap.parse_args()

eng = V41Engine(a.model_dir, max_seq=8192, trace_stats=a.trace_stats, spec=True, prune_keep=a.prune_keep)
sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
from encoding import encode_messages
prompt = encode_messages([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number, with tests."}], thinking_mode="chat")
ids = eng.tokenizer.encode(prompt if isinstance(prompt, str) else prompt[0], add_special_tokens=False)

# warm up: a few steps of real generation so caches, kernels and the LRU are settled
gen = eng.generate(ids, max_tokens=24, temperature=0.0)
for _ in gen:
    pass
m = eng.model
torch.cuda.synchronize()

# now time individual pieces of one 6-token verify step at the current position
pos = m.c.len
tok = 128799  # any token; timing only
t = {}
def timed(name, fn, n=3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    torch.cuda.synchronize(); t[name] = (time.perf_counter() - t0) / n * 1000
    return r

drafts, q, conf = timed("dspark_draft(5 tok, 3 blocks)", lambda: m.dspark_draft(tok, pos - 1, 0.0))
block = torch.cat([torch.tensor([tok], device=eng.device), drafts])
def fwd():
    m.c.len = pos
    return m.forward(block, pos, prefill=False)
logits, mh = timed("main forward (6 tok, 40 layers)", fwd)
timed("dspark_seed(6 pos)", lambda: m.dspark_seed(mh, pos))
print("\n== wall per call (ms), all experts resident ==")
for k, v in t.items():
    print(f"  {k:36s} {v:8.1f}")
print(f"  per layer (main forward / 40)          {t['main forward (6 tok, 40 layers)'] / 40:8.2f}")
st = m.stats
print("  model.stats:", {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()})

# torch profiler on the main forward
from torch.profiler import profile, ProfilerActivity
m.c.len = pos
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(a.steps):
        m.c.len = pos
        m.forward(block, pos, prefill=False)
    torch.cuda.synchronize()
tab_cuda = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
tab_cpu = prof.key_averages().table(sort_by="cpu_time_total", row_limit=30)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    f.write("wall per call (ms):\n" + json.dumps(t, indent=1) + "\n\n== by CUDA time ==\n" + tab_cuda + "\n\n== by CPU time ==\n" + tab_cpu)
print("\n== top ops by CUDA time (%d steps) ==" % a.steps)
print(tab_cuda)
ev = prof.key_averages()
n_launch = sum(e.count for e in ev if e.device_time_total > 0 and "cuda" not in e.key.lower())
print(f"\nkernel-ish ops per step: ~{n_launch / a.steps:.0f}")
eng.close() if hasattr(eng, "close") else None
