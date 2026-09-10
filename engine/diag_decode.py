"""Decode-vs-prefill consistency: logits after prefill(N) + K single-token decode steps must equal
the last logits of prefill(N+K) (same tokens). Per-layer taps localise the first divergence."""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
os.environ["DSV41_FAST"] = "0"
from engine.v41_engine import V41Engine, log
md = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, trace_stats="results/trace-full-20260910/stats/coverage.json", spec=False)
m = eng.model
sys.path.insert(0, os.path.join(md, "encoding")); from encoding import encode_messages
text = '```python\n"""\nlru_ttl_cache.py\n\nA thread-safe LRU (Least Recently Used) cache with per-entry time-to-live\n(TTL) expiration and an optional eviction callback.\n\nFeatures\n--------\n- LRU eviction when the cache exceeds a maximum size.\n'
pr = encode_messages([{"role": "user", "content": "Write a Python LRU cache module."}], thinking_mode="chat")
ids = torch.tensor(eng.tokenizer.encode((pr if isinstance(pr, str) else pr[0]) + text, add_special_tokens=False), device="cuda")
N = ids.numel(); K = int(os.environ.get("K", "8")); n0 = N - K
print("tokens", N, "prefill", n0, "decode steps", K)
def fresh():
    eng._reset()
    if hasattr(m, "begin_prompt"): m.begin_prompt()
# A: full single chunk, tapping the last position
tapsA = {}
m.tap = lambda name, L, t: tapsA.__setitem__((name, L), t[-1].clone() if torch.is_tensor(t) and t.dim() >= 1 and t.size(0) == N else None)
fresh(); lgA, _ = m.forward(ids, 0, prefill=True); lgA = lgA[-1].clone(); m.tap = None
# B: prefill n0 then K decode steps, tapping the last step
fresh(); m.forward(ids[:n0], 0, prefill=True)
tapsB = {}
for k in range(K):
    if k == K - 1:
        m.tap = lambda name, L, t: tapsB.__setitem__((name, L), t[-1].clone() if torch.is_tensor(t) and t.dim() >= 1 else None)
    lgB, _ = m.forward(ids[n0 + k:n0 + k + 1], n0 + k, prefill=False)
m.tap = None; lgB = lgB[-1]
def rel(a, b): return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-9))
print(f"last-position logits: rel err {rel(lgB, lgA):.5f}  argmax A={int(lgA.argmax())} B={int(lgB.argmax())}  top5 A={lgA.topk(5).indices.tolist()} B={lgB.topk(5).indices.tolist()}")
for L in range(40):
    parts = []
    for nm in ("attn_x", "attn_out", "moe_in"):
        a_, b_ = tapsA.get((nm, L)), tapsB.get((nm, L))
        if a_ is not None and b_ is not None: parts.append(f"{nm} {rel(b_, a_):.4f}")
    a_, b_ = tapsA.get(("route_idx", L)), tapsB.get(("route_idx", L))
    if a_ is not None and b_ is not None: parts.append("route eq %.2f" % float((a_ == b_).float().mean()))
    a_, b_ = tapsA.get(("topk", L)), tapsB.get(("topk", L))
    if a_ is not None and b_ is not None:
        sa, sb = set(a_[a_ >= 0].tolist()), set(b_[b_ >= 0].tolist()); parts.append(f"topk |A|={len(sa)} |B|={len(sb)} common={len(sa & sb)}")
    print(f"  L{L:2d}: " + "  ".join(parts))
