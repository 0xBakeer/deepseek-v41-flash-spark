"""Compare FastDecoder (graphed) against Model.forward on identical state, then time it."""
import os, sys, time, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("DSV41_FAST", "0")
from engine.v41_engine import V41Engine, log
from engine.fastdecode import FastDecoder
md = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
keep = float(os.environ.get("KEEP", "0.25"))
eng = V41Engine(md, max_seq=8192, trace_stats="results/trace-full-20260910/stats/coverage.json", spec=True, prune_keep=keep)
m = eng.model
sys.path.insert(0, os.path.join(md, "encoding")); from encoding import encode_messages
pr = encode_messages([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number, with tests."}], thinking_mode="chat")
ids = torch.tensor(eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False), device="cuda")
# prefill with the existing path
eng._reset(); m.begin_prompt() if hasattr(m, "begin_prompt") else None
logits, mh = m.forward(ids, 0, prefill=True)
if hasattr(m, "decoder_replay"):
    pass
P = m.c.len; log(f"prefill done, len={P}")
tok = int(logits[-1].argmax())
def snapshot():
    return {L: (None if p is None else (p[0].clone(), p[1].clone())) for L, p in m.c.pending.items()}
def restore(snap):
    for L, p in snap.items(): m.c.pending[L] = p
    m.c._chunk_inputs.clear()
for parity_test in (0, 1):
    pos = P + parity_test
    if parity_test == 1:
        # advance one position with the reference path so the start parity flips
        lg, mh1 = m.forward(torch.tensor([tok], device="cuda"), P, prefill=False); tok1 = int(lg[0].argmax())
        m.dspark_seed(mh1, P)
        tok_use = tok1
    else:
        tok_use = tok
    snap = snapshot(); n0 = m.c.len
    drafts, q, conf = m.dspark_draft(tok_use, pos - 1, 0.0)
    block = torch.cat([torch.tensor([tok_use], device="cuda"), drafts])
    ref_logits, ref_mh = m.forward(block, pos, prefill=False); ref_logits = ref_logits.clone(); ref_mh = ref_mh.clone()
    m.c.rollback(n0); restore(snap)
    if parity_test == 0 and os.environ.get("DIAG", "1") == "1":
        ref = {}
        m.tap = lambda name, L, t: ref.__setitem__((name, L), (t.clone() if torch.is_tensor(t) else t))
        m.forward(block, pos, prefill=False); m.c.rollback(n0); restore(snap); m.tap = None
        fde = FastDecoder(m, eng, use_graphs=False); mine = {}
        fde.tap = lambda name, L, t: mine.__setitem__((name, L), t.clone())
        fde.step(block, pos, rows) if False else None
        hashes_d = m.hash_state(block[None], pos)[0]
        rows_d = {L: eng.tables[L].rows(hashes_d[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}
        fde.step(block, pos, rows_d); m.c.rollback(n0); restore(snap)
        def rel_(x, y): return float((x.float() - y.float()).norm() / (y.float().norm() + 1e-9))
        for L in range(40):
            parts = []
            for nm in ("h_in", "attn_x", "attn_out", "moe_in"):
                if (nm, L) in ref and (nm, L) in mine: parts.append(f"{nm} {rel_(mine[(nm, L)], ref[(nm, L)]):.4f}")
            if ("route_idx", L) in ref and ("route_idx", L) in mine:
                parts.append("route eq %.2f" % float((mine[("route_idx", L)] == ref[("route_idx", L)]).float().mean()))
            if ("topk", L) in ref and ("topk", L) in mine:
                parts.append("topk eq %.2f" % float((mine[("topk", L)] == ref[("topk", L)]).float().mean()))
            print(f"  L{L:2d}: " + "  ".join(parts))
        del fde
    fd = FastDecoder(m, eng, use_graphs=True)
    d2, q2 = fd.draft(tok_use, pos - 1, 0.0)
    hashes = m.hash_state(block[None], pos)[0]
    rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}
    lg2, mh2 = fd.step(block, pos, rows)
    same_draft = bool((d2 == drafts).all())
    top1 = float((lg2.argmax(-1) == ref_logits.argmax(-1)).float().mean())
    rel = float((lg2 - ref_logits).norm() / ref_logits.norm())
    relh = float((mh2 - ref_mh).norm() / ref_mh.norm())
    print(f"parity {parity_test}: drafts equal={same_draft}  logits rel err {rel:.4f}  argmax agree {top1:.2f}  main_hidden rel {relh:.4f}")
    # timing: repeat the step at the same position
    torch.cuda.synchronize(); t0 = time.perf_counter(); n = 5
    for _ in range(n):
        m.c.rollback(n0); restore(snap)
        fd.step(block, pos, rows)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fd.draft(tok_use, pos - 1, 0.0)
    torch.cuda.synchronize(); dd = (time.perf_counter() - t0) / n
    print(f"   fast step {dt*1000:.1f} ms  draft {dd*1000:.1f} ms  -> {(1+2.2)/(dt+dd):.1f} tok/s at accept 2.2, {(1+3.0)/(dt+dd):.1f} at 3.0")
    m.c.rollback(n0); restore(snap)
    # leave state advanced by the reference path for the next parity test
    m.forward(block, pos, prefill=False); m.c.rollback(pos + 1); m.dspark_seed(ref_mh[:1], pos)
    tok = int(ref_logits[0].argmax()); P = pos + 1
    del fd; torch.cuda.empty_cache()
