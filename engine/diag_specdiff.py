"""Where does a verify block's position-0 prediction differ from a single-token decode?

Speculative decoding is only lossless if the logits the verifier checks are the logits the model
would produce anyway. This walks one real prompt to the point where the two paths disagree and
prints, for the same cache state and the same token: the single-token logits, the un-graphed
6-token block logits, and the graphed fast-path block logits, with their top-5 tokens.
"""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("DSV41_FAST", "0")   # build the engine without a FastDecoder; we make one here
from engine.v41_engine import V41Engine, log
from engine.fastdecode import FastDecoder

md = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
kw = json.loads(os.environ.get("KW", "{}"))
eng = V41Engine(md, max_seq=8192, trace_stats="results/trace-full-20260910/stats/coverage.json", spec=True, **kw)
m = eng.model
sys.path.insert(0, os.path.join(md, "encoding")); from encoding import encode_messages
prompt = os.environ.get("PROMPT", "Write a complete single-file HTML tic-tac-toe game. Output only the HTML.")
pr = encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")
ids = torch.tensor(eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False), device="cuda")
eng._reset()
logits, mh = m.forward(ids, 0, prefill=True)
P = m.c.len
tok = int(logits[-1].argmax())
log(f"prefill len={P}, first token {tok} {eng.tokenizer.decode([tok])!r}")

def snapshot():
    return {L: (None if p is None else (p[0].clone(), p[1].clone())) for L, p in m.c.pending.items()}
def restore(snap):
    for L, p in snap.items(): m.c.pending[L] = p
    m.c._chunk_inputs.clear()

def top5(lg):
    v, i = lg.float().topk(5)
    return [(eng.tokenizer.decode([int(t)]), round(float(s), 3)) for s, t in zip(v, i)]

fd = FastDecoder(m, eng, use_graphs=os.environ.get("GRAPHS", "1") == "1")
if getattr(eng, "store", None) is not None and os.environ.get("LUT", "1") == "1":
    try:
        fd.build_lut()
    except Exception as e:  # noqa: BLE001
        log(f"no LUT: {e}")

pos = P
for stepi in range(int(os.environ.get("STEPS", "3"))):
    snap, n0 = snapshot(), m.c.len
    # 1. single-token decode (what spec-off does)
    lg1, mh1 = m.forward(torch.tensor([tok], device="cuda"), pos, prefill=False)
    lg1 = lg1[0].clone()
    m.c.rollback(n0); restore(snap)
    # 2. drafts from the same state, then the un-graphed block
    drafts, q, conf = m.dspark_draft(tok, pos - 1, 0.0)
    drafts = drafts.clone()
    block = torch.cat([torch.tensor([tok], device="cuda"), drafts])
    lg6, mh6 = m.forward(block, pos, prefill=False)
    lg6_0 = lg6[0].clone(); am6 = lg6.argmax(-1).clone()
    m.c.rollback(n0); restore(snap)
    # 3. the graphed fast path on the same block
    hashes = m.hash_state(block[None], pos)[0]
    rows = {L: eng.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(eng.args.engram_layer_ids)}
    lgF, mhF = fd.step(block, pos, rows)
    lgF_0 = lgF[0].clone(); amF = lgF.argmax(-1).clone()
    m.c.rollback(n0); restore(snap)

    d = lambda t: eng.tokenizer.decode([int(t)])
    log(f"--- step {stepi} at pos {pos}, tok {d(tok)!r}, drafts {[d(x) for x in drafts.tolist()]}")
    log(f"    single-token argmax {d(lg1.argmax())!r}   top5 {top5(lg1)}")
    log(f"    block[0]  argmax    {d(lg6_0.argmax())!r}   top5 {top5(lg6_0)}   rel to single {float((lg6_0-lg1).norm()/lg1.norm()):.4f}")
    log(f"    fast[0]   argmax    {d(lgF_0.argmax())!r}   top5 {top5(lgF_0)}   rel to single {float((lgF_0-lg1).norm()/lg1.norm()):.4f}  rel to block {float((lgF_0-lg6_0).norm()/lg6_0.norm()):.4f}")
    log(f"    block argmax row   {[d(t) for t in am6.tolist()]}")
    log(f"    fast  argmax row   {[d(t) for t in amF.tolist()]}")
    # advance one token with the single-token path (the trusted one)
    lg1b, mh1b = m.forward(torch.tensor([tok], device="cuda"), pos, prefill=False)
    m.dspark_seed(mh1b, pos)
    tok = int(lg1b[0].argmax()); pos += 1
