"""Smoke test on the shards present (layers 0-3).

Four numbers, all on the residual stream after layer 3:
  * single-chunk vs the v41_ref reference   -- absolute correctness (bf16 / fp8-cache deviations)
  * two-chunk vs single-chunk               -- chunked prefill must be the same computation
  * odd-sized chunks vs single-chunk        -- ... for arbitrary boundaries, incl. a 1-token chunk
  * rollback vs a straight run              -- cache rollback after a 6-token verify block

The reference comes from results/trace-*/state/after_layer{NL-1}.pt when tools/expert_trace.py
happens to have left one there; otherwise it is computed once with tools/v41_ref (plain, untiled
GEMMs, exactly what expert_trace.py runs) and cached under results/engine-ref/.
"""
import glob, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from engine.model import Caches, Model, Weights
from engine import experts as EX, moe_fallback as K
from engine.engram import make_hash_state
import numpy as np
import v41_ref as R

md = os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash"; dev = "cuda"
index = json.load(open(f"{md}/model.safetensors.index.json"))
args = R.Args.from_json(f"{md}/inference/config.json")
NL = 4
SEQS = [0, 5, 20]
W = Weights(md, index, args, dev, act_quant=True, n_layers=NL, load_mtp=False)
arena = K.ExpertArena(1600, dev)
store = EX.ExpertStore(md, index, arena, NL, transient_slots=400, io_threads=8)
caches = Caches(args, 4096, dev)
m = Model(W, store, caches, K.moe_forward, act_quant=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(md)
m.hash_state = make_hash_state(md, tok, 4096, dev)
# engram rows from the fetched row store (same files the trace used)
def rows_from_store(L, hashes):
    z = np.load(f"engram_rows/layer{L}_rows.npz"); rid, vals, sc = z["row_ids"], z["vals"], z["scales"]
    ids = hashes.reshape(-1).cpu().numpy(); pos = np.searchsorted(rid, ids); assert np.all(rid[pos] == ids)
    v = torch.from_numpy(vals[pos]).to(dev).view(torch.float8_e4m3fn).float(); s = torch.exp2(torch.from_numpy(sc[pos]).to(dev).float() - 127)
    return (v.unflatten(-1, (8, 32)) * s.unsqueeze(-1)).flatten(-2).view(hashes.shape[0], hashes.shape[1], 256)
m.engram_rows = rows_from_store
TRACE = sorted(glob.glob("results/trace-*"))[-1]  # newest results/trace-<name>/
meta = json.load(open(f"{TRACE}/meta.json"))
corpus = {json.loads(l)["id"]: json.loads(l)["text"] for l in open("corpus/trace_corpus.jsonl")}
enc = {i: torch.tensor(tok.encode(corpus[meta["seqs"][i]["id"]], add_special_tokens=False), device=dev) for i in SEQS}


# ------------------------------------------------------------------ reference (cached)
class _OneSlot(dict):
    """block_forward's expert cache, capped at one entry: 384 dequantized experts do not fit."""
    def __setitem__(self, k, v):
        self.clear(); dict.__setitem__(self, k, v)


def _store_expert(L, e):
    w1, s1, w2, s2, w3, s3 = store.read_expert(L, e)
    return (R.dequant_fp4_packed(w1.view(*EX.W13_SHAPE).to(dev), s1.view(*EX.S13_SHAPE).to(dev)),
            R.dequant_fp4_packed(w2.view(*EX.W2_SHAPE).to(dev), s2.view(*EX.S2_SHAPE).to(dev)),
            R.dequant_fp4_packed(w3.view(*EX.W13_SHAPE).to(dev), s3.view(*EX.S13_SHAPE).to(dev)))


def build_reference():
    """tools/v41_ref, layer by layer, exactly as tools/expert_trace.py runs it."""
    tile, R.MM_TILE = R.MM_TILE, 0  # the reference is the plain, untiled implementation
    try:
        out = {}
        for i in SEQS:
            ids = enc[i]; T = ids.numel()
            hashes = m.hash_state(ids[None], 0)[0]
            h = W.embed[ids].unsqueeze(1).repeat(1, args.hc_mult, 1)
            st = R.SeqState(h, torch.zeros(T, args.hc_mult, device=dev)); st.pre_mix[:, 0] = 1
            for L in range(NL):
                if L in W.engram:
                    li = list(args.engram_layer_ids).index(L)
                    st.h = R.engram_forward(st.h, rows_from_store(L, hashes[:, li, :]), W.engram[L], args)
                R.block_forward(st, W.layers[L], lambda e, _L=L: _store_expert(_L, e), args, _OneSlot())
            out[i] = st.h.cpu()
        return out
    finally:
        R.MM_TILE = tile


REF_CACHE = f"results/engine-ref/after_layer{NL - 1}.pt"
trace_state = f"{TRACE}/state/after_layer{NL - 1}.pt"
if os.path.exists(trace_state):
    ref_h = {i: torch.load(trace_state, map_location="cpu")["states"][i]["h"] for i in SEQS}
elif os.path.exists(REF_CACHE):
    ref_h = torch.load(REF_CACHE, map_location="cpu")
else:
    print(f"no reference state (expert_trace has moved past layer {NL - 1}); computing one ...")
    t0 = time.time(); ref_h = build_reference()
    os.makedirs(os.path.dirname(REF_CACHE), exist_ok=True)
    torch.save(ref_h, REF_CACHE)
    print(f"reference built and cached in {REF_CACHE} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ runs
def run(ids, chunks):
    caches.len = 0; caches._chunk_inputs.clear()
    for L in caches.pending: caches.pending[L] = None
    s = 0
    for T in chunks:
        m.forward(ids[s:s + T], s, prefill=True, need_logits=False); s += T
    return m.last_h.clone(), m.last_pre_mix.clone()


def rel(a, b):
    b = b.to(a.device)[-a.shape[0]:]  # forward returns the LAST chunk's rows; compare against the tail
    return float((a.float() - b.float()).norm() / b.float().norm())


worst = 0.0
fails = []


def chunkings(T):
    """A spread of splittings: even, odd, 1-token chunks, many chunks, a long tail."""
    out = [[T // 3, T - T // 3], [T // 2, T - T // 2], [1, T - 1], [T - 1, 1]]
    if T > 20:
        out += [[7, 1, 6, T - 14], [1, 1, 1, T - 3], [T - 7, 1, 6], [5, 6, 6, 6, T - 23]]
    return out


for i in SEQS:
    sid = meta["seqs"][i]["id"]; ids = enc[i]; T = ids.numel()
    t0 = time.time(); h1, _ = run(ids, [T]); dt = time.time() - t0
    errs = []
    for ch in chunkings(T):
        e = rel(run(ids, ch)[0], h1)
        errs.append((ch, e))
        worst = max(worst, e)
        if e >= 0.01:
            fails.append((sid, ch, e))
    two = dict((tuple(c), e) for c, e in errs)
    print(f"{sid}: T={T} single-chunk vs ref rel err {rel(h1, ref_h[i]):.4f}  "
          f"two-chunk vs single {two[(T // 3, T - T // 3)]:.4f}  "
          f"odd-chunks vs single {max(e for c, e in errs if len(c) > 2) if T > 20 else two[(1, T - 1)]:.4f}  "
          f"worst of {len(errs)} splittings {max(e for _, e in errs):.4f}  ({dt:.1f}s)")

# rollback: forward a prefix, then a 6-token "verify block", roll back to an accepted prefix,
# forward the rest -- must equal the straight run. Even and odd rollback points.
ids = enc[SEQS[0]]; T = ids.numel()
h_ref, _ = run(ids, [T])
roll_errs = []
for pre, blk, keep in ((5, 6, 8), (5, 6, 9), (7, 6, 11), (4, 6, 10), (9, 6, 9)):
    caches.len = 0; caches._chunk_inputs.clear()
    for L in caches.pending: caches.pending[L] = None
    m.forward(ids[:pre], 0, prefill=True, need_logits=False)
    m.forward(ids[pre:pre + blk], pre, prefill=False, need_logits=False)
    caches.rollback(keep)
    m.forward(ids[keep:T], keep, prefill=True, need_logits=False)
    e = rel(m.last_h, h_ref)
    roll_errs.append((pre, blk, keep, e))
    worst = max(worst, e)
    if e >= 0.01:
        fails.append(("rollback", (pre, blk, keep), e))
print("rollback test rel err " + "  ".join(f"[{p}+{b}->{k}] {e:.4f}" for p, b, k, e in roll_errs))
print(f"WORST chunk-invariance error {worst:.4f}  ({'OK' if worst < 0.01 else 'FAIL'}, target < 0.01)")
for f in fails:
    print("  FAIL", f)
