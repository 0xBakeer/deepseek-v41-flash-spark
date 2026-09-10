"""Per-layer / per-tensor divergence diagnostic.

Two modes:
  * `python engine/diag_layers.py ref [seq]`     -- engine vs tools/v41_ref, single chunk
  * `python engine/diag_layers.py chunk [seq] [c1,c2,...]`
        -- single-chunk vs chunked, comparing EVERY tapped tensor (residual stream, window KV
           rows/mask, compressed rows/mask, compressor latents, indexer top-k, engram rows)
           position by position, so the first deviating (layer, tensor, position) is visible.
"""
import glob, json, os, sys, time, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from engine.model import Caches, Model, Weights, Shared
from engine import experts as EX, moe_fallback as K
from engine.engram import make_hash_state
from safetensors import safe_open
import v41_ref as R

md = os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash"; dev = "cuda"
index = json.load(open(f"{md}/model.safetensors.index.json")); args = R.Args.from_json(f"{md}/inference/config.json")
NL = 4
MODE = sys.argv[1] if len(sys.argv) > 1 else "chunk"
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CHUNKS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None

W = Weights(md, index, args, dev, act_quant=True, n_layers=NL, load_mtp=False)
arena = K.ExpertArena(1600, dev); store = EX.ExpertStore(md, index, arena, NL, transient_slots=400, io_threads=8)
caches = Caches(args, 4096, dev); m = Model(W, store, caches, K.moe_forward, act_quant=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(md); m.hash_state = make_hash_state(md, tok, 4096, dev)

def rows_from_store(L, hashes):
    z = np.load(f"engram_rows/layer{L}_rows.npz"); rid, vals, sc = z["row_ids"], z["vals"], z["scales"]
    ids = hashes.reshape(-1).cpu().numpy(); pos = np.searchsorted(rid, ids); assert np.all(rid[pos] == ids)
    v = torch.from_numpy(vals[pos]).to(dev).view(torch.float8_e4m3fn).float(); s = torch.exp2(torch.from_numpy(sc[pos]).to(dev).float() - 127)
    return (v.unflatten(-1, (8, 32)) * s.unsqueeze(-1)).flatten(-2).view(hashes.shape[0], hashes.shape[1], 256)
m.engram_rows = rows_from_store

TRACE = sorted(glob.glob("results/trace-*"))[-1]  # newest results/trace-<name>/
meta = json.load(open(f"{TRACE}/meta.json")); corpus = {json.loads(l)["id"]: json.loads(l)["text"] for l in open("corpus/trace_corpus.jsonl")}
sid = meta["seqs"][SEQ]["id"]
ids = torch.tensor(tok.encode(corpus[sid], add_special_tokens=False), device=dev); T = ids.numel()
print(sid, "T", T, "mode", MODE)


def reset():
    caches.len = 0; caches._chunk_inputs.clear()
    for L in caches.pending: caches.pending[L] = None
    for t in caches.win: t.zero_()
    for L in caches.ckv: caches.ckv[L].zero_(); caches.ik[L].zero_()


def run_taps(ids, chunks):
    """Run `ids` in `chunks` and collect every tapped tensor keyed by (name, L), stored with
    absolute token positions on dim 0 (latents keyed by absolute compressed group index)."""
    reset()
    store_ = {}
    s = 0
    for T_ in chunks:
        cur = {"S": s}
        def tap(name, L, t, cur=cur):
            key = (name, L)
            if name == "latent":
                j0, lat = t
                d = store_.setdefault(key, {})
                for i in range(lat.shape[0]): d[j0 + i] = lat[i]
            elif name == "n_c":
                store_.setdefault(key, {})[cur["S"]] = t
            else:
                d = store_.setdefault(key, {})
                for i in range(t.shape[0]): d[cur["S"] + i] = t[i].clone()
        m.tap = tap
        m.forward(ids[s:s + T_], s, prefill=True, need_logits=False)
        m.tap = None
        s += T_
    return store_


def rel(a, b):
    a, b = a.float(), b.float()
    n = b.norm()
    return float((a - b).norm() / n) if n > 0 else float((a - b).norm())


if MODE == "ref":
    handles = {}
    def get(name):
        f = index["weight_map"][name]
        if f not in handles: handles[f] = safe_open(f"{md}/{f}", "pt", device="cpu")
        return handles[f].get_tensor(name)
    hashes = m.hash_state(ids[None], 0)[0]
    h_ref = W.embed[ids].unsqueeze(1).repeat(1, 4, 1); st = R.SeqState(h_ref, torch.zeros(T, 4, device=dev)); st.pre_mix[:, 0] = 1
    a = args; reset()
    h = W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1); pre = torch.zeros(T, a.hc_mult, device=dev); pre[:, 0] = 1.0; sh = Shared()
    for L in range(NL):
        w = W.layers[L]
        if L in W.engram:
            li = list(a.engram_layer_ids).index(L); rows = rows_from_store(L, hashes[:, li, :])
            st.h = R.engram_forward(st.h, rows, W.engram[L], a); h = R.engram_forward(h, rows, W.engram[L], a)
        R.block_forward(st, w, R.ExpertLoader(get, L, dev), a, {})
        freqs = m.freqs_c if w.ratio else m.freqs_w
        h, pre = m.block(h, pre, w, L, 0, sh, caches.win[L], freqs, True, store, arena, 384)
        print(f"L{L} (ratio {w.ratio}) rel err h {rel(h, st.h):.4f} pre_mix {rel(pre, st.pre_mix):.4f}")
        if os.environ.get("ISOLATE"): h = st.h.clone(); pre = st.pre_mix.clone()
    sys.exit(0)

# ---------------------------------------------------------------- chunked vs single
chunks = CHUNKS or [T // 3, T - T // 3]
print("chunks", chunks)
A = run_taps(ids, [T])
B = run_taps(ids, chunks)
bound = set(np.cumsum(chunks[:-1]).tolist())
print("chunk boundaries at absolute positions", sorted(bound))

names = ["engram_rows", "engram_out", "attn_x", "q", "kv_new", "win_kv", "win_mask", "attn_out",
         "moe_in", "route_w", "moe_routed", "moe_shared", "h", "pre_mix"]
for L in range(NL):
    # latents first (indexed by compressed group)
    key = ("latent", L)
    if key in A:
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        errs = [(j, rel(db[j], da[j])) for j in common]
        nex = sum(1 for j in common if not torch.equal(da[j], db[j]))
        print(f"  L{L} latent: {len(common)} groups, max err {max(e[1] for e in errs):.2e} at j={max(errs, key=lambda e: e[1])[0]}, "
              f"{nex} not bit-equal")
    # compressed visibility as a set of absolute compressed positions per token
    key = ("topk", L)
    if key in A:
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        diff = [p for p in common if set(da[p][da[p] >= 0].tolist()) != set(db[p][db[p] >= 0].tolist())]
        print(f"  L{L} visible-compressed-set: {len(diff)}/{len(common)} tokens differ {diff[:10]}")
    key = ("route_idx", L)
    if key in A:
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        diff = [p for p in common if not torch.equal(da[p].sort().values, db[p].sort().values)]
        print(f"  L{L} router expert set: {len(diff)}/{len(common)} tokens differ {diff[:10]}")
    for nm in names:
        key = (nm, L)
        if key not in A: continue
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        if nm in ("win_mask", "route_idx"):
            diff = [p for p in common if not torch.equal(da[p], db[p])]
            print(f"  L{L} {nm}: {len(diff)}/{len(common)} positions differ {diff[:10]}")
            continue
        errs = np.array([rel(db[p], da[p]) for p in common])
        i = int(errs.argmax())
        nex = sum(1 for p in common if not torch.equal(da[p], db[p]))
        first = [common[k] for k in range(len(common)) if errs[k] > 1e-2][:10]
        print(f"  L{L} {nm}: max {errs.max():.2e} @pos {common[i]}, median {np.median(errs):.2e}, "
              f"{nex}/{len(common)} not bit-equal, first>1e-2 {first}")
