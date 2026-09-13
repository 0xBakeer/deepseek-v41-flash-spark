#!/usr/bin/env python3
"""
expert_trace.py -- record which routed experts DeepSeek-V4.1-Flash picks, per layer,
over a teacher-forced corpus, by streaming the model one layer shard at a time.

Why layer streaming: the model is 510 GB and this box has 128 GB. Every sequence is
pushed through layer L (one 7.4 GB safetensors shard, mmap'd) before layer L+1 is
touched, so at most one layer's weights are live. The residual stream of all sequences
(T x 4 x 5120 bf16 per token, ~40 KB) plus the shared compressed KV is all that is
carried across layers. State is checkpointed after every layer, so the trace can be
resumed as more shards are downloaded (`--resume`).

The forward pass is the pure-torch port in v41_ref.py (exact for sequences <= 512
tokens; see its docstring). Engram rows for layers 1 and 14 come from
tools/engram_rows.py (rows fetched on demand, the two 101 GB tables are never
downloaded).

Outputs (in --out):
  trace/layer{L}.npz     per-token top-6 expert ids, routing weights, expert-output norms
                         + full gate scores (optional)
  state/after_layer{L}.pt residual stream + compress-kv checkpoints (for --resume)
  meta.json               corpus + run metadata
  logits/ (if head shard present after layer 39): teacher-forced top-1 accuracy + NLL per
                          category -- the end-to-end correctness check of the port.

Usage (on the DGX):
  python tools/expert_trace.py --model-dir ./models/DeepSeek-V4.1-Flash \
      --corpus corpus/trace_corpus.jsonl --engram-dir engram_rows --out results/trace-YYYYMMDD
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41_ref as R  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def shard_for(index: dict, name: str) -> str | None:
    return index["weight_map"].get(name)


class ShardGetter:
    """get(name) -> CPU tensor from the right (mmap'd) shard; keeps handles open."""

    def __init__(self, model_dir: str, index: dict):
        self.model_dir, self.index, self.handles = model_dir, index, {}

    def has(self, name: str) -> bool:
        f = shard_for(self.index, name)
        return f is not None and os.path.exists(os.path.join(self.model_dir, f))

    def __call__(self, name: str) -> torch.Tensor:
        f = shard_for(self.index, name)
        if f is None:
            raise KeyError(name)
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.model_dir, f), "pt", device="cpu")
        return self.handles[f].get_tensor(name)

    def close(self, name_prefix: str | None = None):
        for f, h in list(self.handles.items()):
            del self.handles[f]
        import gc
        gc.collect()


def load_corpus(path: str, tokenizer, max_len: int):
    seqs = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ids = tokenizer.encode(d["text"], add_special_tokens=False)
        assert len(ids) <= max_len, (d["id"], len(ids))
        seqs.append({"id": d["id"], "category": d["category"], "ids": ids})
    return seqs


def load_engram_rows(engram_dir: str, layer: int, seq_id: str, device: str) -> torch.Tensor:
    """Rows fetched by engram_rows.py: {seq_id: int64 [T,24]} hash ids + a row store."""
    store = np.load(os.path.join(engram_dir, f"layer{layer}_rows.npz"))
    ids = np.load(os.path.join(engram_dir, f"layer{layer}_hashes.npz"))[seq_id]  # [T, 24]
    row_ids, vals, scales = store["row_ids"], store["vals"], store["scales"]  # sorted row ids, uint8 [n,256], uint8 [n,8]
    pos = np.searchsorted(row_ids, ids.reshape(-1))
    assert np.all(row_ids[pos] == ids.reshape(-1)), "missing engram rows for this sequence"
    v = torch.from_numpy(vals[pos]).to(device).view(torch.float8_e4m3fn).float()  # [T*24, 256]
    s = R.e8m0_to_float(torch.from_numpy(scales[pos]).to(device))  # [T*24, 8]
    v = (v.unflatten(-1, (8, 32)) * s.unsqueeze(-1)).flatten(-2)
    return v.view(ids.shape[0], ids.shape[1], 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--engram-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--layers", default="0-39", help="inclusive range to run, e.g. 0-3")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-scores", action="store_true", help="also store full 384-way gate scores (fp16)")
    ap.add_argument("--no-act-quant", action="store_true", help="skip fp8 activation fake-quant (debug)")
    a = ap.parse_args()

    if a.no_act_quant:
        R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)

    torch.set_grad_enabled(False)
    dev = a.device
    os.makedirs(os.path.join(a.out, "trace"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "state"), exist_ok=True)

    index = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))
    args = R.Args.from_json(os.path.join(a.model_dir, "inference", "config.json"))
    get = ShardGetter(a.model_dir, index)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    seqs = load_corpus(a.corpus, tok, a.max_len)
    n_tok = sum(len(s["ids"]) for s in seqs)
    log(f"corpus: {len(seqs)} sequences, {n_tok} tokens")

    lo, hi = (int(x) for x in a.layers.split("-"))
    start_layer = lo
    states: list[R.SeqState] = []
    if a.resume:
        ck = sorted(glob.glob(os.path.join(a.out, "state", "after_layer*.pt")),
                    key=lambda p: int(re.search(r"after_layer(\d+)", p).group(1)))
        if ck:
            st = torch.load(ck[-1], map_location=dev)
            start_layer = st["layer"] + 1
            for s in st["states"]:
                x = R.SeqState(s["h"].to(dev), s["pre_mix"].to(dev))
                x.compress_kv = None if s["compress_kv"] is None else s["compress_kv"].to(dev)
                x.compress_ratio = s["compress_ratio"]
                states.append(x)
            log(f"resumed after layer {st['layer']} from {ck[-1]}")
    if not states:
        if not get.has("embed.weight"):
            sys.exit("embed.weight shard missing")
        embed = get("embed.weight").to(dev)
        for s in seqs:
            ids = torch.tensor(s["ids"], device=dev)
            h = embed[ids].to(torch.bfloat16).unsqueeze(1).repeat(1, args.hc_mult, 1)
            pre = torch.zeros(len(ids), args.hc_mult, device=dev, dtype=torch.float32)
            pre[:, 0] = 1.0
            states.append(R.SeqState(h, pre))
        del embed
        start_layer = max(start_layer, 0)

    json.dump({"corpus": a.corpus, "n_seqs": len(seqs), "n_tokens": n_tok, "max_len": a.max_len,
               "seqs": [{"id": s["id"], "category": s["category"], "n": len(s["ids"])} for s in seqs],
               "act_quant": not a.no_act_quant},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)

    for L in range(start_layer, hi + 1):
        probe = f"layers.{L}.ffn.gate.weight"
        if not get.has(probe):
            log(f"layer {L}: shard {shard_for(index, probe)} not present -- stopping here (resume later)")
            break
        if L in args.engram_layer_ids:
            need = [os.path.join(a.engram_dir, f"layer{L}_rows.npz"), os.path.join(a.engram_dir, f"layer{L}_hashes.npz"),
                    os.path.join(a.engram_dir, f"layer{L}_weights.safetensors")]
            if not all(os.path.exists(p) for p in need):
                log(f"layer {L}: engram rows/weights missing in {a.engram_dir} -- run tools/engram_rows.py first")
                break
        t0 = time.time()
        w = R.LayerWeights(get, L, args, dev)
        experts = R.ExpertLoader(get, L, dev)
        ew = None
        if L in args.engram_layer_ids:
            eg = safe_open(os.path.join(a.engram_dir, f"layer{L}_weights.safetensors"), "pt", device="cpu")
            ew = R.EngramWeights(lambda n, eg=eg: eg.get_tensor(n), L, dev)

        rec_idx, rec_w, rec_norm, rec_scores, rec_cat, rec_tok = [], [], [], [], [], []
        expert_cache: dict = {}
        big_norm = [0]
        for s, st in zip(seqs, states):
            if ew is not None:
                rows = load_engram_rows(a.engram_dir, L, s["id"], dev)
                st.h = R.engram_forward(st.h, rows, ew, args)

            def record(indices, weights, scores, s=s):
                rec_idx.append(indices.to(torch.int16).cpu().numpy())
                rec_w.append(weights.to(torch.float16).cpu().numpy())
                if a.save_scores:
                    rec_scores.append(scores.to(torch.float16).cpu().numpy())
                rec_cat.extend([s["category"]] * indices.size(0))
                rec_tok.extend(s["ids"])

            def record_norms(contrib_norms):
                # fp32: [tokens, topk] x 4 bytes is a few KB a sequence, and the first fp16 attempt
                # overflowed on 366 picks in the deepest three layers (2026-09-13). Still counted if
                # anything is non-finite, so a broken layer is named rather than summed in silence.
                arr = contrib_norms.to(torch.float32).cpu().numpy()
                big_norm[0] += int((~np.isfinite(arr)).sum())
                rec_norm.append(arr)

            R.block_forward(st, w, experts, args, expert_cache, record, record_norms)
        n_uniq = len(expert_cache)
        del expert_cache, w, experts, ew
        torch.cuda.empty_cache() if dev.startswith("cuda") else None

        if big_norm[0]:
            log(f"layer {L}: WARNING {big_norm[0]} contribution norms are not finite; "
                f"the layer's saliency ranking cannot be trusted")
        np.savez_compressed(os.path.join(a.out, "trace", f"layer{L}.npz"),
                            indices=np.concatenate(rec_idx), weights=np.concatenate(rec_w),
                            # [tokens, topk], aligned with `indices`: ||gate_weight * expert_e(x_t)||,
                            # REAP's saliency per pick (arXiv 2510.13999), which
                            # tools/expert_stats.py sums into `saliency_<topic>`.
                            contrib_norms=np.concatenate(rec_norm),
                            scores=(np.concatenate(rec_scores) if a.save_scores else np.zeros(0, np.float16)),
                            category=np.array(rec_cat), token=np.array(rec_tok, dtype=np.int32))
        torch.save({"layer": L, "states": [{"h": st.h.cpu(), "pre_mix": st.pre_mix.cpu(),
                                             "compress_kv": None if st.compress_kv is None else st.compress_kv.cpu(),
                                             "compress_ratio": st.compress_ratio} for st in states]},
                   os.path.join(a.out, "state", f"after_layer{L}.pt"))
        prev = os.path.join(a.out, "state", f"after_layer{L - 1}.pt")
        if os.path.exists(prev):
            os.remove(prev)
        get.close()
        log(f"layer {L}: {n_tok} tokens, {n_uniq}/{args.n_routed_experts} experts touched, {time.time() - t0:.1f}s")

    # optional end-to-end check: teacher-forced next-token prediction through the real head
    last = max((int(re.search(r"layer(\d+)", p).group(1)) for p in glob.glob(os.path.join(a.out, "trace", "layer*.npz"))),
               default=-1)
    if last == args.n_layers - 1 and get.has("head.weight"):
        log("running final norm + head for the teacher-forced accuracy check")
        norm_w = get("norm.weight").to(dev).to(torch.bfloat16)
        head = get("head.weight").to(dev).float()
        res = {}
        for s, st in zip(seqs, states):
            h = R.hc_pre(st.h, st.pre_mix)
            h = R.rmsnorm(h, norm_w, args.norm_eps)
            logits = h.float() @ head.T  # [T, V]
            ids = torch.tensor(s["ids"], device=dev)
            tgt = ids[1:]
            lp = torch.log_softmax(logits[:-1], dim=-1)
            nll = -lp.gather(1, tgt[:, None]).squeeze(1)
            top1 = (logits[:-1].argmax(-1) == tgt).float()
            c = res.setdefault(s["category"], {"nll": [], "top1": []})
            c["nll"].append(nll.cpu()); c["top1"].append(top1.cpu())
        summary = {k: {"mean_nll": float(torch.cat(v["nll"]).mean()), "top1_acc": float(torch.cat(v["top1"]).mean()),
                       "n": int(torch.cat(v["nll"]).numel())} for k, v in res.items()}
        json.dump(summary, open(os.path.join(a.out, "teacher_forced_check.json"), "w"), indent=1)
        log("teacher-forced check:", summary)


if __name__ == "__main__":
    main()
