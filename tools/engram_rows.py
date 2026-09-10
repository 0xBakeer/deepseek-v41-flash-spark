#!/usr/bin/env python3
"""
engram_rows.py -- fetch exactly the Engram table rows a corpus needs, without downloading
the two 101 GB table shards.

The Engram hash ids depend ONLY on the token ids (inference/engram.py, tech report 3.1.3),
so for a fixed corpus the set of rows is known before any forward pass. Each row is
256 bytes of FP8 plus 8 bytes of UE8M0 scales. This tool computes the hash ids with the
reference NgramHashState, de-duplicates them, and pulls the rows straight out of the
safetensors shards on Hugging Face with HTTP range requests (or, with --local-shard, from a
local copy). It also pulls the small non-table engram weights (wkv 157 MB, q/k weights).

This is the same access pattern a serving recipe would use against NVMe (24 rows per
token per engram layer, i.e. 48 random ~264 B reads per token), just over HTTP.

Outputs in --out:
  layer{L}_hashes.npz    {seq_id: int64 [T, 24]} hash ids per sequence
  layer{L}_rows.npz      row_ids (sorted int64 [n]), vals (uint8 [n,256]), scales (uint8 [n,8])
  layer{L}_weights.safetensors  layers.L.engram.{wkv.weight,wkv.scale,q_weight,k_weight}
  fetch_stats.json       request counts / bytes / wall time (the I/O measurement)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"


def hf_headers():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        for p in (os.path.expanduser("~/.cache/huggingface/token"),):
            if os.path.exists(p):
                tok = open(p).read().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


class RemoteShard:
    """Range reads against one safetensors file on the Hub (resolves the CDN URL once)."""

    def __init__(self, fname: str, session_factory):
        import requests
        self.fname = fname
        self.headers = hf_headers()
        self.session_factory = session_factory
        self._local = threading.local()
        self.bytes, self.reqs = 0, 0
        r = requests.head(BASE + fname, headers=self.headers, allow_redirects=True)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers.get("Content-Length", 0))
        hdr = self.read(0, 8)
        n = struct.unpack("<Q", hdr)[0]
        self.header = json.loads(self.read(8, 8 + n))
        self.header.pop("__metadata__", None)
        self.data_base = 8 + n

    def session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._local.s = self.session_factory()
        return s

    def read(self, start: int, end: int) -> bytes:
        """[start, end) absolute file offsets."""
        for attempt in range(6):
            try:
                r = self.session().get(self.url, headers={**self.headers, "Range": f"bytes={start}-{end - 1}"}, timeout=60)
                if r.status_code in (200, 206) and len(r.content) == end - start:
                    self.bytes += end - start
                    self.reqs += 1
                    return r.content
                if r.status_code == 403:  # signed CDN url expired
                    import requests
                    self.url = requests.head(BASE + self.fname, headers=self.headers, allow_redirects=True).url
                if r.status_code == 429:
                    time.sleep(3.0 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                err = repr(e)
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"range read failed {self.fname} {start}-{end}: {err if 'err' in dir() else r.status_code}")

    def read_multi(self, spans):
        """Multipart range request; spans = [(start, end_inclusive), ...] in request order.
        Returns the bodies in the same order (parsed from the Content-Range headers)."""
        import re as _re
        hdr = "bytes=" + ",".join(f"{s}-{e}" for s, e in spans)
        want = {s: e - s + 1 for s, e in spans}
        for attempt in range(8):
            try:
                r = self.session().get(self.url, headers={**self.headers, "Range": hdr}, timeout=90)
                ct = r.headers.get("Content-Type", "")
                if r.status_code == 206 and ct.startswith("multipart/byteranges"):
                    boundary = ct.split("boundary=", 1)[1].strip().encode()
                    parts = r.content.split(b"--" + boundary)
                    got = {}
                    for p in parts:
                        m = _re.search(rb"Content-Range: bytes (\d+)-(\d+)/\d+\r\n\r\n", p)
                        if not m:
                            continue
                        s0 = int(m.group(1))
                        body = p[m.end():m.end() + want[s0]]
                        got[s0] = body
                    if all(len(got.get(s, b"")) == n for s, n in want.items()):
                        self.bytes += sum(want.values())
                        self.reqs += 1
                        return [got[s] for s, _ in spans]
                elif r.status_code == 206 and len(spans) == 1 and len(r.content) == want[spans[0][0]]:
                    self.bytes += len(r.content); self.reqs += 1
                    return [r.content]
                if r.status_code == 403:
                    import requests
                    self.url = requests.head(BASE + self.fname, headers=self.headers, allow_redirects=True).url
                if r.status_code == 429:
                    time.sleep(3.0 * (attempt + 1))
                err = f"status {r.status_code} ct {ct} len {len(r.content)}"
            except Exception as e:  # noqa: BLE001
                err = repr(e)
            time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"multi-range read failed {self.fname} ({len(spans)} spans): {err}")

    def tensor_span(self, name: str):
        t = self.header[name]
        s, e = t["data_offsets"]
        return self.data_base + s, self.data_base + e, t

    def full_tensor(self, name: str) -> torch.Tensor:
        s, e, t = self.tensor_span(name)
        buf = self.read(s, e)
        dt = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16,
              "F32": torch.float32, "I8": torch.int8}[t["dtype"]]
        return torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(dt).view(*t["shape"]).clone()


class LocalShard:
    def __init__(self, path: str):
        self.fname = os.path.basename(path)
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.header.pop("__metadata__", None)
        self.data_base = 8 + n
        self.lock = threading.Lock()
        self.bytes, self.reqs = 0, 0

    def read(self, start, end):
        with self.lock:
            self.f.seek(start)
            b = self.f.read(end - start)
        self.bytes += end - start
        self.reqs += 1
        return b

    tensor_span = RemoteShard.tensor_span
    full_tensor = RemoteShard.full_tensor


def compute_hashes(model_dir: str, corpus: str, layer_ids):
    """Returns {layer: {seq_id: int64 [T, 24]}} using the reference NgramHashState."""
    sys.path.insert(0, os.path.join(model_dir, "inference"))
    import engram as E  # noqa: E402
    from transformers import AutoTokenizer

    cfg = json.load(open(os.path.join(model_dir, "inference", "config.json")))

    class A:  # what EngramLayout / NgramHashState read from ModelArgs
        engram_layer_ids = tuple(cfg["engram_layer_ids"])
        engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]
        engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = tuple(cfg["engram_num_embeddings"])
        engram_head_dim = cfg["engram_head_dim"]
        engram_pad_id = cfg["engram_pad_id"]
        engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]
        max_batch_size = 1
        max_seq_len = 4096

    tok = AutoTokenizer.from_pretrained(model_dir)
    layout = E.EngramLayout.from_args(A)
    state = E.NgramHashState(A, layout, tok)
    out = {L: {} for L in layer_ids}
    for line in open(corpus):
        if not line.strip():
            continue
        d = json.loads(line)
        ids = torch.tensor([tok.encode(d["text"], add_special_tokens=False)])
        h = state(ids, 0)[0]  # [T, n_layers, 24]
        for li, L in enumerate(layout.layer_ids):
            out[L][d["id"]] = h[:, li, :].numpy().astype(np.int64)
    return out, layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with inference/ config + tokenizer")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-shard", action="append", default=[], help="local path of model-00047/48 if present")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--layers", default="1,14")
    ap.add_argument("--spans", type=int, default=100, help="rows per multipart range request")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    layer_ids = [int(x) for x in a.layers.split(",")]

    index = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
    hashes, layout = compute_hashes(a.model_dir, a.corpus, layer_ids)
    stats = {}
    import requests
    for L in layer_ids:
        fname = index[f"layers.{L}.engram.embed.weight"]
        local = [p for p in a.local_shard if os.path.basename(p) == fname]
        shard = LocalShard(local[0]) if local else RemoteShard(fname, lambda: requests.Session())
        np.savez_compressed(os.path.join(a.out, f"layer{L}_hashes.npz"), **hashes[L])
        rows = np.unique(np.concatenate([v.reshape(-1) for v in hashes[L].values()]))
        n_tok = sum(v.shape[0] for v in hashes[L].values())
        log = lambda *x: print(time.strftime("%H:%M:%S"), f"[L{L}]", *x, flush=True)
        log(f"{n_tok} tokens -> {n_tok * 24} lookups -> {len(rows)} unique rows "
            f"({len(rows) * 264 / 1e6:.1f} MB) from {fname} ({'local' if local else 'HTTP range'})")

        ws, we, wt = shard.tensor_span(f"layers.{L}.engram.embed.weight")
        ss, se, stt = shard.tensor_span(f"layers.{L}.engram.embed.scale")
        assert wt["shape"][1] == 256 and stt["shape"][1] == 8
        vals = np.zeros((len(rows), 256), np.uint8)
        scales = np.zeros((len(rows), 8), np.uint8)

        # One multipart range request per batch of rows: CloudFront (the Hub's CDN) answers
        # `Range: bytes=a-b,c-d,...` with multipart/byteranges, so a batch of --spans rows costs one
        # round trip for the 256 B rows and one for the 8 B scales. Single-range requests got
        # rate-limited (HTTP 429) at ~400 req/s.
        done = [0]
        t0 = time.time()
        batches = [(i, min(len(rows), i + a.spans)) for i in range(0, len(rows), a.spans)]

        def multi_read(base, row_bytes, i0, i1):
            spans = [(base + int(r) * row_bytes, base + int(r) * row_bytes + row_bytes - 1) for r in rows[i0:i1]]
            out = shard.read_multi(spans)
            return np.frombuffer(b"".join(out), np.uint8).reshape(-1, row_bytes)

        def fetch(b):
            i0, i1 = b
            if local:
                for k in range(i0, i1):
                    r = int(rows[k])
                    vals[k] = np.frombuffer(shard.read(ws + r * 256, ws + r * 256 + 256), np.uint8)
                    scales[k] = np.frombuffer(shard.read(ss + r * 8, ss + r * 8 + 8), np.uint8)
            else:
                vals[i0:i1] = multi_read(ws, 256, i0, i1)
                scales[i0:i1] = multi_read(ss, 8, i0, i1)
            done[0] += 1
            if done[0] % 50 == 0:
                el = time.time() - t0
                log(f"{done[0]}/{len(batches)} batches, {shard.reqs} reqs, {shard.bytes / 1e6:.1f} MB, {el:.0f}s, "
                    f"{shard.reqs / el:.1f} req/s, {(i1) / el:.0f} rows/s")

        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(fetch, batches))
        runs = batches
        el = time.time() - t0
        np.savez_compressed(os.path.join(a.out, f"layer{L}_rows.npz"), row_ids=rows, vals=vals, scales=scales)
        log(f"rows done: {shard.reqs} requests, {shard.bytes / 1e6:.1f} MB in {el:.1f}s ({shard.reqs / el:.0f} req/s)")

        from safetensors.torch import save_file
        ws_t = {n: shard.full_tensor(f"layers.{L}.engram.{n}") for n in ("wkv.weight", "wkv.scale", "q_weight", "k_weight")}
        save_file({f"layers.{L}.engram.{k}": v for k, v in ws_t.items()},
                  os.path.join(a.out, f"layer{L}_weights.safetensors"))
        stats[L] = {"tokens": n_tok, "lookups": n_tok * 24, "unique_rows": int(len(rows)), "runs": len(runs),
                    "requests": shard.reqs, "bytes": shard.bytes, "seconds": el, "source": "local" if local else "http",
                    "table_rows": wt["shape"][0]}
        log("engram weights saved")
    json.dump(stats, open(os.path.join(a.out, "fetch_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
