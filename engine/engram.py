"""
engram.py -- Engram table rows at serve time: 24 random 264-byte reads per token per engram
layer, straight from the two 101 GB safetensors shards on NVMe (page cache, buffered preadv in a
thread pool; the reads are far too small for O_DIRECT to help). Hash ids come from the reference
`NgramHashState`, which depends on the token ids only.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class EngramTable:
    def __init__(self, model_dir: str, index: dict, layer: int, device: str, threads: int = 16):
        wm = index["weight_map"]
        self.path = os.path.join(model_dir, wm[f"layers.{layer}.engram.embed.weight"])
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        w = hdr[f"layers.{layer}.engram.embed.weight"]
        s = hdr[f"layers.{layer}.engram.embed.scale"]
        assert w["shape"][1] == 256 and s["shape"][1] == 8
        self.w_off = base + w["data_offsets"][0]
        self.s_off = base + s["data_offsets"][0]
        self.rows = w["shape"][0]
        self.fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.pool = ThreadPoolExecutor(threads)
        self.device = device
        self.stats = {"rows": 0, "seconds": 0.0, "calls": 0}
        # small process-local row cache (exact n-gram repeats inside a conversation hit here)
        self.cache: dict[int, bytes] = {}
        self.cache_max = 200_000

    def _read_rows(self, ids: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), 264), np.uint8)
        for i, r in enumerate(ids):
            r = int(r)
            b = self.cache.get(r)
            if b is None:
                b = os.pread(self.fd, 256, self.w_off + r * 256) + os.pread(self.fd, 8, self.s_off + r * 8)
                if len(self.cache) < self.cache_max:
                    self.cache[r] = b
            out[i] = np.frombuffer(b, np.uint8)
        return out

    def rows(self, hashes: torch.Tensor) -> torch.Tensor:
        """hashes: int64 [T, 24] -> float32 [T, 24, 256] dequantized rows."""
        t0 = time.perf_counter()
        flat = hashes.reshape(-1).cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        n = len(uniq)
        chunk = max(8, n // (self.pool._max_workers * 2) + 1)
        parts = list(self.pool.map(self._read_rows, [uniq[i:i + chunk] for i in range(0, n, chunk)]))
        raw = np.concatenate(parts) if parts else np.empty((0, 264), np.uint8)
        raw = torch.from_numpy(raw).to(self.device)
        vals = raw[:, :256].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(raw[:, 256:].float() - 127.0)
        deq = (vals.unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2)  # [n, 256]
        out = deq[torch.from_numpy(inv).to(self.device)].view(hashes.shape[0], hashes.shape[1], 256)
        self.stats["rows"] += int(n); self.stats["seconds"] += time.perf_counter() - t0; self.stats["calls"] += 1
        return out


def make_hash_state(model_dir: str, tokenizer, max_seq: int, device: str):
    """The reference NgramHashState (engram.py from the checkpoint's inference/ folder)."""
    sys.path.insert(0, os.path.join(model_dir, "inference"))
    import engram as E  # noqa: E402
    cfg = json.load(open(os.path.join(model_dir, "inference", "config.json")))

    class A:
        engram_layer_ids = tuple(cfg["engram_layer_ids"])
        engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]
        engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = tuple(cfg["engram_num_embeddings"])
        engram_head_dim = cfg["engram_head_dim"]
        engram_pad_id = cfg["engram_pad_id"]
        engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]
        max_batch_size = 1
        max_seq_len = max_seq + 16

    layout = E.EngramLayout.from_args(A)
    st = E.NgramHashState(A, layout, tokenizer)
    return st.to(device)
