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
    def __init__(self, model_dir: str, index: dict, layer: int, device: str, threads: int = 32):
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
        self.n_rows = w["shape"][0]
        self.fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.pool = ThreadPoolExecutor(threads)
        self.device = device
        self.stats = {"rows": 0, "seconds": 0.0, "calls": 0}
        # small process-local row cache (exact n-gram repeats inside a conversation hit here)
        self.cache: dict[int, bytes] = {}
        self.cache_max = 200_000
        # Pinned staging for the H2D (see to_device), OFF by default: it does make `to_device`
        # itself ~10x cheaper (0.27 s vs 3.16 s of a 200-token run, because the pageable copy
        # synchronises the stream and absorbs the queued graph work) but it does NOT make the run
        # faster -- measured 12.04-13.57 s of decode against 12.03-12.14 s pageable, i.e. the host
        # simply blocks somewhere else instead. Kept behind the switch rather than deleted.
        self.pinned = os.environ.get("DSV41_ENGRAM_PINNED", "0") == "1"
        self._stage = [None, None]
        self._inv_stage = [None, None]
        self._ev = [None, None]
        self._cur = 0

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

    def read_raw(self, hashes_np: np.ndarray):
        """Host-only part (safe in a background thread: no CUDA calls): NVMe reads of the unique rows.
        hashes_np: int64 [T, 24]. Returns (raw uint8 [n, 264], inv, shape)."""
        t1 = time.perf_counter()
        flat = hashes_np.reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        n = len(uniq)
        chunk = max(8, n // (self.pool._max_workers * 2) + 1)
        parts = list(self.pool.map(self._read_rows, [uniq[i:i + chunk] for i in range(0, n, chunk)]))
        raw = np.concatenate(parts) if parts else np.empty((0, 264), np.uint8)
        self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
        self.stats["rows"] += int(n); self.stats["calls"] += 1
        return raw, inv, hashes_np.shape

    def to_device(self, raw, inv, shape) -> torch.Tensor:
        """GPU part (main thread): dequantize the rows and expand to [T, 24, 256] float32.

        The raw rows and the row-index vector go through PINNED staging buffers and are copied
        non-blocking. A plain `.to(device)` from pageable numpy memory synchronises the calling
        stream, which inside a decode step means blocking the host until every graph queued so far
        has finished -- exactly the overlap this pipeline exists to avoid. The buffers are reused
        across steps, so the previous step's copy out of them is waited on first (`_ev`); a whole
        step elapses in between, so that wait is free.
        """
        t0 = time.perf_counter()
        n = raw.shape[0]
        if not self.pinned:
            rawt = torch.from_numpy(raw).to(self.device)
            invt = torch.from_numpy(inv.reshape(-1)).to(self.device)
        else:
            i = self._cur
            self._cur ^= 1  # two buffers: the wait is on the copy from two calls ago
            if self._stage[i] is None or self._stage[i].shape[0] < n:
                cap = max(n, 4096)
                self._stage[i] = torch.empty(cap, 264, dtype=torch.uint8, pin_memory=True)
                self._inv_stage[i] = torch.empty(cap * 64, dtype=torch.int64, pin_memory=True)
                self._ev[i] = torch.cuda.Event()
                self._ev[i].record()
            self._ev[i].synchronize()  # the last H2D out of these buffers is done
            self._stage[i][:n].copy_(torch.from_numpy(raw))
            ni = inv.size if hasattr(inv, "size") else len(inv)
            self._inv_stage[i][:ni].copy_(torch.from_numpy(inv.reshape(-1).astype(np.int64)))
            rawt = self._stage[i][:n].to(self.device, non_blocking=True)
            invt = self._inv_stage[i][:ni].to(self.device, non_blocking=True)
        vals = rawt[:, :256].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(rawt[:, 256:].float() - 127.0)
        deq = (vals.unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2)
        out = deq[invt].view(shape[0], shape[1], 256)
        if self.pinned:
            self._ev[i].record()
        self.stats["seconds"] += time.perf_counter() - t0
        return out

    def rows(self, hashes: torch.Tensor) -> torch.Tensor:
        """hashes: int64 [T, 24] -> float32 [T, 24, 256] dequantized rows."""
        t0 = time.perf_counter()
        flat = hashes.reshape(-1).cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        n = len(uniq)
        chunk = max(8, n // (self.pool._max_workers * 2) + 1)
        t1 = time.perf_counter()
        parts = list(self.pool.map(self._read_rows, [uniq[i:i + chunk] for i in range(0, n, chunk)]))
        self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
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
