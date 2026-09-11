"""
experts.py -- the routed-expert store for one-box serving.

15,360 routed experts x 18.8 MB (FP4 + UE8M0 scales) = 288.8 GB do not fit next to everything
else, so the experts live in three places:

  * the ARENA: a fixed number of GPU slots holding packed FP4 experts exactly as stored in the
    checkpoint (no re-quantization). Sized at start-up from the memory that is left.
  * the LRU: a map (layer, expert) -> slot, least-recently-used eviction, warm-started from a
    routing trace so the hottest experts are resident before the first request.
  * NVMe: every expert is read straight out of its layer's safetensors shard with O_DIRECT
    preadv (no page-cache pollution, ~5.5 GB/s with 8+ reads in flight on this box), into a
    pinned staging buffer, then copied into its slot. Two runs per expert, not six: see
    `ShardFile.expert_runs`. Each run is split into `read_chunk_mb` aligned pieces issued in
    parallel on a second thread pool, because a decode layer misses only ~4 experts and two
    serial 18.8 MB reads cannot keep the device busy on their own (see NOTES "Speed work").

Prefill chunks touch almost every expert of a layer; letting them stream through the LRU would
evict the hot set each prompt. So misses during prefill go through a small TRANSIENT ring of
slots instead, and only decode misses enter the LRU.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ALIGN = 4096
# every counter reset between requests (engine/v41_engine.py::_reset) lives here
ZERO_STATS = {"hits": 0, "misses": 0, "prefill_misses": 0, "bytes_read": 0, "read_s": 0.0,
              "resolve_s": 0.0, "route_s": 0.0, "load_s": 0.0, "lease_s": 0.0, "h2d_s": 0.0,
              "loads": 0, "promoted": 0}
W13_SHAPE = (2304, 2560)
S13_SHAPE = (2304, 160)
W2_SHAPE = (5120, 1152)
S2_SHAPE = (5120, 72)
NAMES = ("w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale")
EXPERT_BYTES = 3 * (2304 * 2560 + 2304 * 160)


class ShardFile:
    """One safetensors shard: header spans + an O_DIRECT fd."""

    def __init__(self, path: str):
        self.path = path
        self._runs: dict[str, list] = {}
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        self.base = 8 + n
        self.spans = {k: (self.base + v["data_offsets"][0], self.base + v["data_offsets"][1]) for k, v in hdr.items()}
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.fd_buffered = os.open(path, os.O_RDONLY)

    def expert_span(self, prefix: str):
        """Byte span covering the 6 tensors of one expert."""
        s = [self.spans[prefix + n] for n in NAMES]
        lo, hi = min(a for a, _ in s), max(b for _, b in s)
        return lo, hi, s

    def expert_runs(self, prefix: str):
        """The 6 tensors of one expert grouped into maximal contiguous file ranges.

        These shards store ALL the scale tensors near the front and all the weight tensors far
        behind them, but within each group the three tensors of an expert are adjacent. So an
        expert is exactly two runs -- a 1.1 MB scale run and a 17.7 MB weight run -- not six, and
        not one. Reading it as six separate `preadv`s costs six O_DIRECT round trips; at a large
        arena a decode step misses only about one expert per layer, so nothing else is in flight to
        hide that latency and the read rate collapses from ~4.7 GB/s to ~0.6 GB/s.

        Returns [(file_lo, file_hi, [(name_index, offset_in_run, nbytes), ...]), ...].
        """
        cached = self._runs.get(prefix)
        if cached is not None:
            return cached
        spans = [self.spans[prefix + n] for n in NAMES]
        order = sorted(range(len(spans)), key=lambda i: spans[i][0])
        runs = []
        for i in order:
            a, b = spans[i]
            if runs and runs[-1][1] == a:
                lo, _, members = runs[-1]
                members.append((i, a - lo, b - a))
                runs[-1] = (lo, b, members)
            else:
                runs.append((a, b, [(i, 0, b - a)]))
        self._runs[prefix] = runs
        return runs


def _pread_chunk(fd: int, view: memoryview, off: int, need: int) -> None:
    """O_DIRECT-read `need` bytes at file offset `off` into `view` (an aligned slice of a pinned
    buffer). The request length is always the full aligned slice -- O_DIRECT rejects unaligned
    lengths -- and the loop stops as soon as the bytes that actually exist have arrived, which is
    what makes the aligned tail of the last tensor in a shard safe."""
    got = 0
    while got < need:
        r = os.preadv(fd, [view[got:]], off + got)
        if r <= 0:
            raise IOError(f"short read at {off}+{got}/{need}")
        got += r


class ExpertStore:
    def __init__(self, model_dir: str, index: dict, arena, n_layers: int, transient_slots: int = 400,
                 io_threads: int = 12, mtp_prefix: str | None = None, read_threads: int | None = None,
                 read_chunk_mb: float | None = None):
        self.model_dir = model_dir
        self.arena = arena  # tools.fp4_moe.ExpertArena or a compatible object with .slots and load_slot_bytes
        self.n_slots = arena.slots
        self.transient_slots = transient_slots
        self.lru_slots = self.n_slots - transient_slots
        assert self.lru_slots > 0
        # a prefill chunk can touch all 384 experts of a layer; a smaller ring is only safe when every routable
        # expert is resident (pruned all-resident mode). resolve() asserts on slot collisions either way.
        assert transient_slots >= 8, 'transient ring too small'
        self.shards: dict[str, ShardFile] = {}
        self.index = index["weight_map"]
        self.lru: OrderedDict[tuple, int] = OrderedDict()  # (layer, expert) -> slot
        self.slot_key: dict[int, tuple] = {}
        self.free_lru = list(range(self.lru_slots))
        self.transient_ring = list(range(self.lru_slots, self.n_slots))
        self.transient_index = {s: i for i, s in enumerate(self.transient_ring)}
        self.transient_pos = 0
        self.transient_map: dict[tuple, int] = {}
        io_threads = int(os.environ.get("DSV41_IO_THREADS", io_threads))
        if read_threads is None:
            read_threads = int(os.environ.get("DSV41_READ_THREADS", 24))
        if read_chunk_mb is None:
            read_chunk_mb = float(os.environ.get("DSV41_READ_CHUNK_MB", 4))
        self.io_threads = io_threads
        self.read_threads = read_threads
        self.read_chunk = int(read_chunk_mb * 1024 * 1024) // ALIGN * ALIGN
        # Two pools on purpose. `pool` runs one task per expert (it owns a pinned staging buffer for
        # the whole read + H2D); `read_pool` runs the individual aligned pieces of that expert's two
        # file runs. A single pool would deadlock as soon as every worker sat waiting for a piece
        # that has no worker left to run it.
        self.pool = ThreadPoolExecutor(io_threads, thread_name_prefix="expert-io")
        self.read_pool = ThreadPoolExecutor(max(1, read_threads), thread_name_prefix="expert-read")
        self.lock = threading.Lock()
        # pinned, aligned staging buffers, one per io thread
        self.stage = [torch.empty(EXPERT_BYTES + 8 * ALIGN, dtype=torch.uint8, pin_memory=True) for _ in range(io_threads)]
        self.stage_mv = [memoryview(b.numpy()) for b in self.stage]
        self.stage_free = list(range(io_threads))
        self.stage_sem = threading.Semaphore(io_threads)
        self._tls = threading.local()
        self.n_experts = 384
        self.stats = dict(ZERO_STATS)

    # ------------------------------------------------------------------ io
    def _shard(self, name: str) -> ShardFile:
        f = self.index[name]
        if f not in self.shards:
            self.shards[f] = ShardFile(os.path.join(self.model_dir, f))
        return self.shards[f]

    def _lease(self) -> int:
        self.stage_sem.acquire()
        with self.lock:
            return self.stage_free.pop()

    def _release(self, sid: int) -> None:
        with self.lock:
            self.stage_free.append(sid)
        self.stage_sem.release()

    def _read_leased(self, layer: int, expert: int, prefix: str | None, sink):
        """O_DIRECT-read one expert into a pinned staging buffer and call ``sink(views)`` while the
        buffer is still leased. ``views`` are 6 uint8 tensors that ALIAS the pinned buffer, so the
        sink must be done with them before it returns.

        The expert's two contiguous file runs (see `ShardFile.expert_runs`) are cut into
        `self.read_chunk`-sized aligned pieces and all but the first are handed to `read_pool`, so
        one expert alone keeps ~5 requests in flight. A decode layer misses ~4 experts; at two
        serial reads each the queue depth was ~4-8 and the device only gave ~2.6 GB/s of its
        5.5 GB/s, which is the whole reason decode was slower than its byte count implies.
        """
        p = prefix or f"layers.{layer}.ffn.experts.{expert}."
        sh = self._shard(p + "w1.weight")
        runs = sh.expert_runs(p)
        t_lease = time.perf_counter()
        sid = self._lease()
        self.stats["lease_s"] += time.perf_counter() - t_lease
        buf = self.stage[sid]
        try:
            mv = self.stage_mv[sid]
            base_addr = buf.data_ptr()
            cur = (-base_addr) % ALIGN
            out = [None] * len(NAMES)
            jobs = []
            nbytes = 0
            t0 = time.perf_counter()
            for (a, b, members) in runs:
                alo = a - a % ALIGN
                ahi = (b + ALIGN - 1) // ALIGN * ALIGN
                n = ahi - alo
                step = self.read_chunk if 0 < self.read_chunk < n else n
                off = 0
                while off < n:
                    m = min(step, n - off)
                    need = min(b - alo - off, m)  # the aligned tail may run past EOF
                    if need > 0:
                        jobs.append((sh.fd, mv[cur + off: cur + off + m], alo + off, need))
                    off += m
                nbytes += n
                base = cur + (a - alo)
                for (i, o, nb) in members:
                    out[i] = buf[base + o: base + o + nb]
                cur += n
            futs = [self.read_pool.submit(_pread_chunk, *j) for j in jobs[1:]]
            _pread_chunk(*jobs[0])
            for f in futs:
                f.result()
            assert all(t is not None for t in out)
            self.stats["bytes_read"] += nbytes
            self.stats["read_s"] += time.perf_counter() - t0
            self.stats["loads"] += 1
            return sink(out)
        finally:
            self._release(sid)

    def read_expert(self, layer: int, expert: int, prefix: str | None = None):
        """The 6 tensors (CPU uint8) of one expert, copied out of the staging buffer."""
        return self._read_leased(layer, expert, prefix, lambda v: [t.clone() for t in v])

    def _copy_stream(self):
        """One CUDA stream per io thread.

        The arena copy must not run on the default stream: every worker would then have to
        synchronise the stream the model is computing on, once per miss. On its own stream a worker
        only has to (a) wait for whatever was queued on the compute stream when the lease started --
        the previous layer's MoE kernel may still be reading the slot we are about to overwrite --
        and (b) synchronise its own stream before releasing the pinned buffer.
        """
        st = getattr(self._tls, "stream", None)
        if st is None:
            st = self._tls.stream = torch.cuda.Stream()
        return st

    def _load_into_slot(self, key: tuple, slot: int, prefix: str | None = None):
        """Read one expert straight from NVMe into its arena slot.

        The pinned staging buffer is handed to `arena.load_slot` directly instead of being cloned
        first: the clone was a second 18.8 MB CPU memcpy per expert AND it made the H2D copy run
        from pageable memory, which PyTorch has to stage through a bounce buffer of its own.
        """
        stream = self._copy_stream()
        compute = torch.cuda.current_stream()  # capture OUTSIDE the `with`, where it is still ours

        def sink(v):
            t0 = time.perf_counter()
            w1, s1, w2, s2, w3, s3 = v
            with torch.cuda.stream(stream):
                stream.wait_stream(compute)
                self.arena.load_slot(slot, w1.view(*W13_SHAPE), s1.view(*S13_SHAPE), w2.view(*W2_SHAPE),
                                     s2.view(*S2_SHAPE), w3.view(*W13_SHAPE), s3.view(*S13_SHAPE),
                                     non_blocking=True)
                sim = getattr(self, "requant", None)  # simulated low-bit format (engine/codebook_sim.py)
                if sim is not None:
                    bits = sim.get(key)
                    if bits:
                        self.requant_sims[bits].requant_slot(self.arena, slot)
            stream.synchronize()  # the staging buffer is leased to another expert right after
            self.stats["h2d_s"] += time.perf_counter() - t0
            return slot

        return self._read_leased(key[0], key[1], prefix, sink)

    # ------------------------------------------------------------------ cache policy
    def _lru_slot_for(self, key: tuple, used: set | frozenset = frozenset()) -> int:
        """Reserve an LRU slot for `key` (evicting if needed). Caller loads it.
        `used` holds the slots already promised to other experts of the SAME resolve() call; they
        must never be evicted, or two experts would end up sharing one slot."""
        if self.free_lru:
            slot = self.free_lru.pop()
        else:
            parked = []
            while True:
                if not self.lru:
                    raise RuntimeError("LRU exhausted: more experts in one call than lru_slots")
                old_key, slot = self.lru.popitem(last=False)
                if slot not in used:
                    self.slot_key.pop(slot, None)
                    break
                parked.append((old_key, slot))
            for k, s in reversed(parked):  # put the protected entries back, oldest first
                self.lru[k] = s
                self.lru.move_to_end(k, last=False)
        self.lru[key] = slot
        self.slot_key[slot] = key
        return slot

    def _transient_slot_for(self, key: tuple, used: set | frozenset = frozenset()) -> int:
        """Next slot of the transient ring, skipping any slot already promised in this call."""
        n = len(self.transient_ring)
        for _ in range(n):
            slot = self.transient_ring[self.transient_pos % n]
            self.transient_pos += 1
            if slot not in used:
                break
        else:
            raise RuntimeError("transient ring exhausted: more experts in one call than transient_slots")
        old = self.slot_key.pop(slot, None)
        if old is not None:
            self.transient_map.pop(old, None)
        prev = self.transient_map.get(key)
        if prev is not None and prev != slot:  # stale mapping from an earlier, recycled slot
            self.slot_key.pop(prev, None)
        self.transient_map[key] = slot
        self.slot_key[slot] = key
        return slot

    def _promote_transient(self, key: tuple, slot: int, used: set) -> bool:
        """Give a transient-ring slot to the LRU without re-reading its 18.8 MB.

        A prefill chunk loads almost every expert of a layer into the transient ring; when decode
        then routes to one of those the old code counted a hit, used the slot, and left it in the
        ring -- so the ring wrapped over it a few requests later and the expert was read again even
        though it was demonstrably hot at decode time. The ring is only a list of slot ids, so the
        fix is a pointer swap: this slot joins the LRU where it lies, and an LRU victim's slot takes
        its place in the ring.
        """
        i = self.transient_index.get(slot)
        if i is None:
            return False
        if self.free_lru:
            donor = self.free_lru.pop()
        else:
            donor, parked = None, []
            while self.lru:
                k2, s2 = self.lru.popitem(last=False)
                if s2 not in used and s2 != slot:
                    self.slot_key.pop(s2, None)
                    donor = s2
                    break
                parked.append((k2, s2))
            for k2, s2 in reversed(parked):
                self.lru[k2] = s2
                self.lru.move_to_end(k2, last=False)
            if donor is None:
                return False
        self.transient_ring[i] = donor
        self.transient_index.pop(slot, None)
        self.transient_index[donor] = i
        self.transient_map.pop(key, None)
        self.lru[key] = slot
        self.slot_key[slot] = key
        self.stats["promoted"] += 1
        return True

    def resolve(self, layer: int, experts: torch.Tensor, prefill: bool) -> torch.Tensor:
        """experts: int tensor [T, K] of expert ids for `layer`. Returns the slot ids [T, K],
        loading misses (in parallel) first."""
        t_res = time.perf_counter()
        # One device->host copy, and the set/LUT work in numpy on the host. The old path ran
        # torch.unique on the GPU, synchronised on .tolist(), built a 384-entry LUT, copied that
        # back up and gathered it there: two extra launches and a second sync per layer, 40 layers
        # per token, for 36 numbers.
        ex = experts.to("cpu", dtype=torch.int32, non_blocking=False).numpy()
        uniq = np.unique(ex)
        slot_of = {}
        to_load = []
        used: set[int] = set()  # slots already promised in this call -- never recycle one of them
        # pass 1: residents. Reserving them before any allocation is what keeps a later miss from
        # running the transient ring over a slot an earlier hit is already using (which used to
        # give two experts the same slot: the second load overwrote the first expert's weights and
        # the duplicate index in moe_forward's `y[t] +=` dropped one contribution).
        for e in uniq.tolist():
            key = (layer, e)
            s = self.lru.get(key)
            if s is None:
                s = self.transient_map.get(key)
                if s is not None and not prefill:
                    self._promote_transient(key, s, used)
            else:
                self.lru.move_to_end(key)
            if s is not None:
                slot_of[e] = s
                used.add(s)
                self.stats["hits"] += 1
        # pass 2: misses
        for e in uniq.tolist():
            if e in slot_of:
                continue
            key = (layer, e)
            if prefill:
                self.stats["prefill_misses"] += 1
                s = self._transient_slot_for(key, used)
            else:
                self.stats["misses"] += 1
                s = self._lru_slot_for(key, used)
            slot_of[e] = s
            used.add(s)
            to_load.append((key, s))
        assert len(set(slot_of.values())) == len(slot_of), "slot collision in resolve()"
        lut = np.full(self.n_experts, -1, dtype=np.int32)
        for e, s in slot_of.items():
            lut[e] = s
        slots = torch.from_numpy(lut[ex.astype(np.intp)]).to(experts.device)
        self.stats["route_s"] += time.perf_counter() - t_res
        if to_load:
            t0 = time.perf_counter()
            list(self.pool.map(lambda ks: self._load_into_slot(*ks), to_load))
            self.stats["load_s"] += time.perf_counter() - t0
        self.stats["resolve_s"] += time.perf_counter() - t_res
        return slots

    def warm_start(self, ranked_keys: list[tuple], log=print):
        """Fill the LRU with `ranked_keys` (most important first) up to capacity."""
        keys = [k for k in ranked_keys[: self.lru_slots]]
        t0 = time.time()
        jobs = []
        for k in keys:
            jobs.append((k, self._lru_slot_for(k)))
        done = 0
        for _ in self.pool.map(lambda ks: self._load_into_slot(*ks), jobs):
            done += 1
            if done % 500 == 0:
                log(f"warm start {done}/{len(jobs)} experts, {self.stats['bytes_read'] / 1e9:.1f} GB, {time.time() - t0:.0f}s")
        log(f"warm start done: {len(jobs)} experts resident ({len(jobs) * EXPERT_BYTES / 1e9:.1f} GB) in {time.time() - t0:.0f}s")

    def hit_rate(self):
        h, m = self.stats["hits"], self.stats["misses"]
        return h / max(1, h + m)


def category_counts(trace_stats_json: str, profile: str, n_experts: int = 384) -> dict[int, np.ndarray]:
    """Per-layer expert histogram restricted to one corpus category.

    `coverage.json` only carries the mixed histogram (`counts`) plus the two coverage *curves*, so a
    workload-specific hot set has to be recomputed from the raw traces the stats were made from:
    `results/<name>/trace/layer<L>.npz` with `indices` [tokens, 6] and `category` [tokens].
    """
    trace_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(trace_stats_json))), "trace")
    out: dict[int, np.ndarray] = {}
    if not os.path.isdir(trace_dir):
        return out
    import glob
    import re
    for path in glob.glob(os.path.join(trace_dir, "layer*.npz")):
        L = int(re.search(r"layer(\d+)", os.path.basename(path)).group(1))
        z = np.load(path)
        idx, cat = z["indices"], z["category"]
        sel = idx[cat.astype("U") == profile]
        if sel.size == 0:
            continue
        out[L] = np.bincount(sel.reshape(-1).astype(np.int64), minlength=n_experts).astype(np.float64)
    return out


def rank_from_trace(trace_stats_json: str, n_layers: int = 40, fallback_uniform: bool = True,
                    profile: str = "mixed") -> list[tuple]:
    """(layer, expert) ranked by frequency from tools/expert_stats.py coverage.json. Layers not in the
    trace get their experts appended in a round-robin so every layer has some residents.

    `profile` picks which slice of the traced corpus ranks the experts: "mixed" (the whole corpus,
    the default and what the coverage.json histogram is), "coding" or "general". The coding and
    general top-25% sets overlap by only 0.18-0.31 Jaccard, so the profile is a real lever on the
    hit rate of a workload that is all one kind.
    """
    ranked = []
    counts = {}
    if profile and profile != "mixed":
        counts = category_counts(trace_stats_json, profile)
    if not counts:
        try:
            d = json.load(open(trace_stats_json))
            for L, v in d["per_layer"].items():
                counts[int(L)] = np.array(v["counts"], dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
    known = sorted(counts)
    if known:
        # normalize per layer so a layer with more traced tokens is not favoured
        keys = []
        for L in known:
            c = counts[L] / counts[L].sum()
            keys += [(float(c[e]), L, e) for e in range(384)]
        keys.sort(reverse=True)
        ranked = [(L, e) for _, L, e in keys]
    missing = [L for L in range(n_layers) if L not in counts]
    if missing and fallback_uniform:
        # untraced layers: interleave a uniform share so the LRU can learn them
        share = [(L, e) for e in range(384) for L in missing]
        # interleave: after every traced key, one untraced key
        out = []
        it = iter(share)
        for k in ranked:
            out.append(k)
            try:
                out.append(next(it))
            except StopIteration:
                pass
        out += list(it)
        ranked = out
    return ranked
