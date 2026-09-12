"""
v41_engine.py -- the generation loop: chunked prefill, DSpark block drafting + verification with
rejection sampling, cache rollback, and the Engine API the server uses.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from engine import experts as EX  # noqa: E402
from engine.engram import EngramTable, make_hash_state  # noqa: E402
from engine.model import MAX_CHUNK, Caches, Model, Weights  # noqa: E402
import v41_ref as R  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), "[engine]", *a, flush=True)


# Per-phase host wall-clock timing of the decode loop (DSV41_STEP_TIMING=1, off by default: with
# the flag off nothing but one `if` per phase runs). Everything on this loop is host time -- the
# graphs are queued asynchronously, so a phase's number is the time the HOST spent in it, including
# any wait for GPU work queued earlier. That is the quantity that explains the gap between
# engine/profile_fast.py's step+draft and the tok/s the generator actually reaches.
STEP_TIMING = os.environ.get("DSV41_STEP_TIMING", "0") == "1"

# Lean step: keep the per-step host work off the critical path -- the greedy accept/reject decision
# is taken on the GPU and read back as ONE 7-element tensor instead of up to eleven separate
# device->host syncs, and the per-step `torch.tensor` / `torch.cat` / `torch.arange` allocations are
# replaced by preallocated buffers (see also DSV41_LEAN_STEP in engine/fastdecode.py, which is the
# same switch). DSV41_LEAN_STEP=0 restores the original per-position Python loop exactly.
# Sampling semantics are untouched: the temperature > 0 path is the ORIGINAL code, RNG draw for RNG
# draw, and the greedy path computes argmax of the same logits in the same order.
LEAN_STEP = os.environ.get("DSV41_LEAN_STEP", "1") == "1"


class StepPhases:
    """Accumulating per-phase timer, printed as a table over a whole run."""

    __slots__ = ("acc", "order", "steps", "_t", "t0")

    def __init__(self):
        self.acc, self.order, self.steps = {}, [], 0
        self.t0 = self._t = time.perf_counter()

    def start(self):
        self._t = time.perf_counter()

    def mark(self, name):
        t = time.perf_counter()
        if name not in self.acc:
            self.acc[name] = 0.0
            self.order.append(name)
        self.acc[name] += t - self._t
        self._t = t

    def table(self, notes=()) -> str:
        n = max(1, self.steps)
        total = sum(self.acc.values())
        wall = time.perf_counter() - self.t0
        rows = [f"  {'phase':<14} {'ms/step':>9} {'% of step':>10} {'total s':>9}"]
        for k in self.order:
            v = self.acc[k]
            rows.append(f"  {k:<14} {v / n * 1e3:9.3f} {v / total * 100:10.1f} {v:9.3f}")
        rows.append(f"  {'-- accounted':<14} {total / n * 1e3:9.3f} {100.0:10.1f} {total:9.3f}")
        rows.append(f"  {'-- loop wall':<14} {wall / n * 1e3:9.3f} {'':>10} {wall:9.3f}"
                    f"   ({n} steps)")
        for name, v in notes:
            rows.append(f"  {name:<14} {v / n * 1e3:9.3f} {'':>10} {v:9.3f}   (already inside a phase)")
        return "\n".join(rows)


class MemoryWatchdog:
    """Kill this process rather than let the host run out of memory.

    A 121 GiB box holding a ~95 GB pinned expert arena has no slack. If something -- a second
    engine, the transient scratch of the 3-bit packer, a large prompt -- pushes MemAvailable toward
    zero, the kernel does not OOM-kill cleanly: it thrashes, sshd can no longer fork, and the
    machine answers ping while being unusable until someone power-cycles it. That happened three
    times on 2026-09-11/12.

    Exiting immediately is strictly better than continuing: the kernel reclaims everything this
    process holds the moment it dies, and the box stays reachable. The watchdog samples
    /proc/meminfo twice a second from a daemon thread and calls os._exit, which is the only way out
    that does not itself need to allocate.
    """

    def __init__(self, floor_gb: float = 2.5, interval: float = 0.5, strikes: int = 6, log=print):
        # `strikes` consecutive samples below the floor before acting: the warm start dips briefly
        # while the packer's scratch is live and recovers on its own, and killing a configuration
        # that works is its own kind of failure. A real collision (a second engine, a runaway
        # allocation) stays below the floor and trips this within a few seconds.
        self.floor = floor_gb * 1e9
        self.strikes = strikes
        self.interval = interval
        self.log = log
        self._stop = threading.Event()
        self.low_water = None

    def start(self):
        if host_available_bytes() is None:
            return self  # not Linux, nothing to watch
        t = threading.Thread(target=self._run, name="mem-watchdog", daemon=True)
        t.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            avail = host_available_bytes()
            if avail is None:
                return
            if self.low_water is None or avail < self.low_water:
                self.low_water = avail
            if avail >= self.floor:
                self._below = 0
                continue
            self._below = getattr(self, "_below", 0) + 1
            if self._below >= self.strikes:
                try:
                    self.log(f"FATAL: host MemAvailable {avail / 1e9:.1f} GB stayed below the "
                             f"{self.floor / 1e9:.1f} GB floor for {self._below * self.interval:.1f} s; "
                             f"exiting now so the kernel can reclaim this process instead of "
                             f"thrashing the machine")
                    sys.stderr.flush()
                except Exception:  # noqa: BLE001 - never let logging stop the exit
                    pass
                os._exit(3)


def host_available_bytes():
    """MemAvailable from /proc/meminfo, or None off Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class FixedStore:
    """DSpark experts: 3 x 128, all resident, slot = k * 128 + e."""

    def __init__(self, arena):
        self.arena = arena
        self.stats = {"hits": 0, "misses": 0}

    def resolve(self, layer, experts, prefill):
        k = layer - 40
        return (experts.to(torch.int32) + k * 128)


class Penalties:
    """OpenAI presence/frequency penalties, plus a breaker for exact repetition cycles.

    The penalties are the standard ones (a token already generated loses
    `presence_penalty + frequency_penalty * count` from its logit) and default to 0, because a flat
    penalty hurts code, which is legitimately repetitive.

    The cycle breaker is separate and defaults ON. Greedy decoding on this checkpoint can fall into
    an exact repetition -- a run of one token, or a short block repeated verbatim -- and has no way
    out of it, because the same context keeps producing the same argmax (NOTES 2026-09-11). It fires
    only on an *exact* cycle of period <= `cycle_max_period` repeated `cycle_repeats` times, and
    then bans just the one token that would continue the cycle for that single step, which is enough
    to leave the attractor. Normal repetitive code never reaches an exact cycle of whole tokens
    repeated three times over, so nothing else is affected.
    """

    def __init__(self, presence: float = 0.0, frequency: float = 0.0, cycle_repeats: int = 4,
                 cycle_max_period: int = 16, enabled: bool | None = None,
                 no_repeat_ngram: int | None = None):
        self.presence = float(presence or 0.0)
        self.frequency = float(frequency or 0.0)
        self.cycle_repeats = int(cycle_repeats)
        self.cycle_max_period = int(cycle_max_period)
        if enabled is None:
            enabled = os.environ.get("DSV41_CYCLE_BREAK", "1") == "1"
        self.cycle_enabled = bool(enabled)
        self.no_repeat_ngram = int(no_repeat_ngram if no_repeat_ngram is not None
                                   else os.environ.get("DSV41_NO_REPEAT_NGRAM", "0"))
        self.counts: dict[int, int] = {}
        self.hits = 0
        self.ngram_hits = 0

    @property
    def active(self) -> bool:
        return (self.presence != 0.0 or self.frequency != 0.0 or self.cycle_enabled
                or self.no_repeat_ngram > 0)

    def observe(self, tokens) -> None:
        for t in tokens:
            self.counts[t] = self.counts.get(t, 0) + 1

    def _cycle_token(self, history) -> int | None:
        """The token that would continue an exact cycle, or None."""
        if not self.cycle_enabled or len(history) < max(4, self.cycle_repeats):
            return None
        for p in range(1, min(self.cycle_max_period, len(history) // self.cycle_repeats) + 1):
            block = history[-p:]
            if all(history[-p * (i + 1):len(history) - p * i] == block for i in range(1, self.cycle_repeats)):
                return block[0]
        return None

    def _banned_ngram_tokens(self, history) -> list:
        """Tokens that would repeat an n-gram of `no_repeat_ngram` already in `history`.

        Same rule as transformers' NoRepeatNGramLogitsProcessor, which is the reference for this
        behaviour: if the last n-1 tokens have appeared before, every token that followed them is
        refused. A degenerate loop repeats far more than n tokens, so it is cut on the second turn
        of the cycle, while ordinary text and code almost never repeat a whole n-gram verbatim.
        """
        n = self.no_repeat_ngram
        if not n or len(history) < n:
            return []
        prefix = tuple(history[-(n - 1):])
        banned = set()
        for i in range(len(history) - n + 1):
            if tuple(history[i:i + n - 1]) == prefix:
                banned.add(history[i + n - 1])
        return list(banned)

    def apply(self, logits: torch.Tensor, history) -> torch.Tensor:
        """logits [V] or [T, V] fp32 -> the same tensor, penalised in place."""
        if self.presence or self.frequency:
            idx = torch.tensor(list(self.counts), device=logits.device, dtype=torch.long)
            if idx.numel():
                cnt = torch.tensor([self.counts[int(i)] for i in idx.tolist()],
                                   device=logits.device, dtype=logits.dtype)
                pen = self.presence + self.frequency * cnt
                if logits.dim() == 1:
                    logits[idx] -= pen
                else:
                    logits[:, idx] -= pen
        ban = self._banned_ngram_tokens(history)
        if ban:
            idx = torch.tensor(ban, device=logits.device, dtype=torch.long)
            if logits.dim() == 1:
                logits[idx] = float("-inf")
            else:
                logits[:, idx] = float("-inf")
            self.ngram_hits += 1
        t = self._cycle_token(history)
        if t is not None:
            self.hits += 1
            if logits.dim() == 1:
                logits[t] = float("-inf")
            else:
                logits[:, t] = float("-inf")
        return logits


def sample_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """[V] fp32 logits -> probability vector (temperature + nucleus)."""
    if temperature <= 0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    p = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sp, si = p.sort(descending=True)
        cum = sp.cumsum(0)
        keep = cum - sp < top_p
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        p = torch.zeros_like(p).scatter_(0, si, sp)
        p = p / p.sum()
    return p


def build_keep_masks(counts: dict, frac: float, select: str, device, min_per_layer: int = 24):
    """Which experts stay routable under a budget of ceil(frac*384) per layer ON AVERAGE.

    select="uniform": the top ceil(frac*384) experts of every layer (the v0.2.0-wip scheme).
    select="global": one ranking of all (layer, expert) pairs by per-layer-normalized routing
    frequency, cut at the same total count; layers whose routing is flat get more experts and
    layers whose routing is skewed get fewer, with at least `min_per_layer` in every layer. Under
    a total budget this greedy choice maximizes the routing mass that stays routable.
    Returns (masks {L: bool[384] on device}, keep {L: np.ndarray of expert ids, hottest first}).
    """
    import math as _m
    n_keep = max(6, _m.ceil(frac * 384))
    total = n_keep * len(counts)
    keep = {}
    if select == "uniform":
        for L, c in counts.items():
            keep[int(L)] = np.argsort(np.asarray(c))[::-1][:n_keep]
    elif select == "global":
        keys = []
        for L, c in counts.items():
            c = np.asarray(c, dtype=np.float64); c = c / c.sum()
            order = np.argsort(c)[::-1]
            keep[int(L)] = [int(e) for e in order[:min_per_layer]]
            keys += [(float(c[e]), int(L), int(e)) for e in order[min_per_layer:]]
        keys.sort(reverse=True)
        remaining = total - sum(len(v) for v in keep.values())
        for _, L, e in keys[:remaining]:
            keep[L].append(e)
        keep = {L: np.array(v) for L, v in keep.items()}
    else:
        raise ValueError(f"unknown prune_select {select!r}")
    masks = {}
    for L, k in keep.items():
        m = torch.zeros(384, dtype=torch.bool, device=device)
        m[torch.as_tensor(np.array(k), device=device)] = True
        masks[L] = m
    return masks, keep


class V41Engine:
    #: this engine can constrain sampling with a decoding gate (``generate(grammar=...)``)
    supports_grammar = True

    def __init__(self, model_dir: str, max_seq: int = 32768, arena_gb: float | None = None, device: str = "cuda",
                 trace_stats: str | None = None, act_quant: bool = False, spec: bool = True, io_threads: int = 12,
                 transient_slots: int = 400, keep_free_gb: float = 20.0, swa_replay: bool | None = None,
                 hot_profile: str | None = None, prune_keep: float | None = None,
                 sim_bits: int | None = None, sim_cold_frac: float = 1.0, prune_select: str = "uniform",
                 expert_format: str = "fp4", sim_cb2_frac: float = 0.0,
                 expert_topics: str | None = None):
        self.model_dir = model_dir
        self.device = device
        self.spec = spec
        self.swa_replay = (os.environ.get("DSV41_SWA_REPLAY", "1") != "0") if swa_replay is None else swa_replay
        self.hot_profile = hot_profile or os.environ.get("DSV41_HOT_PROFILE", "mixed")
        self.lock = threading.Lock()
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.args = R.Args.from_json(os.path.join(model_dir, "inference", "config.json"))
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.eos_token_id = 1
        self.max_context = max_seq

        self.act_quant = act_quant
        self.trace_stats = trace_stats
        self.expert_format = (expert_format or "fp4").lower()
        assert self.expert_format in ("fp4", "cb3"), self.expert_format
        self.sim_cb2_frac = float(sim_cb2_frac or 0.0)
        assert 0.0 <= self.sim_cb2_frac <= 1.0, self.sim_cb2_frac
        try:
            import fp4_moe as K
            fp4_moe_fn, fp4_arena_cls = K.moe_forward, K.ExpertArena
            self.kernel = "triton-fp4"
            log("using Triton FP4 MoE kernel")
        except Exception as e:  # noqa: BLE001
            from engine import moe_fallback as K
            fp4_moe_fn, fp4_arena_cls = K.moe_forward, K.ExpertArena
            self.kernel = "dequant-fallback"
            log(f"Triton kernel unavailable ({e!r}); using the slow dequant fallback")
        cb3_cls = None
        if self.expert_format == "cb3":
            import cb3_moe as C3
            from engine.codebook_sim import CodebookSim
            assert not sim_bits, "--sim-bits simulates a low-bit format inside an FP4 arena; it is " \
                                 "meaningless with --expert-format cb3, which IS the packed format"
            cb3_cls = C3.CB3ArenaV2
            self._cb3_sim = CodebookSim(3, device)
            cb3_moe_fn = C3.moe_forward_v3
            self.kernel = "triton-cb3"
            log("using Triton CB3 (3-bit per-row codebook) MoE kernel for the routed experts")

        def moe_fn(x, slots, weights, arena, limit):
            """Dispatch on the arena's format. The DSpark draft arena stays FP4 whatever the main
            arena is -- it is 384 experts (7.2 GB), it is read five times per step, and a 3-bit
            drafter would cost acceptance for nothing."""
            if cb3_cls is not None and isinstance(arena, cb3_cls):
                return cb3_moe_fn(x, slots, weights, arena, limit)
            return fp4_moe_fn(x, slots, weights, arena, limit)

        self.moe_fn = moe_fn
        arena_cls = fp4_arena_cls

        def make_expert_arena(n_slots):
            if cb3_cls is None:
                return fp4_arena_cls(n_slots, device)
            a = cb3_cls(n_slots, device)
            a.sim = self._cb3_sim
            return a

        self.expert_bytes = (EX.EXPERT_BYTES if cb3_cls is None
                             else __import__("cb3_moe").CB3_BYTES_PER_SLOT)

        self.W = Weights(model_dir, index, self.args, device, log=log, act_quant=act_quant)
        if R.dense_fp4_groups():
            # the load-time fp8 -> fp4 re-quantization leaves ~2.4 GB of fp32 scratch blocks in the
            # caching allocator; hand them back before the expert arena asks for its 90 GB.
            torch.cuda.empty_cache()
            log(f"dense fp4 groups: {','.join(sorted(R.dense_fp4_groups()))}; "
                f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated after weights")
        self.caches = Caches(self.args, max_seq, device)
        # DSpark experts: all resident
        self.W.dspark_arena = arena_cls(384, device)
        self.W.dspark_store = FixedStore(self.W.dspark_arena)
        # main expert arena: size from what is left.
        # On GB10 the GPU and the host share one pool, and `torch.cuda.mem_get_info()` counts the
        # *page cache* as used -- after a 470 GB download it reports 30 GB free on a box with 118 GiB
        # actually available, which silently gives a 23 GB arena. /proc/meminfo's MemAvailable is the
        # honest number (it counts reclaimable page cache), so take the larger of the two and keep a
        # hard floor of `keep_free_gb` under it: this box hard-resets if MemAvailable goes negative.
        free, total = torch.cuda.mem_get_info()
        host_avail = host_available_bytes()
        budget = float(max(free, host_avail or 0))
        # A prefill chunk is the largest transient this process ever holds; the indexer's
        # score tiles grow with the compressed cache, hence the max_seq term.
        reserve = MAX_CHUNK * 5e6 + 0.25e9 * (max_seq / 8192)
        auto = arena_gb is None
        if auto:
            arena_gb = max(10.0, (budget - reserve) / 1e9 * 0.82)
        if host_avail is not None:
            cap = (host_avail - keep_free_gb * 1e9) / 1e9
            if cap < 10.0:
                # Refuse rather than squeeze. A cap this low means the host has no memory to give:
                # another engine is still resident, or the kernel has not finished reclaiming the
                # last one's arena (it is page-cache backed and released lazily). Loading anyway is
                # how this box gets wedged -- it answers ping and accepts TCP on 22 while sshd can
                # no longer fork, and only a power cycle brings it back.
                raise RuntimeError(
                    f"not enough host memory to start: MemAvailable {(host_avail or 0) / 1e9:.1f} GB "
                    f"leaves {cap:.1f} GB for the expert arena after keep_free {keep_free_gb} GB. "
                    f"Wait for the previous engine's memory to be reclaimed (watch MemAvailable in "
                    f"/proc/meminfo) or lower --arena-gb/KEEP_FREE_GB.")
            if arena_gb > cap:
                log(f"arena {arena_gb:.1f} GB capped to {cap:.1f} GB (MemAvailable {host_avail / 1e9:.1f} GB, "
                    f"keep_free {keep_free_gb} GB)")
                arena_gb = max(10.0, cap)
            # A pinned arena is checked against what is free *now*, but the warm start needs more on
            # top of it: the 3-bit packer works on the GPU in scratch buffers, and a chunked prefill
            # allocates activations. Reserve that too, and refuse up front rather than discovering it
            # halfway through a 3-minute warm start with 0.2 GB left.
            # This check runs after the dense weights are resident, so MemAvailable already
            # excludes them: what is still to come is the arena itself plus the transient scratch of
            # the warm start (the 3-bit packer works in GPU buffers). Calibrated against what this
            # box actually does -- a 98 GB arena loads with ~110 GB available and settles with 6 GB
            # free -- rather than a pessimistic sum, because refusing a configuration that works is
            # its own failure. The MemoryWatchdog below is the backstop if this estimate is wrong.
            # What is still to come once the arena is full: a prefill chunk. At ~5 MB per
            # prefill token this engine needs about 10 GB at the default 2,048-token chunk,
            # and `keep_free_gb` defaults to less than that -- so the old check could pass a
            # configuration that loaded, logged `ready`, and was killed by the watchdog on the
            # first request. Measured 2026-09-12 on this box, MemAvailable 111.0 GB with the
            # dense weights resident: a 98 GB arena left 5.5 GB and died on request one; an
            # 87 GB arena left 16.5 GB and served. Reserve the larger of the two floors.
            pack_scratch = 3e9 if self.expert_format != "fp4" else 1e9
            prefill_reserve = MAX_CHUNK * 5e6
            floor = max(keep_free_gb * 1e9, prefill_reserve)
            need = arena_gb * 1e9 + pack_scratch + floor
            if need > host_avail:
                raise RuntimeError(
                    f"refusing to start: arena {arena_gb:.1f} GB + {pack_scratch / 1e9:.1f} GB "
                    f"warm-start scratch + {floor / 1e9:.1f} GB floor "
                    f"({'a prefill chunk' if prefill_reserve > keep_free_gb * 1e9 else 'keep_free'}) "
                    f"= {need / 1e9:.1f} GB, but MemAvailable is {host_avail / 1e9:.1f} GB. "
                    f"Lower --arena-gb by at least {(need - host_avail) / 1e9:.1f} GB (./tune.sh "
                    f"shows what fits), lower DSV41_PREFILL_CHUNK, or stop whatever else holds "
                    f"memory (on this box a boot-time vLLM container used to take 85 GB).")
        slots = int(arena_gb * 1e9 / self.expert_bytes)
        self.arena_gb, self.slots = round(arena_gb, 1), slots
        log(f"CUDA free {free / 1e9:.1f} GB of {total / 1e9:.1f}; host MemAvailable "
            f"{(host_avail or 0) / 1e9:.1f} GB; arena {arena_gb:.1f} GB = {slots} {self.expert_format} "
            f"expert slots of {self.expert_bytes / 1e6:.2f} MB "
            f"({slots / 15360 * 100:.0f}% of all routed experts, {'auto' if auto else 'pinned'})")
        # from here on, anything that drives the host out of memory kills this process instead of
        # the machine (see MemoryWatchdog)
        self.mem_watchdog = MemoryWatchdog(floor_gb=float(os.environ.get("DSV41_MEM_FLOOR_GB", "2.5")),
                                           log=log).start()
        self.arena = make_expert_arena(slots)
        self.store = EX.ExpertStore(model_dir, index, self.arena, self.args.n_layers, transient_slots=transient_slots,
                                    io_threads=io_threads)
        # load the DSpark experts (all 3 x 128 resident in their own arena)
        for k in range(3):
            for e in range(128):
                w1, s1, w2, s2, w3, s3 = self.store.read_expert(40 + k, e, prefix=f"mtp.{k}.ffn.experts.{e}.")
                self.W.dspark_arena.load_slot(k * 128 + e, w1.view(*EX.W13_SHAPE), s1.view(*EX.S13_SHAPE),
                                              w2.view(*EX.W2_SHAPE), s2.view(*EX.S2_SHAPE), w3.view(*EX.W13_SHAPE),
                                              s3.view(*EX.S13_SHAPE))
        log("DSpark experts resident")

        self.model = Model(self.W, self.store, self.caches, self.moe_fn, act_quant=act_quant)
        self.model.hash_state = make_hash_state(model_dir, self.tokenizer, max_seq, device)
        self.tables = {L: EngramTable(model_dir, index, L, device) for L in self.args.engram_layer_ids}
        from concurrent.futures import ThreadPoolExecutor as _TPE
        self.eg_pool = _TPE(len(self.args.engram_layer_ids))
        self.model.engram_rows = lambda L, h: self.tables[L].rows(h)
        self.prune_keep = prune_keep
        if prune_keep and prune_keep < 1.0:
            # pruned, all-resident mode: per layer only the top-N experts (by trace frequency) stay
            # routable, and exactly those are warm-started, so decode never touches NVMe
            # Which topics the resident set must serve. Each topic is one per-layer expert
            # histogram measured on a corpus of that topic alone; the keep-set is the top-N of their
            # per-layer-normalised sum, so composing topics is arithmetic and needs no new trace.
            # Normalising per topic before summing is the point: it gives a topic that contributed
            # few tokens the same vote as one that contributed many, which is what "serve this
            # workload too" means. Unset = every topic present in the coverage file.
            topics = [t.strip() for t in (expert_topics or os.environ.get("DSV41_EXPERT_TOPICS", "")).split(",") if t.strip()]
            if not topics:
                topics = EX.available_topics(trace_stats)
            per = {}
            for t in topics:
                c = EX.category_counts(trace_stats, t)
                if len(c) != 40:
                    raise ValueError(f"topic {t!r} is not in {trace_stats} (have: {', '.join(EX.available_topics(trace_stats))})")
                per[t] = c
            if not per:
                raise ValueError(f"no topics found in {trace_stats}")
            self.expert_topics_used = list(per)
            log(f"keep-set topics: {', '.join(per)}")
            # How the two workloads are combined into one ranking. "sum" (the 0.2.0-wip default)
            # adds the per-layer-normalized frequencies, which favours experts moderately used by
            # both and drops each workload's specialists -- the coding and prose top sets overlap by
            # a Jaccard of only 0.18-0.31, so that is most of the routing. Prose suffered for it:
            # at keep 31 % a story prompt degenerated into a repeated phrase while the same budget
            # ranked on prose alone wrote clean text (NOTES 2026-09-12). "max" keeps an expert that
            # matters to EITHER workload, which is what a general-purpose server needs.
            rank = os.environ.get("DSV41_PRUNE_RANK", "sum")  # "max" was measured worse (NOTES 2026-09-12)
            def _norm(c, L):
                s_ = c[L].sum()
                return c[L] / s_ if s_ > 0 else c[L]
            if rank == "sum":
                counts = {L: sum(_norm(c, L) for c in per.values()) for L in range(40)}
            elif rank == "max":
                counts = {L: np.maximum.reduce([_norm(c, L) for c in per.values()]) for L in range(40)}
            else:
                raise ValueError(f"unknown DSV41_PRUNE_RANK {rank!r} (max | sum)")
            self.prune_select = prune_select
            masks, keep = build_keep_masks(counts, prune_keep, prune_select, device)
            n_keep = max(len(v) for v in keep.values())
            ranked = [(float(counts[L][e]), L, int(e)) for L, ks in keep.items() for e in ks]
            ranked.sort(reverse=True)
            ranked = [(L, e) for _, L, e in ranked]
            self.model_prune_mask = masks
            per_layer = [len(keep[L]) for L in range(40)]
            if sim_bits:
                # simulated low-bit format on the coldest `sim_cold_frac` of the KEPT experts of every layer
                from engine.codebook_sim import CodebookSim
                self.store.requant_sims = {sim_bits: CodebookSim(sim_bits, device)}
                pol = {}
                for L, ks in keep.items():
                    n_cold = int(round(sim_cold_frac * len(ks)))
                    for e in ks[len(ks) - n_cold:]:
                        pol[(L, int(e))] = sim_bits
                self.store.requant = pol
                log(f"simulated {sim_bits}-bit codebook format on {len(pol)} of {len(ranked)} kept experts "
                    f"(coldest {sim_cold_frac:.0%} per layer)")
            if self.sim_cb2_frac > 0:
                # Quality-only simulation of a 2-bit tier INSIDE the CB3 arena: the coldest
                # `sim_cb2_frac` of every layer's kept set get a 4-of-16 row codebook repeated up to
                # the slot's 8 entries, so the slot's bytes and the CB3 kernel are untouched and the
                # arithmetic is a real 2-bit format's. Nothing is saved -- what it answers is whether
                # saving it would be worth building.
                assert self.expert_format == "cb3", "--sim-cb2-frac needs --expert-format cb3"
                from engine.codebook_sim import CodebookSim
                self.store.cb_sims = {2: CodebookSim(2, device)}
                pol2 = {}
                for L, ks in keep.items():
                    n_cold = int(round(self.sim_cb2_frac * len(ks)))
                    for e in ks[len(ks) - n_cold:]:
                        pol2[(L, int(e))] = 2
                self.store.cb_bits = pol2
                log(f"simulated 2-bit codebook on {len(pol2)} of {len(ranked)} kept experts "
                    f"(coldest {self.sim_cb2_frac:.0%} per layer); slot size unchanged")
            if len(ranked) > self.store.lru_slots:
                log(f"WARNING: pruned set {len(ranked)} experts > {self.store.lru_slots} LRU slots; the tail will stream")
            log(f"pruned mode ({prune_select}): keep {prune_keep:.2f}, {len(ranked)} experts total "
                f"({min(per_layer)}-{max(per_layer)} per layer), {len(ranked) * self.expert_bytes / 1e9:.1f} GB")
        else:
            self.model_prune_mask = None
            ranked = (EX.rank_from_trace(trace_stats, profile=self.hot_profile) if trace_stats
                      else [(L, e) for e in range(384) for L in range(40)])
        self.model.prune_mask = self.model_prune_mask
        self.store.warm_start(ranked, log=log)
        self.fast = None
        if spec and os.environ.get("DSV41_FAST", "1") == "1":
            from engine.fastdecode import FastDecoder
            self.fast = FastDecoder(self.model, self, use_graphs=os.environ.get("DSV41_GRAPHS", "1") == "1")
            resident = (self.prune_keep and self.prune_keep < 1.0 and
                        len(self.store.lru) >= 40 * max(6, int(np.ceil(self.prune_keep * 384))) and
                        os.environ.get("DSV41_LUT", "1") == "1")
            if resident:
                self.fast.build_lut()
                # prefill routes through Model.moe, which takes the same table when it is there
                self.model.slot_lut = self.fast.lut
            log("fast decode path enabled (CUDA graphs=%s, device slot LUT=%s)" % (self.fast.use_graphs, self.fast.lut is not None))
        # preallocated staging for the lean decode step (see LEAN_STEP): the verify block, the
        # [n_accepted, argmax x 6] readback and its pinned host landing buffer.
        from engine.fastdecode import T_VERIFY as _TV
        self._tv = _TV
        self._blk = torch.empty(_TV, dtype=torch.long, device=device)
        self._vout = torch.empty(_TV + 1, dtype=torch.long, device=device)
        self._vhost = torch.empty(_TV + 1, dtype=torch.long).pin_memory()
        torch.cuda.synchronize()
        self.last_stats = {}
        log("ready")

    # ------------------------------------------------------------------ generation
    def _reset(self):
        c = self.caches
        c.len = 0
        c._chunk_inputs.clear()
        for L in c.pending:
            c.pending[L] = None
        self.model.stats = {"attn_s": 0.0, "moe_s": 0.0, "engram_s": 0.0, "tokens": 0}
        for t in self.tables.values():
            t.stats = {"rows": 0, "seconds": 0.0, "calls": 0}
        self.store.stats.update(EX.ZERO_STATS)

    def generate(self, prompt_ids, *, max_tokens=4096, temperature=1.0, top_p=0.95, stop_token_ids=None, seed=None,
                 penalties=None,
                 ignore_eos=False, grammar=None):
        """Yield bursts of new token ids.

        ``ignore_eos``: keep decoding until ``max_tokens`` even if EOS/stop ids come up. The
        benchmark needs runs of a fixed output length -- otherwise a decode-rate comparison is
        really a comparison of how early each config decided to stop.

        ``grammar``: optional decoding gate (see server/engine_api.py). It is shown every token
        this loop settles on (``observe``) and is asked to mask the verify block's logit rows
        (``mask_rows``) before the accept/reject decision. It masks nothing until the model starts
        a tool-calls block, and with ``grammar=None`` not one instruction of this loop changes.
        """
        with self.lock:
            yield from self._generate(list(prompt_ids), max_tokens, temperature, top_p, set(stop_token_ids or ()),
                                      seed, ignore_eos, grammar, penalties)

    def _generate(self, prompt, max_tokens, temperature, top_p, stop_ids, seed, ignore_eos=False, grammar=None,
                  penalties=None):
        if seed is not None:
            torch.manual_seed(seed)
        stop_ids = set() if ignore_eos else (set(stop_ids) | {self.eos_token_id})
        self._reset()
        m = self.model
        P = len(prompt)
        assert P + max_tokens + 8 <= self.max_context, f"prompt {P} + max_tokens {max_tokens} > context {self.max_context}"
        ids = torch.tensor(prompt, dtype=torch.long, device=self.device)
        t_start = time.perf_counter()
        # Every counter the stats epilogue reads is initialised here, because the epilogue runs in a
        # `finally`: the server closes the generator on a stop string or a client disconnect, which
        # raises GeneratorExit at the pending `yield`, and an aborted request must still report the
        # work it did instead of the previous request's numbers.
        n_out = 0
        steps = 0
        pos = P
        accepted_hist = []
        t_prefill = 0.0
        t_decode0 = t_start
        try:
            yield from self._decode_loop(ids, P, max_tokens, temperature, top_p, stop_ids,
                                         _st := {}, grammar, penalties)
        finally:
            n_out = _st.get("n_out", n_out)
            steps = _st.get("steps", steps)
            accepted_hist = _st.get("accepted", accepted_hist)
            t_prefill = _st.get("t_prefill", t_prefill)
            t_dec = max(time.perf_counter() - _st.get("t_decode0", t_start), 1e-9)
            _ph = _st.get("phases")
            if _ph is not None and _ph.steps:
                _notes = []
                _fd0 = _st.get("fd0")
                if _fd0 is not None and self.fast is not None:
                    _notes = [("of it: engram", self.fast.stats["engram_s"] - _fd0["engram_s"]),
                              ("of it: graphs", self.fast.stats["graph_s"] - _fd0["graph_s"]),
                              ("of it: draft g", self.fast.stats["draft_s"] - _fd0["draft_s"])]
                print(f"[step timing] host wall clock per decode step, {_ph.steps} steps "
                      f"(DSV41_STEP_TIMING=1)\n{_ph.table(_notes)}", flush=True)
            st = self.store.stats
            m = self.model
            self.last_stats = {
                "prompt_tokens": P, "completion_tokens": n_out, "prefill_s": round(t_prefill, 3),
                "prefill_tok_s": round(P / t_prefill, 2) if t_prefill > 0 else None,
                "decode_s": round(t_dec, 3), "decode_tok_s": round(max(n_out - 1, 0) / t_dec, 2),
                "steps": steps,
                "accept_len_mean": round(float(np.mean(accepted_hist)) + 1, 2) if accepted_hist else None,
                "expert_hit_rate": round(self.store.hit_rate(), 4), "expert_misses": st["misses"],
                "prefill_expert_misses": st["prefill_misses"], "nvme_gb": round(st["bytes_read"] / 1e9, 2),
                "nvme_read_s": round(st["read_s"], 2),
                "engram_rows": sum(t.stats["rows"] for t in self.tables.values()),
                "engram_s": round(sum(t.stats["seconds"] for t in self.tables.values()), 3),
            "engram_read_s": round(sum(t.stats.get("read_s", 0.0) for t in self.tables.values()), 3),
                "attn_s": round(m.stats["attn_s"], 2), "moe_s": round(m.stats["moe_s"], 2),
                # where the MoE time actually goes: routing+slot bookkeeping, waiting for the
                # NVMe loads of this layer, and the Triton kernel itself (moe_s minus the rest).
                "route_s": round(st["route_s"], 2), "load_wait_s": round(st["load_s"], 2),
                "lease_s": round(st["lease_s"], 2), "h2d_s": round(st["h2d_s"], 2),
                "kernel_s": round(m.stats["moe_s"] - st["resolve_s"], 2),
                "nvme_gb_per_token": round(st["bytes_read"] / 1e9 / max(n_out, 1), 3),
                "promoted": st["promoted"],
            }

    def _decode_loop(self, ids, P, max_tokens, temperature, top_p, stop_ids, out_st, grammar=None, penalties=None):
        m = self.model
        t_start = time.perf_counter()
        out_st["t_decode0"] = t_start
        # prefill in chunks
        logits = None
        m.begin_prompt()
        if self.swa_replay:
            # CED + Decoder SWA Bounded Replay: the prompt runs through the encoder half only
            # (layers 0..20, which is everything that writes global KV), and the decoder half is
            # replayed once over the last `window_size` prompt tokens.
            for s in range(0, P, MAX_CHUNK):
                m.forward(ids[s:s + MAX_CHUNK], s, prefill=True, need_logits=False, encoder_only=True)
            logits, mh, s_rep = m.decoder_replay(need_logits=True)
            if self.spec:
                m.dspark_seed(mh, s_rep)
        else:
            for s in range(0, P, MAX_CHUNK):
                chunk = ids[s:s + MAX_CHUNK]
                last = s + len(chunk) >= P
                logits, mh = m.forward(chunk, s, prefill=True, need_logits=last)
                if self.spec:
                    m.dspark_seed(mh, s)
        t_prefill = time.perf_counter() - t_start
        out_st["t_prefill"] = t_prefill
        pen = penalties if (penalties is not None and penalties.active) else None
        p = sample_probs(logits[-1], temperature, top_p)
        tok = int(torch.multinomial(p, 1)) if temperature > 0 else int(p.argmax())
        out = [tok]
        n_out = 1
        pos = P  # position of `tok` (not yet forwarded)
        accepted_hist = []
        ph = StepPhases() if STEP_TIMING else None
        if ph is not None and self.fast is not None:
            # the engram row wait and the graph queueing both live inside the "step" phase; record
            # where they started so the table can break that phase down
            out_st["fd0"] = dict(self.fast.stats)
        out_st.update(n_out=1, steps=0, accepted=accepted_hist, t_decode0=time.perf_counter(), phases=ph)
        steps = 0
        # The gate cannot be constraining anything yet (it engages on a marker that takes several
        # tokens to write), but it has to see every settled token to stay in step with the stream.
        if grammar is not None:
            grammar.observe([tok])
        yield [tok]
        while n_out < max_tokens and tok not in stop_ids:
            if ph is not None:
                ph.start()
            if self.spec:
                # the lean path applies to greedy decoding only; temperature > 0 keeps the original
                # sequential rejection-sampling loop so its RNG stream is bit-for-bit unchanged
                lean = LEAN_STEP and self.fast is not None and temperature <= 0
                if self.fast is not None:
                    drafts, q = self.fast.draft(tok, pos - 1, temperature)
                    if not lean:
                        # the drafter's static buffers survive until the next draft() call, which is
                        # after this step's verification, so the lean path does not need the copies
                        drafts = drafts.clone(); q = q.clone()
                    if ph is not None:
                        ph.mark("draft")
                    if lean:
                        # no H2D and no allocation: fill_ bakes the token into the kernel argument
                        block = self._blk
                        block[0].fill_(tok)
                        block[1:].copy_(drafts)
                    else:
                        block = torch.cat([torch.tensor([tok], device=self.device), drafts])
                    if ph is not None:
                        ph.mark("block")
                    hashes = m.hash_state(block[None], pos)[0]
                    if ph is not None:
                        ph.mark("hash")
                    # both tables' rows are read in background threads (NVMe only, no CUDA calls there) and
                    # each is dequantized on the main thread when its layer needs it. The hash ids go to the
                    # host HERE, before any graph is queued: a .cpu() later would wait for the whole step.
                    h_np = hashes.cpu().numpy()
                    if ph is not None:
                        ph.mark("hash_d2h")
                    futs = {L: (self.eg_pool.submit(self.tables[L].read_raw, h_np[:, li, :]), self.tables[L].to_device)
                            for li, L in enumerate(self.args.engram_layer_ids)}
                    if ph is not None:
                        ph.mark("submit")
                    logits, mh = self.fast.step(block, pos, lambda: futs)
                    if ph is not None:
                        ph.mark("step")
                else:
                    drafts, q, conf = m.dspark_draft(tok, pos - 1, temperature)
                    block = torch.cat([torch.tensor([tok], device=self.device), drafts])  # T_VERIFY tokens at pos..
                    logits, mh = m.forward(block, pos, prefill=False)
                # Constrained decoding, before anything reads the logits. Row i is the
                # distribution after block[0..i], so its legal set depends on the drafts accepted
                # ahead of it; the gate walks the block token by token and leaves its own state
                # where it found it. Rows after the first illegal draft are left alone: that draft
                # is masked out of its own row, so it is rejected there and nothing reads further.
                if grammar is not None:
                    grammar.mask_rows(logits, block)
                    if ph is not None:
                        ph.mark("grammar")
                if pen is not None:
                    pen.apply(logits, out)
                # verify drafts[i] (position pos+1+i) against logits[i]
                if lean:
                    # Greedy verification, entirely on the GPU: argmax of the six logit rows, the
                    # five draft comparisons, and the number of LEADING accepts (cumprod-then-sum,
                    # which is the index of the first mismatch). The bonus token is argmax[a] for
                    # every a, including a = 5 -- exactly what the loop below computes. One 7-wide
                    # D2H is the only host sync of the whole step.
                    am = logits.argmax(-1)                                   # [6]
                    acc = am[:self._tv - 1].eq(drafts).to(torch.int32).cumprod(0)  # 1 while still accepting
                    self._vout[0] = acc.sum()
                    self._vout[1:].copy_(am)
                    self._vhost.copy_(self._vout)
                    v = self._vhost.tolist()
                    a, cand = v[0], v[1:]
                    new = cand[:a]
                    bonus = cand[a]
                    for j, t in enumerate(new):   # an accepted stop token ends the block, no bonus
                        if t in stop_ids:
                            a, new, bonus = j + 1, new[:j + 1], None
                            break
                    if ph is not None:
                        ph.mark("verify")
                    m.c.rollback(pos + a + 1)
                    if ph is not None:
                        ph.mark("rollback")
                    accepted_hist.append(a)
                    emitted = list(new)
                    if bonus is not None:
                        emitted.append(bonus)
                    pos = pos + a + 1
                    tok = emitted[-1] if emitted else tok
                    if emitted:
                        if pen is not None:
                            pen.observe(emitted)
                        out += emitted
                        n_out += len(emitted)
                        out_st["n_out"] = n_out
                        if grammar is not None:
                            grammar.observe(emitted)
                        if ph is not None:
                            ph.mark("emit")
                        yield emitted
                        if ph is not None:
                            ph.mark("consumer")
                        if any(t in stop_ids for t in emitted):
                            break
                    steps += 1
                    out_st["steps"] = steps
                    if ph is not None:
                        ph.steps = steps
                    continue
                a = 0
                new = []
                bonus = None
                for i in range(5):
                    pt = sample_probs(logits[i], temperature, top_p)
                    d = int(drafts[i])
                    if temperature <= 0:
                        ok = int(pt.argmax()) == d
                    else:
                        r = torch.rand((), device=self.device)
                        ok = bool(r < (pt[d] / q[i][d].clamp_min(1e-20)).clamp(max=1.0))
                    if ok:
                        a += 1
                        new.append(d)
                        if d in stop_ids:
                            break
                    else:
                        if temperature <= 0:
                            bonus = int(pt.argmax())
                        else:
                            resid = (pt - q[i]).clamp_min(0)
                            if float(resid.sum()) <= 0:
                                resid = pt
                            bonus = int(torch.multinomial(resid / resid.sum(), 1))
                        break
                if bonus is None and not (new and new[-1] in stop_ids):
                    pt = sample_probs(logits[a] if a < 5 else logits[5], temperature, top_p)
                    bonus = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())
                if ph is not None:
                    ph.mark("verify")
                # caches valid for positions < pos + a + 1 (tok + accepted drafts)
                m.c.rollback(pos + a + 1)
                if self.fast is None:
                    m.dspark_seed(mh[:a + 1], pos)
                if ph is not None:
                    ph.mark("rollback")
                accepted_hist.append(a)
                emitted = list(new)
                if bonus is not None:
                    emitted.append(bonus)
                pos = pos + a + 1
                tok = emitted[-1] if emitted else tok
                if emitted:
                    if pen is not None:
                        pen.observe(emitted)
                    out += emitted
                    n_out += len(emitted)
                    out_st["n_out"] = n_out
                    if grammar is not None:
                        grammar.observe(emitted)
                    if ph is not None:
                        ph.mark("emit")
                    yield emitted
                    if ph is not None:
                        ph.mark("consumer")
                    if any(t in stop_ids for t in emitted):
                        break
                steps += 1
                out_st["steps"] = steps
                if ph is not None:
                    ph.steps = steps
            else:
                logits, mh = m.forward(torch.tensor([tok], device=self.device), pos, prefill=False)
                if grammar is not None:
                    grammar.mask_rows(logits, None)
                if pen is not None:
                    pen.apply(logits, out)
                pt = sample_probs(logits[0], temperature, top_p)
                tok = int(torch.multinomial(pt, 1)) if temperature > 0 else int(pt.argmax())
                pos += 1
                if pen is not None:
                    pen.observe([tok])
                out.append(tok)
                n_out += 1
                steps += 1
                out_st.update(n_out=n_out, steps=steps)
                if grammar is not None:
                    grammar.observe([tok])
                yield [tok]
        out_st.update(n_out=n_out, steps=steps)

    # ------------------------------------------------------------------ introspection
    supports_penalties = True

    def config(self):
        """Static engine configuration -- everything a measured number has to be quoted with."""
        return {
            "engine": "v41",
            "arena_gb": self.arena_gb,
            "arena_slots": self.slots,
            "lru_slots": self.store.lru_slots,
            "transient_slots": self.store.transient_slots,
            "resident_expert_pct": round(self.slots / (self.args.n_layers * self.args.n_routed_experts) * 100, 1),
            "max_seq": self.max_context,
            "spec": self.spec,
            "trace_stats": self.trace_stats,
            "kernel": self.kernel,
            "expert_format": self.expert_format,
            "expert_mb": round(self.expert_bytes / 1e6, 2),
            "dense_fp4": ",".join(sorted(R.dense_fp4_groups())) or "off",
            "head_fmt": R.head_fmt(),
            "routed_topk": self.args.n_activated_experts,
            "sim_cb2_frac": self.sim_cb2_frac,
            "act_quant": self.act_quant,
            "swa_replay": self.swa_replay,
            "prune_keep": self.prune_keep,
            "prune_select": getattr(self, "prune_select", None),
            "expert_topics": getattr(self, "expert_topics_used", None),
            "hot_profile": self.hot_profile,
            "prefill_chunk": MAX_CHUNK,
            "io_threads": self.store.io_threads,
            "read_threads": self.store.read_threads,
            "read_chunk_mb": round(self.store.read_chunk / 1024 / 1024, 2),
        }

    def stats(self):
        return {**self.config(), **self.last_stats}

    def close(self):
        for pool in [self.store.pool, self.store.read_pool] + [t.pool for t in self.tables.values()]:
            try:
                pool.shutdown(wait=False)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ replay verification
    @torch.inference_mode()
    def verify_replay(self, texts: list[str]):
        """Full 40-layer prefill vs CED encoder + Decoder SWA Bounded Replay, same prompt, one load.

        For a prompt of at most `window_size` tokens the replay covers the whole prompt and the two
        paths are the *same* arithmetic in the same order, so the logits must agree to the last bit;
        anything else means the state carried across the split (hidden states, the shifted HC pre-mix,
        layer 20's top-k and candidate pool) is not what the fused pass had. For a longer prompt the
        replay is an approximation by construction -- the decoder layers see a 128-token window
        instead of the full prefix -- and what matters is that the next-token distribution barely
        moves.
        """
        m = self.model
        out = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            P = len(ids)
            t = torch.tensor(ids, dtype=torch.long, device=self.device)

            def full():
                self._reset(); m.begin_prompt()
                lg, _ = m.forward(t, 0, prefill=True, need_logits=True)
                return lg[-1].float()

            def replay():
                self._reset(); m.begin_prompt()
                for s in range(0, P, MAX_CHUNK):
                    m.forward(t[s:s + MAX_CHUNK], s, prefill=True, need_logits=False, encoder_only=True)
                lg, _, _ = m.decoder_replay(need_logits=True)
                return lg[-1].float()

            a_, b_ = full(), replay()
            la, lb = torch.log_softmax(a_, -1), torch.log_softmax(b_, -1)
            kl = float((la.exp() * (la - lb)).sum())
            r = {"tokens": P, "replayed": min(P, self.args.window_size),
                 "exact_expected": P <= self.args.window_size,
                 "max_abs_logit_delta": round(float((a_ - b_).abs().max()), 6),
                 "top1_full": int(a_.argmax()), "top1_replay": int(b_.argmax()),
                 "kl_nats": round(kl, 6),
                 "nll_delta_on_full_top1": round(float(la[a_.argmax()] - lb[a_.argmax()]), 6)}
            log(json.dumps(r))
            out.append(r)
        return out

    # ------------------------------------------------------------------ teacher forcing
    @torch.inference_mode()
    def teacher_forced(self, corpus_path: str, max_len: int = 512, replay: bool = False, tail: bool = False):
        """Run every corpus sequence through `Model.forward` in ONE chunk and report mean NLL and
        top-1 next-token accuracy per category -- the same measurement
        `tools/expert_trace.py` makes with the pure-torch tracer, so the two are directly
        comparable and any drift between `engine/model.py` and `tools/v41_ref.py` shows up here.

        ``replay`` prefills through the CED encoder + Decoder SWA Bounded Replay path instead, which
        only produces logits for the last `window_size` positions; ``tail`` scores only those
        positions in either mode, so the two are measured on exactly the same predictions."""
        res = {}
        per_seq = []
        W = self.args.window_size
        for line in open(corpus_path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids = self.tokenizer.encode(d["text"], add_special_tokens=False)
            assert len(ids) <= max_len, (d["id"], len(ids))
            self._reset()
            self.model.begin_prompt()
            t0 = time.perf_counter()
            P = len(ids)
            t = torch.tensor(ids, dtype=torch.long, device=self.device)
            if replay:
                for s0 in range(0, P, MAX_CHUNK):
                    self.model.forward(t[s0:s0 + MAX_CHUNK], s0, prefill=True, need_logits=False,
                                       encoder_only=True)
                logits, _, off = self.model.decoder_replay(need_logits=True)
            else:
                logits, _ = self.model.forward(t, 0, prefill=True, need_logits=True)
                off = 0
            if tail:
                keep = max(0, P - W) - off  # first scored row inside `logits`
                if keep > 0:
                    logits, off = logits[keep:], off + keep
            tgt = torch.tensor(ids[off + 1:], device=self.device)
            lp = torch.log_softmax(logits[:-1].float(), dim=-1)
            nll = -lp.gather(1, tgt[:, None]).squeeze(1)
            top1 = (logits[:-1].argmax(-1) == tgt).float()
            c = res.setdefault(d["category"], {"nll": [], "top1": []})
            c["nll"].append(nll.cpu()); c["top1"].append(top1.cpu())
            per_seq.append({"id": d["id"], "category": d["category"], "n": len(ids), "scored_from": off,
                            "nll_by_pos": [round(v, 4) for v in nll.tolist()],
                            "mean_nll": round(float(nll.mean()), 4), "top1_acc": round(float(top1.mean()), 4),
                            "seconds": round(time.perf_counter() - t0, 2)})
            log(f"{d['id']}: n={len(ids)} nll={float(nll.mean()):.4f} top1={float(top1.mean()):.4f} "
                f"({time.perf_counter() - t0:.1f}s)")
        summary = {k: {"mean_nll": float(torch.cat(v["nll"]).mean()), "top1_acc": float(torch.cat(v["top1"]).mean()),
                       "n": int(torch.cat(v["nll"]).numel())} for k, v in res.items()}
        return {"summary": summary, "config": self.config(), "replay": replay, "tail": tail,
                "per_seq": per_seq}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash")
    ap.add_argument("--max-seq", type=int, default=32768)
    ap.add_argument("--arena-gb", type=float, default=None)
    ap.add_argument("--trace-stats", default=None)
    ap.add_argument("--prompt", default="Write a Python function that returns the n-th Fibonacci number.")
    ap.add_argument("--prompt-file", default=None, help="read the prompt from this file instead of --prompt")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--no-swa-replay", action="store_true",
                    help="run all 40 layers over the whole prompt instead of CED + bounded replay")
    ap.add_argument("--hot-profile", default=None, choices=["mixed", "coding", "general"],
                    help="which slice of the traced corpus ranks the warm-start hot set")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--act-quant", action="store_true",
                    help="fake-quantize activations to fp8 like the reference (and like the tracer)")
    ap.add_argument("--verify-replay", action="store_true",
                    help="one load: full 40-layer prefill vs encoder + decoder SWA bounded replay")
    ap.add_argument("--teacher-forced", default=None,
                    help="corpus jsonl: run each sequence through Model.forward in one chunk and report NLL/top-1")
    ap.add_argument("--tf-out", default=None, help="write the teacher-forced result JSON here")
    ap.add_argument("--prune-sweep", default=None,
                    help="with --teacher-forced: comma list of keep fractions (e.g. 0.25,0.4,1.0); per layer only the "
                         "top-N experts by trace frequency stay routable; writes --tf-out with one entry per fraction")
    ap.add_argument("--prune-profile", default="mixed", choices=["mixed", "coding", "general"])
    ap.add_argument("--expert-topics", default=None,
                    help="comma-separated topics the resident set must serve; unset = every topic in the coverage file")
    ap.add_argument("--prune-select", default="uniform", choices=["uniform", "global"],
                    help="uniform: top-N per layer; global: one cross-layer ranking under the same total budget")
    ap.add_argument("--transient-slots", type=int, default=None,
                    help="prefill-miss ring slots (default 400 = a whole layer; 16 is enough when every kept expert is resident)")
    ap.add_argument("--keep-free-gb", type=float, default=None, help="host memory to leave free when auto-sizing the arena (default 20)")
    ap.add_argument("--expert-format", default=os.environ.get("EXPERT_FORMAT") or "fp4", choices=["fp4", "cb3"],
                    help="routed-expert arena format: fp4 = the checkpoint's packed FP4 (18.80 MB/expert); "
                         "cb3 = the 3-bit per-row codebook format (14.45 MB/expert, 0.769x), packed at warm "
                         "start, which fits ~40.8%% of all routed experts in 90.5 GB instead of 31.3%%")
    ap.add_argument("--sim-bits", type=int, default=None, help="simulate a 2/3-bit per-row codebook expert format (quality only)")
    ap.add_argument("--sim-cold-frac", type=float, default=1.0, help="fraction of the kept experts (coldest first) that get --sim-bits")
    ap.add_argument("--sim-cb2-frac", type=float, default=float(os.environ.get("SIM_CB2_FRAC") or 0.0),
                    help="with --expert-format cb3: give the coldest fraction of the kept experts a 2-bit "
                         "row codebook inside their (unchanged) CB3 slot. Quality only -- no bytes are "
                         "saved; it measures what a real 2-bit tier would cost")
    ap.add_argument("--prune-keep", type=float, default=None,
                    help="serve a REAP-style pruned model: only the top-F experts per layer are routable and all of "
                         "them are resident (F <= ~0.25 fits the arena on a 128 GB box)")
    ap.add_argument("--tf-ab", action="store_true",
                    help="with --teacher-forced: score the last window_size positions of every "
                         "sequence twice in one load -- full 40-layer prefill vs encoder + decoder "
                         "SWA bounded replay -- and report the NLL delta the approximation costs")
    ap.add_argument("--spec-ab", action="store_true",
                    help="one load, three runs: greedy without spec, greedy with spec (must match) and "
                         "one sampled spec run at --temperature/--top-p")
    ap.add_argument("--ab-out", default=None, help="write the --spec-ab result JSON here")
    a = ap.parse_args()
    if a.prompt_file:
        a.prompt = open(a.prompt_file).read()
    eng = V41Engine(a.model_dir, max_seq=a.max_seq, arena_gb=a.arena_gb, trace_stats=a.trace_stats,
                    spec=not a.no_spec, act_quant=a.act_quant,
                    swa_replay=(False if a.no_swa_replay else None), hot_profile=a.hot_profile, prune_keep=a.prune_keep, prune_select=a.prune_select,
                    expert_topics=a.expert_topics,
                    sim_bits=a.sim_bits, sim_cold_frac=a.sim_cold_frac, expert_format=a.expert_format,
                    sim_cb2_frac=a.sim_cb2_frac,
                    **({"transient_slots": a.transient_slots} if a.transient_slots else {}),
                    **({"keep_free_gb": a.keep_free_gb} if a.keep_free_gb else {}))
    if a.verify_replay:
        short = "def fib(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
        res = eng.verify_replay([short, open(os.path.join(HERE, "..", "corpus", "sources", "dsv41_readme.md")).read()[:1400]])
        print(json.dumps(res, indent=1))
        raise SystemExit(0)
    if a.teacher_forced and a.prune_sweep:
        import math as _m
        if a.prune_profile == "mixed":
            cc, cg = EX.category_counts(a.trace_stats, "coding"), EX.category_counts(a.trace_stats, "general")
            counts = {L: cc[L] / cc[L].sum() + cg[L] / cg[L].sum() for L in cc if L in cg}
        else:
            counts = EX.category_counts(a.trace_stats, a.prune_profile)
        assert len(counts) == 40, f"trace npz files missing next to {a.trace_stats}: {len(counts)} layers"
        out = {}
        for frac in [float(x) for x in a.prune_sweep.split(",")]:
            n_keep = max(6, _m.ceil(frac * 384))
            masks, keep = build_keep_masks(counts, frac, a.prune_select, eng.device)
            eng.model.prune_mask = masks if frac < 1.0 else None
            t0 = time.time()
            res = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq))
            res["keep_frac"] = frac; res["experts_per_layer"] = n_keep; res["resident_gb_fp4"] = round(n_keep * 40 * EX.EXPERT_BYTES / 1e9, 1)
            res["select"] = a.prune_select; res["per_layer_counts"] = [int(len(keep[L])) for L in range(40)]
            res["seconds"] = round(time.time() - t0, 1)
            out[str(frac)] = res
            log(f"prune keep={frac} {a.prune_select} ({n_keep}/384 per layer avg, {res['resident_gb_fp4']} GB): {json.dumps({k: v for k, v in res.items() if k in ('coding', 'general')})}")
            if a.tf_out:
                json.dump(out, open(a.tf_out, "w"), indent=1)
        raise SystemExit(0)
    if a.teacher_forced:
        if a.tf_ab:
            full = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq), replay=False, tail=True)
            rep = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq), replay=True, tail=True)
            # A bounded replay is worst at the START of the replayed window (the query at the
            # replay start sees a one-token window) and exact-ish at the END, and only the LAST
            # position's logits ever produce a token, so a single mean over the whole window
            # says almost nothing about serving. Bucket the delta by distance from the end.
            def _buckets(run):
                out = {}
                for k in (1, 8, 32, 128):
                    vals = [v for r in run["per_seq"] for v in r["nll_by_pos"][-k:]]
                    out[f"last{k}"] = (round(float(np.mean(vals)), 4), len(vals))
                return out
            fb, rb = _buckets(full), _buckets(rep)
            by_dist = {k: {"full_nll": fb[k][0], "replay_nll": rb[k][0],
                           "delta_nats": round(rb[k][0] - fb[k][0], 4), "n": fb[k][1]} for k in fb}
            print(json.dumps({"nll_by_distance_from_end": by_dist}, indent=1))
            delta = {k: {"full_nll": round(full["summary"][k]["mean_nll"], 4),
                         "replay_nll": round(rep["summary"][k]["mean_nll"], 4),
                         "delta_nats": round(rep["summary"][k]["mean_nll"] - full["summary"][k]["mean_nll"], 4),
                         "full_top1": round(full["summary"][k]["top1_acc"], 4),
                         "replay_top1": round(rep["summary"][k]["top1_acc"], 4),
                         "n": full["summary"][k]["n"]}
                     for k in full["summary"]}
            print(json.dumps(delta, indent=1))
            res = {"summary": delta, "nll_by_distance_from_end": by_dist, "config": eng.config(),
                   "full": full, "replay": rep}
        else:
            res = eng.teacher_forced(a.teacher_forced, max_len=min(512, a.max_seq))
            print(json.dumps(res["summary"], indent=1))
        if a.tf_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.tf_out)), exist_ok=True)
            json.dump(res, open(a.tf_out, "w"), indent=1)
            print("wrote", a.tf_out)
        raise SystemExit(0)
    sys.path.insert(0, os.path.join(a.model_dir, "encoding"))
    from encoding import encode_messages
    prompt = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="thinking" if a.thinking else "chat")
    if isinstance(prompt, tuple):
        prompt = prompt[0]
    ids = eng.tokenizer.encode(prompt, add_special_tokens=False)
    def run(tag, spec, temperature, top_p):
        eng.spec = spec
        print(f"\n===== {tag}: spec={spec} temperature={temperature} top_p={top_p} =====", flush=True)
        toks, text = [], []
        for burst in eng.generate(ids, max_tokens=a.max_tokens, temperature=temperature, top_p=top_p,
                                  ignore_eos=a.ignore_eos):
            toks += list(burst)
            piece = eng.tokenizer.decode(burst)
            text.append(piece)
            print(piece, end="", flush=True)
        print()
        st = eng.stats()
        print(json.dumps(st, indent=1), flush=True)
        return {"tag": tag, "spec": spec, "temperature": temperature, "top_p": top_p,
                "tokens": toks, "text": "".join(text), "stats": st}

    if a.spec_ab:
        # One process, one load: `spec` is just a flag on the engine, so the two runs share the
        # arena and the LRU state. Greedy speculative decoding must reproduce greedy autoregressive
        # decoding token for token (the target verifies every draft), so the first divergence index
        # is the whole test.
        runs = [run("greedy-nospec", False, 0.0, 1.0), run("greedy-spec", True, 0.0, 1.0)]
        A, B = runs[0]["tokens"], runs[1]["tokens"]
        first = next((i for i in range(min(len(A), len(B))) if A[i] != B[i]), None)
        n_eq = len(A) if first is None else first
        print(f"\n== greedy spec vs non-spec: {n_eq}/{min(len(A), len(B))} identical leading tokens; "
              f"first divergence at {first}")
        if first is not None:
            print("  nospec:", repr(eng.tokenizer.decode(A[max(0, first - 8):first + 8])))
            print("  spec  :", repr(eng.tokenizer.decode(B[max(0, first - 8):first + 8])))
        if a.temperature > 0:
            runs.append(run(f"sampled-spec-t{a.temperature}", True, a.temperature, a.top_p))
        if a.ab_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.ab_out)), exist_ok=True)
            json.dump({"prompt": a.prompt, "max_tokens": a.max_tokens, "config": eng.config(),
                       "identical_leading_tokens": n_eq, "first_divergence": first, "runs": runs},
                      open(a.ab_out, "w"), indent=1)
            print("wrote", a.ab_out)
    else:
        run("run", not a.no_spec, a.temperature, a.top_p)
