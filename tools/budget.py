"""budget.py -- what a topic selection costs on this box.

The engine refuses to start when the arena plus its working set will not fit
(engine/v41_engine.py, the pre-flight around `pack_scratch`), and finding that
out costs a three-minute load. This module answers the same question in a
millisecond, from the same arithmetic, so a configuration can be chosen before
it is paid for.

Two things are computed here.

MEMORY is exact. Every term below is either a shape out of the checkpoint's own
config or a constant the engine itself uses, and the total is compared against
`MemAvailable` the way the engine compares it.

COVERAGE is the interesting one. A keep-set is a cache policy learned from a
workload sample: per layer, only the top-N experts stay routable. Coverage is
the fraction of a topic's measured routing that lands on an expert that stayed.
It is the number that predicts whether generation holds together -- a topic the
trace never saw routes off the keep-set and the output degenerates, which is
exactly what a 0.03 coverage on markup did before the corpus was widened.

So the useful reading is not "how many topics" but "what is the weakest
selected topic's coverage, and what keep fraction does it need". Fewer topics
do not make a step faster on their own: step time is set by the bytes of the
experts a token activates, and that does not change. Fewer topics reach a given
coverage at a LOWER keep fraction, and a lower keep fraction is a smaller
arena -- which is where the memory, and the context window, come from.

No torch, no CUDA, no model load. numpy if it is there, plain Python if not.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field

try:
    import numpy as _np
except Exception:  # noqa: BLE001
    _np = None

# --- constants, each with where it comes from -------------------------------

N_LAYERS = 40
N_EXPERTS = 384          # routed experts per layer (config.json n_routed_experts)
N_ROUTED = N_LAYERS * N_EXPERTS   # 15,360
TOPK = 6                 # routed experts per token (config.json n_activated_experts)

# One expert as it sits in an arena slot.
#   fp4: 3 x (2304x2560 weights + 2304x160 UE8M0 scales) -- engine/experts.py EXPERT_BYTES
#   cb3: the 3-bit row-codebook layout -- tools/cb3_moe.py CB3_BYTES_PER_SLOT
EXPERT_BYTES = {"fp4": 18_800_640, "cb3": 14_454_784}

# The DSpark drafter's own experts: 3 MTP blocks x 128, always fully resident,
# always in the fp4 layout (engine/v41_engine.py loads them into dspark_arena).
DSPARK_BYTES = 3 * 128 * EXPERT_BYTES["fp4"]

# Everything that is not a routed expert: attention, shared experts, embeddings,
# the LM head, the Engram projections. Measured at load ("N GiB allocated after
# weights"), so it moves with DSV41_DENSE_FP4 and DSV41_HEAD_FMT.
DENSE_BYTES = {
    ("attn,wo_a", "fp8"): 7.61e9,   # measured 2026-09-12: "7.09 GiB allocated after weights"
    ("attn,wo_a", "bf16"): 8.94e9,  # + the bf16 head (129280x5120x2 = 1.32 GB)
    ("", "fp8"): 18.1e9,            # measured 2026-09-11 before the dense fp4 work
    ("", "bf16"): 19.4e9,
}
DENSE_DEFAULT = 7.61e9


def dense_key_from_env() -> tuple:
    """Which dense-weight figure applies, from the same two variables start.sh
    reads. Hardcoding the shipped pair understated resident memory by up to
    11.8 GB for anyone who changed either one in .env, which is enough to make
    the verdict wrong rather than merely imprecise."""
    groups = ",".join(sorted(g for g in os.environ.get("DSV41_DENSE_FP4", "attn,wo_a").split(",") if g))
    return (groups, os.environ.get("DSV41_HEAD_FMT", "fp8"))

# The KV and indexer caches are allocated for MAX_SEQ up front (engine/model.py
# Caches). Per token: for every kv_source_layer, one compressed-KV row of
# head_dim and one index row of index_head_dim, both bf16, at that layer's
# compression ratio.  ratios {2:2, 8:2, 14:2, 20:1}, head_dim 512, index 128:
#   (3 x 1/2 + 1) x (512 + 128) x 2 = 3,200 bytes per token.
KV_BYTES_PER_TOKEN = 3200
# The sliding-window rings do not scale with MAX_SEQ: RING 4096 x head_dim 512
# x bf16 x (40 layers + 3 MTP).
WINDOW_BYTES = 43 * 4096 * 512 * 2

# What the warm start needs on top of the arena while it packs experts into it,
# and the floor the launcher keeps free. Both are the engine's own numbers.
PACK_SCRATCH_BYTES = {"cb3": 3e9, "fp4": 1e9}
KEEP_FREE_GB_DEFAULT = 6.0

# Prefill misses go through a small ring of slots instead of the LRU, so the
# arena has to hold the kept set PLUS that ring: engine/experts.py sets
# lru_slots = n_slots - transient_slots, and a kept set larger than lru_slots
# streams its tail from NVMe on every step -- which is exactly the property a
# fully resident keep-set exists to buy.
TRANSIENT_SLOTS_DEFAULT = 8

PREFILL_CHUNK_DEFAULT = 2048
N_INDEX_LAYERS = 8               # config.json index_source_layers

# What one prefill chunk needs on top of everything resident. This is the term
# that decides whether a configuration serves or gets killed, and it is
# measured, not derived: at MAX_SEQ 32768 and chunk 2048 an arena of 87 GB left
# 16.5 GB free and served; 98 GB left 5.5 GB and the memory watchdog killed the
# process on the first request, with MemAvailable at 0.4 GB.
#
#   2026-09-12 14:07  "FATAL: host MemAvailable 0.4 GB stayed below the 2.5 GB
#                      floor for 3.0 s"   (arena 98.0 GB, keep 0.44)
#
# Measured 2026-09-12 by prefilling prompts of 8k to 128k tokens in one load at
# max_seq 262144 and tracking the low-water mark of MemAvailable throughout:
#
#     context   prefill held
#         8k        7.3 GB
#        16k        7.4 GB
#        32k        7.5 GB
#        64k        7.5 GB
#       128k        9.2 GB
#
# Flat to 64k, then a step. A linear fit over the whole range gives 7.2 GB of
# chunk cost plus 15.1 KB per token of context, which over-states the flat
# region and is the safe direction to be wrong in. It also explains the kill
# that started all this: that configuration left 5.5 GB against a 7.5 GB need.
PREFILL_BYTES_PER_TOKEN = 7.2e9 / PREFILL_CHUNK_DEFAULT

# And how much that grows with the CONTEXT, separately from the chunk. The
# indexer's score tiles are shaped [chunk, compressed positions] and the
# compressed cache is half the sequence, so a longer context makes every chunk
# more expensive even though the chunk itself is the same size.
#
PREFILL_BYTES_PER_CONTEXT_TOKEN = 15.1 * 1024

# The watchdog kills the process below this, so a configuration has to leave the
# prefill reserve AND this on top of it, not one or the other.
WATCHDOG_FLOOR_GB = 2.5

# Contexts that have been loaded and generated from on a GB10. Above the last
# one the KV arithmetic still holds, but the prefill path has not been run
# there: a 64k attempt tripped the memory watchdog at an arena that had room
# for the cache many times over, because the indexer's score tiles grow with
# the compressed cache and that term is not characterised yet. So the tool
# marks those lengths rather than predicting them.
VALIDATED_MAX_SEQ = 131072   # prefilled and measured at this length, 2026-09-12

GB = 1e9


def prefill_bytes(max_seq: int = 32768, chunk: int = PREFILL_CHUNK_DEFAULT) -> float:
    """Peak transient memory of one prefill chunk -- the reserve a configuration
    must leave free, or the watchdog kills the server on the first request."""
    return chunk * PREFILL_BYTES_PER_TOKEN + max_seq * PREFILL_BYTES_PER_CONTEXT_TOKEN


def kv_bytes(max_seq: int) -> float:
    """Exact: the caches engine/model.py allocates up front for MAX_SEQ."""
    return max_seq * KV_BYTES_PER_TOKEN + WINDOW_BYTES


# --- the box ----------------------------------------------------------------

@dataclass
class Host:
    name: str
    total_bytes: float
    available_bytes: float
    is_spark: bool
    cores: int = 0
    note: str = ""
    busy: str = ""      # a process already holding an arena, if there is one
    boxes: int = 1      # machines pooled by DSV41_BOXES; > 1 is sizing, not serving
    checked: float = 0.0    # when `busy` was last probed

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB

    @property
    def available_gb(self) -> float:
        return self.available_bytes / GB


def _meminfo() -> tuple:
    total = avail = 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                avail = float(line.split()[1]) * 1024
    except OSError:
        pass
    return total, avail


ARENA_SCRIPTS = ("v41_engine.py", "app.py", "expert_trace.py", "engram_rows.py")


def _busy(exclude_pid: int | None = None) -> str:
    """A process already holding an arena, if there is one. This box is
    single-tenant: two ~88 GB arenas wedge it past the point where sshd can
    fork (docs/gotchas.md).

    Read argv out of /proc rather than matching a pgrep pattern against whole
    command lines -- a shell *watching* for these names matches such a pattern
    and is not an engine. What counts is a Python interpreter whose script
    argument is one of them.
    """
    me = exclude_pid if exclude_pid is not None else os.getpid()
    try:
        pids = [int(d) for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return ""
    for pid in pids:
        if pid == me:
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        argv = [a for a in argv if a]
        if len(argv) < 2 or "python" not in os.path.basename(argv[0]):
            continue
        script = next((a for a in argv[1:] if os.path.basename(a) in ARENA_SCRIPTS), None)
        if script:
            return f"{os.path.basename(script)} (pid {pid})"
    return ""


def refresh(host: Host, busy_every: float = 4.0) -> Host:
    """Re-read what changes while the screen is open. MemAvailable moves on its
    own, so a budget checked against a startup reading is already stale;
    /proc/meminfo is two lines and costs nothing per frame. The process probe
    forks, so it is throttled."""
    if not (_env_gb("DSV41_HOST_TOTAL_GB") or _env_gb("DSV41_HOST_AVAIL_GB") or host.boxes > 1):
        total, avail = _meminfo()
        if total:
            host.total_bytes, host.available_bytes = total, avail
    now = time.monotonic()
    if now - host.checked >= busy_every:
        host.busy = _busy()
        host.checked = now
    return host


# The machine, overridable. The defaults come from /proc, but a keep-set has to
# be sized for the box it will run on, which is often not the box you are
# sitting at. These let you plan for one:
#
#   DSV41_HOST_TOTAL_GB   physical memory, in GB
#   DSV41_HOST_AVAIL_GB   memory free for the engine, in GB
#   DSV41_BOXES           how many such machines (see below)
#   DSV41_HOST_NAME       what to call it on screen
#
# DSV41_BOXES multiplies the memory and says so. It answers "what could two of
# these hold", which is a real question: the whole expert set is 222 GB in the
# 3-bit format, so two 121 GiB machines hold all of it and no keep-set is
# needed at all. What it does NOT do is make this engine serve across them --
# there is no pipeline or expert parallelism here, one process, one machine.
# Treat a BOXES > 1 answer as sizing for a system you would still have to build.
def _env_gb(name):
    v = os.environ.get(name)
    try:
        return float(v) * GB if v else None
    except ValueError:
        return None


def read_host() -> Host:
    """What this machine has, right now, unless the environment describes a
    different one. Linux reads /proc; anything else is a stand-in so the tool
    still runs where it is being edited."""
    total, avail = _meminfo()
    name, is_spark = platform.node(), False
    for p in ("/proc/device-tree/model", "/sys/devices/virtual/dmi/id/product_name"):
        try:
            m = open(p, "rb").read().decode("utf-8", "replace").strip("\x00 \n")
            if m:
                name = m
                break
        except Exception:  # noqa: BLE001
            pass
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=4).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    if gpu:
        name = gpu if gpu.lower() not in name.lower() else name
    is_spark = bool(re.search(r"GB10|DGX Spark|GX10", f"{name} {gpu}", re.I))
    note = "" if is_spark else "not a GB10 -- numbers are the model's, not this machine's"
    busy = _busy()
    if not total:  # not Linux: show the Spark so the arithmetic is still the real one
        total, avail = 130.6e9, 117.0e9
        note = "no /proc/meminfo here; showing a GB10's 121 GiB"

    t_over, a_over = _env_gb("DSV41_HOST_TOTAL_GB"), _env_gb("DSV41_HOST_AVAIL_GB")
    if t_over:
        total = t_over
        avail = a_over or t_over * (avail / total if total else 0.9)
    elif a_over:
        avail = a_over
    if t_over or a_over:
        note = "memory from the environment, not this machine"

    try:
        boxes = max(1, int(os.environ.get("DSV41_BOXES", "1")))
    except ValueError:
        boxes = 1
    if boxes > 1:
        total, avail = total * boxes, avail * boxes
        note = (f"{boxes} machines pooled: sizing only, this engine serves from one "
                f"(no pipeline or expert parallelism)")

    return Host(name=os.environ.get("DSV41_HOST_NAME") or name, total_bytes=total,
                available_bytes=avail, is_spark=is_spark, cores=os.cpu_count() or 0,
                note=note, busy=busy, boxes=boxes, checked=time.monotonic())


# --- topics -----------------------------------------------------------------

def _cumsum_desc(counts_by_layer, order_by_layer, topic_counts):
    """For one topic: curve[n] = routing mass of that topic captured by keeping
    the top-n experts of every layer. Pure Python, fine at 40x384."""
    curve = [0.0] * (N_EXPERTS + 1)
    for L in range(N_LAYERS):
        c = topic_counts[L]
        acc = 0.0
        for n, e in enumerate(order_by_layer[L], start=1):
            acc += c[e]
            curve[n] += acc
    return curve


def topic_names(path: str) -> list:
    """The topic names in a coverage file, without loading its histograms --
    find_stats compares every candidate in the checkout and there can be many."""
    try:
        d = json.load(open(path))
        any_layer = next(iter((d.get("per_layer") or {}).values()), {})
        return sorted(k[len("counts_"):] for k in any_layer if k.startswith("counts_"))
    except Exception:  # noqa: BLE001
        return []


class TopicIndex:
    """The per-topic expert histograms in a coverage.json, and everything that
    can be derived from a selection of them without touching the model."""

    def __init__(self, path: str):
        self.path = path
        d = json.load(open(path))
        pl = d.get("per_layer") or {}
        any_layer = next(iter(pl.values()), {})
        self.topics = sorted(k[len("counts_"):] for k in any_layer if k.startswith("counts_"))
        self.counts = {}
        for t in self.topics:
            per = {}
            ok = True
            for L in range(N_LAYERS):
                v = pl.get(str(L), {}).get(f"counts_{t}")
                if v is None:
                    ok = False
                    break
                per[L] = [float(x) for x in v]
            if ok:
                self.counts[t] = per
        self.topics = [t for t in self.topics if t in self.counts]
        self.totals = {t: sum(sum(v) for v in self.counts[t].values()) for t in self.topics}
        # Every token routes to `n_activated_experts` experts in each of the 40
        # layers, so the histogram totals divide back to the tokens the trace
        # actually saw for that topic. A topic sampled thinly ranks noisily, and
        # its coverage bar is no more trustworthy than the sample under it.
        self.tokens = {t: int(round(v / (N_LAYERS * TOPK))) for t, v in self.totals.items()}
        self._cache: dict = {}

    THIN = 2000              # tokens below which a ranking is mostly noise

    def __bool__(self) -> bool:
        return bool(self.topics)

    def curves(self, selection: tuple, select: str = "uniform", only: tuple | None = None):
        """coverage curves for a selection: {topic: [384+1 floats]}, index n =
        keeping the top-n experts per layer. Computed once per selection, so a
        slider move is a lookup."""
        key = (tuple(sorted(selection)), select, only)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        sel = [t for t in selection if t in self.counts]
        if not sel:
            return {}
        # the ranking the engine builds: per-layer-normalised counts, summed
        combined = {}
        for L in range(N_LAYERS):
            acc = [0.0] * N_EXPERTS
            for t in sel:
                c = self.counts[t][L]
                s = sum(c)
                if s > 0:
                    for e in range(N_EXPERTS):
                        acc[e] += c[e] / s
            combined[L] = acc
        order = {L: sorted(range(N_EXPERTS), key=lambda e: -combined[L][e]) for L in range(N_LAYERS)}
        out = {}
        # every topic in the file gets a curve, so an unselected one can be read
        # off too -- that is how you see what a selection costs the rest.
        # `only` narrows that when the caller just wants the selected ones,
        # which is ten times cheaper when ten selections are compared at once.
        for t in (only if only is not None else self.topics):
            tot = self.totals[t] or 1.0
            curve = _cumsum_desc(combined, order, self.counts[t])
            out[t] = [x / tot for x in curve]
        self._cache[key] = (out, order, combined)
        return self._cache[key]

    def coverage(self, selection: tuple, keep: float, select: str = "uniform") -> dict:
        got = self.curves(selection, select)
        if not got:
            return {}
        curves = got[0]
        n = keep_n(keep)
        return {t: c[n] for t, c in curves.items()}

    def keep_for(self, selection: tuple, target: float, select: str = "uniform") -> float | None:
        """Smallest keep fraction at which every selected topic reaches `target`.
        An empty selection means every topic, which is what the engine ranks on."""
        selection = tuple(selection) or tuple(self.topics)
        got = self.curves(selection, select, only=tuple(sorted(selection)))
        if not got:
            return None
        curves = got[0]
        sel = [t for t in selection if t in curves]
        for n in range(1, N_EXPERTS + 1):
            if all(curves[t][n] >= target for t in sel):
                return n / N_EXPERTS
        return None


def keep_n(keep: float) -> int:
    """Experts kept per layer at this fraction -- the engine's own rounding."""
    return max(6, math.ceil(keep * N_EXPERTS))


# --- the plan ---------------------------------------------------------------

@dataclass
class Plan:
    keep: float
    n_keep: int
    kept: int
    slots: int
    transient: int
    fmt: str
    max_seq: int
    arena: float
    dense: float
    dspark: float
    kv: float
    prefill: float
    scratch: float
    floor: float
    available: float
    coverage: dict = field(default_factory=dict)
    selection: tuple = ()

    @property
    def resident_frac(self) -> float:
        return self.kept / N_ROUTED

    @property
    def resident(self) -> float:
        """Everything that stays in memory for the whole run."""
        return self.arena + self.dense + self.dspark + self.kv

    # --- gate 1: the launcher's own pre-flight ------------------------------
    # engine/v41_engine.py refuses to start unless
    #     arena + pack_scratch + keep_free <= MemAvailable
    # measured after the dense weights are already resident. Reproduced here
    # against MemAvailable as it is now, so the dense term is explicit.
    @property
    def launch_need(self) -> float:
        # engine/v41_engine.py: floor = max(keep_free_gb, MAX_CHUNK * 5 MB).
        # Mirroring it exactly matters -- reading `keep_free_gb` alone made this
        # 4.2 GB more generous than the launcher's real margin at the default.
        return self.arena + self.scratch + self.dense + max(self.floor, self.need_free)

    @property
    def launch_slack(self) -> float:
        return self.available - self.launch_need

    # --- gate 2: what is left once it is up ---------------------------------
    @property
    def free_after_load(self) -> float:
        return self.available - self.resident

    @property
    def need_free(self) -> float:
        """A prefill chunk, plus the floor the watchdog kills below."""
        return self.prefill + WATCHDOG_FLOOR_GB

    @property
    def fits(self) -> bool:
        return self.launch_slack >= 0 and self.free_after_load >= self.need_free

    @property
    def verdict(self) -> str:
        # The engine's own gate lets a configuration start that the first
        # request then kills, because that gate does not know about the drafter
        # experts, the cache, or a prefill chunk. This one does.
        if self.launch_slack < 0 or self.free_after_load < self.need_free:
            return "over"
        if self.launch_slack < 3.0 or self.free_after_load < self.need_free + 3.0:
            return "tight"
        return "ok"

    @property
    def weakest(self):
        sel = {t: c for t, c in self.coverage.items() if t in self.selection}
        if not sel:
            return None, None
        t = min(sel, key=lambda k: sel[k])
        return t, sel[t]

    def max_arena(self) -> float:
        """Largest arena that both starts AND survives a prefill chunk."""
        launch = self.available - self.scratch - self.dense - self.floor
        serve = self.available - self.dense - self.dspark - self.kv - self.need_free
        return max(0.0, min(launch, serve))

    def max_keep(self) -> float:
        slots = self.max_arena() * GB / EXPERT_BYTES[self.fmt] - self.transient
        return max(0.0, slots / N_ROUTED)

    @property
    def everything_fits(self) -> bool:
        """Enough memory for every routed expert, so no keep-set is needed at
        all and the router is never restricted. Two GB10s get to 93 %; three
        clear it outright."""
        return self.max_keep() >= 1.0

def plan(host: Host, index: TopicIndex | None, selection, keep: float, max_seq: int,
         fmt: str = "cb3", select: str = "uniform", arena_gb: float | None = None,
         keep_free_gb: float = KEEP_FREE_GB_DEFAULT, dense_key=None,
         chunk: int = PREFILL_CHUNK_DEFAULT,
         transient_slots: int = TRANSIENT_SLOTS_DEFAULT) -> Plan:
    # the engine's own rounding: ceil(keep * 384) experts in every layer
    dense_key = dense_key or dense_key_from_env()
    kept = keep_n(keep) * N_LAYERS
    slots = kept + transient_slots
    arena = (arena_gb * GB) if arena_gb else slots * EXPERT_BYTES[fmt]
    if arena_gb:
        slots = int(arena / EXPERT_BYTES[fmt])
        kept = min(kept, slots - transient_slots)
    kv = kv_bytes(max_seq)
    cov = index.coverage(tuple(selection), keep, select) if index else {}
    return Plan(
        keep=keep, n_keep=keep_n(keep), kept=kept, slots=slots, transient=transient_slots,
        fmt=fmt, max_seq=max_seq,
        arena=arena / GB,
        dense=DENSE_BYTES.get(dense_key, DENSE_DEFAULT) / GB,
        dspark=DSPARK_BYTES / GB,
        kv=kv / GB,
        prefill=prefill_bytes(max_seq, chunk) / GB,
        scratch=PACK_SCRATCH_BYTES[fmt] / GB,
        floor=keep_free_gb,
        available=host.available_gb,
        coverage=cov, selection=tuple(selection),
    )
