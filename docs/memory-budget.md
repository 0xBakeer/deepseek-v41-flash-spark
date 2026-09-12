# The memory budget

Every term in `./tune.sh`'s budget panel, where each number comes from, and the two gates a
configuration has to pass. The design this arithmetic serves is in
[`docs/architecture.md`](architecture.md); the screen is in [`docs/tune.md`](tune.md) and
[`docs/tune-reference.md`](tune-reference.md); the sharp edges are in
[`docs/gotchas.md`](gotchas.md).

The model is `tools/budget.py`. It holds no torch, loads nothing and answers in a millisecond, which
is the point: the engine answers the same question by loading for three minutes first.

## What is being budgeted

Memory on a GB10 is unified. The GPU allocator and the page cache draw on one pool, and
`torch.cuda.mem_get_info()` counts the page cache as used — right after a checkpoint download it
reported 32.0 GB free on a box with 99.9 GiB of `MemAvailable`. `nvidia-smi` answers `[N/A]` for
every memory field. So the honest number, for the tool and for the engine alike, is `MemAvailable`
from `/proc/meminfo`, and the engine takes the larger of that and `mem_get_info()`.

There is no OOM-kill moment to catch: push `MemAvailable` towards zero on this hardware and the
machine stops being reachable. Everything below is arranged around not doing that.

## The terms

| term | value | where it comes from |
|---|---|---|
| routed experts | 40 layers x 384 = 15,360 | `config.json` `n_routed_experts`, 40 layers |
| activated per token | 6 per layer | `config.json` `n_activated_experts` |
| expert slot, `cb3` | 14,454,784 B (14.45 MB) | `CB3_BYTES_PER_SLOT` in `tools/cb3_moe.py` |
| expert slot, `fp4` | 18,800,640 B (18.80 MB) | `EXPERT_BYTES` in `engine/experts.py`, `3 x (2304x2560 + 2304x160)` |
| kept experts | `ceil(keep x 384) x 40`, never fewer than 6 per layer | `build_keep_masks` in `engine/v41_engine.py` |
| transient ring | 8 slots as the tool writes it; 400 is the engine's default | `TRANSIENT_SLOTS`, `engine/experts.py` |
| expert arena | `(kept + ring) x slot` | |
| dense weights | 7.61 GB | measured at load, `7.09 GiB allocated after weights`, 2026-09-12 |
| drafter experts | `3 x 128 x 18,800,640` = 7.22 GB | the DSpark MTP blocks, `engine/v41_engine.py` |
| KV + indexer cache | `max_seq x 3,200 B` | `Caches` in `engine/model.py` |
| sliding-window rings | `43 x 4096 x 512 x 2` = 180,355,072 B | same, and independent of `max_seq` |
| warm-start pack scratch | 3 GB for `cb3`, 1 GB for `fp4` | the 3-bit packer's GPU buffers |
| keep-free floor | 6 GB as the tool writes it; 20 GB is the engine's default | `KEEP_FREE_GB` |
| prefill chunk | 5 MB per token, so 10.24 GB at the default 2,048-token chunk | measured; see below |

Everything except the last two rows stays resident for the whole run.

### Kept experts

`PRUNE_KEEP` is a fraction of each layer's 384 experts, and the engine rounds it up:
`ceil(keep x 384)` experts in every layer, never fewer than six. At `PRUNE_KEEP=0.39` that is 150 per
layer and 6,000 in all, 39.1 % of the 15,360.

The tool uses the same rounding, so the count on screen is the count the engine will build.

### The transient ring

Prefill touches nearly every expert of a layer — 370 to 381 of 384, measured at layer 0 — so a
prefill miss must not go through the LRU or every prompt would evict the hot set. Misses go to a
small ring of slots outside it instead, and the arena has to hold **the kept set plus that ring**:
`engine/experts.py` sets `lru_slots = n_slots - transient_slots`, and `warm_start` fills only
`lru_slots`. A kept set larger than that has its tail left cold, streaming from NVMe on every step,
with nothing but the tok/s to say so.

The size of the ring depends on which mode the engine is in, and the two answers are far apart:

* **Streaming mode**, where experts are read on demand, needs at least 384 slots, because below that
  the ring wraps inside a single layer and the engine computes that layer with the wrong experts —
  no exception and no warning, just an 0.88 relative error in the output. The default is 400 and it
  must not be lowered.
* **Pruned all-resident mode**, which is what `./tune.sh` configures, never misses during prefill,
  because every routable expert is already in the LRU. Eight slots is enough, and `engine/experts.py`
  asserts that as the hard minimum.

The difference is not small: 392 extra slots is 5.7 GB of arena, which is more than a full step of
the keep slider. Sizing an arena against 8 and running with 400 leaves 392 kept experts outside the
LRU. That is why `./tune.sh` writes `TRANSIENT_SLOTS` into `.env` alongside `ARENA_GB` — the two
numbers are only correct together.

### The arena

`(kept + ring) x slot_bytes`, written into `.env` as `ARENA_GB` rounded up to a whole GB. The engine
divides it back into slots by flooring, and `tools/test_budget.py` checks at every keep step and
every ring size that the round trip never leaves a kept expert outside the arena.

Format matters more than anything else on the screen. At `PRUNE_KEEP=0.39` the same 6,008 slots are
86.8 GB in `cb3` and 113.0 GB in `fp4` — the difference between serving on this box and not fitting
on it at all.

### Dense weights

Everything that is not a routed expert: attention, the shared experts, the embeddings, the LM head
and the Engram projections. This is measured at load rather than derived, and it moves with two
switches:

| `DSV41_DENSE_FP4` | `DSV41_HEAD_FMT` | dense weights |
|---|---|---|
| `attn,wo_a` | `fp8` | 7.61 GB |
| `attn,wo_a` | `bf16` | 8.94 GB |
| unset | `fp8` | 18.1 GB |
| unset | `bf16` | 19.4 GB |

`env.example` ships the first row, and the budget panel always assumes it. If those two switches are
changed in `.env`, the panel understates resident memory by up to 11.8 GB and the verdict cannot be
trusted.

### Drafter experts

The DSpark drafter has its own 384 experts — 3 MTP blocks of 128 — and they are loaded into their
own arena, fully resident, always in the FP4 layout whatever `EXPERT_FORMAT` says. 7.22 GB, fixed,
for every configuration.

They are also the largest term the engine's own pre-flight does not know about, because that check
runs before they are allocated.

### KV cache

Exact, and much smaller than people expect. `engine/model.py` allocates the caches for `MAX_SEQ`
up front. Per token, for each of the four `kv_source_layers` — 2, 8, 14 and 20 — one compressed-KV
row of `head_dim` 512 and one index row of `index_head_dim` 128, both bf16, at that layer's
compression ratio (2, 2, 2 and 1):

```
(3 x 1/2 + 1) x (512 + 128) x 2 = 3,200 bytes per token
```

The sliding-window rings are a separate, constant term: `RING` 4096 positions of `head_dim` 512 in
bf16 for 40 layers plus 3 MTP blocks, 180,355,072 B, the same at any context length. The panel's
`KV cache` row is the two added.

| `MAX_SEQ` | rows | + window rings | panel |
|---|---|---|---|
| 4,096 | 0.013 GB | 0.193 GB | 193 MB |
| 32,768 | 0.105 GB | 0.285 GB | 285 MB |
| 131,072 | 0.419 GB | 0.600 GB | 600 MB |
| 262,144 | 0.839 GB | 1.019 GB | 1.0 GB |
| 1,048,576 | 3.355 GB | 3.536 GB | — |

The cache is never what limits the context window on this box. Prefill is. The whole range the tool
offers, 4k to 256k, is 0.83 GB — less than a fifth of one 2-point step of the keep slider. The tool
therefore marks 32,768 as the longest context that has actually been loaded and generated from,
rather than predicting a ceiling from the cache arithmetic: a 64k attempt tripped the memory
watchdog at an arena with room for the cache many times over, because the indexer's score tiles grow
with the compressed cache and that term is not characterised.

### What the panel leaves out

* The **Engram row cache**, capped at 200,000 rows of 264 bytes per table across two tables, so
  105.6 MB at most. It is also the only thing still read from NVMe once a keep-set is fully
  resident: 24 rows per token per table, about 13 KB a token, against zero bytes of expert weights.
* The **pack scratch** and the **prefill chunk**, which are transient rather than resident. They are
  not in the resident total but they are in the gates, which is where they belong.

## The two gates

A configuration has to pass two separate checks, and they are not the same check.

### Gate 1 — the engine's own pre-flight

`engine/v41_engine.py` refuses to start when

```
arena + pack scratch + floor  >  MemAvailable
```

measured **after the dense weights are already resident**, with `floor = max(KEEP_FREE_GB, one
prefill chunk)`. The tool reproduces it against `MemAvailable` as it is now, before the weights are
loaded, so the dense term is explicit:

```
room to launch = MemAvailable − (arena + pack scratch + dense weights + keep-free floor)
```

That is the `room to launch` row. One difference is worth knowing: the row uses the keep-free floor
alone, while the engine takes the larger of that floor and a prefill chunk, so at `KEEP_FREE_GB=6`
the engine's own margin is 4.2 GB smaller than the number shown. It never changes the verdict,
because gate 2 below is strictly tighter — but that is only true while `KEEP_FREE_GB` is what the
tool writes. Pin an arena by hand and leave the engine's 20 GB default in place and gate 1 becomes
the binding one.

### Gate 2 — a prefill chunk still fits

Everything resident, subtracted from what the box has:

```
resident        = arena + dense weights + drafter experts + KV cache
free after load = MemAvailable − resident
```

and that has to leave room for one prefill chunk. This engine's activation cost is about **5 MB per
prefill token**, so the default 2,048-token chunk needs about 10 GB. This number is measured, not
derived, and the measurement is the reason the gate exists:

| arena | free after load | what happened |
|---|---|---|
| 98 GB | 5.5 GB | loaded, logged `ready`, and the memory watchdog killed it on the **first** request |
| 87 GB | 16.5 GB | served two 2,400-token generations |

Same box, same `.env` except the keep fraction, `MemAvailable` 111.0 GB with the dense weights
resident, 2026-09-12:

```
FATAL: host MemAvailable 0.4 GB stayed below the 2.5 GB floor for 3.0 s
```

**Gate 1 accepts both.** It runs before the drafter experts, before the cache and before anything
has prefilled, so it cannot see the 7.2 GB of drafter and the 10 GB of chunk that turn 98 GB from a
configuration that starts into one that dies. That asymmetry is the single most useful thing the
tool does: it applies gate 2 and refuses configurations the engine would have accepted.

### The verdict

```
will not load   room to launch < 0  or  free after load < one prefill chunk
tight           room to launch < 3 GB  or  free after load < that chunk + 3 GB
fits            otherwise
```

### A third, coarser gate

Before either of them, `start.sh` and the container entrypoint refuse to start at all below
`MIN_FREE_GIB` (90 GiB of `MemAvailable`), and name the processes and containers holding the pool.
That one is about whether the box is free, not about whether the configuration is sized right.

## The ceiling

The largest arena that both starts and survives a prefill chunk is the smaller of the two gates:

```
max_arena = min( MemAvailable − pack scratch − dense − keep-free floor,
                 MemAvailable − dense − drafter − KV − one prefill chunk )
max_keep  = (max_arena / slot_bytes − 8) / 15,360
```

With `cb3` experts and `MemAvailable` at 118.6 GB — the box before the dense weights, the same
reading the 111.0 GB above was taken after — that is a 93.2 GB arena and **about 42 % of the routed
experts**. It is not a constant: it moves with `MemAvailable` while the screen is open, and it is
lower in `fp4` (32 %) because each slot is 30 % larger.

The shipped default sits at 39 %, below the ceiling on purpose. `RESULTS.md` §4.3 was taken at 44 %,
and that configuration does not reliably serve.

## The ladder

`cb3` experts, an 8-slot ring, `MAX_SEQ=32768`, `MemAvailable` 118.6 GB:

| keep | per layer | kept | arena | resident | free after load | verdict |
|---|---|---|---|---|---|---|
| 25 % | 96 | 3,840 | 55.6 GB | 70.7 GB | 47.9 GB | fits |
| 32 % | 123 | 4,920 | 71.2 GB | 86.3 GB | 32.3 GB | fits |
| 39 % | 150 | 6,000 | 86.8 GB | 102.0 GB | 16.6 GB | fits |
| 40 % | 154 | 6,160 | 89.2 GB | 104.3 GB | 14.3 GB | fits |
| 41 % | 158 | 6,320 | 91.5 GB | 106.6 GB | 12.0 GB | tight |
| 42 % | 162 | 6,480 | 93.8 GB | 108.9 GB | 9.7 GB | will not load |
| 44 % | 169 | 6,760 | 97.8 GB | 112.9 GB | 5.7 GB | will not load |
| 51 % | 196 | 7,840 | 113.4 GB | 128.6 GB | −10.0 GB | will not load |

The 44 % row is the configuration that was killed on its first request, and the 5.7 GB here is the
5.5 GB that was measured. The 39 % row is the shipped default, and its 16.6 GB is the 16.5 GB that
served.

Reproduce any row without a terminal:

```bash
./tune.sh --keep 0.42 --print
```

## Where this is checked

```bash
python3 tools/test_budget.py
```

Cross-checks the slot sizes against the kernel's own constant, the KV formula against two measured
lengths, the launch gate against an arena the box accepted and one it did not, the `ARENA_GB`
rounding at every keep step and ring size, and — by reading `engine/v41_engine.py` with a regular
expression — that the engine still reserves a prefill chunk at the same rate this model assumes. If
the two ever drift, the tool would start advising configurations the engine refuses, or worse, ones
it accepts and the watchdog then kills.
