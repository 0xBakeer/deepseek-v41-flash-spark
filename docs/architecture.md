# Architecture — how 510 GB is served from 128 GB

DeepSeek-V4.1-Flash is 510 GB on disk. This box has 128 GB of unified memory, ~121 GiB of
it visible, shared between the GPU and the host. Nothing about that gap is closed by
being clever with allocation; it is closed by deciding, per weight, **which ones are
allowed to be resident** — and then making the rest cheap enough to fetch.

Everything below is derived from [`NOTES.md`](../NOTES.md) and the engine's own
docstrings. Numbers marked *measured* were taken on the box; the rest is arithmetic that
needs no measurement.

## The arithmetic that decides the design

The routed experts are the whole problem:

```
15,360 routed experts  (384 per layer × 40 layers)
   × 3 matrices each (w1, w2, w3)
   = 543.6 billion weights
   at FP4 E2M1 + one UE8M0 scale per 32 values = 4.25 bits/weight
   = 288.8 GB          (18.8 MB per expert)
```

Against that, what has to be resident no matter what:

| resident, always | GB |
|---|---|
| attention + indexer + hyper-connections + router + norms | 5.2 |
| shared experts | 1.4 |
| embedding | 1.3 |
| LM head | 1.3 |
| MTP / DSpark blocks (3 × 128 experts, drafting every step) | 7.9 |
| Engram non-table weights (`wkv`, `q_weight`, `k_weight`) | 0.3 |
| vision tower (skippable for text-only serving) | 1.0 |
| **total** | **~18.5** |

KV is not the constraint on this model — it is unusually small. Global KV is 890 B/token
and the per-layer 128-token sliding window is 2.8 MB for a whole sequence, so 256k tokens
of context cost about 0.23 GB. (For scale: vLLM on four Sparks fit 1.03M tokens into
4.84 GiB.)

So on a 121 GiB box, after ~18.5 GB of always-resident weights, activations, the indexer
slices and a floor of free memory, **the routed experts get about 85–90 GB** of the
288.8 GB they want. Spread evenly that is:

> **~1.3 bits per weight, averaged over all 15,360 experts.**

That single number kills the obvious designs:

* **Everything resident, mixed precision** (hot experts at FP4, cold ones at 2–3 bpw) —
  the whole set at 2.0 bpw is still 136 GB, and even then only ~9% could stay at FP4. It
  only fits below ~1.3 bpw average, which is below any quality floor worth shipping.
* **Everything resident, pruned** — 136 GB → ~88 GB means dropping ~35% of the experts
  outright. Useful as a speed reference, not as the recipe.
* **A resident hot set at full FP4 + NVMe streaming for the rest** — the only design that
  keeps every expert at the precision DeepSeek shipped. This is what the engine does, and
  it lives or dies on the miss rate.

## The three tiers

```
                 ┌──────────────────────────────────────────────┐
   resident      │ non-expert weights ~18.5 GB                   │  never evicted
   (GPU/unified) │ DSpark expert set 7.2 GB (3 × 128, all)       │
                 ├──────────────────────────────────────────────┤
   the ARENA     │ N slots × 18.8 MB of packed FP4 experts       │  ~4,300 slots
                 │   ├─ LRU region   (decode misses land here)   │  measured: 80.7 GB
                 │   └─ TRANSIENT ring (prefill misses only)     │  = 28% of experts
                 ├──────────────────────────────────────────────┤
   NVMe          │ the other ~11,000 experts, read on miss with  │  measured 5.4–5.6 GB/s
                 │ O_DIRECT preadv straight out of the shards    │  at 8–32 reads in flight
                 └──────────────────────────────────────────────┘
```

### The arena

A fixed number of GPU slots holding experts **exactly as they are stored in the
checkpoint** — packed FP4 plus UE8M0 scales, no re-quantisation on the way in, which is
what makes "resident" and "streamed" the same numerics. The slot count is sized at
start-up from what is free.

Sizing is subtle on this hardware, and getting it wrong is silent. `torch.cuda.mem_get_info()`
counts the page cache as used, so right after a large download it reported *32.0 GB free
on a box with 99.9 GiB `MemAvailable`* — which would have handed the engine a 23 GB arena
and a permanently NVMe-bound server with no error anywhere. The engine takes the larger of
`mem_get_info()` and `/proc/meminfo`'s `MemAvailable`, keeps a hard `keep_free_gb` floor
(20 GB) under the latter, and logs both. The floor is not politeness: this box hard-resets
when `MemAvailable` goes negative.

*Measured:* `MemAvailable` 99.9 GB at sizing time → arena 80.7 GB = 4,291 slots
(3,891 LRU + 400 transient) = **28% of the 15,360 routed experts**.

### The LRU, and the warm start

A map `(layer, expert) → slot` with least-recently-used eviction. It is warm-started
before the socket is bound, filling the arena in the order given by a measured routing
trace (`results/trace-*/stats/coverage.json` from `tools/expert_stats.py`) rather than in
`(layer, expert)` index order. That ranking is worth a lot — see the coverage table below.

*Measured:* 3,891 experts / 73.2 GB in **16 s at 4.6 GB/s** (`O_DIRECT`, 12 IO threads),
after ~63 s for the 19 GB of non-expert weights.

### The transient ring

A prefill chunk touches nearly every expert of a layer — the routing trace saw 370–381 of
384 used at layer 0 over 10,760 tokens. Letting that stream through the LRU would evict
the entire hot set on every prompt, so **prefill misses go to a small ring of transient
slots and never enter the LRU**; only decode misses do.

The ring has a hard lower bound: it must hold at least 384 slots, one full layer's worth.
A 64-slot ring wrapped *inside* a layer and the engine silently computed that layer with
the wrong experts — an 0.88 relative error in the first smoke test, and no exception
anywhere. The default is 400.

### NVMe streaming

Every miss is a read straight out of the layer's safetensors shard with `O_DIRECT`
`preadv` into a pinned, 4 KiB-aligned staging buffer, then a copy into the slot. No page
cache: these are 18.8 MB objects and polluting the cache with them costs more than it
saves.

The one thing that matters here is **how many reads an expert costs**. Its six tensors
(`w1/w2/w3` × weight and scale) are not adjacent in the shard — all the scale tensors sit
near the front, all the weight tensors far behind — but *within* each group the three
tensors of one expert are contiguous. So an expert is exactly **two runs**: a 1.1 MB scale
run and a 17.7 MB weight run. Reading it as six separate `preadv`s costs six `O_DIRECT`
round trips, and at a large arena a decode step misses only about one expert per layer, so
there is nothing else in flight to hide the latency: the read rate collapses from
~4.7 GB/s to ~0.6 GB/s. Two runs, not six, not one.

*Measured, `O_DIRECT`, 18.8 MB objects:* 1 in flight 4.08 GB/s (3.9 ms/read); 8 in flight
5.43 GB/s; 32 in flight 5.59 GB/s.

## What the coverage curve says

The full 40-layer routing trace (10,760 teacher-forced tokens, coding + general) gives the
hit rates the design depends on:

| resident experts | GB at FP4 | static coverage | LRU hit / token | LRU hit / 6-token block |
|---|---|---|---|---|
| 3,000 | 56.4 | 0.670 | 0.805 | 0.681 |
| 4,000 | 75.2 | 0.748 | 0.855 | 0.764 |
| **4,500** | **84.6** | **0.780** | **0.875** | **0.796** |
| 5,000 | 94.0 | 0.810 | 0.891 | 0.822 |
| 6,000 | 112.8 | 0.859 | 0.917 | 0.865 |

Per layer, the top 25% of experts cover 59–83% of routed slots (flattest at layer 0, most
skewed around layer 28); a 6-token DSpark block touches 18–28 unique experts of a possible
36 per layer. The decoder layers are only mildly more skewed than the encoder — the
"deeper layers will save us" hope did not pay off.

Read that against the budget: at ~4,500 resident experts a DSpark step misses ~20% of its
~1,000 `(layer, expert)` slots — about 200 loads × 18.8 MB ≈ **3.8 GB per step**, ~0.7 s
at the measured 5.5 GB/s ceiling. **The streaming design lands around 4–6 tok/s on a
general workload** before any smarter placement. The levers that are left: workload-specific
hot sets (a coding-only top set covers noticeably more of coding slots — the Jaccard
overlap between the coding and general top-25% sets is only 0.18–0.31), LRU adaptation
during a session, and prefetching the next layer's likely experts.

## The other three things that make this model unusual

### Engram: 48 random 264-byte reads per token

Two of the 40 layers (1 and 14) carry an n-gram lookup table of 101.5 GB each. For each
token, each engram layer reads **24 rows** — 3 n-gram sizes × 8 heads — and a row is 256 B
of FP8 plus 8 B of UE8M0 scales. That is 48 rows and ~12.7 KB of random reads per token
across the model.

The saving grace is that the addresses depend on the **token ids alone**: the reference
`NgramHashState` maps ids through a compressed vocabulary (99,092 ids, NFKC + lowercase +
whitespace collapse, so `The`/`the`/` THE` hash identically), XORs the previous 3 ids under
per-`(layer, lookback)` odd multipliers, and reduces each hash by a distinct prime into a
disjoint bucket range. No forward pass is needed to know what to read. The engine keeps
the tables on NVMe and reads rows with **buffered** `preadv` in a thread pool — 264 bytes
is far too small for `O_DIRECT` to help — plus a small process-local row cache, which
exact n-gram repeats inside a conversation hit.

*Measured:* 3,792 rows in 0.65 s on a 64-token generation. Engram is not the bottleneck.

### Attention: the prompt only runs half the model

MLA-style with one 512-dim KV latent per token and 64 query heads. Every layer has a
128-token sliding window over its own KV; layers 2–19 add compressed global KV at ratio 2,
layers 20–39 at ratio 1, and only layers 2, 8, 14 and 20 *produce* global KV — everyone
else reads the last producer's cache. Because layer 20's global KV feeds layers 20–39
(CED), **a prompt only needs layers 0–20**; the decoder's window KV is rebuilt from the
last 128 prompt tokens. For a streaming engine that is not a detail: it roughly halves the
expert traffic of prefill.

### DSpark: the drafter is resident, always

Three extra blocks with their own 128-expert MoE (top-3) — 7.2 GB, small enough to keep
resident permanently, and they must be: they run on every step. The drafter takes the
attention inputs of layers 37–39, proposes a block of 5 tokens, and the target model
verifies them. It is lossless, so a speculative run and a plain run differ in speed only;
`SPEC=0` gives the A/B baseline and is the first thing to turn off when chasing a numerics
problem.

## Chunk invariance, and what it cost

The engine is **bit-exact under chunking** for sequences ≤ 512 tokens: for every splitting
tested, and for cache rollback after a 6-token speculative block, the residual stream is
identical, not merely close. That is not free, and the reasons are worth knowing before
touching a GEMM:

* cuBLAS reduces split-K partials in bf16 unless
  `allow_bf16_reduced_precision_reduction` is turned off — the attention softmax then
  amplifies the difference ~2× per layer.
* cuBLAS picks tiling *and* split-K from M, and for several shapes even a row's offset
  inside the tile changes its last bits. So activation GEMMs and last-dim reductions run
  in fixed 16-row tiles regardless of the chunk length. Cost: a 512-token prefill chunk
  issues 32 GEMM launches per projection instead of one, ~+10% on the smoke test.
* The MoE kernel writes one row per `(k, token)` and sums the experts afterwards rather
  than `tl.atomic_add`-ing them in block-scheduling order.

Beyond 512 tokens the guarantee lapses — and so does the reference implementation's, for
the same reason. [`LIMITATIONS.md`](../LIMITATIONS.md) has the exact boundary.

## The FP4 MoE kernel

`tools/fp4_moe.py` is a Triton grouped-MoE forward over the packed FP4 experts, JIT-compiled
on the first call. Three launches, no host sync: group the `(token, k)` pairs by expert slot,
stream each expert's packed rows **once** while decoding FP4 in registers with the hardware
`cvt.rn.f16x2.e2m1x2` instruction, then scatter-add the down projection. Because a dot
product is order-invariant along K, the even and odd nibbles never have to be interleaved —
the activation tile is loaded with stride 2 instead, so the weights need no layout shuffle.

*Measured on GB10:* 193 GB/s effective at decode size (T=1, 6 experts, 0.58 ms), 197 GB/s
(T=6, 30 experts, 2.9 ms), against a ~210 GB/s practical copy ceiling on a nominally
273 GB/s box; relative error 4.4e-3 against the dequantised reference. Prefill (T=512) is
compute-bound at ~23 TFLOPs and unoptimised. Tile width matters more than it should here:
16-byte-wide tiles cap at ~130 GB/s on GB10, 64-byte tiles do not — see
[gotchas](gotchas.md).

If Triton is unavailable the engine falls back to a dequantise-then-GEMM path
(`engine/moe_fallback.py`), which is correct and slow; `/health`'s `engine_config.kernel`
says which one you got.

## Where to go next

* [`NOTES.md`](../NOTES.md) — the running log, with the checkpoint layout, the router,
  hyper-connections, the full landscape survey, and every measurement in context.
* [`LIMITATIONS.md`](../LIMITATIONS.md) — what does not work and exactly why.
* [gotchas](gotchas.md) — the failure modes this design has, in the order you will meet them.
* [benchmarking](benchmarking.md) — how to measure it without fooling yourself.
