# Gotchas

Every one of these was hit for real, or is written down because the code had to be shaped
around it. Sources: [`NOTES.md`](../NOTES.md) and [`LIMITATIONS.md`](../LIMITATIONS.md).

## The box hard-resets when host memory goes negative

There is no OOM killer moment to catch and no log line to find afterwards. On unified
memory the GPU allocator and the page cache draw on the same pool; push `MemAvailable`
towards zero and the machine goes down. Everything below follows from that:

* The engine keeps a hard `keep_free_gb` floor (20 GB) under `MemAvailable` when sizing the
  arena, and the auto factor was lowered from 0.88 to 0.82 after a warm start peaked at
  111 GiB of 121 with 10 GiB left.
* `start.sh` and the container entrypoint **refuse to start** below `MIN_FREE_GIB` (90),
  naming the processes that hold the pool.
* `stop.sh` and `./run.sh stop` block until the pool is actually back. Pinned and
  page-cache-backed memory is reclaimed lazily; starting the next server before that lands
  is the classic way to wedge the box.
* `compose.yaml` uses `restart: on-failure:1`, not `unless-stopped`. A load that fails
  usually fails because the pool is spoken for, and restarting into the same condition
  turns "the server did not start" into "the power button is the only way back".

## Do not set a memory limit on the container

`--memory` / `mem_limit` on this hardware is a cap on **GPU** allocations too, and the
arena auto-sizer will silently shrink to fit it — giving a small hot set and a permanently
NVMe-bound server with no error anywhere. There is no cgroup knob that means "host RAM but
not GPU" here. Leave it unset.

## `torch.cuda.mem_get_info()` lies on GB10

It counts the page cache as used. Right after the checkpoint download it reported **32.0 GB
free on a box with 99.9 GiB `MemAvailable`** — which would have handed the engine a 23 GB
arena, about 7% of the routed experts, and a server that streams every token. The engine
now takes the larger of `mem_get_info()` and `/proc/meminfo`'s `MemAvailable` and logs
both. If you write anything that sizes a buffer from free memory on this box, do the same.

## The transient ring must hold a full layer

Prefill touches nearly every expert of a layer (370–381 of 384 measured at layer 0), so
prefill misses go to a small ring of transient slots instead of through the LRU — otherwise
every prompt evicts the hot set. The ring's size is not a tuning knob with a soft floor:
below 384 slots it **wraps inside a single layer** and the engine computes that layer with
the wrong experts. No exception, no warning — just an 0.88 relative error in the output,
which is exactly what the first smoke test produced. Default 400. Do not lower it.

## An expert is two reads, not six and not one

The six tensors of an expert (`w1/w2/w3` × weight and scale) are not adjacent in the shard:
all the scales sit near the front, all the weights far behind, but within each group an
expert's three tensors are contiguous. Read it as six `preadv`s and you pay six `O_DIRECT`
round trips — and at a large arena a decode step misses only about one expert per layer, so
nothing else is in flight to hide the latency. The rate collapses from ~4.7 GB/s to
~0.6 GB/s. `ShardFile.expert_runs` groups them into exactly two runs: 1.1 MB of scales,
17.7 MB of weights.

## `O_DIRECT` alignment, and the last tensor of a shard

`O_DIRECT` reads must start and end on 4 KiB boundaries and land in an aligned buffer, so
every read is widened outward to `ALIGN` and the wanted bytes are sliced out afterwards. Two
consequences that bite:

* The staging buffers are pinned and over-allocated by `8 × ALIGN` for exactly that
  widening.
* The **aligned tail can run past EOF** on the last tensor of a shard, and the short read
  that comes back is not an error — the loop tops up until it has what it asked for and
  accepts a short final read. Code that treats a short `preadv` as a failure will break here
  only on some shards, which is the worst way to find out.

And the whole path needs a real filesystem: `O_DIRECT` does not work on overlayfs, so the
checkpoint must be a **bind mount**, never a `COPY` into the image or a network filesystem.

## Engram rows are too small for `O_DIRECT`

48 random 264-byte reads per token. `O_DIRECT` on those is all overhead: the engram reader
uses **buffered** `preadv` in a thread pool with `POSIX_FADV_RANDOM` and a small
process-local row cache, and it is not the bottleneck (3,792 rows in 0.65 s on a 64-token
generation). Do not "optimise" it into the `O_DIRECT` path.

## The CDN rate-limits single-range requests

`tools/engram_rows.py` fetches only the n-gram rows a corpus needs, straight out of the two
101 GB shards on the Hub, without downloading them. Single-range requests get HTTP 429 at
around 400/s. **Multipart** range requests are the way: 500 rows per request, ~314 requests
per layer, 10–11 s for a 10,760-token corpus. If you rewrite that fetcher, keep the
multipart batching or you will be rate-limited into hours.

## cuBLAS makes the same GEMM give different bits

Two separate mechanisms, both of which had to be disarmed for the engine to be chunk-invariant:

* **split-K reduction in bf16.** `F.linear(x[:6], w) != F.linear(x, w)[:6]` by ~2.4e-3 on
  the N=512 `wkv` projection, and the attention softmax amplifies that ~2× per layer.
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` cuts it to
  ~9e-5.
* **tiling chosen from M.** cuBLAS picks its tiling and split-K from the number of rows, and
  for several shapes even a row's *offset inside* the tile changes its last bits — 8 and 16
  are offset-invariant, 32/64/128 are not. So every activation GEMM and last-dim reduction
  runs in fixed 16-row tiles. It costs about +10%: a 512-token prefill chunk issues 32 GEMM
  launches per projection instead of one.

## Bit-exactness stops at 512 tokens

Beyond ~1024 tokens of context the indexer's top-512 stops keeping every visible compressed
position, and which 512 it keeps is decided by scores computed against a key cache whose
*length* differs between a chunked and a single-chunk run. Ties then break differently and
the two runs can select different positions. The engine is close but not identical past that
point — and so is the reference implementation, for exactly the same reason. Do not write a
long-context regression test that asserts equality.

Related: `Caches.rollback(n)` only restores the compressor's pending token if `n` lies inside
the last forwarded chunk (or exactly at its start). Further back it raises rather than
silently producing a wrong latent. That covers the speculative-decoding use and nothing else.

## 16-byte tiles cap at ~130 GB/s on GB10

The FP4 MoE kernel is bandwidth-bound on the 18.8 MB every active expert weighs, so tile
width is not a micro-optimisation. Narrow (16-byte) tiles top out around 130 GB/s on this
part; the 64-byte-wide tiles the kernel uses reach 193–197 GB/s effective at decode sizes,
against a ~210 GB/s practical copy ceiling on a nominally 273 GB/s box. If you retune
`BM`/`BN`, watch the effective bandwidth, not the occupancy.

## `persistent_topk` does not fit in GB10's shared memory

This one is inherited, and it is the reason no upstream engine runs this model on one
Spark. The sparse-attention indexer's `persistent_topk` kernel wants 128 KB of shared memory
per block; GB10 has 99 KB. The four-Spark vLLM build works around it with
`top_k_per_row_decode` instead, plus block size 64/128 for the sparse SWA and indexer caches.
Anyone porting a datacenter kernel to this box should expect to meet the same wall — and
`DeepSelect`, DeepSeek's own top-k kernel, is `sm_100a`/`sm_103a` only.

## Triton has to be able to compile for `sm_121a`

The MoE kernel is JIT-compiled on the first call and emits inline PTX
(`cvt.rn.f16x2.e2m1x2`), so `ptxas` inside the container must know the target. Triton wheels
bundle their own `ptxas`, and that copy has shipped behind the driver before — the CUDA-12.8
one in the triton 3.5 wheels could not name `sm_121` at all. The image is therefore built on
`nvidia/cuda:13.0.2-devel-ubuntu24.04` with `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
If a kernel launch dies with something about `gpu-name` or an unknown target, that variable
is the first thing to check.

If Triton cannot compile at all the engine falls back to a dequantise-then-GEMM path — it
is correct and very slow, and it will not announce itself except in `engine_config.kernel`
on `/health`. Check that field before believing a slow row.

## The warm start is not optional (but it is silent)

Without a `coverage.json` the arena is filled in `(layer, expert)` index order, which is a
measurably worse hot set than the traced one, and the only sign is a lower `expert_hit_rate`
in the benchmark. The container's entrypoint auto-discovers the newest
`results/trace-*/stats/coverage.json` and logs which one it used, or says it is falling back
to index order. Read that line.

## First token can be minutes, and startup can be tens of minutes

The socket is bound only after the warm start has read tens of GB off NVMe (measured:
~63 s for the 19 GB of non-expert weights, then 73.2 GB of experts in 16 s), and a cold
prompt misses almost every expert. Health waits are 20 minutes native, 45 in the container,
and any client timeout has to be set accordingly. Nothing is wrong at minute 6.

## Chunk counts lie under speculative decoding

Several accepted tokens arrive per SSE chunk. Use `usage.completion_tokens`; counting chunks
under-reports decode speed by 3–5×. See [benchmarking](benchmarking.md).
