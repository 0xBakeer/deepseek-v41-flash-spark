# LIMITATIONS

Honest list of what this repo does not (yet) do, with the exact reason. Updated per phase.

## Status: it serves (2026-09-10), and it is slow

The engine, the server, the launcher and the bench all run on the full checkpoint. Measured numbers
are in RESULTS.md. What follows is what is still missing or broken.

### Speed

* **2.6-2.7 tok/s decode, ~11 s TTFT on a 62-token prompt** (RESULTS.md 4). That is 15-30x slower
  than the same model on four DGX Sparks with every expert resident. The cause is not a mystery and
  not fixable by tuning: at a 25.6% resident expert set the engine streams **0.92 GB of expert
  weights per generated token** and the NVMe delivers ~2.5 GB/s at the read sizes and queue depths
  a decode step produces. Attention, the engram lookups and the Triton MoE kernel together are
  under 8% of decode time.
* **Nothing overlaps the expert reads with compute.** `resolve()` blocks the whole model while a
  layer's misses are read, then the GPU computes with the NVMe idle, forty times per step. Real
  prefetching is not possible layer-to-layer (layer L+1's routing does not exist until layer L has
  run), but a router-lookahead or a speculative prefetch of the DSpark block's likely experts was
  never attempted.
* **Prefill has no Decoder SWA Bounded Replay.** The engine runs all 40 layers over the prompt like
  the reference does; CED means layers 21-39 only need the last 128 prompt tokens, so a correct
  implementation would roughly halve prefill expert traffic (NOTES.md 0.2). Not implemented, so
  TTFT is about twice what it should be.
* **The arena is sized once at load and never adapts.** No per-workload hot set (the trace shows
  coding-only and general-only top sets overlap by a Jaccard of only 0.18-0.31), no promotion of
  experts a long session keeps hitting beyond plain LRU, no prefetch on a session's first turn.

### Not measured this session

* **Only the `code` benchmark row exists.** `prose`, both one-shot games (`angry-birds`, `mario`)
  and the thinking-on run were stopped by the owner before they produced a number, so
  `results/oneshots/` is empty and **there is no measured evidence in this repo that the engine can
  produce a long (thousands of tokens) generation**, nor any thinking-mode number at all. Two
  earlier attempts were killed externally and produced nothing.
* **Long context is unmeasured.** Everything here ran at prompts of 16-62 tokens. `MAX_SEQ` is
  32768 and the caches are allocated for it, but no run has gone past a few hundred positions, so
  the indexer/candidate path, the compressed-KV growth and the hit rate at 8k+ context are
  untested at serving time.
* **No quality evaluation beyond teacher forcing.** The ±0.05 nats agreement with the pure-torch
  port (RESULTS.md 2) proves the *engine* matches the *port*; it does not prove either matches the
  reference tilelang kernels, and no benchmark suite (MMLU, HumanEval, ...) was run.
* **Batch size is 1 and there is no concurrency story.** The server serialises requests on one
  lock. It also only notices a dead client when it next writes a chunk, so a request that was
  already queued when its client died keeps the engine busy for its whole `max_tokens` budget;
  there is no cancel endpoint and no queue cap. `/health` reports `busy` honestly, and the only
  recovery is a restart.

### Still true from Phase 0

* **The tracer, and the engine's exactness guarantee, are for sequences of <= 512 tokens.** Beyond
  ~1024 tokens the indexer's top-512 starts pruning and a chunked run can select different
  compressed positions than a single-chunk run. The reference has the same property.
* **The corpus is small** (10,760 tokens, 50 sequences, two categories) — enough for a coverage
  shape and for a teacher-forced check, not for per-expert frequencies in the tail.
* **Images are rejected.** The vision tower is not loaded and `/v1/chat/completions` returns 400
  for image content.
* **The container image is untested.** `Dockerfile` / `compose.yaml` / `run.sh` /
  `scripts/entrypoint.sh` exist and `docker compose config` resolves, but no image has been
  built or run on the box yet — the first build is the GitHub Actions arm64 job on the
  `v0.1.0-wip` tag, and nothing has served a request from a container. The native path
  (`./start.sh`) is the one every number in RESULTS.md came from.
* **There is no `setup.sh` and no lockfile.** The native path expects an interpreter that
  already has torch (CUDA 13 / sm_121), triton, transformers and safetensors; docs/install.md
  lists the versions that were used, but nothing pins them.

## Known blockers for a single-box recipe (from the size arithmetic, NOTES.md 0.8)

* 288.8 GB of FP4 routed experts against ~85-90 GB of expert budget = ~1.3 bits per weight average
  if everything must be resident. **No all-resident scheme meets the Q4-class quality floor.** The
  only quality-preserving path is the resident hot set plus NVMe streaming this repo implements —
  and the measured price of that choice is 2.6 tok/s.
* No engine has a single-GPU expert-streaming path for `deepseek_v41` today. vLLM (`dsv41-feat`) and
  SGLang both assume all experts resident across TP ranks; the tonyd2wild 4x Spark build is the
  closest working code (engram-on-disk + SM12x fixes) but states "TP2 does not fit either way".

## Bugs found and fixed during bring-up (2026-09-10)

Listed because the exact errors are useful to whoever reads the code next; all four are fixed.

* `engine/engram.py`: `self.rows = w["shape"][0]` in `__init__` shadowed the `rows()` method →
  `TypeError: 'int' object is not callable` on the first engram layer of the first forward.
  Renamed to `n_rows`.
* `engine/v41_engine.py`: the arena was auto-sized from `torch.cuda.mem_get_info()`, which on GB10
  counts the host page cache as used — it reported 32.0 GB free on a box with 99.9 GiB
  MemAvailable, i.e. a 23 GB arena (7% resident) instead of ~75 GB. Now takes the larger of that
  and `/proc/meminfo` MemAvailable, with a hard `keep_free_gb` floor (20 GB).
* `engine/model.py`: the LM head and the DSpark Markov head were converted bf16 → fp32 on every
  token / every drafted token (a 2.65 GB allocation per token). Stored fp32 once at load.
* `engine/model.py::dspark_draft`: the confidence head was fed the RMS-normed hidden and squashed
  with a sigmoid; the reference (`inference/model.py::DSparkBlock.forward_head`) feeds it the
  un-normed `hc_pre` output and returns the raw projection. Fixed — no effect on any measurement
  here, because adaptive verification is off (the confidence is reported, not acted on).
* `engine/experts.py`: `ShardFile.expert_span`'s docstring claimed an expert's six tensors are
  contiguous in the shard. They are not — they are two runs (scales at the front of the file,
  weights far behind). Corrected, and `expert_runs()` now reads two ranges instead of six.

## Chunk invariance of the engine (2026-09-10)

`engine/model.py` is now bit-exact under chunking: for every splitting tested in
`engine/test_layers.py` (even, odd, 1-token chunks, many chunks) and for cache rollback after a
6-token speculative block, the residual stream after layer 3 is **identical**, not merely close.
That took making every op independent of how many rows are in the call:

* `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` -- cuBLAS otherwise
  reduces split-K partials in bf16.
* all activation GEMMs and all last-dim reductions run in fixed 16-row tiles (`v41_ref.mm`,
  `v41_ref.tiled_rows`); cuBLAS picks tiling AND split-K from M, and for several shapes even a
  row's *offset inside* the tile changes its last bits (8 and 16 are offset-invariant, 32/64/128
  are not).
* the compressed KV block handed to the attention softmax is always `index_topk` wide, and the
  indexer scores in fixed 512-key blocks, so the attention/score GEMM shapes do not grow with the
  chunk.
* `tools/fp4_moe.py` writes one row per (k, token) and sums the experts afterwards instead of
  `tl.atomic_add`-ing them together in block-scheduling order.

Known remaining inexactness:

* **Long context is not covered by this guarantee.** Once the number of compressed positions
  exceeds `index_topk` (512, i.e. ~1024 tokens at ratio 2) the indexer's top-k stops keeping every
  visible position, and which 512 it keeps is decided by scores computed against a key cache whose
  *length* differs between a chunked and a single-chunk run. Equal scores are then broken
  differently and the two runs can select different compressed positions. The bit-exactness above
  is verified only for T <= 512; beyond that the engine is close but not identical, and so is the
  reference (`inference/model.py` has exactly the same property).
* `Caches.rollback(n)` can only restore the compressor's pending token if `n` lies inside the last
  forwarded chunk (or exactly at its start); rolling back further raises rather than silently
  producing a wrong latent. That covers the speculative-decoding use (roll back into the verify
  block) and nothing more.
* The fixed-tile GEMMs cost throughput: a 512-token prefill chunk issues 32 GEMM launches per
  projection instead of 1, and the always-512-wide compressed KV block does more attention work
  than a short prompt needs. Measured cost on the 4-layer smoke test is roughly +10%.
