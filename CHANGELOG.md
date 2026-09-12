# Changelog

What a version means here: this repository is not a library and nothing imports it. What
you depend on is **the defaults the recipe ships and the measurements taken on them**, so
a release is a measurement epoch — the configuration as it stood, and the figures that
belong to it.

- **MAJOR** — the measurement basis changes (different hardware, model or checkpoint).
- **MINOR** — a shipped default changes, or the recipe gains a capability. Your numbers move.
- **PATCH** — documentation, corrections, tooling. Your numbers do not move.

`./run.sh` and the container print the version they were launched from, and the image is
tagged with it: `ghcr.io/<owner>/deepseek-v41-flash-spark:<version>`. A `-wip` version means
exactly what it says: the defaults are not settled and the measurements are incomplete.

## 0.1.0-wip — 2026-09-10

**Work in progress, not a release.** The recipe serves, and the numbers it served at are
in [`RESULTS.md`](RESULTS.md), but the benchmark sweep is one row long, the container has
never been run, and nothing here has been repeated on a second day. This entry records
what exists on the day the engine first served and the container/documentation
scaffolding was added, so that the first real release has something to be a delta from.

Box: one DGX Spark class machine — NVIDIA GB10, `sm_121a`, 128 GB unified memory
(~121 GiB visible), 20 cores, one local NVMe — Ubuntu 24.04 / DGX OS, CUDA 13.

### What works

- **Phase 0 is complete.** Checkpoint layout, architecture notes and the public landscape
  survey in [`NOTES.md`](NOTES.md); the expert-routing tracer (`tools/expert_trace.py`,
  layer-streaming and resumable), the Engram row fetcher (`tools/engram_rows.py`,
  multipart HTTP ranges against the two 101 GB shards), the corpus builder and the
  coverage/LRU analysis (`tools/expert_stats.py`).
- **The full 40-layer routing histogram** over 10,760 teacher-forced tokens, in
  `results/trace-full-20260910/`. At ~4,500 resident experts (84.6 GB): 0.780 static
  coverage, 0.875 LRU hit per token, 0.796 per 6-token block.
- **The teacher-forced check of the pure-torch port**, all 40 layers plus the head: coding
  NLL 2.15 / top-1 63.8%, general NLL 3.41 / top-1 47.4%. A broken port would sit near 10%.
- **The engine runs.** `engine/` loads the real checkpoint and produces coherent greedy
  text: the arena + LRU + transient ring over `O_DIRECT` NVMe streaming
  (`engine/experts.py`), the chunked-prefill/decode-block model with caches
  (`engine/model.py`), Engram rows at serve time (`engine/engram.py`), and the generation
  loop with DSpark drafting and verification (`engine/v41_engine.py`).
- **The Triton FP4 grouped-MoE kernel** (`tools/fp4_moe.py`): 193–197 GB/s effective at
  decode sizes on GB10, relative error 4.4e-3 against the dequantised reference.
- **Chunk invariance.** `engine/model.py` is bit-exact under every chunking tested for
  sequences ≤ 512 tokens, including cache rollback after a 6-token speculative block.
- **The OpenAI-compatible server** (`server/app.py`, standard library only) with the
  thinking/effort mapping, `reasoning_content` streaming, DSML tool-call parsing and
  `x_engine_stats` on every response; 15 end-to-end tests against the mock engine.
- **The launcher and the harness**: `start.sh` / `stop.sh` with port and memory guards,
  `bench/bench.py` with the `x_engine_stats` medians.

### Added in this entry

- `Dockerfile` — arm64, `nvidia/cuda:13.0.2-devel-ubuntu24.04`, torch 2.13.0+cu130 from
  the PyTorch cu130 aarch64 index (which is also where the matching `triton` comes from),
  plus transformers / tokenizers / safetensors / numpy / sympy / huggingface_hub. No
  compile step: the only kernel is JIT-compiled on the box. The devel base rather than
  `-runtime` because Triton needs a `ptxas` that knows `sm_121a`, and
  `TRITON_PTXAS_PATH` points at the toolkit's.
- `compose.yaml` — loopback-only `127.0.0.1:8000`, `./models:/models` and
  `./results:/app/results`, `.env` pass-through, `ipc: host`, `memlock` unlimited, all
  GPUs, and `restart: on-failure:1` so a failing load can never loop the box.
- `run.sh` — `setup` / `serve` / `logs` / `stop` / `shell` / `bench` / `config`, reading
  the same `.env` as `start.sh`.
- `scripts/entrypoint.sh` — the container's `start.sh`: the same env knobs as
  `env.example`, the `MemAvailable` guard, and auto-discovery of the newest
  `results/trace-*/stats/coverage.json` to rank the warm start.
- `scripts/download-model.sh` — resumable `snapshot_download` of
  `deepseek-ai/DeepSeek-V4.1-Flash`, with the 510 GB warning and a free-space check.
- `.github/workflows/image.yml` — build and push to GHCR on `v*` tags and on demand,
  `ubuntu-24.04-arm`, `docker/build-push-action`, GHA cache.
- `.dockerignore`, `VERSION`, `docs/` (install, architecture, openai-api, benchmarking,
  gotchas), this file and `CREDITS.md`.

### Known not to work

Everything in [`LIMITATIONS.md`](LIMITATIONS.md). The short version: the container image
has never been built or run; decode is NVMe-bound at 2.6–2.7 tok/s and nothing overlaps the
expert reads with compute; bit-exactness stops at 512 tokens; long context, concurrency and
model quality beyond teacher forcing are all unmeasured; and one benchmark row is not a
benchmark.

### Measured on it

Everything in [`RESULTS.md`](RESULTS.md), all of it on 2026-09-10 on one GB10 box with the
pool to itself. The headline row, at a 73.8 GB arena (3,926 slots, 25.6 % of the routed
experts) on the `code` workload with DSpark on and thinking off:

| load to `/health` | TTFT (62-token prompt) | decode | acceptance | expert hit rate | NVMe per token |
|---|---|---|---|---|---|
| ~90 s | 11.05 s | 2.68 tok/s | 3.03 | 0.830 | 0.92 GB |

Plus the load breakdown (§1), the teacher-forced agreement with the pure-torch port (§2,
within 0.03 nats), the DSpark spec-on/spec-off A/B (§3, greedy output token-for-token
identical), and the two performance bugs the A/Bs caught (§5).

**Benchmarks are work in progress.** One workload row (`code`) exists. `prose`, both
one-shot generations and every thinking-on run were stopped before they produced a number,
so there is no measured long generation and no thinking-mode figure in this repo at all.
The earlier bring-up figures in `NOTES.md` taken on a 20 GB debug arena (6.9 % of the
routed experts) are a measurement of that arena, not of the recipe — do not quote them.

## 0.5.0-wip — 2026-09-12 (unreleased)

**Choosing what the box is good at becomes a thing you can see.** About 40 % of the routed experts
fit in memory at once, and which 40 % decides both what the model is good at and whether it loads.
Until now that was two numbers in a file and a three-minute wait to find out.

### Added
- **`./tune.sh`** — pick topics, watch what they cost against the memory the box has free right
  now, and start the server from the same screen. Coverage per topic (the fraction of its measured
  routing the budget keeps resident) with the number of tokens each was traced on; the memory
  the selection costs, line by line; the largest keep fraction this box will take.
  `--list`, `--print` and `--write` need no terminal. [`docs/tune.md`](docs/tune.md).
- **`tools/budget.py`** — the cost model behind it, with no torch dependency: slot sizes from the
  kernel's own constant, the KV cache from the checkpoint's shapes, the launch gate from
  `engine/v41_engine.py`.
- **`EXPERT_TOPICS`** composes a keep-set from named topics instead of one of three fixed profiles.
  Each topic is a per-layer expert histogram measured on a corpus of that topic alone, stored in
  `coverage.json`, so composing is arithmetic on numbers already in the checkout.
- **`corpus/fetch_topics.py`** gathers the sources a 35-topic catalogue needs — sixteen programming
  languages from a tree you name, eleven natural languages and eight domain registers from
  Wikipedia — and prints the flags `make_corpus.py` wants.
- **Four checks that need no GPU, no checkpoint and no torch**: `tools/test_budget.py`,
  `tools/test_engine_kwargs.py`, `tools/test_tune_draw.py` and the existing `server/test_server.py`.

### Fixed
- **`EXPERT_TOPICS` had never worked.** `expert_topics` was read inside `V41Engine.__init__` and
  passed by the engine's own CLI, but was never a parameter of it, so every launch through
  `start.sh` raised `TypeError` three minutes in, with the weights already on the GPU.
  `tools/test_engine_kwargs.py` now checks every launcher kwarg against the signature.
- **`VERSION` had said `0.1.0-wip` since the day it was written**, through four tags. The container
  is tagged from that file, so every image built from 0.2.0 onward carried the wrong version.

### Changed
- The recommended keep fraction on a 121 GiB box drops to **42 %**. One prefill chunk needs about
  10 GB on top of everything resident, and the engine's own pre-flight does not know that — it runs
  before the drafter experts, the KV cache and any prefill exist. A 98 GB arena passes it, reports
  ready, and is killed by the memory watchdog on the first request. See
  [`LIMITATIONS.md`](LIMITATIONS.md).

### Known, unfixed
- Long generations still degenerate past the gate's reach: clean at 1,200 tokens, collapsed into
  repeated corrupted CSS by 2,400, with penalties at 0. Not attributed to a layer of the stack yet.

## 0.4.1-wip — 2026-09-12

**Guard rails, after the box had to be power-cycled three times.**

### Added
- A pre-flight that refuses to start when the arena plus its warm-start scratch plus the free-memory
  floor exceeds `MemAvailable`, and a watchdog thread that kills this process rather than let the
  machine thrash when host memory runs out.
- A device-side slot table for the chunked prefill path.

### Fixed
- A published routing-timer reading was wrong and is corrected: `route_s` is nested inside `moe_s`,
  so removing it changed wall-clock time by nothing.

## 0.4.0-wip — 2026-09-12

**A configuration that writes whole files and whole stories.** v0.3.0-wip shipped a default that
scored better on teacher-forced loss and degenerated in free generation; this tag replaces the metric
that allowed it, rebuilds the expert keep-set on a corpus that contains the workloads, and fixes a
real routing bug in the graphed decode path.

### Fixed
- `engine/fastdecode.py`: the router gate runs in fp32, as `Model.moe` does. In bf16 it selected
  different experts for 11-31 % of tokens (RESULTS 4.4).
- `DSV41_FUSED_ATTN` now defaults to 0; the kernel measurably degrades agreement with the reference.
- The expert keep-set is ranked on `results/trace-union` (web, code, configuration, technical prose
  and narrative fiction; 190 sequences, 36,250 tokens) instead of a 50-document Python-only corpus.

### Added
- `corpus/trace_corpus_v2.jsonl`, `corpus/trace_corpus_v3.jsonl` and their sources under
  `corpus/sources/web` and `corpus/sources/prose`.
- Per-category histograms in `coverage.json`, so a keep-set can be built without the raw trace.
- `presence_penalty` / `frequency_penalty` per request and `DSV41_NO_REPEAT_NGRAM`, all defaulting
  to 0 — a frequency penalty fixes prose repetition and corrupts CSS, so none of them ships on.
- `engine/test_spec_lossless.py`; grammar-constrained DSML tool calls via xgrammar
  (`server/tool_grammar.py`, `DSV41_TOOL_GRAMMAR=1`, off by default).
- The engine refuses to start when host memory cannot hold the arena instead of squeezing.

### Changed
- Shipped defaults: `PRUNE_KEEP=0.44 EXPERT_FORMAT=cb3 ARENA_GB=98 TRANSIENT_SLOTS=8 KEEP_FREE_GB=6`
  with the union trace. 44.1 % of routed experts resident, hit rate 1.0.

### Measured on it
17-37 tok/s across nine workloads, prefill 337 tok/s on a 5,014-token prompt, every case passing a
900-2,000-token generation gate with structural checks (RESULTS.md v0.4.0-wip).

## 0.3.0-wip — 2026-09-11

**The shipped default changes: keep 40 % of the routed experts, all resident in the 3-bit CB3
format.** Decode 16.6 → 19.0 tok/s and held-out loss −0.03 (code) / −0.17 (prose) nats against
the 0.2.0-wip default, on the same box (RESULTS.md v0.3.0-wip).

### Added
- `EXPERT_FORMAT=cb3` / `--expert-format cb3`: the resident arena holds 3-bit per-row codebook
  experts (`tools/cb3.py`, 14.45 MB each), packed on the GPU at warm start from the FP4 shards;
  decode runs the CB3 v3 Triton kernel (`tools/cb3_moe.py`, 182 GB/s of expert bytes, 0.79x the
  FP4 kernel's time per expert); prefill unpacks to FP4 codes and runs the FP4 kernel. Unit test
  `tools/test_cb3_moe.py`.
- `tools/fp8_linear.py::fp8_grouped_linear`: the `wo_a` projection runs from its stored FP8
  (`DSV41_WOA_FP8=0` restores the bf16 einsum).
- `tools/decode_attn.py`: fused decode attention (bf16 keys, fp32 softmax with the sink, two KV
  segments without a copy; `DSV41_FUSED_ATTN=0` restores the fp32 torch path).
- `tools/fp32_skinny.py`: split-K fp32 kernel for the Hyper-Connection mixing GEMMs
  (`DSV41_HC_KERNEL=0` restores `F.linear`).
- `--prune-select global` / `PRUNE_SELECT`: cross-layer keep-set ranking; measured worse than
  uniform on the held-out corpus (NOTES.md 2026-09-11 10:00) and left as a documented option.
- CUDA graphs per step segmented at the Engram layers (`DSV41_GRAPH_SEGMENTS=0` restores per-layer
  graphs; no measurable gain either way), pinned Engram staging (`DSV41_ENGRAM_PINNED=1`, off).
- LM head and DSpark Markov head loaded in their stored bf16 (`DSV41_HEAD_FP32=1` restores fp32).

### Changed
- Default configuration in `env.example`/`docs/install.md`: `PRUNE_KEEP=0.40 EXPERT_FORMAT=cb3`
  for a 128 GB box. Warm start is 183 s in this format (19 s for FP4).
- Verify step with everything resident: 168 → 147 ms (FP4, keep 31 %); RESULTS.md addenda 2.9-2.11.

### Measured on it
RESULTS.md v0.3.0-wip: 18.98 tok/s greedy decode, TTFT 9.81 s on a 1,806-token prompt, held-out
1.5384 / 3.2087 nats. Not measured: thinking-on in this configuration, 8k+ prompts in this
configuration, sampled A/B, the image end to end.

## 0.2.0-wip — 2026-09-11

**The model math fix and the resident pruned configuration.** Everything in 0.1.0-wip ran on a port
with a transposed Hyper-Connection residual mix; this tag fixes it and rebuilds the decode path.

### Fixed
- `tools/v41_ref.py::hc_post`: sum over the first index of `comb` (combᵀ · residual), as in the
  reference `Block.hc_post`. Teacher-forced coding loss 2.16 -> 1.37 nats; greedy output no longer
  stutters; DSpark acceptance 2.4 -> 3.75 on code. (commit bd24743, 2026-09-11 00:50)
- Transient prefill ring must hold a whole layer (>= 384) unless every routable expert is resident.

### Added
- `engine/fastdecode.py`: CUDA-graph decode path (per-layer graphs, host slot resolve between them),
  fused HC Sinkhorn Triton kernel (`engine/hc_sinkhorn.py`), bf16 head, masked fixed-length indexer.
  Verify step 436 -> 173 ms with everything resident.
- `tools/fp8_linear.py`: dense projections in stored FP8 (Triton, 1.9x bf16 GEMM at decode size);
  `v41_ref.dense()` dispatch; `DSV41_DENSE_FP8=0` restores bf16 copies.
- Pruned all-resident serving: `--prune-keep F` (router restricted to the top-F experts per layer by
  trace frequency, exactly those warm-started), `--prune-sweep` teacher-forced ladder,
  `--transient-slots`, `--keep-free-gb`, `--arena-gb` pinned sizing.
- `engine/codebook_sim.py`, `tools/cb3.py`, `tools/cb3_moe.py`: 3-bit per-row codebook expert format
  (simulation, packer, and a correct-but-slow kernel).
- `engine/diag_decode.py` (decode == prefill consistency, per layer), `engine/test_fastdecode.py`,
  `engine/profile_decode.py`, `engine/profile_fast.py`.
- Held-out corpus `corpus/heldout_corpus.jsonl` (sources in `corpus/heldout_sources/`).

### Measured (RESULTS.md §v0.2.0-wip)
keep 31 % resident: 12.9 tok/s at +0.07 / +0.19 nats; unpruned streaming 3.5 tok/s; full ladder there.

### Process
Conventions applied from this tag on: no benchmark sweeps (single decode numbers only); docs
append-only with dates and per-tag sections; credits limited to the model vendor, the author's other
Spark recipes and the toolchain.
