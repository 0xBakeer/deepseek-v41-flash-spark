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
