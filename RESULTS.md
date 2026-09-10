# RESULTS — DeepSeek-V4.1-Flash on one DGX Spark (GB10, 121 GiB)

**Measured 2026-09-10 on the box described below. Every number here was produced by a run on this
machine; nothing is extrapolated, scaled or quoted from elsewhere.** Where a planned measurement
was not taken it says so instead of guessing. The running log with the bug hunt behind these
numbers is NOTES.md ("Bring-up"); what still does not work is LIMITATIONS.md.

> **Status: work in progress.** Steps 1-5 of the bring-up (smoke, correctness, DSpark, engine/server
> API, serve) are complete and measured. Step 6 (the benchmark sweep) was stopped by the owner after
> the `code` row because the box was needed interactively, so **`prose`, the `angry-birds`/`mario`
> one-shots and the thinking-on run are not measured** and `results/oneshots/` is empty.

## Box and build

| | |
|---|---|
| machine | ASUS Ascent GX10, NVIDIA GB10 (sm_121a), 128 GB unified / 121 GiB visible, 20 cores, 1 NVMe (916 GB) |
| OS / driver | Ubuntu 24.04 (DGX OS base), driver 580.173.02, CUDA 13 |
| python | a venv with torch 2.13.0+cu130, triton 3.7.1, transformers 5.12.1 (docs/install.md) |
| model | `deepseek-ai/DeepSeek-V4.1-Flash`, full 48-shard checkpoint (510 GB) on local NVMe, FP4 routed experts read straight out of the shards |
| engine | this repo: `server/app.py --engine v41` -> `engine/v41_engine.py`, MoE on the Triton FP4 kernel `tools/fp4_moe.py` (`kernel: triton-fp4`) |
| other load | none — the box's other inference container was stopped for the whole session, so the unified pool was ours alone |

The engine streams routed experts: only a resident hot set lives in the GPU arena and every miss is
an O_DIRECT read from the checkpoint. **A speed number from this recipe is meaningless without the
expert hit rate and the GB read that produced it**, so every table below carries them.

## 1. Load

Auto-sized arena, warm start ranked by `results/trace-full-20260910/stats/coverage.json`.

| stage | `max_seq` 8192 | `max_seq` 32768 (the served config) |
|---|---|---|
| non-expert weights (~19 GB) to GPU | 63 s | 63 s |
| DSpark experts (3 x 128 = 7.2 GB) resident | 5 s | 5 s |
| warm start | 3,891 experts / 73.2 GB in **16 s** (4.6 GB/s) | 3,526 experts / 66.3 GB in **14 s** (4.7 GB/s) |
| **total, process start to `ready` / `/health`** | ~85 s | **~90 s** |

Arena at `max_seq 32768`: **73.8 GB = 3,926 slots = 25.6 % of the 15,360 routed experts**
(3,526 LRU + 400 transient). Peak host use 99-101 GiB of 121; MemAvailable never below 19 GiB.

## 2. Correctness — teacher-forced NLL / top-1 vs the pure-torch reference port

`engine/v41_engine.py --teacher-forced corpus/trace_corpus.jsonl --act-quant`: 50 sequences,
each pushed through `Model.forward` in one chunk, next-token NLL and top-1 from the real head.
`--act-quant` matches the tracer's fp8 activation fake-quant (`results/trace-full-20260910/meta.json`);
serving runs with it off, which is strictly more precision.

Config: arena 80.7 GB / 4,291 slots (27.9 % resident), max_seq 8192, spec off, kernel triton-fp4,
act_quant on. 965 s of forward time.
Raw: `results/engine-tf-20260910/teacher_forced_engine_actquant.json`.

| corpus | tokens | reference (`tools/v41_ref.py`) NLL | **engine NLL** | delta | reference top-1 | **engine top-1** |
|---|---|---|---|---|---|---|
| coding | 5,459 | 2.1527 | **2.1586** | **+0.0059** | 0.6384 | **0.6410** |
| general | 5,251 | 3.4124 | **3.4380** | **+0.0256** | 0.4738 | **0.4769** |

Both well inside the ±0.05 nats bar. The serving engine's math — expert arena, Triton FP4 grouped
MoE, engram rows read off NVMe, fixed-tile GEMMs — agrees with the pure-torch port end to end.

## 3. DSpark speculative decoding

`--spec-ab`: one load, `spec` toggled between runs, so all three runs share the arena and the LRU
state. Config: **arena 74.5 GB / 3,960 slots (25.8 % resident), max_seq 8192, kernel triton-fp4,
act_quant off, thinking off**, prompt 16 tokens, 64 output tokens.
Raw: `results/engine-tf-20260910/spec_ab2.json`.

| run | temperature | decode tok/s | steps | accept_len_mean | expert hit rate | NVMe GB | attn s | moe s |
|---|---|---|---|---|---|---|---|---|
| greedy, spec **off** | 0 | 1.75 | 63 | — | 0.826 | 70.4 | 2.86 | 31.42 |
| greedy, spec **on** | 0 | **2.64** | 17 | **3.71** | 0.771 | 86.6 | 1.02 | 24.69 |
| sampled, spec **on** | 1.0 / top_p 0.95 | **2.64** | 20 | **3.40** | 0.794 | 92.0 | 1.21 | 25.55 |

**Greedy speculative output is token-for-token identical to greedy autoregressive output: 64 of 64,
first divergence `None`.** The verify loop is lossless as implemented. The sampled run is coherent.

DSpark is worth **1.5x** here: a 6-token verify block reads more expert bytes than a single token
does (86.6 vs 70.4 GB for the same 64 tokens) but amortises them over 3.7 accepted tokens.

## 4. Served throughput — `code` workload

Server: `./start.sh` with `.env` = `MAX_SEQ=32768`, `DEFAULT_THINKING=off`, `SPEC=1`,
`TRACE_STATS=results/trace-full-20260910/stats/coverage.json`, `ARENA_GB` auto -> **73.8 GB /
3,926 slots / 25.6 % resident, kernel triton-fp4, act_quant off**.
Bench: `bench/bench.py --workload code --runs 2 --osl 512 --ignore-eos`, thinking **off**,
temperature 0.6, top_p 0.95, 1 warm-up + 2 measured runs, **every run exactly 512 completion
tokens** (`finish_reason: length`). Raw: `results/bench-20260910/code.json`.

| run | TTFT | TPOT | decode tok/s | accept_len | expert hit rate | NVMe GB | engram rows |
|---|---|---|---|---|---|---|---|
| warm-up | 12.48 s | 338 ms | 2.96 | 3.47 | 0.820 | 496.7 | 45,440 |
| run 1 | 10.98 s | 369 ms | 2.71 | 3.02 | 0.834 | 517.6 | 51,792 |
| run 2 | 11.12 s | 379 ms | 2.64 | 3.04 | 0.827 | 543.1 | 51,408 |
| **median of the 2 measured runs** | **11.05 s** | **374 ms** | **2.68** | **3.03** | **0.830** | **530.3** | **51,600** |

Where the time goes (run 1): prefill 62 tokens in 10.82 s — 2,473 prefill expert misses = 46 GB at
4.3 GB/s; decode 514 tokens in 188.3 s over 170 DSpark steps — 25,044 decode expert misses = 471 GB
at **2.5 GB/s effective**, `moe_s` 160.5 s, `attn_s` 9.1 s, `engram_s` 3.6 s.

**The headline of this recipe: 0.92 GB of expert weights are streamed from NVMe per generated
token** at a 25.6 % resident set. Attention, the engram lookups and the Triton MoE kernel together
are under 8 % of the decode time; everything else is the SSD.

### Not measured

| planned row | status |
|---|---|
| `prose`, `--runs 2 --osl 512 --ignore-eos` | **not measured** — stopped by the owner |
| `angry-birds` one-shot, thinking off, 8192 max output | **not measured** — stopped by the owner |
| `mario` one-shot, thinking off, 8192 max output | **not measured** — stopped by the owner |
| `angry-birds` one-shot, thinking on, effort 75 | **not measured** — stopped by the owner |
| `results/oneshots/*.html` | **empty** — no one-shot completed |
| long-context (`random --isl 8192`) | never attempted this session |

## 5. Performance work done during bring-up (A/B, same box, same work)

Two bugs outside the model math dominated the first runs. Both A/Bs are honest in the way that
matters here: greedy decoding at a fixed arena size reads the *same* expert bytes before and after,
so only the time changed.

| measurement | before | after |
|---|---|---|
| expert read, **1** in flight (the large-arena decode regime) | 8.11 ms/expert, 2.32 GB/s | **4.78 ms/expert, 3.93 GB/s** |
| expert read, 12 in flight | 4.02 ms/expert, 4.68 GB/s | 3.95 ms/expert, 4.76 GB/s |
| decode, 20 GB arena, 64 greedy tokens, no spec (152.05 GB read both times) | 0.93 tok/s | **1.21 tok/s** |
| decode, ~75 GB arena, 64 greedy tokens, no spec (~70 GB read both times) | 0.76 tok/s | **1.75 tok/s** |
| decode, ~75 GB arena, 64 greedy tokens, DSpark on | 1.74 tok/s | **2.64 tok/s** |

* **The LM head was converted bf16 -> fp32 on every token** — a 2.65 GB allocation per token
  (`head` is [129280, 5120]), plus 132 MB per drafted token for the Markov head. Next to a 74 GB
  arena that pushes the caching allocator into `cudaFree`/`cudaMalloc`. Both are stored fp32 once
  at load now (+1.33 GB and +66 MB resident), which is also what the reference does.
* **The expert reader issued six O_DIRECT reads per expert and synchronised the compute stream six
  times per miss.** An expert is now read as its **two** maximal contiguous file runs (a 1.1 MB
  scale run and a 17.7 MB weight run — the shards group all scales at the front and all weights
  behind them), verified byte-exact against `safetensors.safe_open`; and the pinned staging buffer
  goes straight into the arena with `non_blocking=True` on a **per-io-thread CUDA stream** that
  first waits on the compute stream.

## 6. Reference points (not our measurements)

For scale only — different hardware, all experts resident, no streaming: a public **4x** DGX Spark
TP4 vLLM build reports 39-77 tok/s single stream, TTFT 0.27-0.58 s, DSpark acceptance 3.56
(NOTES.md 0.5). That build needs four boxes and states "TP2 does not fit either way". This repo runs the same model, at FP4 expert quality, on **one** box, at 2.6-2.7 tok/s.
