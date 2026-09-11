# RESULTS — DeepSeek-V4.1-Flash on one DGX Spark (GB10, 121 GiB)

> **This file is append-only history.** Each tag has its own section with date, time and the exact
> configuration. Superseded numbers stay in place and are annotated; nothing is deleted.
> Sections: [v0.1.0-wip (2026-09-10)](#v010-wip--2026-09-10) · [v0.2.0-wip (2026-09-11)](#v020-wip--2026-09-11)

---

## v0.1.0-wip — 2026-09-10

> **Superseded (annotated 2026-09-11 09:20):** every number in this section was measured with a
> bug in the ported model math (`v41_ref.hc_post` mixed the Hyper-Connection residual with the
> transposed matrix). The engine ran, the numbers are what it did that day, but the model quality
> behind them was wrong (teacher-forced coding loss 2.16 nats instead of 1.37) and the DSpark
> acceptance was depressed (~2.4-3.0 instead of 3.0-3.75). See v0.2.0-wip below for the corrected
> state; NOTES.md ("2026-09-11 00:50") has the bug hunt.

**Measured 2026-09-10 on the box described below. Every number here was produced by a run on this
machine; nothing is extrapolated, scaled or quoted from elsewhere.** Where a planned measurement
was not taken it says so instead of guessing. The running log with the bug hunt behind these
numbers is NOTES.md ("Bring-up"); what still does not work is LIMITATIONS.md.

> **Status: work in progress.** Steps 1-5 of the bring-up (smoke, correctness, DSpark, engine/server
> API, serve) are complete and measured. Step 6 (the benchmark sweep) was stopped after
> the `code` row because the box was needed for interactive use, so **`prose`, the `angry-birds`/`mario`
> one-shots and the thinking-on run are not measured** and `results/oneshots/` is empty.

## Box and build

| | |
|---|---|
| machine | ASUS Ascent GX10, NVIDIA GB10 (sm_121a), 128 GB unified / 121 GiB visible, 20 cores, 1 NVMe (916 GB) |
| OS / driver | Ubuntu 24.04 (DGX OS base), driver 580.173.02, CUDA 13 |
| python | a venv with torch 2.13.0+cu130, triton 3.7.1, transformers 5.12.1 (docs/install.md) |
| model | `deepseek-ai/DeepSeek-V4.1-Flash`, full 48-shard checkpoint (510 GB) on local NVMe, FP4 routed experts read straight out of the shards |
| engine | this repo: `server/app.py --engine v41` -> `engine/v41_engine.py`, MoE on the Triton FP4 kernel `tools/fp4_moe.py` (`kernel: triton-fp4`) |
| other load | none — the box's other inference container was stopped for the whole of these runs, so the unified pool was ours alone |

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
| `prose`, `--runs 2 --osl 512 --ignore-eos` | **not measured** — not run in this tag (box needed for interactive use) |
| `angry-birds` one-shot, thinking off, 8192 max output | **not measured** — not run in this tag (box needed for interactive use) |
| `mario` one-shot, thinking off, 8192 max output | **not measured** — not run in this tag (box needed for interactive use) |
| `angry-birds` one-shot, thinking on, effort 75 | **not measured** — not run in this tag (box needed for interactive use) |
| `results/oneshots/*.html` | **empty** — no one-shot completed |
| long-context (`random --isl 8192`) | never attempted in this tag |

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


---

## v0.2.0-wip — 2026-09-11

Measured 2026-09-11 00:50-09:15 on the same box (Qwen container stopped, pool ours alone), same
checkpoint. Commits `bd24743` (hc_post fix) .. `22bd9a8`+ (FP8 dense, pruning, CB3). Python venv as
in v0.1.0-wip. Every row below is one run of the stated command; no benchmark sweeps were run
(this recipe records a single decode number per configuration).

### 2.1 The bug and what it changed (2026-09-11 00:50, commit bd24743)

`tools/v41_ref.py::hc_post` summed the 4x4 Hyper-Connection `comb` matrix over the wrong index
(comb @ residual instead of the reference's combᵀ @ residual). Found by proving decode == single-chunk
prefill bit-for-bit at every layer (so caches were innocent) and re-reading the reference line by line.

| teacher-forced, trace corpus (engine, one chunk per sequence) | before fix | after fix |
|---|---|---|
| coding NLL / top-1 (5,459 tokens) | 2.1599 / 63.9 % | **1.3708 / 74.4 %** |
| general NLL / top-1 (5,251 tokens) | 3.4263 / 47.1 % | **2.8635 / 55.1 %** |

Same code prompt, greedy: before the fix every path stuttered ("LRLR", "time-to-llive"); after it,
clean production-quality code. DSpark acceptance length on that prompt 2.4 -> 3.75.

### 2.2 Decode paths (2026-09-11 00:10-07:05)

`engine/fastdecode.py`: CUDA graphs per layer (attention+HC+router graph, host slot resolve, MoE+residual
graph), fused Sinkhorn Triton kernel, bf16 head, fixed-length masked indexer scoring.
`tools/fp8_linear.py`: dense projections read in their stored FP8 form (Triton, 223 GB/s of FP8 at
M=6, 1.9x the bf16 GEMM); the bf16 copies are gone, which grew the auto arena from 74 to 79 GB.

| verify step (6 tokens), everything resident | wall |
|---|---|
| reference path (`Model.forward`), 2026-09-10 23:5x | 436 ms |
| fast path, bf16 dense (00:10) | 183 ms + 16 ms draft |
| fast path, FP8 dense (07:00) | **173 ms + 15 ms draft** |

Greedy argmax agreement fast vs reference path: 100 % on the tested positions; hidden states differ
2-5 % from bf16 GEMM noise amplified by near-tie router flips (documented in fastdecode.py).

### 2.3 Speed ladder (greedy, temperature 0, same 40-token code prompt, 160-200 output tokens, DSpark on, fast path, FP8 dense)

| configuration (all 2026-09-11) | resident experts | decode tok/s | accept len | hit rate | NVMe GB / request |
|---|---|---|---|---|---|
| unpruned, streaming, arena 79 GB (07:01) | 27 % | 3.5 | 3.24 | 0.826 | 208 |
| keep 40 % (07:06) | 68 % of kept | 6.4 | 3.02 | 0.943 | 78 |
| keep 30 % (07:04) | 91 % of kept | 9.5 | 2.76 | 0.986 | 23 |
| **keep 31 %, arena 90.5 GB = 4,813 slots, transient ring 16 (08:12)** | **100 %** | **12.9** | 2.99 | 1.000 | 0.08 |
| keep 25 %, arena 79 GB (07:00) | 100 % | 13.6 | 3.09 | 0.999 | 7 |

Prefill (from the 2026-09-10 23:xx prefill work, still valid): 1,860-token prompt TTFT
118.5 s -> 33.7 s with 2048-token chunks + Decoder SWA Bounded Replay; short prompts 5-11 s.

### 2.4 Quality ladder of pruning (teacher-forced, held-out corpus `corpus/heldout_corpus.jsonl`: code and prose the trace never saw; 5,444 + 5,270 tokens)

Router restricted per layer to the top-N experts by trace frequency (mixed profile); loss in nats.

| kept / layer | coding NLL (Δ) | general NLL (Δ) | time |
|---|---|---|---|
| 384 (100 %) | 1.5067 | 3.1884 | 02:45 |
| 192 (50 %) | 1.5232 (+0.017) | 3.2528 (+0.064) | 02:45 |
| 154 (40 %) | 1.5285 (+0.022) | 3.3017 (+0.113) | 02:45 |
| 154 (40 %) + all kept experts at simulated 3-bit codebook | 1.5392 (+0.033) | 3.2122 (+0.024) | 07:50 |
| 120 (31 %) — the resident configuration above | 1.5729 (+0.066) | 3.3788 (+0.190) | 09:13 |
| 116 (30 %) | 1.5962 (+0.090) | 3.4241 (+0.236) | 02:45 |
| 116 (30 %) + coldest 40 % at simulated 3-bit | 1.5817 (+0.075) | 3.4187 (+0.230) | 08:38 |
| 96 (25 %) | 1.6687 (+0.162) | 3.5079 (+0.320) | 02:45 |

In-sample (trace corpus) deltas are in NOTES.md and are slightly smaller. The simulated 3-bit rows
use `engine/codebook_sim.py` (per-row 8-of-16 subset of the FP4 grid, 21 % relative weight error);
the packed format `tools/cb3.py` is bit-exact with it, its kernel `tools/cb3_moe.py` is correct but
not yet fast (54 GB/s vs 190 for FP4), so no CB3 speed row exists yet.

### 2.5 NVMe (2026-09-10 16:5x, O_DIRECT, 18.8 MB objects; unchanged)
1 in flight 4.1 GB/s · 8 in flight 5.4 GB/s · 32 in flight 5.6 GB/s.

### What is not measured in this tag
Thinking-on decode, long-context (>2k) serving, sampled (temperature 1.0) quality A/B, any bench
sweep, the CB3 format at speed, the container image end to end.

### 2.6 Addendum 2026-09-11 09:30-09:50 — decode step after the routing fix (same config as the keep-31 % row)

| change (commit) | verify step, everything resident | e2e decode tok/s (greedy, code prompt, 200 tokens) |
|---|---|---|
| baseline of 2.3 (22bd9a8) | 195 ms + 15 ms draft | 12.9-13.1 |
| torch routing for decode-sized calls instead of the per-arena-slot Triton router, bf16 gate GEMM, device slot LUT (9f172fb) | **168 ms + 15 ms draft** | **15.2-15.7** (acceptance 2.97-3.12) |
| Engram rows read in background threads, overlapped with the graphs (next commit) | unchanged | 15.4 (within run-to-run noise; the reads were 16 ms/step, now hidden) |

Profile of the 168 ms: expert kernels ~86 ms (at the 273 GB/s floor for 30 experts x 18.8 MB x 40
layers), FP8 dense ~29 ms (at floor), remaining bf16 GEMMs (wo_a, head, draft) ~20 ms, fp32 mixing
GEMMs ~8 ms, ~3,000 small elementwise/reduction kernels ~25 ms. Run-to-run spread of the e2e number
is ±5 % (greedy acceptance varies with bf16 nondeterminism: 2.97-3.12 on the same prompt).

### 2.7 Addendum 2026-09-11 09:50 — thinking on (served, keep 31 % resident, arena 90.5 GB, transient 8, LUT)

One request through the gateway, `chat_template_kwargs.thinking=true`, `reasoning_effort=high`,
temperature 0, 400 tokens (all reasoning): **TTFT 3.9 s, decode 21.4 tok/s, DSpark acceptance 4.11**,
hit rate 1.0. Same prompt with thinking off (2.6 addendum): 15.2-15.7 tok/s at acceptance ~3.

### 2.8 Addendum 2026-09-11 10:20 — long prompt through the served resident config (keep 31 %, arena 90.5 GB, transient 8, LUT)

One request through the gateway with an 8,192-token prompt (the server's context clamp), greedy,
thinking off, 200 output tokens: **TTFT 39.6 s (207 prompt tok/s), decode 16.9 tok/s, acceptance
3.28**, output a coherent summary of the prompt. The 8k prefill runs through the chunked encoder +
decoder-replay path (2048-token chunks); decode at an 8k KV is not slower than at 100 tokens
because the CSA2 index keeps the attended set at 512 tokens.

### 2.9 Addendum 2026-09-11 10:55 — FP8 grouped `wo_a` kernel and fused decode attention (same served config)

Step A/B on `engine/profile_fast.py` (keep 31 %, arena 90.5 GB): **165.7 → 152.7 ms** verify step,
draft 14.3 → 13.7 ms. Two hundred greedy tokens, same prompt and flags as 2.6: **16.86 tok/s**
(acceptance 3.06) with the new kernels vs 15.28 (acceptance 2.97) with `DSV41_WOA_FP8=0
DSV41_FUSED_ATTN=0` back to back. The `wo_a` projection now runs from its stored FP8 (7.95 ms/step,
was 13.89 as a bf16 einsum) and attention scores/softmax/PV run in one Triton kernel with bf16
keys and fp32 math (1.0 ms/step, was 3.1 fp32 SIMT). Unit tests in `engine/test_kernels.py`;
details and caveats in NOTES.md (2026-09-11 10:25-10:55).

### 2.10 Addendum 2026-09-11 11:20 — split-K fp32 kernel for the HC mixing projections, no `kv_all` copy

The two Hyper-Connection mixing GEMMs per layer (M=6, N=24, K=20480, fp32) ran on a cuBLAS kernel
at 29 GB/s (84 µs each); a split-K Triton kernel (`tools/fp32_skinny.py`, fp32 math, 4e-7 relative
to cuBLAS, both at the fp32 floor against an fp64 check) runs them at 108 GB/s (22.7 µs). The
compressor projections (N=512) stay on cuBLAS, which is faster there. The attention kernel now
reads the window ring and the CSA2 rows through two base pointers instead of a concatenated copy
(bit-identical output). Verify step on `engine/profile_fast.py`: **152.7 → 147.2 ms**, draft
13.7 → 13.0 ms; GPU time per step −9.1 ms.

The single 200-token greedy decode line moved the other way: 16.71 tok/s (acceptance 2.83, 71
steps) vs 17.11 (acceptance 3.06, 65 steps) with `DSV41_HC_KERNEL=0`. The 4e-7 change in the
mixing values flips borderline routing decisions, the greedy text diverges after a few tokens (both
outputs are coherent), and this prompt landed on a lower-acceptance trajectory; one sample cannot
separate that from run-to-run acceptance spread (±5 %, see 2.6). Every prompt-independent number
(step time, GPU time, the un-graphed comparison in `engine/test_fastdecode.py`) improved, so the
kernel stays on by default. Details in NOTES.md (2026-09-11 11:00-11:20).
