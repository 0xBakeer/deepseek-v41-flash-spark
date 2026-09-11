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

### 2.11 Addendum 2026-09-11 12:10 — CB3 (3-bit) expert kernel at speed; graph merging measured as no gain

**CB3 kernel (`tools/cb3_moe.py` v3, unit test `tools/test_cb3_moe.py`):** one real expert at decode
shapes, expert bytes per second:

| kernel | ms | GB/s of expert bytes |
|---|---|---|
| FP4 (18.80 MB/expert) | 2.14 | 184.1 |
| CB3 v1 (the parked gather variant) | 17.41 | 19.9 |
| CB3 v2 (new plane layout, Triton byte ops) | 1.97 | 154.0 |
| **CB3 v3 (512-weight blocks, inline PTX decode)** | **1.67** | **181.5** (best run 182.0) |

A 3-bit expert now costs 0.787× the time of an FP4 one; dequant is bit-identical to
`engine/codebook_sim.py`, kernel output within 8.6e-5 of the FP4 kernel on the same re-quantized
weights. The decisive factor was tile width, not instruction count: on this box a row-strided read
runs at 101 GB/s for 16-32 B tiles and 185-218 GB/s from 64 B up, so the format's blocks were
widened to 512 weights (w2's K=2304 is packed as 4×512 + 256). Arena arithmetic at 90.5 GB: 4,813
experts all-FP4 (31.3 %), 6,260 all-CB3 (40.8 %). Keep 40 % at simulated 3-bit was measured in 2.4
at coding 1.539 / general 3.212 nats held-out, better than the served keep-31 % FP4 on both. The
kernel is not yet wired into the serving path (a second arena tier); that is the next step.

**Graph merging:** 41 graph replays per step → 3 (at the Engram boundaries) and pinned staging for
the Engram rows: step 147.2 → 146.6 ms, decode 16.56 vs 16.63 tok/s (bit-identical output). GPU
busy time is 144.9 of the 146.6 ms; the rest is per-kernel latency inside the graphs (about 5,300
kernels per step), not launch count. Segmentation stays on (`DSV41_GRAPH_SEGMENTS=0` restores);
pinned staging is off by default (`DSV41_ENGRAM_PINNED=1`).


## v0.3.0-wip — 2026-09-11

Measured 2026-09-11 10:00-14:10 on the same box and checkpoint, the pool ours alone. Commits
`94a96a6` .. this tag. Every row is one run of the stated command; no benchmark sweeps were run.
The kernel work that led here is in the dated addenda 2.8-2.11 above; this section is the shipped
configuration that changed.

### 3.1 Shipped default: keep 40 %, every resident expert in the 3-bit CB3 format

`PRUNE_KEEP=0.40 EXPERT_FORMAT=cb3 ARENA_GB=90.5 TRANSIENT_SLOTS=8 KEEP_FREE_GB=10`, everything
resident, CUDA-graph decode path, device slot LUT. Same prompt and flags as 2.6 for the decode
line; teacher-forced on `corpus/heldout_corpus.jsonl`; TTFT on the 1,806-token prompt of 2.x.

| | keep 31 %, FP4 (v0.2.0-wip default) | **keep 40 %, CB3 (this tag)** |
|---|---|---|
| resident experts | 4,800 = 90.2 GB (31.3 %) | **6,160 = 89.0 GB (40.8 %)** |
| warm start (packing on the GPU) | 19 s | 183 s |
| decode, 200 greedy tokens | 16.61 tok/s (acceptance 2.83, 71 steps) | **18.98 tok/s** (acceptance 3.03, 66 steps) |
| TTFT, 1,806-token prompt | 11.11 s | **9.81 s** |
| held-out coding NLL | 1.5705 | **1.5384** (−0.032) |
| held-out general NLL | 3.3790 | **3.2087** (−0.170) |

The CB3 row reproduces the simulated 3-bit keep-40 % row of 2.4 (1.5392 / 3.2122) to 0.0008 /
0.0035 nats: the packed format, the kernel and the simulation are the same arithmetic. Against the
full unpruned model on the same corpus (2.4: 1.5067 / 3.1884) this configuration costs +0.032
(code) / +0.020 (prose) nats.

Prefill does not run the CB3 decode kernel: above 64 token-expert pairs the experts are unpacked
to FP4 codes on the fly (bit-exact) and the FP4 kernel runs; on one layer at 2,048 tokens that is
1.35x the MoE time of an FP4 arena of the same size. The two-tier arena (hot experts back at FP4,
cold in CB3) is not built; at 40.8 % all-CB3 there is no headroom in 90.5 GB for it.

### What is not measured in this tag
Thinking-on decode in this configuration, sampled quality A/B, long-context (8k+) serving in this
configuration, the container image end to end.

### 3.2 Addendum 2026-09-11 16:00 — attention projections in FP4 (served config, `DSV41_DENSE_FP4=attn`)

Dense bytes per verify step (from the safetensors headers): attention projections 3,734 MB,
shared experts 1,417 MB, `wo_a` 1,344 MB, other 436 MB. A dense FP4 kernel (`tools/fp4_linear.py`,
E2M1 codes + one UE8M0 scale per 32 weights along K, quantized at load from the stored FP8; unit
test `tools/test_fp4_linear.py`) reads 0.53x the bytes and wins on the wide matrices (`wq_b`,
`wo_b`: 199 GB/s of FP4 vs 221 of FP8) but not on the narrow ones. Held-out teacher-forced, one run
per setting, against the keep-40 % CB3 baseline 1.5384 / 3.2087:

| group in FP4 | coding | general | decision |
|---|---|---|---|
| shared experts | 1.5531 (+0.015) | 3.2740 (+0.065) | kept in FP8 |
| attention projections | **1.5403 (+0.002)** | **3.1738 (−0.035)** | **default from this addendum** |
| both | 1.5527 (+0.014) | 3.2323 (+0.024) | kept in FP8 |

The −0.035 on general is within what a 53-sequence corpus can resolve, not a gain. Decode line
back to back: 19.11 tok/s (acceptance 3.03) → **20.85 tok/s** (acceptance 3.23); prefill 4.12 →
3.28 s on the same prompt; verify step on `engine/profile_fast.py` 134.4 → 125.6 ms; 1.75 GiB of
resident weights freed. The shared experts are the one dense FFN every token passes through and
the FP4 weight error (12 % relative) shows there; `wo_a` stays FP8 through the grouped kernel.

### 3.3 Addendum 2026-09-11 17:40 — `wo_a` in FP4 and a leaner decode loop

The output projection's first factor (`attn.wo_a`, 1,343 MB read per verify step) now runs through a
grouped FP4 kernel (`tools/fp4_linear.py::fp4_grouped_linear`, one group per third grid axis),
quantized at load from the stored FP8. Held-out teacher-forced against the 3.1 baseline
(1.5384 / 3.2087): **coding 1.5346 (−0.004), general 3.1498 (−0.059)** — inside what this corpus can
resolve, and on the good side of zero. Resident weights 8.34 → 7.70 GiB. Kernel time in the step
7.95 → 4.56 ms; at T=6 in isolation the FP4 grouped kernel is 150-158 GB/s against the FP8 one's
207-215, and reads 0.53x the bytes, so 1.31-1.44x in wall time.

`DSV41_LEAN_STEP=1` (default; `=0` restores the previous code) computes the greedy accept/reject on
the GPU and reads back one 7-element pinned tensor instead of up to eleven separate syncs, builds
the verify block into a preallocated buffer, and drops redundant clones and a duplicate buffer
preparation. Sampling semantics are unchanged: at temperature 0 both paths produce byte-identical
text, and the temperature path is the original code.

| | verify step (`engine/profile_fast.py`) | wall per step, 200-token run |
|---|---|---|
| 3.2 configuration (`attn`) | 125.6 ms | 156.5 ms |
| this addendum (`attn,wo_a`, lean step) | **118.9 ms** | **149.1 ms** |

The decode line on the standard prompt moved 20.62 → 20.02 tok/s because the DSpark acceptance on
that one prompt fell from 3.23 to 2.99; at equal acceptance the new configuration is 21.7 tok/s.
Per-step time is the number this addendum claims. Prefill on the same prompt 3.27 → 2.61 s.

Instrumenting the loop (`DSV41_STEP_TIMING=1`) also corrected an earlier reading: the gap between
the graph harness and a fresh 200-token run is not removable Python. It is one un-graphed drafter
call on the first step, the Engram host-to-device copy absorbing queued graph work by design, and a
cold Engram row cache — a second decode in the same process costs ~140 ms/step and a third ~133 ms,
tracking the Engram read time and nothing else.

### 3.4 Addendum 2026-09-11 19:00 — the LM head in FP8, and a 2-bit expert tier that was not built

**The head.** `head.weight` is [129280, 5120] bf16 = 1.324 GB and is read in full twice per decode
step (the verify step and the DSpark draft); in 3.3's profile it is 5.79 ms of cutlass at 232 GB/s.
`DSV41_HEAD_FMT` = `bf16` (default) | `fp8` | `fp4` stores it in the dense projections' format
(e4m3 + one UE8M0 scale per 32x32 block, 0.663 GB, `tools/fp8_linear.py::quantize_to_fp8`) or the
routed experts' (E2M1 + one scale per 32 K weights of a row, 0.352 GB), quantized on the GPU at
load; above decode-sized M a quantized head dequantizes 16,384 vocabulary rows at a time into
cuBLAS. Held-out teacher-forced against the 3.3 baseline (1.5346 / 3.1498), one run each:

| head | coding | general | weight rel err | head GEMM at M=6 | decision |
|---|---|---|---|---|---|
| bf16 | 1.5346 | 3.1498 | — | 5.68 ms, 233 GB/s | — |
| **fp8** | **1.5351 (+0.0004)** | **3.1512 (+0.0014)** | 0.027 | **2.96 ms, 224 GB/s** | **default from this addendum** |
| fp4 | 1.5502 (+0.0156) | 3.1572 (+0.0074) | 0.118 | 2.38 ms, 148 GB/s | rejected on coding |

With the fp8 head the 200-token greedy output is byte-identical to the bf16 one at matched positions
(the second decode of each process; in every arm, bf16 included, a process's first decode differs
from its own second at token 9, because the first step drafts eagerly before any graph exists).
`engine/test_fastdecode.py` argmax agreement 1.00 / 1.00 on both parities, step 116.5 / 117.8 →
113.2 / 113.9 ms and draft 14.3 / 13.5 → 11.6 / 10.6. Decode line: 143.0 → 137.2 ms per step,
21.62 → 22.88 tok/s. In one process with the head swapped under a fixed arena, the verify step is
118.9 → 114.6 ms and the draft 13.4 → 10.7, with the two CB3 expert kernels unchanged — the
separate-process form of that A/B measures the arena's page placement instead and reports the
opposite sign (NOTES 2026-09-11).

**The 2-bit tier: measured, not built.** A CB3 slot filled with a four-entry row codebook repeated
to its eight entries carries exactly a 2-bit format's arithmetic at unchanged size, so the quality
question was answered inside the shipped configuration (`--sim-cb2-frac`). Held-out teacher-forced,
keep 0.40, against the same 1.5346 / 3.1498:

| coldest fraction of the kept set at 2 bits | coding | general |
|---|---|---|
| none (shipped) | 1.5346 | 3.1498 |
| 0.30 (1,840 of 6,160 experts) | 1.5471 (+0.0125) | 3.1966 (**+0.0468**) |
| 0.50 (3,080 of 6,160 experts) | 1.5563 (+0.0217) | 3.1940 (**+0.0442**) |

The whole prose penalty is paid by the coldest 30 % and does not grow after it, at 1.6x the budget a
tier would have to fit in. The packed CB2 format and its kernel were built and measured anyway
(`tools/cb3_moe.py::CB2ArenaV2`, 9.99 MB per expert = 0.691x CB3, bit-exact against the FP4 kernel
on the same weights): 143.8 GB/s of its own bytes against CB3's 168.7 in the same run, i.e. 0.81x in
wall time for the cold experts, which carry 22 % of the routed pairs at a 0.50 cold fraction — about
3 ms of a 119 ms step even before the loss is counted. `EXPERT_FORMAT` keeps its two values; nothing
in the serving path changed.

