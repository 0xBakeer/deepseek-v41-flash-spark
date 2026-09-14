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


### 3.5 Addendum 2026-09-11 19:30 — thinking-on decode, the router's top-k, and the verify block size

**Thinking on, in this configuration.** One request each through the gateway, temperature 0, 400
output tokens, the same algorithmic prompt (2.7's number came from the much slower 09:50 engine and
a different prompt):

| | thinking off | thinking on (`reasoning_effort=high`) |
|---|---|---|
| TTFT | 1.67 s | 2.19 s |
| decode | **25.86 tok/s** | **21.44 tok/s** |
| DSpark acceptance | 3.51 | 2.90 |
| wall per step | 135.7 ms | 135.2 ms |

The step costs the same to 0.4 %; the whole difference is acceptance, so a single prompt's tok/s is
a property of the prompt as much as of the engine.

**How many experts a step activates.** `DSV41_ROUTE_STATS=1` (`engine/diag_topk.py`) counts the
DISTINCT routed experts a verify block asks for per layer — the quantity that sets the bytes, since
an expert is read once however many of the block's six tokens route to it. Measured on the real
decode path, keep 0.40 pruned: **20.96 per layer at the checkpoint's top-6**, not the ~30 that 36
routed pairs would suggest, i.e. **12.12 GB of expert reads per step** at 186 GB/s over the two CB3
kernels' 65.0 ms.

**Reducing the router's top-k** (`DSV41_TOPK`, default the checkpoint's 6; the gate weights are
renormalized over the survivors by the line that already normalizes the six; the DSpark drafter's
own top-3 of 128 is untouched). Held-out teacher-forced, one run each, the k=6 row re-measured here:

| `DSV41_TOPK` | distinct experts/layer | expert bytes/step | coding | general | decision |
|---|---|---|---|---|---|
| 6 | 20.96 | 12.12 GB | 1.5351 | 3.1512 | **shipped** |
| 5 | 17.58 (0.839x) | 10.17 GB | 1.5396 (+0.0045) | 3.1762 (**+0.0250**) | rejected on prose |
| 4 | 15.30 (0.730x) | 8.85 GB | 1.5890 (+0.0539) | 3.2330 (+0.0818) | rejected on both |

k=5 would have been worth 8.3 ms of a 114.9 ms verify step (106.6 ms; expert kernels 65.02 → 57.44,
less than proportional because 17.6 experts per layer gives the kernel fewer concurrent programs and
it gives back 5 % of its per-byte rate) and 23.45 → 24.87 tok/s on the standard 200-token prompt.
`engine/test_fastdecode.py` at k=5 is argmax agreement 1.00/1.00 on both parities, so the switch is
the same computation graphed and un-graphed. It fails the +0.015-nat gate on prose by 1.7x and is
not shipped. Prose leans on the tail of the router's distribution and code does not — the same split
the 2-bit expert tier showed in 3.4.

**The verify block size** (`DSV41_BLOCK`, drafted positions, odd, default the checkpoint's 5; the
verify width must stay even for the ratio-2 compressor's parity). One run per setting:

| verify block | step + draft | acceptance | **ms per accepted token** | 200 greedy tokens |
|---|---|---|---|---|
| 4 | 110.7 ms | 2.70 | 41.0 | 23.72 tok/s |
| **6 (shipped)** | 124.8 ms | 3.14 | **39.7** | 23.45 tok/s |
| 8 | 137.0 ms | 3.45 | **39.7** | 23.00 tok/s |

The step grows nearly linearly in the block and acceptance sublinearly, exactly as expected; their
ratio is flat between 6 and 8 (a tie within this box's ±5 % acceptance spread) and worse at 4. Block
8 also drafts beyond the horizon the DSpark head was trained for and lengthens the per-burst
latency, so the block stays at 6. Nothing in the serving configuration changed in this addendum.

### 3.6 Addendum 2026-09-11 23:40 — the 3-bit expert format degenerates in free generation, and is withdrawn

Asked for a single-file HTML game, the configuration shipped in 3.1 (keep 40 %, every resident
expert in the 3-bit CB3 format) writes one token over and over until the output cap:

```
```html
<!<!DOCTYPE><!DOCTYPE><!DOCTYPE><!DOCTYPE> ...
```

Everything else was eliminated one variable at a time, same prompt, greedy, 200-300 tokens each
(`results/htmlbug/`): the graphed decode path, CUDA graphs entirely, the fused attention kernel,
speculative decoding (off: identical loop), the FP4 dense projections and the fp8 head (both
reverted: identical loop), and sampling (temperature 0.0 / 0.3 / 0.6 / 0.8: identical loop, so the
distribution itself is degenerate, not the choice rule). The last variable left was the expert
configuration, and it is decisive:

| experts | output |
|---|---|
| CB3 3-bit, keep 40 % (3.1) | the loop, distinct-token ratio 0.03 |
| FP4, keep 31 % (v0.2.0-wip) | `<!DOCTYPE html / <html> / </html>`, closes and stops, ratio 0.71 |

**The shipped default returns to FP4 experts at keep 31 %** (box `.env`: `PRUNE_KEEP=0.31
EXPERT_FORMAT=fp4`), keeping the changes that are independently gated: the fp32 router (3.7 below),
the FP4 attention and `wo_a` projections, and the fp8 head. A 600-token Python class on that
configuration comes back complete and well-formed at 30.1 tok/s (acceptance 4.72).

**What this says about the gate.** 3.1 measured *better* than the configuration that works —
held-out teacher-forced 1.5384 / 3.2087 against 1.5705 / 3.3790 — and 2.4 predicted it from the
simulator to three decimal places. Teacher-forced loss scores the next token of text the model is
shown; it never lets an error compound, so it cannot see a model that cannot stay on its own
trajectory. Every quantization decision in this repo was taken on that number alone. A format that
improves it can still be unusable, and nothing here measured free-running generation until a user
asked for an HTML file.

`EXPERT_FORMAT=cb3` and its kernel remain in the tree, measured and documented, and must not be a
default again without a generation gate (`engine/test_spec_lossless.py` plus a long-generation
check) in front of it.

### 3.7 Addendum 2026-09-11 21:25 — the graphed decode path routed tokens to the wrong experts

`Model.moe` computes the router gate as `mm(y.float(), gate_w)`; the graphed path computed it in
bf16, added 2026-09-11 morning as a micro-optimization. The gate picks 6 of 384 experts and its
scores are dense with near-ties, so bf16 changed which experts ran: at layer 0, where the inputs are
bit-identical, 11 % of the picks differed, rising to 31 % in the middle layers, and each layer then
ran a different FFN than the reference.

| layer | activation error before / after | routed experts equal before / after |
|---|---|---|
| 0 | 0.0000 / 0.0000 | 0.89 / **1.00** |
| 6 | 0.1070 / **0.0073** | 0.83 / **1.00** |
| 12 | 0.1361 / **0.0087** | 0.69 / **1.00** |
| 36 | 0.0803 / **0.0131** | 0.69 / 0.94 |

Logit error against the reference 0.049 → 0.012, argmax agreement 1.00 on both parities. The fused
attention kernel (`DSV41_FUSED_ATTN`, default 0 since this addendum) is a second, smaller source of
the same divergence: with it on, deep-layer error returns to 0.048-0.093 and routed experts equal
falls to 0.72.


## v0.4.0-wip — 2026-09-12

Measured 2026-09-12 03:00-05:40 on the same box and checkpoint. Every row is one request through the
server; the quality column is the generation gate described in 4.1, not a loss number.

### 4.1 The gate this tag is built on

Teacher-forced loss cannot see a model that has stopped being able to stay on its own trajectory: it
scores the next token of text the model is shown and never lets an error compound. The configuration
shipped in v0.3.0-wip measured **better** on it (1.5384 / 3.2087 against 1.5705 / 3.3790) and wrote
`<!DOCTYPE><!DOCTYPE><!DOCTYPE>` for as long as it was allowed. Every configuration in this tag is
gated on free generation instead: five prompts (a story and an essay at temperature 0.7, a Python
module, a single-file HTML game, a JavaScript module at temperature 0), **900 to 2,000 tokens each**,
and the output must keep a distinct-token ratio above 0.25, repeat no line more than 30 % of the
time, and — where the generation finished on its own — be structurally intact.

Both thresholds were learned the hard way. A 300-token gate passed configurations that collapse at
900. Repetition ratios alone pass output whose CSS has decayed into `inset - 00 1 pix - 00 1 pix`,
so the gate checks balanced tags, closed fences and unit spelling too.

**Added 2026-09-12: every one of those five prompts ran with thinking off.** No gate prompt in this
tag, or in any tag before it, was run with thinking on. Nothing here measures a generation that has
to deliberate first and then leave the think block, which is the exact register §4.5 below finds
was never traced either.

### 4.2 The keep-set is a cache policy, and it must be sampled from every workload

`corpus/trace_corpus.jsonl`, which ranked the experts for every earlier tag, is 50 documents whose
only content marker is Python: no HTML, no JavaScript, no CSS, no SQL, no configuration files. The
experts that write markup never fired while the trace was taken, ranked cold, and were dropped by
every pruned configuration. That single fact explains the degeneration chased through v0.3.0-wip.

| keep-set (all at keep 44 %, CB3, otherwise identical) | story | Python | HTML |
|---|---|---|---|
| original corpus | 0.27 | 0.51 | **0.03** |
| + a 12-gram repeat ban | 0.32 | 0.51 | **0.07** |
| `trace_corpus_v2` (web, code, config, technical prose) | 0.43 | 0.50 | 0.40 |
| `trace_corpus_v3` (narrative fiction and dialogue) | 0.54 | 0.58 | **0.04** |
| **union of both traces** | **0.56** | **0.47** | **0.59** |

(distinct-token ratio; <= 0.15 is degenerate.) A corpus of web and code fixes markup and leaves long
prose repeating; a corpus of fiction fixes prose and loses markup. At 44 % of the experts the two
rankings compete for the same slots, and the answer is to rank on both: an expert trace is a
per-token histogram, so `results/trace-union` is the concatenation of the two traces' per-layer
arrays (190 sequences, 36,250 tokens) with the statistics rebuilt from it. No third trace run.

### 4.3 Shipped configuration

> **Note added 2026-09-12.** This configuration does not reliably serve and is not the working
> point any more. `LIMITATIONS.md` (2026-09-12 14:10) records it loading, reporting ready, and being
> killed by the memory watchdog on its first request: a 98 GB arena plus 7.2 GB of drafter experts
> left 5.5 GB where one 2,048-token prefill chunk needs about 7.5 GB at 32k. The rows below were
> measured on a quieter box, inside that margin. The working point since is **`PRUNE_KEEP=0.36`**,
> 139 experts a layer, arena about 80 GB.

`PRUNE_KEEP=0.44 EXPERT_FORMAT=cb3 ARENA_GB=98 TRANSIENT_SLOTS=8 KEEP_FREE_GB=6`,
`TRACE_STATS=results/trace-union/stats/coverage.json`, `DSV41_DENSE_FP4=attn,wo_a`,
`DSV41_HEAD_FMT=fp8`, `DSV41_FUSED_ATTN=0`, penalties 0. 6,779 experts resident = 44.1 % of all
routed experts, expert hit rate 1.0, no NVMe traffic during decode.

| workload | tok/s | DSpark acceptance |
|---|---|---|
| single-file HTML game | 36.6 | 5.07 |
| SQL schema and query | 31.8 | 4.69 |
| JavaScript module | 28.2 | 4.12 |
| German technical writing | 25.2 | 3.68 |
| arithmetic with working | 25.1 | 3.56 |
| Python module | 24.3 | 3.56 |
| thinking on, effort high | 18.6 | 2.66 |
| explanation (temperature 0.7) | 18.1 | 2.64 |
| long story (temperature 0.7) | 17.1 | 2.46 |

Prefill on a 5,014-token prompt: **14.9 s = 337 tok/s** (the all-resident arena reads nothing from
the SSD; the same prompt through the streaming configuration is 87 tok/s). A tool call returns
`finish_reason: tool_calls` with well-formed arguments. The generated HTML game passes every
structural check — doctype, balanced `<style>` and `<script>`, a 3x3 grid, a win check, a reset
button, click handlers, no corrupted CSS units — and stops on its own at 982 tokens.

The step is ~145 ms in every case; the spread is entirely how well the DSpark drafter predicts each
kind of text, about 5 accepted tokens per step on markup against 2.5 on prose.

### 4.4 What this tag fixes in the engine

* The graphed decode path computed the router gate in bf16 while `Model.moe` computes it in fp32.
  The gate picks 6 of 384 experts and its scores are dense with near-ties, so 11 % of the picks
  differed at layer 0 — where the inputs are bit-identical — and up to 31 % deeper in. Every layer
  after that ran a different FFN than the reference. Now fp32: routed experts agree 1.00 at layer 0,
  logit error against the reference 0.049 -> 0.012.
* `DSV41_FUSED_ATTN` defaults to 0: the fused decode-attention kernel returns deep-layer agreement
  to 0.072-0.093 and routed agreement to 0.72.
* A keep-set can now be built from `coverage.json` alone (the per-category histograms are written
  into it), so a checkout reproduces one without the per-layer trace arrays.

### What is not measured in this tag
Sampled quality A/B at scale, long-context (8k+) generation quality, the container image end to end,
and the tool grammar of `server/tool_grammar.py` on real weights (it is off by default).

### 4.5 Addendum 2026-09-12 — a second ranking rule (`maxmin`), and the register the `sum` rule was starving

All of this at `PRUNE_KEEP=0.36`, 139 of 384 experts a layer, on the 36-topic catalogue, on the same
box and checkpoint.

**0.36 is where the memory pins it, and context is not the lever.** A 121 GiB GB10 holds 0.36 at
256k. The whole range the tool offers moves the ceiling by two points:

| `MAX_SEQ` | largest keep this box holds |
|---|---|
| 262,144 | 0.350 |
| 131,072 | 0.361 |
| 65,536 | 0.366 |
| 8,192 | 0.371 |

Shortening the context to buy experts is not a trade worth making. Separately, 131,072 was prefilled
and measured on this box, so `tools/budget.py` `VALIDATED_MAX_SEQ` is 131,072 (it was 32,768).

**The `sum` rule starves a broad topic, and `maxmin` does not.** `sum` adds the per-layer-normalised
histograms, which divides corpus size out, so a topic that spreads its mass over many experts scores
low on every one of them and loses slot after slot to a peaky specialist.
`DSV41_PRUNE_RANK=maxmin` (`engine/v41_engine.py`, `_maxmin_counts`) instead hands each layer's
slots out one at a time to whichever selected topic is currently least covered; an expert admitted
for one topic counts for every topic that also routes to it, so overlap is paid for once.

Over {english, html, python, reasoning, css, javascript}, the same budget either way:

| rule | english | css | javascript | all six |
|---|---|---|---|---|
| `sum` | **0.556** | 0.803 | 0.814 | a spread of 0.26 |
| `maxmin` | — | — | — | **0.676 – 0.688** |

English is the broad one here — it is the prose inside every other topic — and `sum` is the rule
that makes it the worst-served of the six while css and javascript sit 25 points above it.

What breadth costs, worst-served topic, four selected topics against eighteen:

| rule | 4 topics | 18 topics | the span |
|---|---|---|---|
| `sum` | 0.657 | **0.410** | 0.247 |
| `maxmin` | — | — | **0.06** |

(A dash is a number that was not written down, not a zero: for `maxmin` the span was recorded and
the two endpoints were not.)

Neither rule creates capacity. Under `maxmin` on this box, about **sixteen** topics puts the
worst-served one at 0.671, already under the 0.7 line; 35 of the 36 together run **0.580–0.636**,
english hardest. The same full selection under `sum` runs down to **0.464**, chinese hardest.

**Fault 1 — a register that was traced, never selected, and broke the generation.** The served
selection was {english, html, python, reasoning}. css sat at 0.518 and javascript at 0.564: both
histograms were in the file the whole time and neither was selected, so nothing on the screen was
red. Asked for a styled page the model emitted `* { }`, then `box-sizing: inline-block`, then the
same `<style>` block repeated to the token cap. Reproduced with thinking on **and** off, so it is
not a think-block fault. Selecting css, javascript and typescript under `maxmin`, at the same
80.4 GB arena: **114 correct declarations**, custom properties, no repetition. The lesson is the
selection rather than the rule — a coverage bar can only be red for a topic somebody selected.

**Two candidate causes refuted** (`results/htmlbug/`, one variable per engine load): with
speculative decoding off (`K_nospec`) and with the dense and head quantizations off
(`M_nodensequant`) the degeneration is identical. Neither is the cause.

**What neither rule fixes.** At 0.36 about **31 %** of the routing mass is displaced whichever rule
ranks the set, and what is left is a rare-token failure that no keep-set arrangement here removed: a
token corrupts (`color-scheme` -> `color-s-s-mode`), the one-step cycle breaker breaks up runs of
U+2011, and the model enters a self-correction loop trying to repair it.

Fault 2 — the think-exit register, which was never traced by any corpus in this repo — is recorded
in `NOTES.md` (2026-09-12) and `LIMITATIONS.md`. It is a corpus fault, not a ranking one, and no
gate has been run on the corpus written to close it.

## v0.5.0-wip — 2026-09-13

Measured 2026-09-13 00:55-20:45 on the same box and checkpoint. Every row is one request through
the server, judged by `tools/gate_profile.py` — the successor to the gate of §4.1: free generation
with thinking on, and a prompt passes only when the thing it asked for is in the output.
Two runs of the same profile on the same prompts are the whole measurement — round one in the
morning, the recipe in the evening — and both are appended in each profile's own `GATE.md`, with
the run time in the heading, so any row below can be read back at its source.

### 5.1 The recipe — rank by contribution, not by frequency; keep 0.40; `maxmin`; substitute; 256k

Every keep-set in every tag before this one ranked experts by how **often** the router picked them.
Ranked that way the model corrupts rare tokens at subword boundaries — `clearTimeout` written
`cleartimeout`, `OSError` as `oenerror`, `.some` as `.s.s`, `color-scheme` as `color-s-s-mode`
(`docs/keep-sets.md`, "Why coverage predicts whether long generations hold together") — then loops
trying to repair what it wrote, and on some prompts never leaves the think block at all.

Two controls put that where it belongs before anything was changed.

* **The null** — no router mask, every one of the 384 experts a layer reachable, streamed from
  NVMe. The three prompts that failed on every keep-set (`en-explain`, `js-debounce`,
  `html-page`) all pass: 2 paragraphs and 9 sentences, 12 callables, 71 declarations and 15
  functions, at 1,008-1,549 s per prompt (`results/keepsets/null-unmasked/GATE.md`, 10:38). The
  failures are the keep-set's, not the checkpoint's, and this run is the ceiling every keep-set is
  measured against.
* **The unpruned model at `DSV41_TOPK=4`** — nothing pruned, four routed experts per token instead
  of six. 1 of 3 passes: the page still passes, the explanation picks up a 3x repeat, and the
  debounce corrupts `lastArgs` and `lastThis` at the subword boundary and loops on them
  (`results/keepsets/null-topk4/GATE.md`, 12:22). Identifier handling depends on the whole top-6,
  so a keep-set that serves code has to hold the true six for those tokens and not merely six good
  ones. It is also the honest test of `drop` mode without pruning anything, because with
  `norm_topk_prob` on the survivors are renormalised to the routed scale — the technical report and
  DeepSeek-V3's equation 13 both keep the correction bias for selection only and normalise the
  gating values over the selected set, which is what the engine does.

So the router is faithful and the criterion was wrong. REAP (Lasby et al., Cerebras, ICLR 2026,
[arXiv 2510.13999](https://arxiv.org/abs/2510.13999)) had already published that failure mode on
Kimi-K2 — 384 routed experts, one shared, auxiliary-loss-free routing, this model's shape:
LiveCodeBench 0.434 -> **0.082** at 75 % of experts kept and **0.000** at 50 % under frequency
ranking, against 0.440 and 0.429 under saliency. Saliency is what an expert contributes rather
than how often it is reached: `gate_weight(t, e) · ‖expert_e(x_t)‖₂`, summed here over the tokens
routed to it (`docs/keep-sets.md`, "Frequency is not contribution"; the tracer stores the
contribution norm itself, in fp32, because dividing it back out by the smallest gate weights put
366 picks of the deepest three layers past fp16's range).

The recipe is one line of environment, and nothing in the engine:

```
DSV41_PRUNE_SOURCE=saliency    # rank on saliency_<topic>, not counts_<topic>
DSV41_PRUNE_RANK=maxmin        # each layer's slots to the worst-served selected topic
PRUNE_KEEP=0.40 ARENA_GB=89.2  # 154 experts a layer; see the arithmetic below
MAX_SEQ=262144                 # 256k, and thinking on for every run
```

`DSV41_PRUNE_MODE` is left unset, which is `substitute` — the default, unchanged to the bit.
`PRUNE_KEEP=0.40` is `ceil(0.40 x 384) = 154` experts a layer (`tools/budget.py:keep_n`), so 6,160
slots at 14,454,784 bytes each in the shipped `EXPERT_FORMAT=cb3` layout
(`tools/budget.py:EXPERT_BYTES`) = **89.0 GB** of arena, which is the figure `env.example` already
gives for this keep fraction, and the reason it does not fit in FP4. The budget model does not
allow it: its largest keep at `MAX_SEQ=262144` is 0.350 (§4.5). It was run
anyway, because the engine's own pre-flight accepted it, and it served all seven profiles without
the memory watchdog stirring. What that proves is in §5.4.

### 5.2 Seven narrow profiles, the same prompts, both rankings

Round one is `counts` at `PRUNE_KEEP=0.36`, 139 experts a layer; the recipe is the block above. The
topic column is the profile's own list in `tools/tune.py:PROFILES`, `reasoning` and
`reasoning_code` included (neither carries a prompt, so no run gates them). Times are the gate
headings in each profile's `GATE.md`, all 2026-09-13.

| profile | topics | round one — frequency, keep 0.36 | recipe — contribution, keep 0.40 | gate record |
|---|---|---|---|---|
| Frontend | 9 | 6 of 10 (00:55) | **7 of 10** (18:14) | `results/keepsets/frontend/GATE.md` |
| Backend | 9 | 4 of 10 (01:24) | **5 of 10** (18:42) | `results/keepsets/backend/GATE.md` |
| Chat and explanation | 7 | 5 of 8 (03:12) | **6 of 8** (19:08) | `results/keepsets/chat_and_explanation/GATE.md` |
| Medicine | 6 | 5 of 7 (03:25) | **6 of 7** (19:26) | `results/keepsets/medicine/GATE.md` |
| Law and finance | 6 | 4 of 7 (03:48) | **5 of 7** (19:44) | `results/keepsets/law_and_finance/GATE.md` |
| Data and research | 10 | **7 of 11** (04:24) | 5 of 11 (20:26) | `results/keepsets/data_and_research/GATE.md` |
| Writing | 7 | 3 of 8 (06:07) | **5 of 8** (20:45) | `results/keepsets/writing/GATE.md` |
| **all seven** | | **34 of 61** | **39 of 61** | |

Six profiles up, one down. The strict count is the least interesting half of it, because the
**kind** of miss changed. Counted row by row out of those same fourteen runs:

| what a miss was | round one, 27 misses | recipe, 22 misses |
|---|---|---|
| no answer at all — server cut the generation off for repeating itself | 8 | 4 |
| no answer at all — think-exit: reasoned, then produced nothing | 3 | 2 |
| a finished answer, flagged only for a repeated 12-word window | 15 | **16** |
| a finished answer that was structurally wrong | 1 (`css-card`, no `prefers-color-scheme`) | 0 |
| a corrupted run inside one of those finished answers | 2 | **0** |

Not one rare token corrupts under contribution ranking, where frequency wrote `ttimerid` and
`color-s-s-mode`. The explanation prompt `en-explain` — which failed on **all ten** round-one
profiles, including the three wide ones, and passed only with the router unmasked — passes on 5 of
the 7 recipe runs. And the remaining 16 misses are one failure, not several: a fragment redrafted
**3 to 10 times** inside a long think block, followed by a correct answer that the gate never sees
as correct because the repeat rule fails the run first.

Frontend was run three ways on the identical nine topics, which is the cleanest read on the
criterion because nothing but the ranking moved:

| Frontend, nine topics | passes | what failed |
|---|---|---|
| frequency, keep 0.36 (00:55) | 6 of 10 | 2 think-exits, `ttimerid` corruption, a card with no `prefers-color-scheme` |
| contribution, keep 0.36 (17:19) | 5 of 10 | no corruption, no think-exit; five redrafts, one at 20x |
| contribution, keep 0.40 (18:14) | **7 of 10** | one think-exit (`ts-groupby`), two redrafts at 6x and 3x |

`css-card` passes at 0.40 and had never passed on a pruned keep-set before: it fails in every
other gate record in `results/keepsets/` that ran it — Frontend at 00:55 and 17:19, the ad-hoc
eight-topic set, Programming broadly, Everything. Contribution ranking is what removes the
corruption and the exit failures; the extra 15 experts a layer are what turn "finishes cleanly"
into "passes".

### 5.3 What was refuted, each with the number that refuted it

Every one of these was run on the real server, not reasoned about.

| tried | result |
|---|---|
| `DSV41_PRUNE_MODE=drop` — zero the weight of a non-resident pick instead of substituting | **0 of 6** on Frontend at keep 0.36 with thinking on, including a page that passes under `substitute`. With `norm_topk_prob` on this is top-*k′* routing with about four survivors of six, and a shared-expert-only fallthrough on roughly a token in twenty. Kept as a documented negative result. |
| the unpruned model at `DSV41_TOPK=4` | **1 of 3**; the same identifier corruption with nothing pruned (§5.1) |
| `no_repeat_ngram=8` | does not reach the think-exit; banning an 8-gram leaves the model free to redraft in different words (NOTES.md, 2026-09-12 20:00) |
| `frequency_penalty=0.3` | fixes prose and destroys code on the 900-token gate — the penalty grows with a token's count, and CSS repeats `px` and digits dozens of times (NOTES.md, 2026-09-12 03:50-05:30) |
| `presence_penalty` (flat per distinct token) | does not corrupt code and does not fix prose either; both penalties stay at 0 |
| `reasoning_effort=10` | **worse**: 59,766 characters of deliberation and no answer at all |
| `temperature=0` | "I'll write the code now.", then "Let me write." to the token cap, answer length **0** |

The first two are questions about the keep-set and the last five are settings on the request. The
five are all closed the same way: this is not a sampling problem, and no sampler setting reaches
it.

The profile list itself was refuted the same way. In round one "Programming, broadly" with fifteen
topics passed **4 of 16**, "Many languages" with fourteen passed **2 of 15** — every natural-language
prompt failed — and "Everything", all 37, passed **4 of 39**
(`results/keepsets/programming__broadly/GATE.md`, `many_languages/GATE.md`, `everything/GATE.md`).
Six to ten topics serve on this box and fourteen and up do not, so the two wide bundles became four
narrow ones (Frontend and Systems programming; European languages and World languages) and
Everything is no longer offered as a profile. The topic screen still lets anyone select all 37 by
hand and shows what it costs.

### 5.4 What this proves and what it does not

**Proven.** On this box, at this checkpoint, contribution ranking removes two whole failure classes
that frequency ranking produced — token corruption and the think-exit — and 154 experts a layer at
`MAX_SEQ=262144` load and serve. Seven profiles, 61 prompts, 34 -> 39.

**256k is proven for short prompts only.** No request in any of these gates prefilled anything like
262,144 tokens; the longest prompt in the whole suite is 58 words (`xl-en-fr`,
`tools/gate_profile.py:PROMPTS`). The reserve the budget model keeps back is for exactly the case
none of these runs exercised, which is why it puts this box's
ceiling at 0.350 and this configuration is above it. A long-prefill memory test is running as this
is written, and `env.example` keeps the previous defaults (`PRUNE_KEEP=0.39`, `ARENA_GB=87`,
`MAX_SEQ=32768`, an empty `DSV41_PRUNE_RANK` and `DSV41_PRUNE_SOURCE`) until it lands. Nothing here
justifies raising `tools/budget.py:VALIDATED_MAX_SEQ` above the 131,072 that was measured on
2026-09-12.

**One profile regressed and it is not explained.** Data and research went 7 -> 5 of 11. Every one
of its six misses is a finished answer flagged for a redraft — none corrupt, none fail to close —
so the profile got better in kind and worse in count, and the count is what a user sees.

**One prompt regressed within a profile: Go.** Backend's `go-handler` passed round one at 115 lines
(01:24) and under the recipe was cut off by the server's repeat guard after looping 12x inside the
think block (18:42), while `java-service` went the other way, from cut off to 88 lines. One prompt
each way on one profile is not a pattern yet; it is open.

**The remaining failure class is not a loop.** Sixteen of the 22 recipe misses are a fragment
redrafted 3 to 10 times inside a long think block and then a correct answer. `repeated_ngram` in
`tools/gate_profile.py` fails a run when any 12-word window appears three times, and the harness
cannot tell a model that drafted a function signature four ways before choosing one from a model
that has stopped making progress. Both are worth knowing about and only one is a defect. Until the
gate can separate them, these runs are counted as failures, which is the conservative direction but
makes the 39 a floor rather than a measurement.

### 2026-09-13 22:50 — addendum to §5: the long-prefill test landed, negative

Keep 0.40 at 256k is a short-prompt result only. A 195k-token prefill (4,000 synthetic sections,
194,797 tokens by the tokenizer) took MemAvailable to 0.8 GB and the watchdog killed the engine
after 582 s. The seven-profile table above stands; the configuration it was measured on does not
hold a filled context. `env.example` now ships `DSV41_PRUNE_SOURCE=saliency` and
`DSV41_PRUNE_RANK=maxmin` — proven at 0.36 as well (Frontend: ten of ten prompts finished, no
corruption) — and leaves the keep fraction to the budget model, which was right.

### 2026-09-14 02:55 — addendum to §5: `reasoning_lang` gated, negative

European languages with `reasoning_lang` in the bundle, keep 0.36, saliency, `maxmin`, thinking
on: **3 of 10** (en-note and the two reasoning prompts). The 0.40 record without it was 6 of 10.
Per row: en-explain repeat-looped 50× in reasoning with a sound answer; fr guard-cut at 12k
tokens looping a French sentence; de think-exited at 59k tokens looping 154×; it corrupted into
"e e e e"; pt, es and xl-en-fr guard-cut. The new corpus did not change the failure shape of
French or German deliberation, and it cost Portuguese and the translation row, which had passed.

Why, measured on the Mac from the same coverage file: the topic is spread (`n80` = 8 experts a
layer, against 1 for `reasoning_design`), `maxmin` pours budget into it, and 513 of the 5,560
resident experts were swapped out of the tail — invisible to the coverage number (−0.005 a
topic), decisive for the rows that route there. The two language profiles ship without it again.
What this leaves: languages other than English are served with thinking **off** (4 of 4 on the
same keep-set, §5.4); with thinking on they are the open item, and it is not closed by a corpus
of that shape.

### 2026-09-14 03:40 — World languages with `reasoning_lang`: 6 of 10, the opposite direction

Same bundle shape, same keep 0.36, same night: **6 of 10 strict, 7 of 10 finished** (ar think-exit
at 47k tokens, ja and tr guard-cut on a looping fragment, xl-en-fr repeat-looped with a sound
answer). The 0.40 record without the topic was 3 of 10; Chinese and Russian deliberation now
finish in 2–4k tokens where they looped before. So the 513-expert tail swap that cost European
three rows bought World three. The profiles record that: `reasoning_lang` in World, not in
European. Both language profiles remain the weakest in the catalogue with thinking on; with
thinking off every language in both has passed on the same keep-sets.

### 2026-09-14 06:20 — Backend and Data and research at keep 0.36 (the 256k memory point)

Same recipe (saliency, `maxmin`, substitute), thinking on, effort 45, max 16,000 tokens.

| profile | keep 0.40 (2026-09-13) | keep 0.36, strict | keep 0.36, finished | misses by kind |
|---|---|---|---|---|
| Backend | 5 of 10 | 3 of 10 | **10 of 10** | 7 repeat, 0 exit, 0 guard, 0 corrupt |
| Data and research | 5 of 11 | **8 of 11** | 10 of 11 | 2 repeat, 1 guard |

Backend at 0.36 produced a correct answer on every prompt for the first time — the Go and Python
guard kills of the 0.40 run are gone — but seven deliberations restated themselves 3–8× before
answering, at 18–38k reasoning tokens a prompt against 4–9k where they do not loop. Correct and
slow. Data and research went the other way and is the second-best profile in the catalogue at
this keep. With Frontend (7 of 10 strict, 9 of 10 finished) this makes three of the four code-bearing
profiles measured at the keep that serves a filled 256k context.

### 2026-09-14 07:20 — the design test: a whole page from the Frontend keep-set at 0.36

The brief: a single-file landing page for a small bookshop (header, hero, three books, two events,
about, footer; embedded CSS, dark mode, reduced motion, focus states, 400 px). Four attempts on the
same keep-set (saliency, `maxmin`, 139 experts a layer):

| channel | thinking | result |
|---|---|---|
| coding agent, tool call | on, effort 45 | write closed after `<title>…Hamburg` (172 bytes); retry ran to 32,000 tokens in a parameter format the strict parser rejects |
| plain chat | on, effort 45 | 15,241 reasoning tokens, guard cut, no answer — a 40k-character design plan, then a "Maybe add X? Skip." loop |
| plain chat | on, effort 20 | 9,386 reasoning tokens; the answer was a stylesheet with no markup |
| plain chat | **off** | **11.7 KB page in 158 s**; dark mode, reduced motion, focus-visible present; holds at 400 px with no horizontal scroll |

The thinking-off page is a considered design — paper ground, serif display with one italic accent
word, small-caps labels, one red rule, three-column shelf, a darker events band — and its defects
are all in the text: the about paragraph repeats itself, the hours line is nonsense, one
`</title>` came out as the DSML tool-call end-parameter marker, and `sans-serif` as `ser-serif`.

That marker is the mechanism behind the tool-call failures and behind the `</</p>` corruption of
the earlier run: at a `</` boundary the pruned router drifts from the HTML close tag to the
model's own `</｜DSML｜…>` close marker, which inside a tool call ends the argument. Whole-page
generation with thinking on is a long generation, and long generations are where this keep-set
fails; the same profile passes 7 of 10 (9 finished) on the shorter gate prompts.

### 2026-09-14 08:10 — the close-marker guard, verified live on the Frontend keep-set at 0.36

`server/tool_grammar.py` now keeps the model's own tool-call end marker out of a `</` the value
did not mean: inside a parameter value the marker is masked after `</` while the markup is
unbalanced and the `</` does not follow a newline; outside a calls block the marker is legal only
after a bare `<`; with no tools it is never legal. Same keep-set, same brief, same server otherwise:

| channel | before | after |
|---|---|---|
| plain chat, thinking off | one `</｜DSML｜ parameter>` in the page title | zero markers, `<title>…</title>` closed |
| coding agent, tool call, thinking on | write closed after `<title>…Hamburg` (172 bytes) | five consecutive whole-page writes of ~4,200 tokens each, 12.5 KB, no marker, no `</</` |

What the guard does not fix is behaviour: the coding agent rewrote its clean file five times,
each time announcing the previous one had "arrived corrupted", until the 30-minute cap; and one of
two plain-chat runs with thinking off returned a stylesheet with no markup. Those are the
keep-set's long-generation faults (§5.4), not the transport. The grammar is on by default from
this commit; `DSV41_TOOL_GRAMMAR=0` turns both the constraint and the guard off.

Addendum, 08:20: rendered, the tool-channel page is structurally whole but its prose is
degenerate ("Lumen as lis Lumen a place", blurbs that are word salad) and its layout collapsed
into a third of the width — the thinking-on long-generation fault again, now visible as content
instead of as a cut. The plain-chat page with thinking off remains the only one that reads and
looks like the brief.

Addendum, 08:45 — how often thinking off delivers the page. Five plain-chat runs of the same
brief on the Frontend keep-set at 0.36 (temperature 0.6): three complete pages (12.4 KB, 13.0 KB,
11.7 KB; dark mode, reduced motion and focus states present; 4 sections each), two that stopped
after the stylesheet with no markup (9.4 KB and 10.5 KB). Three of five. None of the four runs
after the guard carried a marker or a `</</`. The stylesheet-only stop is a content fault of the
keep-set, not of the transport; it is the number to beat.

Correction, 09:20 — the 03:40 addendum says that with thinking off "every language in both has
passed". Not so for European: the thinking-off run in `results/keepsets/european_languages/GATE.md`
(2026-09-13 23:07) passed German and Italian and failed French and Spanish on the language-marker
check, 2 of 4. World languages was the 4 of 4. The figures in the World addendum are unchanged.
