# NOTES -- running log

Working notes for `deepseek-v41-flash-spark`, kept as the work happens. Newest entries at the
bottom of each phase. Numbers here are either measured on this box (marked **measured**) or
quoted from a named source with a link. Nothing in RESULTS.md comes from here without a
measurement.

Box: ASUS Ascent GX10 (NVIDIA GB10, sm_121a, 128 GB unified, 121 GiB visible, 20 cores,
1 NVMe 916 GB), Ubuntu 24.04 DGX OS base, driver 580.173.02, CUDA 13. Referred to as "the Spark"
in this repo since the silicon is the same as a DGX Spark.

---

## Phase 0 -- dry work (2026-09-10)

### 0.1 What is in the checkpoint (safetensors headers, no re-derivation)

48 shards, 510.29 GB, 96,085 tensors. Layout is tidy: **one 7.39 GB shard per layer** (shards
3..42 = layers 0..39), embed in shard 2 (1.32 GB, with `image_start/end/newline`), head+norm in
shard 43, MTP/DSpark in 44..46 (7.9 GB), vision in shard 1 (0.97 GB), and the two Engram tables
alone in **shards 47 and 48 (101.54 GB each)** together with the four small non-table engram
tensors of their layer (`wkv.weight` 157 MB fp8 [25600, 6144] + scale, `q_weight`/`k_weight`
bf16 [4, 5120]). So "download everything except Engram" is a clean 307 GB, and "download one
layer" is one file.

HF checkpoint tensor names already use the reference implementation's names (`layers.N.attn.wq_a.weight`,
`layers.N.ffn.experts.E.w1.weight`, `.scale`, `layers.N.hc_attn_fn`, `mtp.K.*`), so no name
mapping is needed for a loader. Per-expert on disk: `w1`,`w3` I8 [2304, 2560] + UE8M0 scale [2304, 160],
`w2` I8 [5120, 1152] + scale [5120, 72] = 3 x (5.90 + 0.37) MB = **18.8 MB per expert**,
384 x 40 = 15,360 experts = 288.8 GB. `wo_a` is stored fp8 [8192, 4096] (convert.py dequantizes it
to bf16, block 32x32). Attention/router/shared-expert/HC weights per layer: ~185 MB.

### 0.2 Architecture facts that matter for serving (from `inference/model.py`, `engram.py`, the tech report)

**Engram lookup path.** `NgramHashState` maps every token id through a *compressed* vocab
(99,092 ids; NFKC/NFD/strip-accents/lowercase/whitespace-collapse, so "The"/"the"/" THE" hash the same),
then for each position takes the previous 3 compressed ids (pad id at sequence start; image spans
are dead), multiplies each by an odd int64 per (layer, lookback) drawn from `default_rng(10007*layer_id)`,
XORs them cumulatively to get the 2-, 3- and 4-gram hashes, and reduces each by a distinct prime
(~16M) per (n-gram size, head) into disjoint bucket ranges of one flat table. Result: **24 rows
per token per engram layer** = (4-1 n-gram sizes) x 8 heads, **48 rows per token total** (layers 1 and 14).
A row is 256 B fp8 + 8 B UE8M0 scales (one per 32 dims) = 264 B, so **~12.7 KB of random reads per
token**, addresses known from the token ids alone (before any forward pass). The 24 rows are
concatenated (6144) and pushed through `wkv` (fp8, 6144 -> 5 x 5120 = one key per HC copy + a value);
the gate is a normalized dot product of the residual copy against its key, signed-sqrt, sigmoid;
`h += gate * value` per HC copy. Tech report 2.4.2/3.1.3 confirms the intent: "deterministic
addressing enables embeddings to be prefetched from host memory via background RDMA transfers".
No convolution (dropped vs the Engram paper).

**Router.** `Gate`: `scores = sqrt(softplus(x_fp32 @ W^T))` (384 x 5120 bf16 weight); selection by
`topk(scores + bias, 6)` with the text bias (`bias_vl` for image tokens); weights = raw scores of the
chosen 6, normalized to sum 1, x `route_scale` 1.5. Plain `torch.topk` over 384 -- DeepSelect is NOT
this (it is the indexer top-k over context positions, see 0.5).

**Hyper-Connections (mHC, hc_mult 4).** The residual stream is 4 parallel copies [T, 4, 5120].
Per sub-block: `mixes = (x.flatten @ hc_fn^T) * rsqrt(mean(x^2))` -> [24] -> `pre` (4, sigmoid+eps),
`post` (4, 2*sigmoid), `comb` (4x4, softmax then 20 Sinkhorn iterations). **Single-Pass shift**: the
`pre` mix computed by a sub-block is used by the *next* sub-block (attention uses the previous
layer's FFN `pre`, the FFN uses this attention's `pre`); the very first uses a one-hot on copy 0.
hc_attn_fn/hc_ffn_fn are fp32 [24, 20480] = 2 MB each per layer. Cheap.

**Attention (CSA2 + CED).** MLA-style with ONE 512-dim KV latent per token (no heads), 64 query
heads of 512, RoPE on the last 64 dims, LoRA q (1280) and grouped LoRA o (8 groups x 1024).
Every layer has a 128-token sliding window over its own fp8 KV. Layers 0,1 (and the 3 MTP layers)
are window-only. Layers 2..19 add compressed global KV at ratio 2, layers 20..39 at ratio 1;
only `kv_source_layers` [2, 8, 14, 20] *produce* global KV (softmax-pooled pairs for ratio 2,
plain projection for ratio 1, RoPE at theta 160000 with YaRN 16x from 64k, then **FP4 E2M1 with an
E4M3 scale per 16 channels**); every other layer reads the last source's cache. Indexer top-k
512 at `index_source_layers` [2, 8, 14, 20, 24, 28, 32, 36]; layer 20 is the candidate source
(2048 blocks x 8 = 16,384-position pool for the decoder indexers). Global KV per token = 890 B
(tech report): with ratio 1 in the decoder that is 512 x 0.5 B + 32 B scales = 288 B for the
layer-20 latent, plus indexer K (128 dims fp4) and the ratio-2 encoder latents.
**CED**: layer 20's global KV feeds layers 20..39, so a prompt only needs layers 0..20 for its
global KV; the decoder's *window* KV is rebuilt from the last 128 prompt tokens only (Decoder SWA
Bounded Replay, report 3.2.2). The reference `model.py` does NOT implement this shortcut (it runs
all 40 layers over the prompt); production engines do (vLLM `--enable-decoder-swa-bounded-replay`
in SGLang, `feat/dsv41-swa-bounded-replay` in vLLM). For us this halves prefill expert traffic.

**DSpark (drafter).** 3 extra blocks under `mtp.*` with their own 128-expert MoE (top-3, hidden 5120,
same expert shape -> 3 x 128 x 18.8 MB = 7.2 GB, which is the 7.9 GB in shards 44..46). Input:
concat of the attention INPUTS of layers 37, 38, 39 (the mean over HC copies) -> `main_proj`
(15360 -> 5120) -> `main_norm`; the block token is [x_t, noise x 4] (block size 5), window-only
attention over the main stream's KV plus the 5 draft positions. Head: the backbone `head` on the
5 positions + a rank-256 Markov head (embed/head 129280 x 256) chained through the sampled draft
tokens, plus a confidence head (5376 -> 1) for adaptive verification. The reference implements
only `forward_spec`; no verify loop. vLLM's `dspark` method: `num_speculative_tokens` 5,
`draft_sample_method`, `rejection_sample_method block`, `enable_adaptive_verification` (off on
GB10 because of FlashInfer #5015, see 0.5).

**What the MTP layers need at decode time**: the last three backbone layers' attention inputs
(free), their own weights (7.9 GB, resident), the shared embed/head, and a 128-token window KV
per MTP block. No engram, no global KV.

**Reference kernels** (`kernel.py`, tilelang 0.1.8): fp8 act quant per 32 with power-of-two
scales (`ue8m0`), fp8 x fp8 and fp8 x fp4 GEMMs with per-(32x32) / per-32 UE8M0 weight scales,
`sparse_attn` (gathered top-k KV, online softmax, learned per-head sink), `hc_split_sinkhorn`,
fp4 act quant (indexer q/k with UE8M0, compressed KV with E4M3). Inference README: "a readable
reference implementation rather than a production serving engine"; `run.sh` defaults MP=8.

### 0.3 Corpus and the tracer (tools/)

* `tools/v41_ref.py` -- pure-torch port of the text forward (no tilelang), **exact for T <= 512
  tokens** because at that length the indexer's top-512 keeps every visible compressed position and
  the 16,384-position candidate pool never prunes; the indexer is therefore skipped. Activations
  are fake-quantized to fp8 exactly like `act_quant`, weights dequantized with their block scales,
  GEMMs in bf16/fp32-accumulate. Compressed KV goes through the FP4/E4M3-per-16 round trip.
* `tools/expert_trace.py` -- **layer streaming**: all sequences go through layer L (one mmap'd
  shard) before layer L+1; state per sequence = residual stream [T, 4, 5120] bf16 + shifted
  `pre_mix` + the current shared compressed KV. Checkpoints after every layer -> resumable as shards
  arrive. Records top-6 ids + weights per token per layer. If the head shard is present after
  layer 39 it also reports teacher-forced top-1 accuracy / NLL, which is the end-to-end check that
  the port is right (routing stats from a broken port would be worthless).
* `tools/engram_rows.py` -- computes the hash ids for the corpus with the reference
  `NgramHashState` and fetches exactly those rows from the two 101 GB shards on the Hub with HTTP
  range requests (+ the four small engram weights). The two shards are never downloaded.
* `tools/make_corpus.py` -- 50 sequences, all <= 512 tokens, V4.1 chat format
  (`<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>...`): **coding** 14 seqs / 5,473 tokens
  (real Python/bash from the V4.1 reference code and the ling3-flash-spark recipe as teacher-forced
  assistant answers, plus the Angry Birds and Mario one-shot prompts), **general** 36 seqs / 5,287
  tokens (tech-report and model-card prose as user documents and as assistant answers, plus 8 short
  QA prompts). Sources listed in `corpus/sources/README.md`.
* Tokenizer check: `<｜User｜>` = 128803, `<｜Assistant｜>` = 128804, `</think>` = 128822, BOS 0, EOS 1 --
  single special tokens, as the encoding README says.

**Measured 2026-09-10 16:58**: layer 0 over 10,760 tokens in 74 s (one 7.4 GB shard, 381/384
experts touched, dequant per expert on the fly); engram row fetch: 10,760 tokens -> 258,240 lookups
-> **156,849 unique rows for layer 1 (61% of lookups are first-time rows on a 10k-token corpus)**,
41.4 MB, ~420 range requests/s from the Hub.

Layer-0-only routing (smoke test, not a result): experts used 381/384, entropy 7.85 bits of 8.58,
top-50% of experts cover 84% of slots, top-25% cover 59%; Jaccard of the coding vs general
top-25% sets = 0.25. Layer 0 is the flattest layer in most MoEs; the deeper layers decide.

### 0.4 Download policy followed

Downloaded to the box: shards 1-6, 43-46 + code = **39 GB** (under the 50 GB rule), into
`$MODEL_DIR` with `snapshot_download(allow_patterns=...)` so a later full
download continues in place. Disk after: 460 GB free. The two engram shards (203 GB) are NOT
downloaded; the trace reads its rows over HTTP. **The full 40-layer histogram needs the remaining
36 layer shards = 266 GB** (307 GB total without engram), which fits the disk (460 GB free) but
is a >50 GB download -> asked in the Phase 0 report.

### 0.5 Landscape as of 2026-09-10 (research, links; nothing here is our measurement)

Nobody serves V4.1-Flash on one 128 GB box. The public state:

* **vLLM**: day-0. Model definitions merged (#56228, `vllm/models/deepseek_v4_1/`), main runtime
  PR #56214 open, branches `dsv41-feat`/`dsv41-optimized` (head e47aa780), image
  `vllm/vllm-openai:deepseekv41-flash-0909` (arm64 exists), recipe page says vLLM >= 0.30.0,
  `vram_minimum_gb: 614`, verified GB200 NVL4 TP4 / 8xH200. Engram: `--engram-config
  '{"cpu_offload": true}'` = pinned host memory, UVA lookup (default). DSpark:
  `--speculative-config '{"method":"dspark","num_speculative_tokens":5,...}'`. SM12x umbrella for
  V4 (#41834) still open; no upstream SM121 work for V4.1.
* **A public 4x DGX Spark TP4 vLLM build** (serving 2026-09-10): the only measured Spark run
  in the wild. vLLM `dsv41-feat` on nightly 8a728663 + `_C_stable_libtorch` rebuilt for
  12.1a + FlashInfer 0.7.0rc1 (0.6.18 lacks the SM120 sparse-MLA decode kernel for V4.1's topk 1152)
  + prebuilt `mxfp8_gemm_cutlass_sm120`. Five SM12x patches (block size 64/128 for the sparse SWA
  and indexer caches, `--block-size 128`, `top_k_per_row_decode` instead of `persistent_topk`
  which needs 128 KB smem per block, GB10 has 99 KB). **Engram-on-disk patch** (`DSV41_ENGRAM_DISK=1`):
  table tensors skipped at load, rows read with `preadv` from the safetensors on NVMe/NFS by a
  32-thread pool in `prepare_inputs` before the forward (so CUDA graphs work), dequantized on CPU,
  copied to a pinned staging buffer. Measured there: 24 serial preads ~17 ms/step on NVMe,
  parallel 3.1 ms (C1). Per rank 81.6 GiB weights (experts all resident, split 4 ways, DeepGEMM
  MXFP4 MoE backend), KV 4.84 GiB = 1.03M tokens. Numbers: 39-77 tok/s single stream
  (counting 77, code 52-57, reasoning 39, prose 23), DSpark acceptance length mean 3.56
  (1.95-5.79), TTFT 0.27-0.58 s. "TP2 does not fit either way." No expert offload of any kind.
* **SGLang**: PR #38798 open, `lmsysorg/sglang:dev-dsv41`, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`
  (host copy of the tables, huge pages advised), `--enable-decoder-swa-bounded-replay`. Datacenter
  GPUs only. Blog: engram host offload +36% KV capacity at same decode on 4x GB300.
* **A 4x RTX PRO 6000 + 128 GB DDR5 SGLang build**: bounded
  64 GiB DDR5 engram row cache with exact NVMe reads on misses; b12x io_uring reader "prerequisites
  met". 200+ tok/s single stream on that hardware.
* **The single-file C inference projects**: no V4.1 branch, only an FYI issue. The public read is
  "not a fit for 128 GB systems ... good fit for a 512 GB Mac ... not really a 'local' model", with
  support "probably yes, initially as an experiment, will 2 bit quants hold up?".
* **llama.cpp**: converter-only draft PR #28696 (Engram written as row-block memmap, 508 GB at
  Q8_0 + MXFP4 experts); "the model won't load until a V4.1 runtime implementation" exists. No
  runtime, no upstream V4 runtime either (fork only).
* **exllamav3 / anemone / TabbyAPI**: V4 Flash only; zero V4.1 mentions.
* **ik_llama.cpp, mlx-lm, transformers main**: nothing for V4.1.
* **Quants on HF (all day-0 stubs)**: GGUF (Q2_K uploading; a MixedQ2 2.25 bpw expert set at
  170 GB, engram excluded), NVFP4 (400 GiB with engram FP8->FP4 lossy; 415 GB ModelOpt), MLX
  (2-bit 239 GB for 256 GiB Macs at 9.5 tok/s; 4/8-bit 427-477 GB; Q4i 14.6 tok/s on a 512 GB
  M3 Ultra). No EXL3, no REAP/pruned.
* **DeepSeek's three new repos** (2026-09-10): `deepseek-recipe` = Rust + Python protocol/chat-template
  layer (Chat Completions/Responses/Messages -> V4.1 prompt, parses thinking + DSML tool calls;
  string `reasoning_effort` only: low=50, high=75, max=100; **no aarch64 wheel**, build from source
  with Rust 1.97.1 + OpenCV 4). `DeepSelect` = the top-k kernel for the sparse-attention INDEXER
  (k=512 over context positions) and the sampler, sm_100a/sm_103a only -- not expert selection.
  `DeepJIT` = header-only JIT runtime extracted from DeepGEMM, ships no kernels, no license file;
  DeepGEMM 26/09 uses it; DeepGEMM sm_121a issues open (#372, #417, #425).
* Prior art for expert caching on the previous model (V4 Flash, not V4.1): bounded expert caches on
  a 48 GB Mac (4.5-5 tok/s) and a 256-slot expert arena with a native packed loader on a Spark
  (~2 tok/s). Rules of thumb, not measured routing traces -- which is why this repo traced it.
* Engram paper (arXiv 2601.07372): offloading a 100B table to host costs <= 2.8% throughput on an
  8B backbone.

### 0.6 First-order memory arithmetic for one Spark (design input, not a measurement)

Non-expert resident set: attention+indexer+HC+router+norms ~5.2 GB, shared experts 1.4 GB, embed
1.3 GB, head 1.3 GB, MTP 7.9 GB, engram non-table weights 0.3 GB, vision 1.0 GB (skippable with
`--language-model-only`) = **~18.5 GB**. KV: 890 B/token global (report) + window KV
(43 layers x 128 x 512 B = 2.8 MB per sequence) -> 256k tokens = 0.23 GB global KV per sequence
(!), so KV is not the constraint on this model -- vLLM on 4 Sparks got 1.03M tokens into 4.84 GiB.
That leaves **~85-90 GB for routed experts** out of 288.8 GB at FP4, i.e. **~4,500-4,800 of 15,360
experts resident at FP4 (29-31%)**. Everything hangs on whether ~30% of experts cover most of the
routed slots -- the coverage curve.

### 0.7 Phase 0 results -- layers 0-3 traced (measured 2026-09-10 17:08)

Trace over 10,760 teacher-forced tokens (5,473 coding / 5,287 general), layers 0-3 only (the
shards on disk). Engram rows for layers 1 and 14 fetched over HTTP in 11 s / 10 s per layer
(314 multipart range requests of 500 rows each; the CDN rate-limits single-range requests at
~400/s with HTTP 429, multipart is the way). Per-layer trace time 46-74 s.

| layer | experts used | top-10% | top-25% | top-30% (= our budget) | top-50% | entropy bits (max 8.58) | unique experts per 6-token block (max 36) |
|---|---|---|---|---|---|---|---|
| 0 | 381 | 0.354 | 0.594 | 0.654 | 0.842 | 7.85 | 28.0 |
| 1 | 380 | 0.386 | 0.614 | 0.672 | 0.846 | 7.85 | 25.1 |
| 2 | 370 | 0.433 | 0.694 | 0.752 | 0.902 | 7.60 | 23.6 |
| 3 | 374 | 0.553 | 0.761 | 0.808 | 0.930 | 7.10 | 22.1 |

Global (1,536 keys): a static resident set of 30% of the (layer, expert) pairs covers **0.724** of
routed slots; an LRU of the same size hits 0.811 per token and 0.724 per 6-token block. Coverage
climbs with depth (layer 3 is markedly more skewed than layer 0), which is the usual MoE shape,
but four layers are not a histogram. **Category-specific resident sets are a real lever**: a
coding-only top-30% set covers 0.70/0.73/0.82/0.88 of coding slots at layers 0-3 vs 0.65-0.81
for the mixed set; the Jaccard overlap of the coding vs general top-25% sets is only 0.18-0.31.

Plots: `results/trace-*/stats/coverage.png`, `layer_hist.png`; tables `coverage.md`.

### 0.8 What the sizes alone already decide (arithmetic, no measurement needed)

Routed experts = 15,360 x 3 x 2304 x 5120 = 543.6 B weights. On disk at FP4 + UE8M0/32 =
**4.25 bits per weight = 288.8 GB**. With ~18.5 GB of non-expert weights resident and KV of a few
GB, the expert budget on this box is **~85-90 GB**, i.e. an average of **~1.3 bits per weight**
over all experts. Consequences:

* **Strategy C (everything resident, hot at FP4 + cold at 2-3 bpw) cannot fit at the quality floor.**
  Even with every cold expert at 2.0 bpw, only ~9% of experts could stay at FP4, and the whole
  set at EXL3 2.0 bpw is still 136 GB. It only fits below ~1.3 bpw average, which is below any
  quality floor worth shipping.
* **Strategy B (2 bpw + prune)**: 136 GB -> ~88 GB means dropping ~35% of experts outright. Speed
  reference only, as planned.
* **Strategy A (hot FP4 cache + NVMe streaming) is the only path that keeps FP4 quality on one
  box**, and it lives or dies by the miss rate: at 10 tok/s a DSpark step (6 tokens) may take
  ~600 ms; NVMe at ~3 GB/s (to be measured) moves ~1.8 GB per step = **~95 expert loads of 18.8 MB
  per 6-token block across 40 layers**, against ~25 unique experts per layer per block = ~1,000
  slots, so the block-level hit rate must be >= ~90% with ~4,500 resident experts. Layers 0-3
  give 0.72 at that share. The deeper layers decide, and the 30 -> 40 layer trend has to be measured,
  not extrapolated.
* Prefill is cheaper than decode here: CED means prompt tokens only run layers 0-20 (the decoder
  runs on the last 128 prompt tokens), so prefill expert traffic is roughly half of a 40-layer model's.

Next: the remaining 36 layer shards (266 GB) are needed for the full histogram; disk has 460 GB
free, so the 307 GB engram-less checkpoint fits with ~155 GB to spare. Waiting for the go-ahead
(the >50 GB rule).

---

## Phase 1/2 -- build log (2026-09-10, brief: "just make it work")

Go-ahead received; the remaining 471 GB were downloaded (85 MB/s, ~95 min; Ling-3.0-flash weights
and the ling3 docker image were deleted to make room, both re-downloadable). Work is split into
parallel agents with one thread orchestrating.

### Full 40-layer routing histogram (measured 19:02, `results/trace-full-20260910/`)

| resident experts | GB (FP4) | static coverage | LRU hit / token | LRU hit / 6-token block |
|---|---|---|---|---|
| 3000 | 56.4 | 0.670 | 0.805 | 0.681 |
| 4000 | 75.2 | 0.748 | 0.855 | 0.764 |
| **4500** | **84.6** | **0.780** | **0.875** | **0.796** |
| 5000 | 94.0 | 0.810 | 0.891 | 0.822 |
| 6000 | 112.8 | 0.859 | 0.917 | 0.865 |

Per layer, the top-25% of experts cover 59-83% of slots (min layer 0, max layer 28); unique experts
per 6-token block 18-28 of 36. The decoder layers are only mildly more skewed than the encoder.
So at the ~4,500-expert budget a DSpark step misses ~20% of its ~1,000 (layer, expert) slots:
~200 loads x 18.8 MB = ~3.8 GB per step, ~0.7 s at the measured 5.5 GB/s NVMe ceiling, i.e.
**the streaming design lands around 4-6 tok/s on a general workload before any smarter placement**.
Levers left: workload-specific hot sets (coding-only top sets cover noticeably more of coding),
LRU adaptation during a session, and prefetching the next layer's likely experts.

**Teacher-forced check of the pure-torch port, all 40 layers + head** (proves the tracer/engine
math end to end): coding NLL 2.15 nats, top-1 63.8% (5,459 tokens); general NLL 3.41, top-1 47.4%
(the "general" chunks are documents pasted into a user turn with no prior context, so they are
inherently unpredictable). A broken port would sit near 10% top-1.

### NVMe (measured, O_DIRECT, 18.8 MB objects, download running concurrently)
1 in flight 4.08 GB/s (3.9 ms/read); 8 in flight 5.43 GB/s; 32 in flight 5.59 GB/s (91 ms/read).

### Engine pieces
* `tools/fp4_moe.py` -- Triton grouped MoE on packed FP4 + UE8M0 (hardware `cvt.rn.f16x2.e2m1x2`
  decode, 64-byte-wide tiles; 16-byte tiles cap at ~130 GB/s on GB10). Decode-size calls: 193 GB/s
  effective (T=1, 6 experts, 0.58 ms), 197 GB/s (T=6, 30 experts, 2.9 ms); rel. error 4.4e-3 vs the
  dequant reference. Prefill (T=512) ~23 TFLOPs, compute-bound, unoptimized.
* `engine/experts.py` -- arena + LRU + transient ring for prefill (must hold >= 384 slots: one
  prefill layer touches ~370 experts; a 64-slot ring wrapped inside a layer and silently computed
  with the wrong experts -- the 0.88 error in the first smoke test).
* `engine/model.py` -- chunked-prefill/decode-block model with caches; single-chunk matches the
  reference trace within 0.7-1.6%; chunk-boundary exactness being fixed (Opus agent).
* `server/app.py` -- stdlib OpenAI-compatible server (14 e2e tests), `start.sh`/`stop.sh`/`bench/`.

---

## Bring-up -- 2026-09-10

First end-to-end run of `engine/` on the full checkpoint. Everything below is **measured on this
box** on 2026-09-10 unless it says otherwise. The box's other inference container was already
stopped (exited 17:2x) when this started, so nothing had to be killed; it is left stopped and not
removed.

### B.1 Smoke test -- two bugs, then coherent greedy text

Command (the one in the brief), `--max-seq 8192 --max-tokens 64 --temperature 0 --no-spec`:

1. **`EngramTable.rows` was shadowed by an int.** `engine/engram.py::__init__` did
   `self.rows = w["shape"][0]`, which overwrote the `rows()` method, so the first engram layer
   (layer 1) raised `TypeError: 'int' object is not callable` on the first forward. Renamed the
   attribute to `n_rows`. This is the only thing that stood between the engine and a first token;
   the Triton kernel import from `engine/` (`sys.path` -> `tools/`), the MTP expert load, the
   `Caches` allocation and the O_DIRECT expert reads all worked first try.
2. **The arena auto-sizing was reading the wrong number.** On GB10 the GPU and the host share one
   pool and `torch.cuda.mem_get_info()` counts the *page cache* as used: after the 470 GB
   checkpoint download it reported **32.0 GB free on a box with 99.9 GiB MemAvailable**, which
   would have silently given a 23 GB arena (7% of the routed experts) and a permanently
   NVMe-bound server. `V41Engine` now takes the larger of `mem_get_info()` and `/proc/meminfo`
   MemAvailable, keeps a hard `keep_free_gb` floor under MemAvailable (default 20 GB; this box
   hard-resets if MemAvailable goes negative) and logs both numbers.

**Measured, load (`--max-seq 8192`, arena pinned to 20 GB for a fast debug loop):**

| stage | time |
|---|---|
| non-expert weights (19 GB) to GPU | 63 s |
| DSpark experts (3 x 128 = 7.2 GB) resident | 5 s |
| warm start, 663 experts / 12.5 GB | 3 s |
| **total to `ready`** | **~72 s** |

**Measured, generation** (prompt 16 tokens, 64 greedy tokens, no spec, 20 GB arena =
1063 slots = 6.9% of the routed experts):

| metric | value |
|---|---|
| output | coherent -- a correct, well-formatted Fibonacci answer with docstrings |
| prefill | 8.02 s for 16 tokens (2.0 tok/s; a cold prefill chunk misses almost every expert) |
| decode | **0.93 tok/s** (67.9 s for 64 tokens) |
| expert hit rate | 0.580 |
| expert misses / prefill misses | 6,459 / 1,618 |
| NVMe read | 152.05 GB for 64 tokens (**2.38 GB/s** effective against a 5.5 GB/s device ceiling) |
| engram | 3,792 rows, 0.65 s total |
| attention / MoE wall time | 6.35 s / 52.78 s |

At this arena size the run is pure NVMe streaming: 2.4 GB of expert weights per generated token.
That is the arena's fault, not the engine's -- see B.3 for the same test with the real arena.

### B.2 Warm start at the real arena size (`--max-seq 8192`, auto)

MemAvailable 99.9 GB at sizing time -> arena 80.7 GB = 4,291 slots (3,891 LRU + 400 transient),
**28% of the 15,360 routed experts**. Warm start: **3,891 experts / 73.2 GB in 16 s = 4.6 GB/s**
(O_DIRECT, 12 io threads, ranked by `results/trace-full-20260910/stats/coverage.json`). Peak host
usage 111 GiB of 121, MemAvailable 10 GiB -- which is why `keep_free_gb` was raised to 20 GB and
the auto factor lowered from 0.88 to 0.82 for the serving runs.

### B.3 Correctness: teacher-forced NLL / top-1 vs the pure-torch tracer (measured 19:45)

New mode `engine/v41_engine.py --teacher-forced corpus/trace_corpus.jsonl`: every corpus sequence
goes through `Model.forward` in ONE chunk (all <= 512 tokens) and the head's next-token NLL and
top-1 are aggregated per category, exactly as `tools/expert_trace.py` does at the end of a full
trace. Run with `--act-quant` so the fp8 activation fake-quant matches the tracer's
(`results/trace-full-20260910/meta.json` has `act_quant: true`); serving runs with it off, which is
strictly more precision.

| corpus | tokens | tracer NLL | **engine NLL** | delta | tracer top-1 | **engine top-1** |
|---|---|---|---|---|---|---|
| coding | 5,459 | 2.1527 | **2.1586** | **+0.0059** | 0.6384 | **0.6410** |
| general | 5,251 | 3.4124 | **3.4380** | **+0.0256** | 0.4738 | **0.4769** |

Both inside the +-0.05 nats bar, so `engine/model.py` (arena + Triton FP4 grouped MoE + engram
rows off NVMe + the fixed-tile GEMMs) agrees with `tools/v41_ref.py` end to end. No bug hunt was
needed. Config: arena 80.7 GB / 4,291 slots (27.9% resident), max_seq 8192, spec off, kernel
triton-fp4. 50 sequences in 965 s of forward time; result in
`results/engine-tf-20260910/teacher_forced_engine_actquant.json`.

### B.4 Two performance bugs found on the way (both measured A/B, same box, same work)

The first end-to-end runs were far slower than the NVMe could explain. Two causes, both outside
the model math:

**(a) The LM head was converted bf16 -> fp32 on every single token.** `Model.forward` ended with
`R.mm(x.float(), self.W.head.float())` and `dspark_draft` with `x.float() @ self.W.head.float().T`
-- a **2.65 GB allocation per token** (`head` is [129280, 5120]), plus a 132 MB one per drafted
token for the Markov head. On a box where the expert arena already holds 74 GB that pushes the
caching allocator into `cudaFree`/`cudaMalloc`, and it cost more than the entire rest of the decode
step. Both are now stored fp32 once at load (+1.33 GB and +66 MB resident), which is also what the
reference does (`ParallelHead`: "kept as fp32 here so the logits come out in fp32 directly").

**(b) The expert reader issued six O_DIRECT reads per expert and synchronised the compute stream
six times per miss.** Two separate fixes:
* `ShardFile.expert_runs` groups the 6 tensors into their **2** maximal contiguous file runs. The
  shards keep all the scale tensors near the front and all the weight tensors far behind, but
  within each group an expert's three tensors are adjacent -- so an expert is a 1.1 MB run and a
  17.7 MB run, not six reads and not one. (The old `expert_span` docstring claimed all six were
  contiguous; they are not.) Verified byte-exact against `safetensors.safe_open` for
  layers 0/7/39, experts 0/123/383 and for `mtp.0.experts.7`.
* `ExpertStore._load_into_slot` now hands the *pinned* staging buffer straight to
  `arena.load_slot(..., non_blocking=True)` on a **per-io-thread CUDA stream** (which first
  `wait_stream`s the compute stream, so a slot cannot be overwritten while the previous layer's
  MoE kernel still reads it) instead of cloning to pageable memory and copying on the default
  stream. `ExpertArena.load_slot` used a plain `.copy_()`, i.e. six synchronisations of the
  stream the model computes on, per miss.

| measurement | before | after |
|---|---|---|
| expert read, **1** in flight (the large-arena decode regime) | 8.11 ms / expert, 2.32 GB/s | **4.78 ms / expert, 3.93 GB/s** |
| expert read, 12 in flight | 4.02 ms, 4.68 GB/s | 3.95 ms, 4.76 GB/s |
| decode, 20 GB arena, 64 greedy tokens, no spec (identical 152.05 GB of reads both times) | 0.93 tok/s | **1.21 tok/s** |
| decode, 74-76 GB arena, 64 greedy tokens, no spec (~70 GB of reads both times) | 0.76 tok/s | **1.75 tok/s** |
| decode, 74-76 GB arena, 64 greedy tokens, DSpark on | 1.74 tok/s | **2.64 tok/s** |

The A/B is honest in the sense that matters here: greedy decoding at a fixed arena size reads the
*same* expert bytes before and after (the tables above quote them), so only the time changed.

Not a win, measured and kept anyway: at 12 reads in flight the fused pinned path is 4.76 vs
4.73 GB/s against clone+H2D -- a wash. It is kept because it is what removes the compute-stream
synchronisation, which the isolated micro-benchmark cannot show.

### B.5 DSpark speculative decoding (measured 19:57, `results/engine-tf-20260910/spec_ab2.json`)

New `--spec-ab` mode: one load, `eng.spec` toggled between runs, so both runs share the arena and
the LRU. Config: arena 74.5 GB / 3,960 slots (25.8% resident), max_seq 8192, kernel triton-fp4,
prompt 16 tokens, 64 output tokens.

**Greedy speculative output is token-for-token identical to greedy autoregressive output:
64 of 64, first divergence `None`.** The DSpark verify loop is lossless as implemented.

| run | decode tok/s | steps | accept_len_mean | expert hit rate | NVMe GB | attn_s | moe_s |
|---|---|---|---|---|---|---|---|
| greedy, no spec | 1.75 | 63 | -- | 0.826 | 70.44 | 2.86 | 31.42 |
| greedy, DSpark | **2.64** | 17 | **3.71** | 0.771 | 86.63 | 1.02 | 24.69 |
| temperature 1.0 / top_p 0.95, DSpark | **2.64** | 20 | **3.40** | 0.794 | 91.98 | 1.21 | 25.55 |

The sampled run is coherent (a correctly structured multi-implementation Fibonacci answer). DSpark
buys 1.5x here: a 6-token verify block reads more expert bytes than a single token does (86.6 vs
70.4 GB) but amortises them over 3.7 accepted tokens.

Semantics were checked against the checkpoint's own `inference/model.py` (`DSparkBlock`,
`DSparkAttention`, `forward_head`, `forward_spec`): block size 5, noise token 128799, target layers
37/38/39 meaned over the HC copies, `main_x` computed once and shared by all three stages, draft
queries at `last_main_pos+1 .. +5`, the Markov head chained through the sampled draft ids. One real
mismatch found and fixed: **the confidence head was being fed the RMS-normed hidden and squashed
with a sigmoid**; the reference feeds it the un-normed `hc_pre` output and returns the raw
projection. It changes nothing measured here because adaptive verification is off in this engine
(the confidence is reported, not acted on).

### B.6 Serving and benchmarking (measured 20:01-20:24)

**Engine API gaps closed for the server and the bench** (`engine/v41_engine.py`, `server/app.py`,
`bench/bench.py`):
* `generate(..., ignore_eos=False)`. With it on, the engine's stop set is empty, and `server/app.py`
  reads a body field `ignore_eos` that empties the stop set on *its* side too (passing it only to
  the engine would have produced a short run anyway, because the server truncates every burst at a
  stop id). `bench/bench.py --ignore-eos` sends it. Without this a "512-token" run really ends
  wherever the model decided to stop, so two configs get compared on two different amounts of work
  -- and, on this recipe, on two different amounts of expert streaming.
* `last_stats` is now written in a `finally` around the decode loop, from counters that are
  initialised before it. The server closes the generator on a stop string or a client disconnect
  (`GeneratorExit` at the pending `yield`), so before this an aborted request reported the
  *previous* request's numbers.
* `V41Engine.config()` returns the static configuration -- arena GB/slots, LRU/transient split,
  resident-expert %, max_seq, spec, trace_stats, kernel, act_quant -- and it is merged into
  `stats()` (hence into `x_engine_stats` on every response) and into `GET /health` as
  `engine_config`. A measured number in RESULTS.md has to be quoted with the config that produced
  it, and a bench should not have to be told what the server was started with.
* `V41Engine.close()` (the server calls `engine.close()` at shutdown; there was no such method).
* `server/test_server.py`: 15 tests pass on the Mac, including a new
  `test_ignore_eos_runs_to_max_tokens` and an `ignore_eos` type-validation case.

**`./start.sh` / `./stop.sh` behaved correctly on the box, unchanged.** `.env` from `env.example`
with `TRACE_STATS=results/trace-full-20260910/stats/coverage.json`, `MAX_SEQ=32768`,
`DEFAULT_THINKING=off`. Start to `/health` **~90 s** (arena 73.8 GB / 3,926 slots = 25.6% resident;
warm start 3,526 experts / 66.3 GB in **14 s**), peak host use 99-101 GiB of 121, MemAvailable
19-22 GiB throughout. `./stop.sh` took **2 s** and MemAvailable came back to 118 GiB. A greedy
`/v1/chat/completions` for "What is 2+2?" answered "2 + 2 equals 4." in 12.2 s wall (6.8 s of that
prefill).

**Bench (`code` only -- see below).** `python3 bench/bench.py --workload code --runs 2 --osl 512
--ignore-eos`, thinking off, temperature 0.6, top_p 0.95, DSpark on, 1 warm-up + 2 measured runs,
every run exactly 512 completion tokens (`finish_reason: length`):

| run | TTFT | TPOT | decode tok/s | accept_len | expert hit | NVMe GB | engram rows |
|---|---|---|---|---|---|---|---|
| warm-up | 12.48 s | 338 ms | 2.96 | 3.47 | 0.820 | 496.7 | 45,440 |
| run 1 | 10.98 s | 369 ms | 2.71 | 3.02 | 0.834 | 517.6 | 51,792 |
| run 2 | 11.12 s | 379 ms | 2.64 | 3.04 | 0.827 | 543.1 | 51,408 |
| **median** | **11.05 s** | **374 ms** | **2.68** | **3.03** | **0.830** | **530.3** | **51,600** |

Per run 1 in detail: prefill 62 tokens in 10.82 s (2,473 prefill expert misses = 46 GB at
4.3 GB/s), decode 514 tokens in 188.3 s over 170 DSpark steps, 25,044 decode expert misses =
471 GB at **2.5 GB/s**, of which `moe_s` 160.5 s, `attn_s` 9.1 s, `engram_s` 3.6 s. So the decode
is squarely NVMe-bound: **0.92 GB of expert weights streamed per generated token** at a 25.6%
resident set, and everything else (attention, engram, the Triton MoE kernel itself) is noise next
to it.

**The rest of step 6 was stopped by the owner** (the box was needed for interactive use) after the
`code` row: no `prose` row, no `angry-birds`/`mario` one-shots, no thinking-on run, and therefore
no `results/oneshots/` artefacts. Two earlier attempts at those rows were killed externally, so
nothing about them is measured and nothing is claimed. The server was left RUNNING on :8000 at the
owner's instruction (`./stop.sh` was verified earlier and not run at the end).

**Operational note found while the benches were being killed**: the server serialises requests on
one lock and only notices a dead client when it next writes a chunk, so a request that was already
queued when its client died keeps the engine busy for its whole `max_tokens` budget. `/health`
reports `busy: true` honestly, but there is no cancel endpoint and no queue cap. See LIMITATIONS.

---

## Speed work (Opus agent) -- 2026-09-10 evening

Brief: make the engine faster, prefill first. Everything below is **measured on this box** with
single short generations against the running server (`<= 64` output tokens, greedy), never a
benchmark run. Two prompts throughout:

* **short** -- "Write a Python function that returns the n-th Fibonacci number." = 16 prompt tokens.
* **long** -- a four-section design document + one question = **1,860 prompt tokens**.

### S.1 Where the time went before any change (measured, instrumented)

New per-phase counters in `ExpertStore.stats` (`route_s`, `load_s`, `lease_s`, `h2d_s`) and in the
engine's `last_stats` (`kernel_s = moe_s - resolve_s`) put numbers on the split for the first time.
On the long prompt, of 117 s of prefill: **100.9 s waiting for expert loads**, 2.5 s routing, 2.0 s
attention, **0.9 s in the Triton MoE kernel**, 0.8 s engram. Prefill is NVMe and nothing else.

The reason is structural: a prefill chunk of any length touches ~370 of the 384 experts of every
layer, and those misses go through the transient ring, so **the expert traffic of a prompt is
`chunks x layers x ~7 GB`, independent of how many tokens are in the chunk**. At 512-token chunks
the 1,860-token prompt was 4 chunks x 40 layers = 23,360 expert reads = **440 GB, i.e. 0.24 GB of
expert weights per prompt token**.

### S.2 Two changes, both aimed at that product

**(a) 2,048-token prefill chunks** (`engine/model.py::MAX_CHUNK`, `DSV41_PREFILL_CHUNK`). Quartering
the number of chunks quarters the traffic. `RING` had to grow from 1,024 to 4,096 slots (the window
gather happens after the whole chunk is written into the ring, so `RING > window + chunk`; 40 layers
x 4,096 x 512 x bf16 = 167 MB, which costs 34 arena slots). The engine implements the indexer, so
chunks past 512 are ordinary work, not an approximation; peak activation at T=2,048 is the gathered
window+compressed KV of one layer, ~2.7 GB. Measured host MemAvailable never fell below 17.4 GiB.

**(b) Decoder SWA Bounded Replay** (tech report 2.2 / 3.2.2), which the reference `inference/model.py`
does not implement: `Model.forward(..., encoder_only=True)` runs layers 0..20 -- everything that
writes global KV -- over the whole prompt, and `Model.decoder_replay()` then runs layers 21..39 once
over the **last 128 prompt tokens** with SWA truncated to that segment (`attention(..., win_lo=S)`),
which is where the prompt's logits and the DSpark seed now come from. 19 of 40 layers stop paying
for the length of the prompt.

Three pieces of state have to cross the split, and they are the whole subtlety of the change:
the residual stream and the *shifted* HC pre-mix at layer 20 (buffered for the tail only), and --
because they are computed per query, not per cache -- layer 20's **top-k** (layers 21-23 reuse it)
and its **candidate pool** (layers 24-39 search inside it). `Shared` carries both within a forward;
`_rep_keep`/`_rep_tail` carry them across the two passes, padding older chunks' narrower candidate
masks with False (a query can only reach columns below its own position, all inside its own chunk's
width). Layer 20 itself keeps running over the whole prompt: it is the CED KV source, and running it
in full is both simpler and strictly more exact than replaying it.

Also in this batch, on the way to the numbers above (not the prefill levers, kept because they are
measured-neutral-or-better and byte-exact):

* `ShardFile.expert_runs` results are cached, and each of an expert's two file runs is cut into
  `DSV41_READ_CHUNK_MB` (default 4 MB) aligned pieces issued on a second thread pool, so one expert
  alone keeps ~5 O_DIRECT requests in flight instead of 2. `engine/test_expert_io.py` checks the
  reader byte-for-byte against `safetensors.safe_open` at chunk sizes 0/1/4/20 MB -- **all exact**.
* `ExpertStore.resolve` does its unique/LUT work in numpy on the host (one D2H copy of the 36 routed
  ids) instead of a GPU `torch.unique` + `.tolist()` sync + a 384-entry LUT copied back and gathered.
* A decode hit that lands in the *transient* ring is now promoted into the LRU by swapping ring
  entries (no re-read): the ring is only a list of slot ids, so the slot joins the LRU where it lies
  and an LRU victim's slot takes its place. Before, an expert that a prompt had loaded and that
  decode then used repeatedly was still overwritten by the next prompt's ring wrap.

### S.3 Before / after (measured, single greedy generations, same box, same arena target)

"before" = the same build with `DSV41_SWA_REPLAY=0 DSV41_PREFILL_CHUNK=512 DSV41_RING=1024`
(arena 73.8 GB / 3,925 slots); "after" = the defaults (arena 73.2 GB / 3,891 slots). DSpark on,
temperature 0.

| 1,860-token prompt, 32 output tokens | before | after | |
|---|---|---|---|
| **TTFT** | **118.5 s** | **33.7 s** | **3.5x** |
| prefill | 117.11 s (15.9 tok/s) | 33.41 s (55.7 tok/s) | 3.5x |
| prefill expert misses | 23,360 | **5,280** | 4.4x |
| NVMe read, whole request | 488.1 GB | **127.1 GB** | 3.8x |
| NVMe per prompt token (prefill only) | 0.237 GB | **0.054 GB** | 4.4x |
| decode | 1.96 tok/s | **2.89 tok/s** | 1.47x |
| decode expert hit rate | 0.877 | 0.896 | |
| first 160 chars of the answer | identical | identical | |

Splitting the two levers on the same prompt (measured separately): replay alone at 512-token chunks
gives **69.0 s** of prefill (13,482 misses), and taking the chunk to 2,048 on top gives **33.4 s**.
So the replay is worth -41% and the chunk size a further -52%.

| 16-token prompt, 64 output tokens | before | after |
|---|---|---|
| TTFT | 4.2-5.4 s | 5.4 s |
| prefill | 4.14 s | 5.38 s |
| decode | 2.73 tok/s | 2.67 tok/s |

Short prompts are a wash, as they must be: at 16 tokens the replay covers the whole prompt and does
exactly the same work as the fused pass (see S.4), so the spread is arena/LRU state, not structure.
Decode itself is untouched by both levers -- it still streams ~0.9 GB of expert weights per generated
token -- except that a prompt now pushes 4x less through the transient ring, which is why the decode
rate on the long prompt improves at all.

After the change the long-prompt prefill is still ~90% NVMe wait (`load_wait_s` 26.7 s of `moe_s`
29.9 s; `kernel_s` 0.42 s, `attn_s` 2.07 s, `route_s` 2.81 s), at ~3.8 GB/s effective against the
5.5 GB/s the device gives at depth. The remaining prefill lever is therefore I/O overlap, not the
model -- deliberately left for the next owner of the engine.

### S.4 Is the replay correct, and what does the approximation cost?

Two checks, both new modes of `engine/v41_engine.py`, both run on the full checkpoint
(arena 73.2 GB / 3,891 slots, serving precision -- no `--act-quant`).

**`--verify-replay`** runs the same prompt through the fused 40-layer prefill and through
encoder + replay in one load and compares the last position's logits. For a prompt of at most 128
tokens the replay covers the whole prompt, so the two paths are the *same* arithmetic in the same
order and must agree bit for bit -- which is the real test of the state that crosses the split:

| prompt | replayed | max abs logit delta | KL(full ‖ replay) | top-1 |
|---|---|---|---|---|
| 46 tokens | 46 | **0.000000** | **0.000000** | same |
| 448 tokens (a dense technical document) | 128 | 4.39 | 0.995 nats | changed |

So the plumbing is exact and the 448-token row is the approximation itself: the decoder layers see a
128-token window instead of the whole prefix. One sample is not a verdict, hence:

**`--teacher-forced corpus/trace_corpus.jsonl --tf-ab`** scores the last <= 128 positions of every
one of the 50 corpus sequences twice in one load -- fused prefill vs replay -- so both are measured
on exactly the same predictions (`results/engine-tf-20260910/tf_replay_ab.json`):

| corpus | tokens scored | full NLL | replay NLL | **delta** | full top-1 | replay top-1 |
|---|---|---|---|---|---|---|
| coding | 1,636 | 1.8867 | 1.9369 | **+0.0502** | 0.687 | 0.675 |
| general | 3,011 | 3.4756 | 3.5168 | **+0.0412** | 0.469 | 0.472 |

By distance from the end of the prompt (the last position is the only one whose logits ever produce
a token): last 1 **+0.0465**, last 8 **+0.0528**, last 32 **+0.0282**, last 128 **+0.0444** nats.
Flat, i.e. the replay is not disproportionately bad at the position that matters. **Everything is
inside the +-0.05 nats bar** this recipe uses for "same model", which is what the tech report claims
for a checkpoint post-trained with the replay simulated -- and the greedy answer to the 1,860-token
test prompt is character-identical with and without it. It is still an approximation and it is on by
default; `DSV41_SWA_REPLAY=0` restores the exact path at 3.5x the TTFT.

The rewritten expert reader is checked separately and byte-for-byte
(`engine/test_expert_io.py`, chunk sizes 0/1/4/20 MB, layers 0/7/39, experts 0/1/123/383: all exact).

### S.5 Not done (deliberately, handed on)

Expert-miss/compute overlap and the `--hot-profile coding|general|mixed` measurement were stopped by
the owner before they were started. The ranking helper for the profile exists and is unit-checked
(`experts.category_counts` reads the per-category counts out of `results/trace-*/trace/layer*.npz`;
the coding top-4,000 differs from the mixed top-4,000 in 27% of its entries) but has never ranked a
warm start in a serving run, and `DSV41_HOT_PROFILE` defaults to `mixed`, i.e. to the old behaviour.

### Expert pruning sweep (2026-09-10 23:07-23:45, measured, in-sample)

Question: can enough experts be dropped (REAP-style: the router only picks among survivors, chosen
per layer by trace frequency, mixed profile) to make the model fully resident? Teacher-forced loss on
the trace corpus, `engine/v41_engine.py --teacher-forced --prune-sweep`:

| kept / layer | FP4 GB | coding NLL | general NLL |
|---|---|---|---|
| 384 (100%) | 288.8 | 2.160 | 3.426 |
| 231 (60%) | 173.7 | 2.185 (+0.025) | 3.418 (-0.009) |
| 192 (50%) | 144.4 | 2.190 (+0.030) | 3.438 (+0.012) |
| 154 (40%) | 115.8 | 2.247 (+0.087) | 3.530 (+0.104) |
| 116 (30%) | 87.2 | 2.319 (+0.159) | 3.755 (+0.329) |

Half the experts are almost free; the cliff is between 40% and 30%, and 30% is the first size that
fits at FP4. Caveat: keep-sets and loss come from the same corpus (in-sample); a held-out corpus
(`corpus/heldout_corpus.jsonl`, code and prose the trace never saw) is being scored next. Also
queued: the all-resident decode speed at keep=25% (the engine's zero-miss ceiling).

### 2026-09-11 00:50 -- the port bug that shaped every earlier number (measured, fixed)

Greedy generation stuttered ("LRLR", "time-to-llive", "ev eviction") on every path, including the
oldest engine commit; decode-vs-prefill was bit-identical at all 40 layers, so it was not a cache
or speculation bug but the ported math. Cause: `v41_ref.hc_post` summed the Hyper-Connection
`comb` matrix over the wrong index (comb @ residual instead of the reference's combᵀ @ residual).
The model stayed coherent enough that teacher-forced loss looked "plausible" (coding 2.16 nats,
top-1 64%) -- it was not. Fix: one einsum in `tools/v41_ref.py::hc_post`, shared by the tracer,
the engine and the fast decode path.

Immediately visible after the fix (same prompt, greedy): clean production-quality code on both
paths; DSpark acceptance length 2.4 -> 3.75; streaming decode 2.86 -> 3.69 tok/s with no other change.

Numbers produced before this fix that are now invalid or biased and get re-measured: the
teacher-forced baselines (tracer and engine), the pruning sweep (loss deltas AND the keep-sets:
the routing trace itself was recorded with the bug, so the hot-set ranking is approximate until
the trace is redone), RESULTS.md speed rows (acceptance was depressed), and the "keep 25% garbles"
observation (to be re-checked).

### Fast decode path (engine/fastdecode.py, measured 2026-09-11 00:10-00:35)
CUDA graphs per layer (A: attention+HC+router, host slot resolve, B: MoE+residual), fused Sinkhorn
Triton kernel, bf16 head, fixed-length masked indexer scoring. Verify step with everything resident:
183 ms + 16 ms draft (was 436 ms). End to end, pruned keep=0.25 all-resident: 9.6-10.5 tok/s (was
4.8); streaming unpruned: NVMe-bound, unchanged. Greedy argmax agreement with the reference path
100% on the tested positions; hidden states differ 2-5% from bf16 GEMM noise amplified by router
near-ties (same class as chunk-boundary noise before the tiling work).
