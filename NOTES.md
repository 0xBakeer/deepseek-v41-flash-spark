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

Downloaded to the box: shards 1-6, 43-46 + code = **39 GB** (under the 50 GB download threshold
this repo keeps to), into
`$MODEL_DIR` with `snapshot_download(allow_patterns=...)` so a later full
download continues in place. Disk after: 460 GB free. The two engram shards (203 GB) are NOT
downloaded; the trace reads its rows over HTTP. **The full 40-layer histogram needs the remaining
36 layer shards = 266 GB** (307 GB total without engram), which fits the disk (460 GB free) but
is a >50 GB download, so it is flagged in the Phase 0 report rather than started there.

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
free, so the 307 GB engram-less checkpoint fits with ~155 GB to spare. Deferred at this point as a
>50 GB download (see 0.4).

---

## Phase 1/2 -- build log (2026-09-10, scope: just make it work)

The remaining 471 GB were downloaded (85 MB/s, ~95 min; Ling-3.0-flash weights
and the ling3 docker image were deleted to make room, both re-downloadable). Work is split across
several parallel tracks.

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
  reference trace within 0.7-1.6%; chunk-boundary exactness being fixed.
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

**The rest of step 6 was not run** (the box was needed for interactive use) after the
`code` row: no `prose` row, no `angry-birds`/`mario` one-shots, no thinking-on run, and therefore
no `results/oneshots/` artefacts. Two earlier attempts at those rows were killed externally, so
nothing about them is measured and nothing is claimed. The server was left RUNNING on :8000
deliberately (`./stop.sh` was verified earlier and not run at the end).

**Operational note found while the benches were being killed**: the server serialises requests on
one lock and only notices a dead client when it next writes a chunk, so a request that was already
queued when its client died keeps the engine busy for its whole `max_tokens` budget. `/health`
reports `busy: true` honestly, but there is no cancel endpoint and no queue cap. See LIMITATIONS.

---

## Speed work -- 2026-09-10 evening

Scope: make the engine faster, prefill first. Everything below is **measured on this box** with
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
model -- deliberately left for future work on the engine.

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

Expert-miss/compute overlap and the `--hot-profile coding|general|mixed` measurement were not run in
this tag. The ranking helper for the profile exists and is unit-checked
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

### 2026-09-11 morning -- results after the fix, quant landscape, the mix question

Post-fix teacher-forced baseline (engine, trace corpus): coding 1.371 nats / top-1 74.4%, general
2.864 / 55.1% (was 2.16 / 3.43 with the bug). Pruning sweep redone (mixed profile keep-sets):

| kept / layer | in-sample coding / general | held-out coding / general |
|---|---|---|
| 100% | 1.371 / 2.864 | 1.507 / 3.188 |
| 60% | +0.011 / +0.008 | -- |
| 50% | +0.012 / +0.021 | +0.017 / +0.064 |
| 40% | +0.015 / +0.086 | +0.022 / +0.113 |
| 30% | +0.067 / +0.140 | +0.090 / +0.236 |
| 25% | +0.130 / +0.189 | +0.162 / +0.320 |

Speed (fast path, FP8 dense, greedy, same code prompt): unpruned streaming 3.5 tok/s (hit 0.83);
keep 40% 6.4 (hit 0.94); keep 30% 9.5 (hit 0.99); keep 25% all-resident 13.6 (accept 3.09).
Graphed step 173 ms + 15 ms draft with everything resident. Arena 78.9 GB = 4,198 slots (27%).

Quant landscape (survey of 33 repos up to 05:00 UTC): nothing fits one CUDA box. Lowest
expert bpw published: 2.25 (GGUF, converter only, no runtime). One EXL3 K3 pack (3.0 bpw measured)
is a 4-Spark TP4 target, 14% uploaded, not loadable. REAP checkpoints are MLX-only (REAP-50 +17%
ppl per its author, consistent with our +0.06-0.11 nats). One 128 GB single-box claim: 15 tok/s on
an M5 Max via SSD streaming, code unpublished.

The mixed-precision idea (2/3/4-bit per layer): the resident budget is 1.16 bit/weight over ALL experts,
so any mix must be paired with pruning to be resident: keep 40% at ~3 bpw (~82 GB) or keep 30%
with the coldest 40% of those at 3 bpw (~77 GB). A per-row 8-of-16 codebook (subset of the FP4
grid, so it runs in the FP4 arena for measurement) has 21% relative weight error (2-bit: 37%) --
near the scalar-quantizer floor, roughly double FP4's own error. Teacher-forced runs of the
candidates are in `results/simq/`.

### 2026-09-11 08:40 -- CB3 (3-bit codebook) format: quality yes, kernel not yet fast

* Held-out, keep 40% of experts at simulated 3-bit: coding 1.539 / general 3.212 (+0.03 / +0.02 vs the
  full FP4 model; the 3-bit step itself ~+0.01-0.03 on top of the pruning). Worth building.
* `tools/cb3.py`: packed format (2-bit plane + 1-bit plane + 8-entry per-row codebook on the FP4 grid,
  UE8M0 scales kept) = 14.45 MB per expert (3.07 bpw); pack/unpack bit-exact vs the simulation.
* `tools/cb3_moe.py`: grouped MoE kernel on CB3 arenas -- CORRECT (rel err 4.4e-3, same as the FP4
  kernel vs its reference) but SLOW: 39-54 GB/s of CB3 bytes with register reshapes (variant 1),
  4-8 GB/s with gather loads (variant 2, current), vs ~190 GB/s for the FP4 kernel. The 3-bit ->
  FP4-nibble rebuild needs cross-lane bit movement; the next attempt is an inline-PTX per-lane decoder
  (one lo word = 16 codes, one hi half-word, the cbword) producing packed bytes with a permuted K
  order matched by the x loads. Parked behind the cheaper win below.
* Cheaper win with FP4 only: in pruned all-resident mode the 400-slot transient ring (7.5 GB) and the
  20 GB host reserve are oversized; `--transient-slots 16 --keep-free-gb 14` should give ~5,000 slots
  (~32% kept, all resident) -- quality between the measured 30% and 40% rows, speed of the resident path.

### 2026-09-11 09:30-09:50 -- resident step 195 -> 168 ms
* The FP4 kernel's `_route_kernel` ran one program per ARENA slot (4,813) and scanned all pairs in
  each: 29 ms per step at decode size. Replaced by `build_routing_small` (torch sort/cumsum, static
  shapes, graph-capturable) for P <= 64 pairs. Unit test (`tools/test_fp4_moe.py`) still passes.
* Gate GEMM back to bf16 tensor cores (weights are bf16 in the checkpoint; fp32 accumulation either
  way) -- my earlier "fp32 to avoid near-tie flips" was chasing noise that came from elsewhere.
* Device slot LUT + one graph per layer in resident mode: measurable but small (13.1 vs 12.95).
* Engram rows: `.cpu()` of the hash ids inside a reader thread synchronized with the queued graphs
  (48 ms waits). Now: ids to host before any replay, threads do only preads (32/table), dequant on
  the main thread at the consuming layer. Reads (~6 ms both tables) are hidden behind layers 0..13.
* e2e: 15.2-15.7 tok/s at keep 31 %; 168 ms step + 15 ms draft => ~16.5 tok/s at acceptance 3.

### 2026-09-11 10:20 -- long-prompt check of the served config (measured)

8,192-token prompt via the gateway (thinking off, greedy): TTFT 39.6 s = 207 tok/s prefill, decode
16.9 tok/s at acceptance 3.28, coherent summary. Recorded as RESULTS 2.8. Decode does not degrade
with the 8k KV (CSA2 index_topk 512 bounds the attended set). Next: better keep-set for the same
4,800-expert budget (global ranking instead of uniform 120 per layer), measured on the held-out
corpus, teacher-forced.

### 2026-09-11 10:00-10:17 -- keep-set selection: global ranking loses to uniform (measured, held-out)

Hypothesis: under the same 4,800-expert budget, one cross-layer ranking (per-layer-normalized
routing frequency, at least 24 experts per layer; `--prune-select global`, `build_keep_masks` in
engine/v41_engine.py) should beat "top-120 of every layer", because it keeps the most routing
mass: flat-routing layers get up to 166 experts, skewed ones as few as 83. Teacher-forced NLL on
`corpus/heldout_corpus.jsonl` says no:

| budget | select | coding NLL | general NLL |
|---|---|---|---|
| 31 % (4,800) | uniform (v0.2.0-wip) | 1.5729 | 3.3788 |
| 31 % (4,800) | global | 1.5983 (+0.025) | 3.4193 (+0.041) |
| 40 % (6,160) | uniform | 1.5285 | 3.3017 |
| 40 % (6,160) | global | 1.5433 (+0.015) | 3.3004 (-0.001) |

Routing mass kept is not the quality objective: a skewed layer's cold experts cost more when they
are dropped than a flat layer's cold experts gain when kept. The option stays in the CLI/launcher
(`PRUNE_SELECT`, default `uniform`) as a documented negative result; the served config is unchanged.

Same window: the LM head and the DSpark Markov head are now loaded bf16 (their stored dtype)
instead of an fp32 copy; `DSV41_HEAD_FP32=1` restores the fp32 reference math. The fast decode path
always ran a bf16 head copy, so served numerics are unchanged, and the 2.65 GB fp32 copy is gone
(~140 expert slots). Teacher-forced check (uniform 31 %, same corpus) with the bf16 head: coding
1.5716 / general 3.3769 vs 1.5729 / 3.3788 with fp32 -- inside run-to-run noise.

### 2026-09-11 10:25-10:55 -- two decode kernels: wo_a stays fp8, fused sinked-softmax attention

Two GPU-side costs of the graphed verify step, both replaced by Triton kernels, both behind an env
switch that restores the old path (`DSV41_WOA_FP8=0`, `DSV41_FUSED_ATTN=0`).

**1. `wo_a` in its stored fp8 format.** `convert.py` dequantizes the attention output LoRA and
everything downstream kept it that way: `[8, 1024, 4096]` bf16 = 67 MB per layer, 2.7 GB read per
step, measured 347 us per layer (120 calls of `cutlass_80_wmma_tensorop_bf16` in the profile) =
13.9 ms per step. `tools/fp8_linear.py` now has `FP8GroupedWeight` (fp8 e4m3 `[G*R, K]` + the UE8M0
32x32 block scales, addressed as G matrices; group g is just rows `g*R..(g+1)*R` of W and rows
`g*R/32..` of the scale table, nothing is copied) and `_fp8_grouped_kernel`, which is
`_fp8_linear_kernel` with a third grid axis for the group. `v41_ref.make_wo_a` / `v41_ref.wo_a_proj`
route both the graphed path (engine/fastdecode.py) and the non-graphed one (engine/model.py
`attention`, tools/v41_ref.py `attention`) through the same weight object, so prefill uses it too
(BLOCK_M=64 there; no transient dequant). Measured in the step: 198.6 us per layer = 7.95 ms, and
1.1 GB less resident (`CUDA free` 95.0 -> 96.1 GB).

The kernel reads 33.5 MB per layer in ~188 us = 178 GB/s. That is not a tiling problem: with the L2
flushed between calls, BLOCK_N 32/64/128/256 x num_stages 2/3/4 all land within 3 % of each other,
and a plain 64 MB device write on this box runs at ~169 GB/s. It is at the achievable bandwidth, so
the remaining factor-of-1.5 to the 273 GB/s nameplate is the box, not the kernel.

Being a Triton kernel it is also row-count- and row-offset-invariant by construction (each output
element is one fp32 accumulation over K in fixed BLOCK_K steps), unlike cuBLAS, so it does not need
the `MM_TILE` row tiling: rows [0:6], [7:13] and [1000:1006] of a 2048-row call come out
bit-identical to the 6-row calls.

**2. Fused decode attention.** `tools/decode_attn.py`: `_dattn_kernel` (flash-decoding over the key
axis, one program per (token, 16 heads, key split)) + `_dattn_combine`. Q and K are read bf16, the
scores and the whole softmax are fp32 and are never rounded to bf16. The PV product is the one place
that has to hand `tl.dot` a tensor-core dtype, so p is split into a bf16 high part and a bf16
remainder and both are accumulated (`PV_SPLIT`): that is the difference between 1.0e-4 and 2.5e-3
relative error against the fp32 torch path, i.e. between "below the bf16 rounding of the output" and
"above it". The all-masked row keeps working the way the torch path did: the max is clamped to
-1e30, so `exp(sink + 1e30) = inf` and the row comes out exactly zero (checked).

Note for anyone reading the old profile: `head_dim` = 512 **already contains** the 64 RoPE dims, so
the kernel's d is 512, not 576. The kernel still carries a DA+DB split for non-power-of-two head
dims (tested at 576); for this checkpoint DB = 0 and the second block is compiled away.

Also a correction to an earlier reading of `results/profile_fast.txt`: the 258 x 122 us
`gemmSN_TN_kernel<float,...>` calls are **not** the attention. They are the fp32 HC-mix GEMMs
(`F.linear(x.flatten(1).float() [6, 20480], hc_attn_fn/hc_ffn_fn [24, 20480])`, 2 per layer) plus
the ratio-2 compressor projections -- 86 launches, 10.5 ms per step, completely unchanged by this
work and now the single largest non-MoE item. 2.5 MB of fp32 weights in 122 us is 20 GB/s; that is
the next thing to fix. The attention itself was the two `cutlass_80_simt_sgemm_128x32` entries,
51.1 + 27.0 us per layer = 3.13 ms per step, now `_dattn_kernel` + `_dattn_combine` = 1.01 ms, plus
2.45 ms less in `unrolled_elementwise_kernel` (the two `.float()` casts and the exp that went away:
3036 -> 2676 launches).

**Measured A/B** (same box, back to back, nothing else running).

`engine/profile_fast.py` (PK=0.31 AG=90.5), wall ms:

| | step | draft |
|---|---|---|
| `DSV41_WOA_FP8=0 DSV41_FUSED_ATTN=0` | 165.7 ms | 14.3 ms |
| both on (default) | 152.7 ms | 13.7 ms |

Served config, 200 greedy tokens, same prompt as RESULTS 2.x
(`--prune-keep 0.31 --arena-gb 90.5 --transient-slots 8 --keep-free-gb 10`):

| | decode_tok_s | accept_len_mean | steps | decode_s | prefill_s |
|---|---|---|---|---|---|
| switches off | 15.28 | 2.97 | 68 | 13.221 | 5.159 |
| both on | **16.86** | 3.06 | 65 | 11.802 | 2.824 |

Output coherent in both (the usual `lru_ttl_cache` module). Part of the 15.28 -> 16.86 is the
acceptance difference (2.97 vs 3.06), which moves by itself run to run; the step-time A/B above is
the cleaner number: 13.0 ms of 165.7.

**Correctness.** New `engine/test_kernels.py`: `wo_a` against the real checkpoint weight (layers 0
and 7, `mtp.0`) dequantized exactly as before, at T = 1/5/6/16/64/2048 -- rel 2.6e-8..6.1e-5, max
abs at bf16 ulp; plus the row-invariance checks above. Attention against the fp32 torch path at
(T=6, n=640), (6, 128), (5, 133), (1, 640) and d=576, split 1/2/4 -- rel 7.4e-5..1.3e-4, max abs
2.4e-4..4.9e-4; all-masked rows exactly zero; one visible key everywhere finite; and the same
numbers after a CUDA-graph capture and two replays with fresh inputs.
`engine/test_fastdecode.py` (which replays one verify block through `FastDecoder` and through the
un-graphed `Model.forward` on identical state, then compares drafts, logits and `main_hidden`)
still passes and if anything agrees better than before: parity 0/1 logits rel err 0.0713 / 0.0458
and argmax agreement 1.00 / 1.00, against 0.1496 / 0.0343 and 1.00 / 0.83 with the switches off.
Its own timing: step 143.1 / 144.7 ms vs 154.3 / 150.1 ms (that harness runs a smaller auto-sized
arena than the served config, so its absolute step time is not comparable to the table above).

**Caveats.**
* `kv_all` is still materialised by a `torch.cat` of the window rows and the CSA2 rows before the
  kernel runs (4.4 MB written and read again per layer, ~1.3 ms per step). Giving the kernel two
  base pointers and a boundary instead would remove it; not done.
* With only T x 64/16 = 24 programs the key axis has to be split to fill 48 SMs; `DSV41_ATTN_SPLIT`
  defaults to 2 (split 4 measured the same, split 1 is ~35 % slower).
* The two paths are not bit-identical to the old ones -- they were never meant to be (the fast
  decode path already differs from `Model.forward`), and the gap to `Model.forward` got smaller,
  not larger.

### 2026-09-11 11:00-11:20 -- the fp32 HC-mix GEMMs, and the kv_all cat

**What the 258 `gemmSN_TN_kernel<float, 128, 16, ...>` calls were.** Exactly two things, and the
count now checks out:

| op | per step | shape (M, N, K) | weight |
|---|---|---|---|
| `_hc_mixes`: `F.linear(h.flatten(1).float(), hc_attn_fn / hc_ffn_fn)` -- one before attention, one before the FFN, x 40 backbone layers | 80 | 6, 24, 20480 | fp32 1.97 MB |
| `_compressed`: `F.linear(x.float(), comp_wkv / comp_wgate)` on the three ratio-2 KV-source layers (2, 8, 14) | 6 | 6, 512, 5120 | fp32 10.5 MB |

86 per step x 3 profiler iterations = 258. (Layer 20 is also a KV source but has ratio 1 and a bf16
`comp_wkv`, so it is not in this row. The draft's six `_hc_mixes` calls are not either --
`profile_fast.py`'s `one()` replays only the step graph.) After the change the row is down to 18
calls, i.e. precisely the 6 compressor calls x 3, which confirms the inventory.

**The HC shape is latency-bound, the compressor shape is not.** Measured standalone on the GB10:

| shape | `F.linear` fp32 | this kernel |
|---|---|---|
| M=6 N=24 K=20480 (hc_fn) | 84.0 us = 29.2 GB/s | 22.7 us = 108 GB/s |
| M=6 N=512 K=5120 (compressor) | 30.8 us = 344 GB/s | 39.0 us = 272 GB/s |

So cuBLAS is only bad when N is tiny; with N=512 it is already near bandwidth and beats the kernel.
`tools/fp32_skinny.py` therefore has a `wins(N)` gate (N <= 64, `DSV41_SKINNY_MAX_N`) and the six
compressor GEMMs deliberately stay on cuBLAS. `DSV41_HC_KERNEL=0` puts everything back on `F.linear`.

**The kernel.** N=24 is a single BLOCK_N tile and M is 6, so the only axis to parallelise is K:
`_skinny_kernel` gives each program `ceil(ceil(K/BLOCK_K)/SPLIT_K)*BLOCK_K` columns (46 programs on
48 SMs, none idle), accumulates fp32, and `_skinny_reduce` sums the partials. No fp32 `atomic_add`:
its summation order is non-deterministic, and this has to replay identically. The first version of
the reduce looped over the splits and cost 15 us for 94 kB -- 46 dependent loads, pure latency; one
wide `[SPLIT, 8, 32]` load plus `tl.sum` over the split axis brought the pair to 22.7 us. `x` is
passed in bf16 and upcast on the loaded tile (the same values `x.float()` produces, half the bytes);
the fp32 copy is still made for the rsqrt.

The two hc GEMMs of a layer are **not** fused into one call: the first reads `h` before attention and
the second reads `h` after `hc_post`, so they are not available at the same time and fusing them
would change the math order.

Accuracy: 4.2e-7 relative to `F.linear` (the gate was 1e-6), bit-reproducible across calls. Against
an fp64 reference over 20 random draws, cuBLAS is 1.8e-7 off and this kernel 4.1e-7 -- both at the
fp32 rounding floor, cuBLAS consistently the closer of the two (its K tree is deeper). That
difference is what makes the greedy text diverge; see the A/B caveat below.

**kv_all.** `decode_attention` now takes the two key pieces as two base pointers plus the boundary
n1 (`_seg` walks a segment; the mask stays one `[T, n1+n2]` tensor -- catting the masks is 3.8 kB per
layer). The `torch.cat` that built `kv_all` is gone from both branches, and the draft's window, which
was a stride-0 `expand`, is now passed as that broadcast view instead of being materialised. In the
profile the 114-call `CatArrayBatchedCopy` row (9.05 us each = 0.34 ms per step; 114 = 38 layers with
a compressed cache x 3) disappears. That is less than the 1.3 ms estimated last round -- the estimate
assumed the cat cost a full write plus a read at DRAM bandwidth, but the 3.9 MB it touched was L2
-resident. The two-segment kernel is **bit-identical** to the single-tensor one (unit test, three
different boundaries), which the A/B below confirms end to end.

**Measured.** `engine/profile_fast.py` (PK=0.31 AG=90.5):

| | step | draft | Self CUDA total (3 iters) |
|---|---|---|---|
| before this round | 152.7 ms | 13.7 ms | 460.164 ms |
| after | **147.2 ms** | 13.0 ms | **432.757 ms** |

`gemmSN_TN_kernel<float>` 258 calls / 31.635 ms -> 18 calls / 1.083 ms; `_skinny_kernel` 240 calls
at 22.086 us = 1.77 ms per step (`_skinny_reduce` falls below the top-28 cut, so < 0.25 ms/step).
GPU time per step is 9.1 ms lower but wall is only 5.5 ms lower: ~3.6 ms of the step is not GPU-busy
time (gaps between the per-layer graph replays and the host work between them), which is now the
next thing worth looking at.

`engine/test_fastdecode.py`: parity 0/1 drafts equal, logits rel err 0.0916 / 0.0350, argmax
agreement **1.00 / 1.00**, main_hidden rel 0.0813 / 0.0564; its own step timing 133.9 / 136.3 ms
against 143.1 / 144.7 ms last round.

Served config, 200 greedy tokens, same prompt:

| | decode_tok_s | accept_len_mean | steps | decode_s | ms per step+draft |
|---|---|---|---|---|---|
| `DSV41_HC_KERNEL=0` | 17.11 | 3.06 | 65 | 11.633 | 178.9 |
| defaults | 16.71 | 2.83 | 71 | 12.031 | **169.5** |

**Read that table carefully.** The step got 9.4 ms (5.3 %) faster and tok/s went *down*, because the
mean accepted length fell from 3.06 to 2.83. The 4e-7 change in the HC mixes flips borderline
decisions in the Sinkhorn and the router, the greedy text diverges after a few tokens (both outputs
are coherent `lru_ttl_cache` modules, worded differently), and this prompt happened to land on a
worse draft-acceptance trajectory. Nothing about the acceptance is caused by the kernel being
"worse" -- 4e-7 is 300x below the tolerance and either direction of rounding would reshuffle the same
decisions. The prompt-independent measurements (profile wall, GPU total, `test_fastdecode` step) all
move the same way, so the default stays on. Note also that the `DSV41_HC_KERNEL=0` arm reproduces
last round's numbers exactly (3.06 acceptance, 65 steps) -- that is the end-to-end proof that the
kv_all change is bit-identical.

**Caveats.**
* One A/B sample cannot separate a real acceptance effect from this kind of coin flip. If acceptance
  on this recipe is ever measured properly it should be over several prompts, not one.
* The six compressor GEMMs (1.08 ms per step) stay on cuBLAS by measurement, not by omission.
* `engine/model.py`'s prefill `hc_mixes` still goes through `R.mm`; at prefill M (up to 2048) cuBLAS
  is in its element and the skinny kernel would not help.

### 2026-09-11 11:20-12:10 -- the non-GPU gap (it is not launches), and a CB3 kernel that is fast

**Part A: the ~3.6 ms/step of non-GPU time is per-KERNEL, not per-graph-launch.**

With the device slot LUT the routing is on-device, so the only host dependency inside a step is the
Engram rows of layers 1 and 14. Two changes, both behind switches:

* `DSV41_GRAPH_SEGMENTS=1` (default): in resident mode `capture()` now captures runs of layers
  between the Engram boundaries as single graphs -- `[0]`, `[1..13]`, `[14..39 + final]` -- so a step
  is 3 graph replays instead of 41. The overlap is unchanged: a segment is queued asynchronously, so
  the host still blocks on the next boundary's NVMe reads while the GPU runs the segment before it.
* `DSV41_ENGRAM_PINNED=1` (default **0**): `EngramTable.to_device` stages the raw rows and the
  row-index vector in pinned buffers and copies them non-blocking, instead of a pageable
  `.to(device)` that synchronises the stream.

Neither is worth anything. `profile_fast.py` wall: 147.2 ms before, 146.8 with segments, 146.6 with
both -- inside noise. The decode A/B, back to back, same binary, shipping config:

| | decode_tok_s | accept | steps | decode_s |
|---|---|---|---|---|
| `DSV41_GRAPH_SEGMENTS=0` | 16.56 | 2.83 | 71 | 12.138 |
| default (segments on) | 16.63 | 2.83 | 71 | 12.083 |

Identical acceptance and step count, i.e. the change is bit-identical, and 0.5 % apart, i.e. nothing.
The pinned staging is the more interesting negative: it *does* make `to_device` itself ~12x cheaper
(0.27 s vs 3.21 s over a 200-token run, because the pageable copy synchronises the stream and
absorbs the queued graph work), and the decode is not faster for it -- 12.04 s single-buffered,
13.57 s double-buffered, against 12.08-12.14 s pageable. The host just blocks somewhere else.
Defaulted off; the code stays behind the switch.

The reason merging launches cannot help: Self CUDA time is 434.571 ms over 3 iterations = 144.9 ms
per step against a 146.6 ms wall, and the top-28 profiler rows alone account for **5,318 kernels per
step**. 1.7 ms spread over >5,300 kernels is ~0.3 us each -- inter-kernel latency inside a graph, not
launch overhead. To close it one has to launch fewer *kernels*, not fewer graphs: the candidates are
the 882 `_fp8_linear_kernel` and the ~1,800-2,700 one-microsecond elementwise kernels per step.

**Part B: CB3 at 182 GB/s -- the target is met.**

The three ideas in order, with what each was actually worth:

*Idea 1, per-lane decode with the codebook in registers.* This is the shape of every variant here and
was never the bottleneck. The per-row codebook lives in one loop-invariant register as eight packed
nibbles, and the lookup is one variable shift (`(cw >> (idx*4)) & 15`) or, in the PTX version, one
`prmt.b32`.

*Idea 2, per-matrix instead of per-row codebook.* Not implemented, and not because of quality: it
**cannot** reduce the instruction count. The per-row codebook word is already loop-invariant and
already in a register, so nothing in the inner loop changes; a per-matrix codebook would only remove
8 bytes per row (0.005 bit/weight) and would cost the per-row adaptivity that the 21 % weight error
depends on. Measuring its quality would have been measuring a change with no upside.

*Idea 3, layout change at pack time.* This is what made it work. `tools/cb3.py` gains a v2 layout
with exactly the same bytes -- 2 bits in the lo plane, 1 in the hi plane, 8 codebook bytes and the
UE8M0 scales per row -- that only changes **which weight sits in which bit**. A dot product is
order-invariant along K and the scale is per 32 consecutive K, so any permutation inside a 32-group
is free provided the activation is loaded to match; the permutation is chosen so that the existing
`_chunk_dot` even/odd x-load pattern still matches. The result is that one `[BN, BW/4]` lo load and
one `[BN, BW/8]` hi load, split in registers, give every scale group its even and its odd index tile
with nothing but shifts and masks -- no gathers, no reshapes, no shared memory.
`pack/unpack/dequant` stay bit-exact with v1 and with `engine/codebook_sim.py`.

Then two more things had to be right, and the second one was the real answer:

* **PTX with pack=4** (`_cb3_asm`): the same arithmetic four bytes per instruction, with the codebook
  lookup as a single `prmt.b32` (it selects one of 8 source bytes per output byte from a nibble
  selector, which is exactly a 3-bit codebook index) and a 4-op byte-lane -> nibble compaction to
  build that selector. 20 instructions per 8 weights against ~10 register ops per weight in Triton.
* **Row-tile width.** Cutting the PTX from 24 to 20 instructions with two `lop3` fusions changed
  nothing (159.6 -> 158.4 GB/s), which said the kernel was not instruction-bound. Measured directly,
  the achievable read bandwidth of a row-strided tile on GB10 is a cliff in its width:

  | row-tile width | 16 B | 32 B | 64 B | 128 B | 256 B | plain copy |
  |---|---|---|---|---|---|---|
  | GB/s | 100.8 | 101.5 | 185.2 | 218.1 | 217.5 | 219.4 |

  A 256-weight block gives a 64 B lo tile but only a **32 B hi tile**, and that one load capped the
  whole kernel. 512-weight blocks give 128 B + 64 B. K = 5120 (w1/w3) takes ten of them; K = 2304
  (w2) is 2^8 * 9 and admits only 4 x 512 + 1 x 256, so w2 is packed ragged and the kernel peels the
  tail (`block_plan`).

Measured on 21-24 real layer-0 experts, T=6 top-6 (`tools/test_cb3_moe.py`):

| kernel | ms | GB/s of expert bytes |
|---|---|---|
| FP4 (18.80 MB/expert) | 2.14 | 184.1 |
| CB3 v1 (gather variant, parked 2026-09-11 08:40) | 17.41 | 19.9 |
| CB3 v2 (new layout, Triton byte ops) | 1.97 | 154.0 |
| CB3 v3 (new layout + PTX + 128/64 B tiles) | **1.67** | **181.5** |

Best single measurement 182.0 GB/s at 0.787x the FP4 kernel's time for 0.769x the bytes, i.e. the
format pays for itself: it costs less time than it saves bytes. Correctness: the dequant is bit-identical to `codebook_sim`, and the kernel
agrees with the FP4 kernel run on the same re-quantized weights to 8.6e-5 (its error against the
dequantized reference is 4.4e-3, which is the FP4 kernel's own error on the same data). Size is
unchanged at 3.26-3.28 bit/weight, 14.45 MB per expert.

**Hazard worth remembering:** at BN=32, `num_warps=8` is not only slower but **wrong** -- rel err ~4
instead of 4.3e-3. Something in Triton/ptxas miscompiles the inline asm at that warp count. The
configs are pinned to `num_warps=4` and `CB3_UP_CFG`/`CB3_DOWN_CFG` carry the warning.

**What a CB3 arena would buy (arithmetic, 90.5 GB, 15,360 routed experts).** FP4 is 18,800,640 bytes
per expert, CB3 14,454,784 (0.769x).

| arena content | experts that fit | % of all routed |
|---|---|---|
| all FP4 (shipped today) | 4,813 | 31.3 % |
| coldest 60 % of the kept set in CB3 | 5,589 | 36.4 % |
| all CB3 | 6,260 | **40.8 %** |

So "keep 40 % with the coldest 60 % in CB3" does **not** fit 90.5 GB: 36.4 % does, and reaching 40 %
needs 93.7 % of the kept set in CB3, i.e. essentially all of it. The quality of that configuration is
already measured (NOTES 2026-09-11 08:40, held-out, teacher-forced): keep 40 % at simulated 3-bit is
coding 1.539 / general 3.212, against the shipped keep-31 % FP4's 1.5729 / 3.3788 -- better on both.

**Not done, and deliberately so.** The kernel is finished and tested but is NOT wired into the
engine. A CB3 arena tier means a second arena in `engine/experts.py`, a tier bit in the device slot
LUT, a two-tier MoE dispatch in `model.moe_fn` (two `build_routing` passes writing disjoint rows of
the same `h`/`parts` buffers, which is clean but touches the graph capture), warm-start assignment of
the coldest kept experts, and the `--cb3-cold-frac` CLI. That is a serving-path change of its own
size and I stopped short of it rather than half-land it. The decode line at keep 40 % and its
teacher-forced check belong with that wiring.

### 2026-09-11 12:15-14:10 -- CB3 wired into the serving path: keep 40 % all-resident

Single-tier: `expert_format` = `fp4` (default, unchanged) or `cb3`, where the WHOLE routed-expert
arena is `CB3ArenaV2` slots packed on the GPU at warm start from the same FP4 shards. Plumbed
through `V41Engine(expert_format=...)`, `--expert-format`, the generic `--engine-kwargs`,
`EXPERT_FORMAT` in `start.sh` and `scripts/entrypoint.sh`, `env.example`, `config()`
(`expert_format` + `expert_mb`) and `docs/install.md`. `--sim-bits` is unchanged and FP4-only; the
engine refuses to combine them, because `cb3` IS the format `--sim-bits` simulates.

The arena arithmetic works out: 90.5 GB is 6,260 CB3 slots of 14.45 MB (40.8 % of the 15,360 routed
experts) against 4,813 FP4 slots (31.3 %). `--prune-keep 0.40` keeps `ceil(0.40*384) = 154` experts
per layer = 6,160 = 89.0 GB, so the LRU (6,252 slots after the 8-slot transient ring) holds all of
them and decode never touches NVMe. The ring stays at 8 slots of the same format.

`moe_fn` became a dispatcher on the arena type rather than a fixed kernel, so the **DSpark draft
arena stays FP4**: it is 384 experts (7.2 GB) read five times per step, and a 3-bit drafter would
cost acceptance for nothing.

**1. Prefill does NOT run the CB3 kernel.** CB3's decode work is paid once per pair BLOCK while its
byte saving is paid once per expert, so the two scale differently. Measured on one layer's real kept
set (154 experts, top-6), against an FP4 arena holding the *same* re-quantized weights:

| | FP4 | CB3 kernel direct | CB3 via the unpack fallback |
|---|---|---|---|
| T=512 | 26.94 ms | 172.97 ms (6.42x) | 50.64 ms (1.88x) |
| T=2048 | 55.31 ms | 368.55 ms (6.66x) | 74.52 ms (1.35x) |

So a call with more than 64 pairs unpacks the experts it needs back into packed FP4 codes -- a
Triton kernel reusing the same PTX decoder, writing bytes instead of dotting -- a batch of 32 at a
time into a 0.6 GB scratch arena, and runs the ordinary FP4 kernel. It is **bit-exact**: the CB3
codes are a subset of the FP4 grid, and the fallback's output is rel 0.0e+00 against the FP4 kernel
on the same weights (`tools/test_cb3_moe.py` T=64 now exercises exactly this path). `h` and `parts`
are written exactly once per (token, k) pair across the batches, so neither needs zeroing and the
reduction runs once. `DSV41_CB3_PREFILL=direct` forces the CB3 kernel instead, for measurement.

Also fixed on the way: `moe_forward_v3` was hard-coded to BM=16. With `_pick_bm` it now matches the
FP4 path, and at prefill sizes the old default was 6.5x FP4 rather than 2.3x.

**2. Warm start: 183 s** for 6,160 experts (89.0 GB resident, 123.1 GB read from NVMe), against 19 s
for 4,800 FP4 experts (90.2 GB, 97.5 GB read). The extra ~160 s is the GPU packing, not the reads.
`fp4_to_cb3_v2` was first made ~cheap: the old version found each code's position in its row
codebook with an `[N, K, 8]` equality test (94 MB of bool temporaries per matrix); `CodebookSim`
now carries a `pos` table (subset, level) -> index inside the subset, so the position is one gather,
and the packer works in uint8/int16 instead of int64. No on-disk cache: it would be 88.8 GB and the
box has 114 GB free, which is too tight to spend on a 3-minute start-up.

**3/4. Graphs and correctness.** CUDA-graph capture and the 6,160-entry device slot LUT work
unchanged (the decode call is P=36 <= 64, so it takes `build_routing_small`, static shapes, as the
FP4 path does). `engine/test_fastdecode.py` in the CB3 config: drafts equal, logits rel err
0.1175 / 0.0344, **argmax agreement 1.00 / 1.00** -- the same quality of agreement as FP4
(0.0916 / 0.0350). Its own step timing was 124.0 ms against 133.9-136.3 ms for keep-31 % FP4.
`engine/diag_decode.py` is a decode-vs-prefill consistency check that forces `DSV41_FAST=0`; with
CB3 the two sides would use different kernels (CB3 for the 1-token decode, the unpack fallback for
the prefill), so it measures the fallback's exactness rather than the decode path -- and that is
already covered bit-exactly by the unit test.

**Measurements** (one run each, same box, `--arena-gb 90.5 --transient-slots 8 --keep-free-gb 10`,
one engine load per config so the decode and the TTFT share a warm arena):

| | keep 0.31, fp4 (shipped) | keep 0.40, cb3 |
|---|---|---|
| resident | 4,800 experts, 90.2 GB, 31.3 % | **6,160 experts, 89.0 GB, 40.8 %** |
| warm start | 19 s | 183 s |
| 200-token greedy `decode_tok_s` | 16.61 | **18.98** (+14.3 %) |
| accepted length / steps | 2.83 / 71 | 3.03 / 66 |
| `decode_s` | 12.103 | 10.537 |
| TTFT, 1,806-token prompt | 11.11 s | **9.81 s** |
| held-out teacher-forced, coding NLL | 1.5705 | **1.5384** (-0.032) |
| held-out teacher-forced, general NLL | 3.3790 | **3.2087** (-0.170) |

The FP4 numbers reproduce RESULTS 2.4 (1.5729 / 3.3788), and the CB3 numbers reproduce the
*simulated* 3-bit keep-40 % row of that table (1.5392 / 3.2122) to 0.0008 and 0.0035 nats -- which is
the end-to-end proof that the packed format, the kernel and the simulation are the same thing.

Faster and better on every axis except warm start, so the box's `.env` moves to
`PRUNE_KEEP=0.40 EXPERT_FORMAT=cb3` (everything else unchanged).

**Caveats.**
* Warm start is 3 minutes instead of 19 seconds. That is the price of packing 123 GB of FP4 into
  89 GB of CB3 on every start; an on-disk cache would fix it at 88.8 GB of disk, which this box does
  not have to spare.
* Prefill pays ~1.35x the MoE time of a same-sized FP4 arena for the unpack. It does not show in the
  TTFT above because keep 0.40 and keep 0.31 are different working sets, but it is real.
* The `num_warps=8` inline-asm miscompile (NOTES 2026-09-11 11:20) is still pinned out of every CB3
  config, and the prefill path avoids the CB3 kernel entirely, so it cannot be hit there either.
* The mixed hot-FP4 / cold-CB3 arena is still not built. At 40.8 % all-CB3 there is no headroom left
  in 90.5 GB anyway; it would be a quality refinement (hot experts back at FP4), not a capacity one.

### 2026-09-11 14:30-16:00 -- the dense projections in FP4: attention yes, shared experts no

The routed experts have been 4-bit since the start (the checkpoint stores them that way) and are now
3-bit (CB3). Everything else -- the dense projections -- is still the checkpoint's fp8: e4m3 weights
with one UE8M0 power-of-two scale per 32x32 block, read by `_fp8_linear_kernel`
(tools/fp8_linear.py). In the profile of one graphed verify step that kernel is 294 calls and
29.4 ms, the second-largest GPU item after the two CB3 kernels.

**What those 294 calls are, and what they weigh.** `tools/dense_inventory.py` reads the safetensors
headers and the config and prints the inventory; it needs no GPU and loads no tensor. The verify
step runs, once: the four `_fp8_linear_kernel` projections of each of the 40 backbone layers
(`attn.wq_a`, `attn.wq_b`, `attn.wkv`, `attn.wo_b`), their shared-expert FFN (`w1`, `w2`, `w3`),
`attn.wo_a` on the grouped kernel, one indexer `wq_b` on each of the 8 index-source layers, one
engram `wkv` on each of the 2 engram layers, and in `_final` `mtp.0.main_proj` plus the three DSpark
blocks' `attn.wkv`. 163 + 120 + 11 = 294, which is exactly the count in the profile.

| group | calls per step | MB per step | the same in fp4 | saved |
|---|---|---|---|---|
| attention (wq_a, wq_b, wkv, wo_b) | 163 | 3734.0 | 1981.7 | 1752.2 |
| shared experts (w1, w2, w3) | 120 | 1417.0 | 752.0 | 664.9 |
| wo_a (grouped kernel, untouched) | 40 | 1343.5 | 713.0 | 630.5 |
| other (indexer wq_b, engram wkv, main_proj) | 11 | 435.6 | 231.2 | 204.4 |
| **total** | **334** | **6930.0** | **3678.0** | **3252.0** |

Per weight: `wq_b` [32768, 1280] and `wo_b` [5120, 8192] are 41.98 MB each and are 90 % of the
attention row; `wq_a` is 6.56 and `wkv` 2.62; the three shared-expert matrices are 11.81 MB each.
The single largest dense read of the step is the engram `wkv` [25600, 6144] at 157.44 MB, twice.

**The format.** Same as the routed experts: E2M1 codes packed two per byte along K (low nibble =
even element) plus one UE8M0 scale per 32 consecutive K weights of a row. That scale table is 32x
finer than the fp8 one (per row per 32 K, not per 32x32 block), so the ratio is not 1/2 but
(1/2 + 1/32) / (1 + 1/1024) = **0.5307**. `tools/fp4_linear.py::quantize_fp8_to_fp4` does the
conversion once at load time, a row block at a time: dequantize to fp32 with the stored block
scales, per-32-group amax, scale = 2^ceil(log2(amax/6)) corrected upward if that would still clip,
then round to nearest on the E2M1 grid with ties to even. Ties are not measure-zero here -- the
input is an e4m3 value divided by a power of two, so it lands exactly on 0.25 / 0.75 / 1.25 / 1.75 /
2.5 / 3.5 / 5.0 often -- so the rounding is done with two `torch.bucketize` calls whose results
differ only at a tie, and the even code is taken there.

**The kernel** (`_fp4_linear_kernel`) is `_fp8_linear_kernel`'s structure with fp4_moe's decoder:
the 64-byte-wide packed row tile, `_split4`, the hardware `cvt.rn.f16x2.e2m1x2` instruction, the
K-permutation trick (even and odd nibbles are two independent dot products against an activation
loaded with stride 2), fp32 accumulation, the group scale applied to the fp32 partial of each
32-wide K step. Two tile shapes, and the N tile is the part that had to be measured:

* At M = 6 the grid is one M-block wide, so a 128-wide N tile leaves 4 (wkv) to 40 (wo_b) programs
  for 48 SMs and the DRAM latency is never hidden. BLOCK_N = 32 is 5-25 % faster on every shape
  measured -- including the ones that were already wide enough -- at `num_warps=4, num_stages=3`.
  `num_warps=8` was bit-identical but 2-2.5x slower on two of the four shapes, and is avoided
  anyway (the inline-asm miscompile of 2026-09-11 11:20).
* At M = 2048 the M axis fills the machine by itself and BLOCK_N = 128 wins by 1.6-1.7x.

Prefill therefore runs the kernel too, unlike the fp8 path, which dequantizes the weight to bf16 on
every prefill call and hands it to cuBLAS. That dequant is the dominant cost there: for `wq_b` at
M = 2048 it is 5.14 ms of the fp8 path's 7.14 ms, against 4.33 ms for the fp4 kernel.
`DSV41_FP4_DENSE_PREFILL=dequant` restores the fp8 path's shape for comparison.

**Correctness** (`tools/test_fp4_linear.py`, nine real checkpoint weights). Against an fp32 matmul
on the same fp4 weights the kernel is 1.61e-3 to 1.73e-3 relative at M = 1, 6, 16, 64, 2048 -- which
is the bf16 rounding of the output, and is exactly what cuBLAS-bf16 scores on the same inputs (the
two differ from each other by 0 to 5e-5, and where they differ it is cuBLAS that has moved: at
M = 64 cuBLAS is 3.5e-3 from the fp32 reference while the kernel stays at 1.66e-3). Rows [0:6],
[7:13], [1000:1006], [0:1] and [0:16] of a 2048-row call come out bit-identical to the short calls,
so the kernel needs no `MM_TILE` row tiling, like the fp8 one.

The weight error itself is the whole story of this round: **0.121 to 0.124 relative** on every
weight (per-32-group mean 0.119-0.122, p99 0.163-0.167, worst group 0.20-0.25). That is what four
bits with eight magnitude levels costs, and it is 60x the fp8 weight error.

Effective bandwidth at M = 6, steady state (the call cycled over ~300 MB of copies of the weight, so
nothing is L2-resident and -- unlike an explicit L2 flush -- no flush write competes for DRAM):

| weight | fp4 | fp8 |
|---|---|---|
| `wq_b` [32768, 1280] | 111.9 us, 199.1 GB/s | 190.1 us, 220.9 GB/s |
| `wo_b` [5120, 8192] | 180.3 us, 123.6 GB/s | 194.0 us, 216.4 GB/s |
| `wq_a` [1280, 5120] | 53.4 us, 65.2 GB/s | 46.4 us, 141.3 GB/s |
| `wkv` [512, 5120] | 49.3 us, 28.2 GB/s | 39.2 us, 67.0 GB/s |
| shared `w1` [2304, 5120] | 66.2 us, 94.6 GB/s | 61.0 us, 193.6 GB/s |
| shared `w2` [5120, 2304] | 41.2 us, 152.0 GB/s | 54.3 us, 217.5 GB/s |
| shared `w3` [2304, 5120] | 57.4 us, 109.2 GB/s | 69.5 us, 169.9 GB/s |

Read that in wall time, not in GB/s: fp4 reads 0.53x the bytes, so it is ahead wherever its GB/s is
above 0.53x the fp8 figure. It wins on the two big attention matrices (1.70x and 1.08x) and loses on
the two small ones, whose grids are too narrow to reach bandwidth in either format.

**The switch.** `DSV41_DENSE_FP4` = `off` (default) | `shared` | `attn` | `shared,attn` (= `all`),
read by `v41_ref.dense_fp4_groups()` and applied in `LayerWeights` and `MTPWeights` as the weights
are loaded; `attn.wo_a`, the indexer `wq_b`, the engram `wkv` and `mtp.0.main_proj` always stay fp8.
`R.dense` and `R.mm` dispatch on the weight object, so the graphed decode path, the un-graphed
`Model.forward` and the prefill all follow automatically. The re-quantization adds ~20 s to the
weight load (57 s -> 78 s for `shared,attn`) and leaves ~0.7 GB of fp32 scratch in the caching
allocator, which the engine hands back with one `empty_cache()` before the arena is built.

**Measurements**, one run each, the served config (`--prune-keep 0.40 --expert-format cb3
--arena-gb 90.5 --transient-slots 8 --keep-free-gb 10`), same box, nothing else running.

Held-out teacher-forced, `corpus/heldout_corpus.jsonl`:

| `DSV41_DENSE_FP4` | coding NLL | delta | general NLL | delta |
|---|---|---|---|---|
| off | 1.5384 | -- | 3.2087 | -- |
| shared | 1.5531 | +0.0147 | 3.2740 | +0.0653 |
| attn | **1.5403** | **+0.0019** | **3.1738** | **-0.0349** |
| shared,attn | 1.5527 | +0.0143 | 3.2323 | +0.0236 |

The `off` row reproduces the CB3 keep-40 % numbers of the 12:15-14:10 section exactly. The shared
experts are the sensitive group: they are the one dense FFN every token goes through, with no
redundancy, and 4 bits costs 0.065 nats of general loss there. The attention projections carry the
same 0.12 weight error and it does not show -- `wq_b` and `wo_b` are wide maps whose output is a sum
over 1280 and 8192 terms, and the error averages out. The two are not additive either: `shared,attn`
is 0.042 nats better on general than `shared` alone, i.e. the two errors partly cancel on this
corpus. One corpus of 53 sequences cannot resolve differences of this size; the `attn` row's -0.035
is a gain only in the sense that it is indistinguishable from zero.

`engine/test_fastdecode.py` (the graphed `FastDecoder` against `Model.forward` on identical state):

| | drafts equal | logits rel err 0/1 | argmax agreement 0/1 |
|---|---|---|---|
| off | yes | 0.1175 / 0.0344 | 1.00 / 1.00 |
| attn | yes | 0.0613 / 0.0249 | **1.00 / 1.00** |
| shared,attn | no (parity 0) | 0.1021 / 0.0371 | 0.83 / 1.00 |

With `attn` the two paths agree better than they did in fp8, on both parities. With the shared
experts in fp4 as well the parity-0 draft diverges by one token, which makes the verify block
different and drops the block's argmax agreement to 5 of 6; the parity-1 numbers are unaffected.

`engine/profile_fast.py`, wall ms of one graphed verify step, and the dense kernels' GPU time per
step from the same profile:

| | step | draft | dense GPU ms per step |
|---|---|---|---|
| off | 134.4 | 13.4 | 29.41 (294 fp8 calls) |
| attn | **125.6** | 13.1 | **25.41** (163 fp4 + 131 fp8) |
| shared,attn | 128.9 | 13.1 | 26.20 (283 fp4 + 11 fp8) |

Putting the shared experts in fp4 as well makes the dense group slower, not faster, for the reason
in the bandwidth table: `w1` and `w3` are among the shapes where fp4 does not pay.

200 greedy tokens, the standard decode line, back to back:

| | decode_tok_s | accept_len_mean | steps | decode_s | prefill_s | ms per step+draft |
|---|---|---|---|---|---|---|
| off | 19.11 | 3.03 | 66 | 10.467 | 4.123 | 158.6 |
| attn | **20.85** | 3.23 | 62 | 9.592 | **3.275** | **154.7** |

(and, from the earlier pair in the same hour, off 19.03 / 3.03 / 66 and `shared,attn` 18.63 / 2.90 /
69 -- the shared-expert arm is slower end to end as well.) As in the 11:00-11:20 round, part of the
19.11 -> 20.85 is the accepted length moving from 3.03 to 3.23, which moves by itself run to run on
a single prompt; the prompt-independent numbers -- the profile wall, the dense GPU time, the
`test_fastdecode` step (120.5/124.8 -> 120.1/117.2 ms) and the 154.7 vs 158.6 ms per step+draft --
all move the same way. Prefill is 21 % faster because the fp8 path's per-call bf16 dequant is gone.

**Memory.** `attn` frees 1.751 GiB of resident weights (measured `CUDA free` before the arena is
allocated: 97.5 -> 99.6 GB); `shared` frees 0.666 GiB; both together 2.417 GiB (97.2 -> 99.8 GB).
At the current 90.5 GB arena that is headroom, not extra experts: 6,260 CB3 slots either way.
1.751 GiB would be 124 more CB3 slots if the arena were resized to take it.

**Verdict.** The attention group passes on every axis and the shared experts fail the loss gate by
6.5x, so the switch stays per-group and `off` remains the default in code; `DSV41_DENSE_FP4=attn` is
the setting worth having (.env is exported by start.sh, so one line there is enough).

**Caveats.**
* The held-out corpus is 53 sequences, 10,661 scored positions. Deltas of 0.002-0.035 nats are
  inside its resolution; only the shared experts' 0.065 is clearly outside it.
* The four small-N decode shapes (`wkv`, `wq_a`, shared `w1`/`w3`) are latency-bound in both
  formats -- 16 to 72 programs for 48 SMs at BLOCK_N = 32 -- and a split-K variant with a fixed-order
  reduction would fix that. Not built; `wkv` and `wq_a` together are only 9.2 MB of the attention
  group's 93.1 MB per layer.
* `attn.wo_a` (1343.5 MB per step, 630.5 MB of it saveable) is still fp8: it needs the grouped
  kernel, which would need the same treatment. Untouched, unmeasured.
* The quantizer's rounding is ties-to-even on the E2M1 grid, which is what the hardware `cvt.rn`
  does but not what `v41_ref.fp4_qdq` does (it rounds ties down). The two differ only on exact ties.
* `DSV41_FP4_DENSE_F16=0` feeds the tensor cores bf16 activations and bf16 (exact) decoded weights
  instead of fp16. Implemented and compiled, not measured: the fp16 path is what the routed-expert
  kernel already does with the same activations.

### 2026-09-11 16:05-17:05 -- `attn.wo_a` in FP4, and where the decode loop's non-graph time goes

Two independent things, each behind its own switch.

#### 1. The grouped FP4 kernel for `attn.wo_a`

`attn.wo_a` is the attention output LoRA: G = 8 independent [1024, 4096] matrices sharing one
row-major buffer. Since the 10:25-10:55 round it has been read in the checkpoint's fp8 format by
`_fp8_grouped_kernel` -- 33.55 MB per layer, 1,343.5 MB per verify step, 198.6 us per layer -- and it
was the last dense group outside the `DSV41_DENSE_FP4` switch (14:30-16:00 above: "still fp8, 630.5 MB
of it saveable, untouched, unmeasured").

`tools/fp4_linear.py` now carries the grouped variant next to the dense one: `FP4GroupedWeight`
(E2M1 codes [G*R, K/2] + one UE8M0 scale per 32 consecutive K weights of a row, [G*R, K/32]),
`_fp4_grouped_kernel`, `fp4_grouped_linear` and `quantize_fp8_grouped_to_fp4`. The kernel is
`_fp4_linear_kernel` with a third grid axis for the group -- exactly the relationship
`_fp8_grouped_kernel` has to `_fp8_linear_kernel`. A group is a row range `g*R .. (g+1)*R` of the
code matrix and the *same* row range of the scale table: the fp4 table has one row per weight row,
so unlike the fp8 32x32 block table it needs no division by 32. Nothing is copied, and the clamp on
the masked N lane keeps a program from reading another group's rows.

The re-quantization is the dense one applied to the flat [8192, 4096] matrix. That is not a
shortcut: the quantizer works per row per 32 consecutive K weights and a group is a row range, so no
amax ever crosses a group boundary and re-quantizing the flat matrix is the same arithmetic as
re-quantizing each group separately.

`DSV41_DENSE_FP4` grew a third group name -- `off | shared | attn | wo_a` and any comma combination,
with `all` now meaning all three rather than `shared,attn`. `v41_ref.make_wo_a` takes the group set
and returns an `FP4GroupedWeight` when `wo_a` is in it; `v41_ref.wo_a_proj` dispatches on the weight
object, so the graphed decode path (engine/fastdecode.py `_attention`), the un-graphed
`Model.attention` and tools/v41_ref.py's own `attention` all follow. Prefill runs the same kernel at
BLOCK_M = 64; `DSV41_FP4_DENSE_PREFILL=dequant` restores a transient dequant + einsum above T = 16,
as it does for the dense groups. `DSV41_WOA_FP8=0` still restores the original bf16 tensor and the
einsum, ahead of either kernel.

**BLOCK_N.** The group axis already supplies 8-fold parallelism, so at T = 6 a 64-wide N tile gives
8 x 1024/64 = 128 programs for 48 SMs -- the dense kernel's reason for preferring 32 does not apply
here. Measured on the real weight at T = 6, 32 / 64 / 128: 114.6 / 119.1 / 123.6 us on layer 0 and
116.7 / 112.9 / 120.1 us on layer 7, i.e. inside this shape's run-to-run spread. `pick_block_n_grouped`
takes 64 at decode M and 128 at prefill M.

**Correctness** (`engine/test_kernels.py`, section 1 extended; three real checkpoint weights).

| weight | fp8 -> fp4 | weight rel err | per-32-group mean / p99 / max |
|---|---|---|---|
| `layers.0.attn.wo_a` | 32.0 -> 17.0 MiB (0.5307x) | 0.1215 | 0.1201 / 0.1643 / 0.2212 |
| `layers.7.attn.wo_a` | 32.0 -> 17.0 MiB | 0.1214 | 0.1203 / 0.1639 / 0.2340 |
| `mtp.0.attn.wo_a` | 32.0 -> 17.0 MiB | 0.1221 | 0.1205 / 0.1647 / 0.2345 |

The same 0.12 the dense weights showed, which is what four bits with eight magnitude levels costs.
Against the bf16 einsum on the same fp4 weights at T = 1, 6, 16, 64 and 2048 the kernel is 0.0 to
4.9e-5 relative with max |delta| at one bf16 ulp of the output (<= 7.8e-3 against outputs of order
2). Against an fp32 matmul on those weights the kernel is 1.62e-3 to 1.66e-3 -- and the bf16 einsum
scores the identical 1.62e-3 to 1.66e-3 on the same inputs, so the kernel is at the output's bf16
rounding, not above it. Rows [0:6], [7:13] and [1000:1006] of a 2048-row call come out bit-identical
to the short calls, so like the other two it needs no `MM_TILE` row tiling.

**Bandwidth at T = 6**, steady state (the call cycled over ~300 MB of copies of the weight, so
nothing is L2-resident and no flush write competes for DRAM):

| weight | bf16 einsum (original) | fp8 grouped | fp4 grouped |
|---|---|---|---|
| `layers.0.attn.wo_a` | 295.0 us, 228 GB/s of 64.1 MiB | 156.4 us, 215 GB/s of 32.0 MiB | 119.1 us, 150 GB/s of 17.0 MiB |
| `layers.7.attn.wo_a` | 301.6 us, 223 GB/s | 162.2 us, 207 GB/s | 112.9 us, 158 GB/s |

Read that in wall time: fp4 reads 0.5307x the bytes at 0.70-0.76x the GB/s, so it is 1.31-1.44x
faster than fp8 standalone. In the real step, where the L2 is thrashed by everything else, the gap
is larger: the profile below has `_fp4_grouped_kernel` at 114.0 us per layer against the fp8
kernel's 198.6 us, i.e. 4.56 ms per step instead of 7.95.

**Resident weights.** `attn` alone reports 8.34 GiB allocated after the weight load; `attn,wo_a`
reports 7.70 GiB -- 0.64 GiB less, which is the predicted 43 x 15.72 MB. At the fixed 90.5 GB arena
that is headroom, not extra experts: 6,260 CB3 slots either way.

#### 2. Where the decode loop's non-graph time goes

`engine/v41_engine.py` grew a per-phase host wall-clock timer (`StepPhases`, `DSV41_STEP_TIMING=1`,
off by default: with the flag off the loop runs one `if ph is not None` per phase and nothing else).
Every number in the tables below is *host* time: the graphs are queued asynchronously, so a phase's
figure is the time the host spent there, including whatever GPU work it waited for. Three derived
rows come from `FastDecoder.stats` and are already inside the `step` phase.

**The first reading was wrong, and the table is what corrected it.** A fresh command-line run at
`DSV41_DENSE_FP4=attn` reports 162.8 ms per step against the graph harness's 125.6 + 13.1 = 138.7 ms,
and the obvious conclusion -- 24 ms per step of removable Python -- does not survive the breakdown:

| phase | ms/step (fresh run, 62 steps) | what it is |
|---|---|---|
| draft | 8.33 | **one** un-graphed `_draft()` call, amortized: on the first step `self.graphs` is still empty, so `draft()` falls through to the eager path (~0.5 s) before any graph exists |
| block | 13.48 | `torch.cat([torch.tensor([tok], device=cuda), drafts])` -- the pageable H2D is stream-ordered, so it blocks until the draft graph has finished; this is the draft's 13.1 ms of GPU time, not overhead |
| hash | 0.39 | `NgramHashState.forward` for 6 positions |
| hash_d2h | 0.02 | the hash ids to the host (free here -- `block` already drained the stream) |
| submit | 0.05 | two `ThreadPoolExecutor.submit` calls |
| step | 58.92 | 3 graph-segment replays + the Engram row waits; of it 48.5 ms is the Engram wait |
| verify | 81.54 | the accept/reject loop -- and the first `.item()` in it waits for the whole verify step |
| rollback / emit / consumer | 0.06 | |

The `step` phase's 48.5 ms of "Engram wait" is `fut.result()` plus `to_device`, and `to_device`'s
pageable H2D is stream-ordered too: at the layer-14 boundary roughly a third of the step's graph
work is queued ahead of it. So both of the two big non-`verify` numbers are GPU time that the host is
correctly using to run the next NVMe read, exactly as the pipeline was designed to. The genuinely
host-side residue of a *warm* step is about 1 ms.

What the 162.8 ms is made of, then: one eager draft (8.3), the CUDA-graph capture inside the first
step, first-call Triton compilation, and a cold Engram row cache (`engram_read_s` 1.33 s over the
run). A second run in the same process costs ~140 ms per step and a third ~133 ms, with the
difference tracking `engram_read_s` (1.208 s on the second, 0.193 s on the third) and nothing else.

**What was removed anyway** (`DSV41_LEAN_STEP`, default 1, `=0` restores the original code path in
both `engine/v41_engine.py` and `engine/fastdecode.py`):

* Greedy accept/reject on the GPU. `logits.argmax(-1)`, the five draft comparisons and the count of
  leading accepts (`cumprod` then `sum`, which is the index of the first mismatch) are one small
  graph of kernels, and the result reaches the host as a single 7-element copy into a pinned buffer
  instead of up to eleven separate `int()`/`bool()` syncs, six `sample_probs` calls and their six
  130k-wide `zeros_like` allocations. The bonus token is `argmax[a]` for every a, including a = 5,
  which is what the loop computed. An accepted stop token still ends the block with no bonus.
* The verify block is filled into a preallocated [6] buffer with `fill_` (a kernel argument, no H2D)
  and a device copy, instead of `torch.cat([torch.tensor([tok], device=cuda), drafts])`.
* `drafts.clone()` and `q.clone()` are gone on the greedy path: the drafter's static buffers are not
  written again until the next `draft()` call, which is after verification. `q` (5 x 130k fp32,
  2.6 MB) is not read at all when the temperature is 0.
* In `FastDecoder.step`: a preallocated `arange` for the positions, `expand` instead of `repeat` for
  the embedding rows, a constant `pre_mix` copy instead of `zero_()` + a scatter, and
  `prepare_pending_buffers()` once per step instead of twice (only the capture run overwrites them).

**Sampling is untouched.** The temperature > 0 path is the original code, RNG draw for RNG draw --
the rejection sampler draws a variable number of scalars depending on where it stops, so batching it
would move the RNG stream. The greedy path takes the argmax of the same logits in the same order.
Verified end to end: at temperature 0 the two paths produce byte-identical output text and identical
token lists (205 tokens, no divergence) in both A/B runs below.

**Phase table before and after**, same process, `DSV41_DENSE_FP4=attn,wo_a`, 200 greedy tokens, the
standard prompt. The two runs of a pair are back to back; the arms were run in both orders because
the second decode of a process still pays Engram NVMe reads that the third does not.

| phase, ms/step | run 2: original | run 3: lean | run 2: lean | run 3: original |
|---|---|---|---|---|
| draft | 0.045 | 0.034 | 0.033 | 0.043 |
| block | 13.233 | 0.012 | 0.012 | 13.243 |
| hash | 0.374 | 0.463 | 0.451 | 0.359 |
| hash_d2h | 0.021 | 12.914 | 12.960 | 0.021 |
| submit | 0.027 | 0.029 | 0.028 | 0.024 |
| step | 48.840 | 41.296 | 47.898 | 41.429 |
| verify | 78.399 | 77.763 | 78.130 | 78.320 |
| rollback + emit + consumer | 0.034 | 0.032 | 0.035 | 0.029 |
| **total** | **140.973** | **132.548** | **139.548** | **133.468** |
| decode_tok_s | 21.93 | 23.32 | 22.15 | 23.16 |
| `engram_read_s` over the run | not recorded | not recorded | 1.208 s | 0.193 s |

Read the columns pairwise by position, not by arm: the 2nd decode of a process costs ~140 ms per
step and the 3rd ~133 ms whichever arm is in it, because the 3rd finds every Engram row in the
process-local cache. At matched positions the lean path is 1.43 ms/step faster (position 2) and
0.92 ms/step faster (position 3). Note also that `block` and `hash_d2h` simply trade places: the wait
for the draft graph has to happen somewhere, because the Engram hash ids depend on the drafted
tokens and have to reach the host before the NVMe reads can start.

Acceptance was 3.09 and the output text identical in all four columns, so unlike the earlier
kernel rounds none of this is an acceptance coin flip.

#### Gates

**`engine/test_fastdecode.py`** (graphed `FastDecoder` against `Model.forward` on identical state),
shipped config, `DSV41_DENSE_FP4=attn,wo_a`:

| parity | drafts equal | logits rel err | argmax agreement | main_hidden rel | step / draft |
|---|---|---|---|---|---|
| 0 | no | 0.0746 | **1.00** | 0.0863 | 116.7 / 14.4 ms |
| 1 | yes | 0.0158 | **1.00** | 0.0211 | 117.9 / 13.5 ms |

Against the `attn` row of the 14:30-16:00 section (0.0613 / 0.0249, 1.00 / 1.00, 120.5 / 124.8 ms
for the two parities in the earlier harness run) the agreement is unchanged at 1.00 / 1.00. The
parity-0 draft now differs by one token, which makes the verify block different; the block's argmax
agreement is still 6 of 6.

**Teacher-forced held-out** (`corpus/heldout_corpus.jsonl`, 5,430 coding + 5,231 general scored
positions), one run, shipped config:

| `DSV41_DENSE_FP4` | coding NLL | delta vs `off` | general NLL | delta vs `off` |
|---|---|---|---|---|
| off (12:15-14:10 section) | 1.5384 | -- | 3.2087 | -- |
| attn (14:30-16:00 section) | 1.5403 | +0.0019 | 3.1738 | -0.0349 |
| **attn,wo_a** | **1.5346** | **-0.0038** | **3.1498** | **-0.0589** |

Top-1 accuracy 0.6862 coding / 0.4580 general. Both deltas are on the good side of zero, which on a
53-sequence corpus means indistinguishable from it -- the same caveat as the `attn` row's -0.035.
The point is the negative sign, not its size: the 0.12 weight error of the output LoRA does not show,
for the same reason it does not show on `wq_b` and `wo_b`, namely that the output is a sum over 4,096
terms and the error averages out. `results/densefp4/wo_a.json`.

**`engine/profile_fast.py`**, one graphed verify step, shipped config:

| | step | draft |
|---|---|---|
| `attn` (14:30-16:00 section) | 125.6 ms | 13.1 ms |
| `attn,wo_a` + lean step | **118.9 ms** | 13.5 ms |

Top eight kernels, per step (the profile runs three iterations):

| kernel | calls/step | ms/step |
|---|---|---|
| `_cb3v3_up_kernel` | 40 | 41.01 |
| `_cb3v3_down_kernel` | 40 | 22.54 |
| `_fp4_linear_kernel` | 163 | 16.15 |
| `_fp8_linear_kernel` | 131 | 9.78 |
| `cutlass_80_wmma_tensorop_bf16` (the LM head, 1.34 GB at 232 GB/s) | 1 | 5.79 |
| `_fp4_grouped_kernel` (`wo_a`) | 40 | 4.56 |
| `unrolled_elementwise_kernel` | 895 | 2.12 |
| `_skinny_kernel` | 80 | 1.85 |

Self CUDA total 116.6 ms per step against 118.9 ms of wall, i.e. ~2.3 ms of the step is not GPU-busy
time. The dense fp8/fp4 group is 30.5 ms per step (163 fp4 + 131 fp8 + 40 fp4-grouped) against
33.4 ms with `wo_a` in fp8.

**Decode line**, two fresh command-line runs back to back, 200 greedy tokens, the standard prompt:

| | decode_tok_s | accept_len_mean | steps | decode_s | prefill_s | ms per step |
|---|---|---|---|---|---|---|
| `attn`, original step | 20.62 | 3.23 | 62 | 9.701 | 3.268 | 156.5 |
| `attn,wo_a`, lean step | 20.02 | 2.99 | 67 | 9.989 | 2.609 | **149.1** |

**Read that table the way the 11:00-11:20 one had to be read.** The step got 7.4 ms (4.7 %) faster
and tok/s went down, because the mean accepted length fell from 3.23 to 2.99 on this one prompt. The
0.12 weight error on `wo_a` moves the logits enough to change which drafts the target accepts, and
this prompt landed on a worse trajectory; at the old run's acceptance the new configuration would be
3.24 / 0.1491 = 21.7 tok/s. Every prompt-independent number moves the same way: the profile wall
(125.6 -> 118.9), the dense GPU time (33.4 -> 30.5 ms), the in-process step time at matched positions
(140.97 -> 139.55 as the 2nd decode of a process, 133.47 -> 132.55 as the 3rd) and prefill (3.27 -> 2.61 s, because `wo_a`'s prefill call is
now the kernel instead of a bf16 dequant plus cuBLAS). Both outputs are coherent `lru_ttl_cache`
modules.

**Shipped.** `.env` on the serving box now carries `DSV41_DENSE_FP4=attn,wo_a`; the code default for
`DSV41_DENSE_FP4` stays `off` and `DSV41_LEAN_STEP` defaults to 1.

**Caveats.**
* One prompt cannot separate the acceptance drop from a coin flip; the two decode lines differ by
  five steps out of sixty-something.
* The four small-N decode shapes noted in the 14:30-16:00 section are still latency-bound, and the
  grouped kernel does not change that -- `wo_a`'s R = 1024 is wide enough.
* The lean step is worth about 1 ms per step, not the 24 ms the first reading of the fresh-run table
  suggested. The rest of that gap is one eager draft call, the graph capture and a cold Engram row
  cache, all of which a long generation amortizes away by itself.
* `DSV41_LEAN_STEP` only changes the greedy path. Sampled decoding still runs the sequential
  rejection loop with its per-position syncs, deliberately: batching it would change the RNG stream.
* The `to_device` H2D of the Engram rows remains the place where a third of the step's GPU time is
  absorbed. `DSV41_ENGRAM_PINNED=1` moves that wait elsewhere without removing it (measured
  2026-09-11 09:36, unchanged here).

### 2026-09-11 17:00-19:00 -- the LM head in fp8, and a 2-bit expert tier that is not worth building

Two independent pieces of work, each behind its own switch.

#### 1. The LM head's stored format

`head.weight` is [129280, 5120] bf16 = 1.324 GB, and it is the one weight read in full on every
decode step -- twice, because the DSpark drafter runs the same head over its five positions. In the
profile of the shipped configuration it is the fifth-largest kernel of the verify step (5.79 ms, a
cutlass bf16 GEMM at 232 GB/s). Every other weight of that size already has a narrower stored
format; this is the same move applied to the head.

**The formats.** `DSV41_HEAD_FMT` = `bf16` (default, the checkpoint's dtype) | `fp8` | `fp4`.
`fp8` is the dense projections' own stored format -- e4m3 codes with one UE8M0 power-of-two scale
per 32x32 block -- 0.663 GB, 0.5005x. `fp4` is the routed experts' format -- E2M1 codes two per byte
along K with one UE8M0 scale per 32 consecutive K weights of a row -- 0.352 GB, 0.2656x. The dense
weights arrive in the fp8 format from disk and the head does not, so `tools/fp8_linear.py` gained
`quantize_to_fp8` (per 32x32 block: amax, scale = 2^ceil(log2(amax/448)) corrected upward if that
would still clip, then the hardware's round-to-nearest-even e4m3 conversion; nothing can overflow,
because a value <= 448 never rounds above 448, whose neighbour in the grid is 512 with the tie at
480). The fp4 direction reuses `tools/fp4_linear.py::quantize_to_fp4` unchanged. Both run on the GPU
as the weight is loaded (0.2 s and 0.4 s) inside `v41_ref.make_head`, and every consumer reaches the
result through `head_logits` or `dense`, which dispatch on the object, so the graphed decode path,
the un-graphed `Model.forward` and the drafter all follow. `DSV41_HEAD_FP32=1` still selects the
reference's fp32 head and refuses to combine with the switch.

**Prefill.** Both kernels are shaped for bandwidth, not for arithmetic -- one fp32 accumulator per
output -- and on this weight at M = 512 they run at 26 and 12 GB/s against
cuBLAS's 155 (25.31 and 28.23 ms against 8.54). So above M = 16 a quantized head dequantizes 16,384
vocabulary rows at a time and hands each block to cuBLAS (`v41_ref._head_blocked`): a transient of
166 MB instead of the 1.324 GB a whole-weight dequant would need, and the extra DRAM traffic is one
read plus one write of the head per call. `DSV41_HEAD_PREFILL=kernel` runs the decode kernel at
every M instead.

**The weight and the head's own GEMM**, measured on the real weight with the call cycled over copies
so nothing is L2-resident (`bf16` is `F.linear`, i.e. the cutlass kernel the profile names):

| format | bytes | weight rel err | M = 6 | M = 512 |
|---|---|---|---|---|
| bf16 | 1.324 GB | -- | 5.68 ms, 233 GB/s | 8.54 ms |
| fp8 | 0.663 GB (0.5005x) | 0.0266 | 2.96 ms, 224 GB/s | 25.31 ms (kernel) |
| fp4 | 0.352 GB (0.2656x) | 0.1180 | 2.38 ms, 148 GB/s | 28.23 ms (kernel) |

fp8 reads half the bytes at 96 % of bf16's bandwidth and is 1.92x in wall time. fp4 reads 0.266x at
64 % and is only 0.58 ms better than fp8, for 4.4x the weight error.

**Held-out teacher-forced** (`corpus/heldout_corpus.jsonl`, 5,430 coding + 5,231 general scored
positions), one run per format, the shipped configuration (keep 0.40, CB3,
`DSV41_DENSE_FP4=attn,wo_a`):

| `DSV41_HEAD_FMT` | coding NLL | delta | general NLL | delta | top-1 coding / general |
|---|---|---|---|---|---|
| bf16 | 1.5346 | -- | 3.1498 | -- | 0.6862 / 0.4580 |
| **fp8** | **1.5351** | **+0.0004** | **3.1512** | **+0.0014** | 0.6856 / 0.4582 |
| fp4 | 1.5502 | +0.0156 | 3.1572 | +0.0074 | 0.6867 / 0.4584 |

The bf16 row reproduces the 16:05-17:05 section's numbers exactly. fp8's two deltas are far below
what a 53-sequence corpus can resolve. fp4's coding delta is not, and it is consistent sequence by
sequence -- on the four coding sequences checked individually it is +0.013 to +0.018 nats, never the
mixed signs a noise-sized difference gives -- so fp4 fails the loss gate on coding while passing it
on prose.

**The generated text.** Two 200-token greedy runs per format in one process, the standard prompt.
The first decode of a process is not the second: in every arm, bf16 included, the first run's token
list leaves the second run's at index 9, because the first step runs an un-graphed drafter call and
the graph capture before any graph exists, so its drafts -- and therefore which tokens share a
verify block -- differ, and a bf16 GEMM is not row-count-invariant across blocks. Compared at
matched positions, the second decode of the fp8 process is **byte-identical to the second decode of
the bf16 process**, 205 tokens, no divergence at all. fp4's text is not compared; it fails the loss
gate first.

| head | 2nd decode tok/s | accept | steps | ms per step | 1st decode tok/s |
|---|---|---|---|---|---|
| bf16 | 21.62 | 3.09 | 66 | 143.0 | 19.32 |
| **fp8** | **22.88** | 3.14 | 65 | **137.2** | 18.75 |
| fp4 | 23.35 | 3.12 | 64 | 133.8 | 20.62 |

5.8 ms per step, which is the head's two calls (5.68 + 5.68 -> 2.96 + 2.96 in isolation = 5.4 ms)
and not an acceptance coin flip: the accepted length is 3.09 against 3.14 and the text is the same
text.

**`engine/test_fastdecode.py`** (the graphed `FastDecoder` against `Model.forward` on identical
state), shipped config, `DSV41_HEAD_FMT=fp8`:

| parity | drafts equal | logits rel err | argmax agreement | main_hidden rel | step / draft |
|---|---|---|---|---|---|
| 0 | no | 0.0746 | **1.00** | 0.0863 | 113.2 / 11.6 ms |
| 1 | yes | 0.0158 | **1.00** | 0.0211 | 113.9 / 10.6 ms |

The three rel-err figures are identical to the bf16 row of the 16:05-17:05 section, as they must be:
both sides of that comparison use whichever head is loaded, so the format cancels. What moves is the
time -- 116.7 / 117.9 ms of step and 14.4 / 13.5 of draft in bf16 against 113.2 / 113.9 and
11.6 / 10.6 here.

**Memory.** The head's resident bytes go from 1.324 to 0.663 GB. At the pinned 90.5 GB arena that is
headroom, not slots. Prefill on the standard prompt is unchanged (1.278 -> 1.266 s) and the
53-sequence teacher-forced pass costs 3.6 s more (134.6 -> 138.2 s) for the blocked dequant.

**Shipped.** `.env` on the serving box carries `DSV41_HEAD_FMT=fp8`; the code default stays `bf16`.
It is worth 4.3 ms of a 118.9 ms verify step and 2.7 ms of a 13.4 ms draft, at +0.0004 / +0.0014
nats and the same generated text.

**`engine/profile_fast.py`, and why it had to be run inside one process.** Run as two separate
processes the harness says the opposite of everything above: 118.5 ms of step with the bf16 head
against 123.2-123.8 with fp8, reproducibly (two runs each; the bf16 runs also reproduce the
16:05-17:05 section's 118.9 to 0.3 %). The head's own line moves exactly as predicted in those runs
-- 5.79 ms of cutlass becomes +2.95 ms inside `_fp8_linear_kernel` -- and what actually moves is the
two CB3 expert kernels, which get 12 % slower for no reason the head can supply.

They do not, in one process. Loading the engine once with the bf16 head, timing the step, then
swapping `FastDecoder.head_bf16` to a `quantize_to_fp8` of the same weight, clearing the captured
graphs and timing again -- same arena, same pages, nothing else touched:

| | wall step | draft | `_cb3v3_up` | `_cb3v3_down` | `_fp4_linear` | `_fp8_linear` | head (cutlass) | self CUDA |
|---|---|---|---|---|---|---|---|---|
| bf16 head | 118.9 | 13.4 | 40.95 | 22.58 | 16.27 | 9.72 | 5.80 | 116.81 |
| fp8 head | **114.6** | **10.7** | 40.84 | 22.35 | 16.08 | 12.64 | -- | **113.26** |

The expert kernels do not move at all (0.3 % and 1.0 %, both inside this box's run-to-run spread),
the head's 5.80 ms of cutlass turns into 2.92 ms inside `_fp8_linear_kernel`, and the step and the
draft fall by 4.3 and 2.7 ms -- 7.0 ms per step + draft, which is what the isolated GEMM timings
predict. The bf16 column reproduces the 16:05-17:05 profile table row for row.

So the cross-process figure is an artifact of the process, not of the head: `head.weight` is 0.66 GB
smaller in fp8, so every later allocation -- including the 89 GB expert arena -- lands at a different
offset, and on this box a purely streaming kernel over 89 GB is sensitive to where its pages sit.
**Any A/B on this engine that changes a resident allocation's size has to be run inside one process,
or the arena's placement will be measured instead of the change.**

**Caveats.**
* The blocked prefill path allocates the fp32 logits for the whole call up front, as the bf16 path
  always did; at 2,048 rows that is 1.06 GB either way.
* fp4 is implemented and measured, not used. Its weight error (0.118) is the same 12 % the dense fp4
  groups carry, and unlike `wq_b` or `wo_a` -- whose outputs are sums over 1,280 to 8,192 terms --
  the head's output is a sum over 5,120 terms that is then taken an argmax over across 129,280
  competing rows, which is where the error shows.

#### 2. A 2-bit tier for the coldest resident experts: measured, not built

The routed experts are 63.5 ms of a 118.9 ms verify step (`_cb3v3_up` 41.0 + `_cb3v3_down` 22.5) and
they are at the bandwidth floor, so the only way down is fewer bytes per expert. CB3 is 3.0 + 0.25
bit/weight; a per-row codebook of FOUR FP4 grid codes instead of eight would be 2.0 + 0.25, i.e.
9.99 MB per expert against 14.45 (0.691x). The cold half of a layer's kept set is the natural place
to spend that, because those experts are rarely routed. The question asked here is what it costs in
loss and what it would buy, in that order.

**The quality question, answered without building anything.** A CB3 slot holds eight codebook
entries per row and three bits per weight. Fill it with a FOUR-entry codebook repeated to eight and
the packer's indices -- which come from `CodebookSim.pos` and never exceed 2^bits - 1 -- only ever
name the first four: the slot's bytes and the CB3 kernel are untouched and the arithmetic is exactly
a 2-bit format's. So the shipped configuration itself can be measured with a 2-bit tier inside it,
all resident, no streaming, the same 90.5 GB arena and the same keep fraction. That is
`--sim-cb2-frac` / `SIM_CB2_FRAC`: it gives the coldest fraction of every layer's kept set a
`CodebookSim(2)` at load time (`cb3._pad_codebook`, `CB3ArenaV2.load_slot(..., sim=)`, and a
per-expert `cb_bits` / `cb_sims` policy on the store). `tools/test_cb3_moe.py` checks that such a
slot dequantizes bit-identically to `codebook_sim(2)`, that no index above 3 is ever written, and
that the 8-entry codebook is the 4-entry one repeated.

**Held-out teacher-forced**, one run per arm, keep 0.40, CB3, `DSV41_DENSE_FP4=attn,wo_a`, bf16 head
-- i.e. the same configuration as the all-3-bit baseline, so the comparison is like for like:

| coldest fraction at 2 bits | experts | coding NLL | delta | general NLL | delta |
|---|---|---|---|---|---|
| none (all 3-bit, the shipped row) | 0 of 6,160 | 1.5346 | -- | 3.1498 | -- |
| 0.30 | 1,840 | 1.5471 | +0.0125 | 3.1966 | **+0.0468** |
| 0.50 | 3,080 | 1.5563 | +0.0217 | 3.1940 | **+0.0442** |

Coding behaves as one would expect -- +0.013 at 30 % of the set, +0.022 at 50 %, roughly with the
routing mass the cold part carries. Prose does not: it is already +0.047 at 30 % and does not get
worse at 50 % (the 0.003 between the two arms is inside this corpus's resolution). The whole prose
penalty is paid by the coldest 30 % of the kept set, and it is 1.6x the budget on its own. Halving
the cold fraction again would halve the byte saving, which at 30 % is only 3.5 % of the expert bytes
per step, so there is no fraction at which this trade is worth a new format.

The weight error is the reason: on a real layer-0 expert the per-row 8-of-16 codebook is 0.21
relative and the 4-of-16 one is 0.37 -- three times FP4's own 0.12. A cold expert is routed rarely,
but when it is routed it is routed because the router wanted it, and on prose the model leans on the
tail of the distribution far more than it does on code.

**The format and the kernel exist, and were measured before the quality answer came back.** CB2 is
CB3 with the high-bit plane left out: the same v2 bit positions in the lo plane, the same UE8M0
scales, a four-entry codebook per row (`cb3.fp4_to_cb2` / `dequant_cb2`, `cb3_moe.CB2ArenaV2`,
`_cb2_up_kernel` / `_cb2_down_kernel`, and the same unpack-to-FP4 fallback for prefill-sized calls).
The lo plane it writes is bit-identical to the lo plane the CB3 packer writes for the same indices,
and the hi plane CB3 would write is all zeros, which the unit test asserts. The PTX decoder loses
three instructions per four weights (no high-bit shift, no lop3 fusion of it) and the codebook
lookup needs only one `prmt` source register, because an index of 0..3 never names a byte above lane
3. Tile widths are unchanged -- 128 B per 512-weight block, 64 B per 256-weight block -- so the
width cliff that made CB3 reach bandwidth still applies.

Correctness and speed, 24 real layer-0 experts, T = 6 top-6 (`tools/test_cb3_moe.py`, one run):

| | bytes/expert | bit/weight | ms | GB/s of its own bytes |
|---|---|---|---|---|
| FP4 (on the same 4-level weights) | 18.80 MB | 4.25 | 2.16 | 182.5 |
| CB3 v3 | 14.45 MB (0.769x) | 3.25 | 1.80 | 168.7 |
| CB2 | 9.99 MB (0.531x of FP4, 0.691x of CB3) | 2.26 | 1.46 | 143.8 |

CB2 is bit-exact against the FP4 kernel run on the same re-quantized weights at T = 1 and T = 64
(rel 0.0) and at bf16 rounding at T = 6 (4.7e-5), and its prefill unpack is bit-exact at T = 128
(max |delta| 0.0). It reads 0.691x of CB3's bytes at 0.85x of CB3's per-byte rate, so it is 0.81x in
wall time -- real, but 15 % of the saving is given back to the decoder being more instruction-bound
per byte than CB3's.

**What a two-tier arena would have bought** (per-layer keep sets from the same trace histogram the
router uses; the eight transient slots ride in the cold tier; arena 90.5 GB):

| keep | cold fraction | CB3 slots | CB2 slots | GB | routing coverage | cold share of routed pairs | expert bytes per step |
|---|---|---|---|---|---|---|---|
| 0.40 | 0.00 (shipped) | 6,160 | 0 | 89.1 | 86.30 % | 0 % | 1.000x |
| 0.40 | 0.30 | 4,320 | 1,840 | 80.9 | 86.30 % | 11.3 % | 0.965x |
| 0.40 | 0.50 | 3,080 | 3,080 | 75.4 | 86.30 % | 21.9 % | 0.932x |
| 0.40 | 0.70 | 1,840 | 4,320 | 69.8 | 86.30 % | 36.5 % | 0.887x |
| 0.45 | 0.50 | 3,480 | 3,440 | 84.8 | 89.27 % | 20.9 % | 0.936x |
| 0.50 | 0.70 | 2,320 | 5,360 | 87.2 | 91.74 % | 34.3 % | 0.894x |
| 0.55 | 0.70 | 2,560 | 5,920 | 96.2 | 93.88 % | -- (does not fit) |

The last column assumes CB2 reaches CB3's GB/s, which it does not: at 0.85x of it the 0.932x row
becomes 0.951x, i.e. 3.1 ms of a 119 ms step. The capacity side is the better half of the trade --
keep 0.50 at cold 0.70 fits in the same 90.5 GB and is 5.4 points of routing coverage -- but it
needs the cold fraction the loss numbers rule out twice over.

**Conclusion: the 2-bit tier is not worth building.** The cheapest arm that would pay for itself
costs 1.6x the loss budget, and the format that would carry it -- which now exists, is unit-tested
and is bit-exact against the FP4 kernel -- reaches 0.85x of CB3's per-byte rate, so even the
arithmetic that ignores quality only buys 3 ms of a 119 ms step. `EXPERT_FORMAT` keeps its two
values (`fp4`, `cb3`); nothing in the serving path changed. `CB2ArenaV2` and its kernels stay in
`tools/cb3_moe.py` as measured, unused code, next to `moe_forward_v2`, and `--sim-cb2-frac` stays as
the cheap way to re-ask the quality question if the keep fraction or the corpus ever changes.

### 2026-09-11 18:20-19:30 -- thinking-on decode in the shipped configuration, the router's top-k, and the verify block size

Three pieces of work. The first is a measurement of the running server, the other two are switches
with the old path as the default.

#### 1. Thinking on and thinking off in the shipped configuration

The only thinking-on decode number on record was taken against the 09:50 engine (keep 31 %, FP4
experts, bf16 head), which no longer exists. One request each through the gateway against the
shipped configuration (keep 0.40, CB3 experts, `DSV41_DENSE_FP4=attn,wo_a`, fp8 head), temperature
0, 400 output tokens, the same prompt both times -- a small algorithmic question ("longest strictly
increasing subsequence: explain the O(n log n) algorithm and why the patience-sorting tails array is
correct"), which the thinking arm spends all 400 tokens reasoning about and never finishes:

| | thinking off | thinking on (`reasoning_effort=high`) |
|---|---|---|
| prompt tokens | 43 | 69 |
| TTFT | 1.67 s | 2.19 s |
| prefill | 1.669 s, 25.8 tok/s | 2.059 s, 33.5 tok/s |
| decode | **25.86 tok/s** | **21.44 tok/s** |
| steps for ~400 tokens | 114 | 138 |
| DSpark acceptance | 3.51 | 2.90 |
| wall per step | 135.7 ms | 135.2 ms |
| expert hit rate | 1.0 | 1.0 |

The per-step time is the same to 0.4 %, which is what one would expect: the step does not know what
kind of text it is producing. The whole 4.4 tok/s is the acceptance -- the drafter predicts this
model's reasoning text less well than its answer text on this prompt (3.51 against 2.90), and a
step emits `acceptance + 1` tokens whatever it costs. The 09:50 number (21.4 tok/s at acceptance
4.11) is the same tok/s reached a different way: a much slower step carrying a much higher
acceptance, on a different prompt. Acceptance is a property of the prompt as much as of the engine,
so a single prompt's tok/s is not a configuration's tok/s.

#### 2. How many experts a decode step activates, and what fewer would cost

The routed experts are 65.0 ms of a 114.9 ms verify step (`_cb3v3_up` 41.8 + `_cb3v3_down` 23.2)
and are at the bandwidth floor, so after the 2-bit tier was ruled out on quality the only remaining
lever is touching fewer of them. The checkpoint routes 6 of 384 per token plus the layer's shared
expert; a verify block holds 6 tokens, so a layer sees 36 routed (token, expert) pairs per step --
but an expert is read once however many of the block's tokens ask for it, so what sets the bytes is
the number of DISTINCT experts, which nothing had measured.

**The count.** `DSV41_ROUTE_STATS=1` (off by default) one-hots each layer's `k x 6` expert ids into
a [384] table and adds the number of rows hit to a per-layer accumulator. Both ops have static
shapes and no host round-trip, so they capture into the layer graphs and the measurement runs at
full speed on the real decode path; `engine/diag_topk.py` drives it (a first generation captures the
graphs, then the accumulator is reset and a 60-token generation is counted), with `TF=` to score the
held-out corpus in the same process so the count and the loss of one setting come from one weight
load. Measured over ~20 verify blocks of the standard code prompt, keep 0.40 pruned:

| router top-k | routed pairs per layer | **distinct experts per layer** | of 36 pairs | expert bytes per step (14.45 MB each) |
|---|---|---|---|---|
| 6 (the checkpoint) | 36 | **20.96** | 0.58 | 12.12 GB |
| 5 | 30 | **17.58** (0.839x) | 0.59 | 10.17 GB |
| 4 | 24 | **15.30** (0.730x) | 0.64 | 8.85 GB |

So the working assumption of ~30 distinct experts per layer was wrong by 30 %: the block's six
tokens are consecutive text and their routing overlaps heavily -- 36 pairs collapse to 21 reads --
and pruning to the top 154 experts of each layer concentrates it further. 12.12 GB over the 65.0 ms
the two expert kernels take is 186 GB/s, which is the rate the CB3 kernel reaches in isolation
(168.7 GB/s of its own bytes on a cycled buffer, 2026-09-11 17:00-19:00), so the floor is real and
the only way down is fewer reads. Layer 0 is the outlier at 28.4 distinct experts; layers 24-28 are
the most concentrated at 16.5-19.1.

**The switch.** `DSV41_TOPK` = 1..6, unset (the default) = the checkpoint's `n_activated_experts`.
It is applied in `v41_ref.Args.from_json`, so every consumer of the args follows: the graphed decode
path, `Model.moe`, the reference `moe`, and the static `[6, k]` routing buffers of `FastDecoder`.
The gate weights are renormalized over the surviving experts by the line that already normalizes the
checkpoint's six (`w / w.sum() * route_scale`), so the routed contribution keeps its scale and only
its composition changes. The DSpark drafter's own router -- top-3 of the 128 experts of each MTP
block -- is a separate hard-coded k in both paths and is untouched by the switch; it was not
measured here.

**Held-out teacher-forced** (`corpus/heldout_corpus.jsonl`, 5,430 coding + 5,231 general scored
positions), one run per setting, the shipped configuration, the k=6 row re-measured in this round
rather than quoted:

| `DSV41_TOPK` | coding NLL | delta | general NLL | delta | top-1 coding / general |
|---|---|---|---|---|---|
| 6 (shipped) | 1.5351 | -- | 3.1512 | -- | 0.6856 / 0.4582 |
| 5 | 1.5396 | +0.0045 | 3.1762 | **+0.0250** | 0.6864 / 0.4542 |
| 4 | 1.5890 | **+0.0539** | 3.2330 | **+0.0818** | 0.6810 / 0.4521 |

The k=6 row reproduces the 17:00-19:00 section's fp8-head row exactly (1.5351 / 3.1512), so the
comparison is like for like. k=5 passes the +0.015 budget on code and misses it on prose by 1.7x;
k=4 misses it on both by 3.6x and 5.5x. The two corpora disagree in the same direction and for the
same reason the 2-bit tier disagreed with itself: prose leans on the tail of the router's
distribution and code does not, so the sixth expert is worth little on code and a lot on prose.
The trend is monotone and steep between k=5 and k=4 on both corpora, which is what a real effect
looks like rather than a corpus-sized wobble.

**What it would have bought.** Measured anyway, one process per setting (the arena is 6,260 slots
either way and no resident allocation changes size, so the cross-process caveat of 17:00-19:00 does
not apply here). `engine/profile_fast.py`'s harness for the step, then two 200-token greedy decodes
on the standard prompt of which the second is quoted:

| | top-k 6 | top-k 5 |
|---|---|---|
| verify step | 114.9 ms | **106.6 ms** |
| draft | 9.9 ms | 10.2 ms |
| `_cb3v3_up` + `_cb3v3_down` per step | 65.02 ms | **57.44 ms** |
| expert bytes per step | 12.12 GB | 10.17 GB |
| effective expert bandwidth | 186.3 GB/s | 177.0 GB/s |
| 200 greedy tokens | 23.45 tok/s, accept 3.14, 133.8 ms/step | 24.87 tok/s, accept 3.06, 123.1 ms/step |

8.3 ms of a 114.9 ms step, 7.2 %. Note that it is less than proportional: 0.839x the bytes buys
0.883x the kernel time, because 17.6 experts per layer gives the kernel fewer concurrent programs
and it gives back 5 % of its per-byte rate. `engine/test_fastdecode.py` at `DSV41_TOPK=5` (the
graphed decoder against `Model.forward` on identical state, both at k=5) reports argmax agreement
**1.00 / 1.00** on both parities, logits rel err 0.0341 / 0.0627, step 105.8 / 103.9 ms and draft
11.5 / 11.0 -- so the switch is the same computation on both paths and the graphs capture it
correctly.

**Decision: not shipped.** The gate for a change to the model's own computation is +0.015 nats on
both corpora and k=5 costs +0.025 on prose for 7 % of a step. `DSV41_TOPK` stays in the tree,
default unset; `.env` on the serving box does not set it. The one thing it does change permanently
is the arithmetic everyone was using: the expert traffic of a step is 12.1 GB, not the 17.3 GB that
30 distinct experts would imply, and a future format has to beat 186 GB/s on 12.1 GB.

#### 3. The verify block size

The block is 6 positions: one accepted token plus 5 DSpark drafts (`dspark_block_size` = 5). Expert
bytes grow with the block because more tokens route to more distinct experts, while acceptance grows
sublinearly, so 6 is a choice and not a constant of the problem.

**The switch.** `DSV41_BLOCK` = the number of drafted positions, odd, 1..15, default 5. It has to be
odd because the verify block (`block + 1`) has to be even: the ratio-2 key compressor groups the
block's positions in pairs around a single pending slot, which is exactly what `capture(S_parity)`'s
two parities encode, and an odd verify width would flip the parity every step. Making it variable
was five constants, not a rewrite -- `T_VERIFY`/`T_DRAFT` in `engine/fastdecode.py` (every static
buffer already derives from them), the `[:6]`/`[5]` of the ratio-2 grouping and of the unpaired
pending token in `FastDecoder._compressed` and `FastDecoder.step`, the `[6]`/`[7]` staging buffers
and `am[:5]` of the lean verification in `engine/v41_engine.py`, and the hard-coded `B = 5` in
`Model.dspark_draft`, which now reads `T_DRAFT`. At the default every one of those is the value it
was.

One run per setting, same harness and same prompt as above, the second decode quoted:

| verify block | drafts | verify step | draft | step + draft | acceptance | **ms per accepted token** | 200 greedy tokens | expert kernels per step |
|---|---|---|---|---|---|---|---|---|
| 4 | 3 | 102.0 ms | 8.7 ms | 110.7 ms | 2.70 | **41.0 ms** | 23.72 tok/s, 114.0 ms/step | 54.3 ms |
| **6 (shipped)** | 5 | 114.9 ms | 9.9 ms | 124.8 ms | 3.14 | **39.7 ms** | 23.45 tok/s, 133.8 ms/step | 65.0 ms |
| 8 | 7 | 126.0 ms | 11.0 ms | 137.0 ms | 3.45 | **39.7 ms** | 23.00 tok/s, 149.9 ms/step | 74.8 ms |

Both halves behave as predicted: the step grows nearly linearly in the block (110.7 / 124.8 / 137.0,
i.e. 12-14 ms per two positions, and the expert kernels 54.3 / 65.0 / 74.8) and acceptance grows
sublinearly (2.70 / 3.14 / 3.45 for 3 / 5 / 7 drafted positions). Their ratio -- the only number
that matters -- is flat between 6 and 8 and worse at 4. Block 8 also drafts two positions beyond the
horizon the DSpark head was trained for, and pays a longer tail latency per emitted burst for no
throughput; block 4 gives up more acceptance than it saves in bytes. The acceptance figures are one
prompt each and carry this box's usual +-5 %, so the 6-versus-8 tie is a tie and not a measurement
that 8 is equal.

**Decision: the block stays at 6.** `DSV41_BLOCK` stays in the tree at its default 5, `.env` does
not set it.

#### What these three pieces did not measure

The thinking-on pair is one prompt; acceptance on reasoning text was not sampled across prompts, and
thinking-on was not measured at long context. `DSV41_TOPK` was scored on the held-out corpus and on
the standard code prompt only -- no sampled-quality A/B, and no per-sequence breakdown of the prose
delta of k=5 (the monotone trend through k=4 is the evidence that it is real, not a per-sequence
sign check of the kind 3.4 ran for the fp4 head). `engine/test_fastdecode.py` was run at k=5 and not
at k=4. The distinct-expert counts were taken on one code prompt at keep 0.40; a prose prompt, a
different keep fraction or a long context would each give a different number, and the count at the
block sizes of section 3 was not taken separately -- the expert-kernel times there stand in for it.
The DSpark drafter's own router (top-3 of 128 per MTP block) was left alone and not measured at any
other k. `DSV41_BLOCK` was measured for throughput only: no held-out loss was scored at block 4 or
8, and the reference (un-graphed) drafter was not compared against the graphed one at those widths,
so only the default 5 has the `engine/test_fastdecode.py` parity evidence behind it.

### 2026-09-11 19:55 -- tool calls were being dropped when the completion's DSML deviated

Symptom, seen in two different clients at once: the model writes the sentence that precedes a tool
call ("Let me check the workspace context.") and the turn ends there with `finish_reason: stop` and
no call. The API itself was fine -- a single-tool request, streamed or not, returns a correct
`tool_calls` delta with `index`, and `engine/test_server.py`'s tool test passed throughout.

What the server log showed on the failing turns:

```
tool-call parse failed (Parameter format error: ' name="query" string="nvidia DGX Spark GB10 128GB unified memory specs"<'); returning raw text
tool-call parse failed (Unexpected content after tool calls); returning raw text as content
```

The completion is parsed by the checkpoint's own routine
(`corpus/sources/dsv41_encoding.py::parse_message_from_completion_text`), which is strict: a
parameter must read `name="x" string="true|false">value<`, and nothing may follow the closing tag of
the calls block. Real completions deviate in two recoverable ways -- the value placed in the
`string` attribute with no value body, and ordinary prose after the block -- and on either the
server discarded every call in the completion and returned the raw text as content. Since the tool
marker is also a stop string, what reached the client was just the preamble sentence.

`server/app.py` now runs a tolerant pass when the strict parser raises, and uses it only if it
recovers at least one call: it reads each `invoke` by name and accepts both parameter spellings,
ignores anything after the block, and returns `[]` on text that holds no call at all (so the
existing raw-text fallback still applies). `DSV41_LOG_TOOL_TEXT=1` logs the first 2,000 characters
of an unparsed completion so a future deviation is visible rather than silent. Regression test:
`server/test_server.py::test_tolerant_tool_call_recovery` (spec form, value-in-attribute, trailing
prose, two calls in one completion, and text with no calls).

Verified after the restart against the same shape that failed: a search-style tool now comes back
with `finish_reason: tool_calls` and two well-formed calls. Not diagnosed: why the model writes the
attribute form at all. It is a formatting deviation of the model, more likely under long
tool-definition prompts, and this configuration is pruned and re-quantized, so the deviation rate
may differ from the full checkpoint's.

### 2026-09-11 20:00-20:45 -- greedy decoding fell into a repetition loop; the fused attention kernel is off by default

Reported from two clients at once: asked for a single-file HTML game, the engine wrote

```
<!DOCTYPE HTML>
<html<!DOCTYPE HTML>
<html<!DOCTYPE HTML>
```

for as many tokens as it was allowed, at a DSpark acceptance of 4.76 against the 3.0-3.2 every other
run showed. The high acceptance is the tell: the drafter was predicting perfectly because the target
was producing the same few tokens every step.

Isolation, same prompt, greedy, 200 tokens, one run each (`results/htmlbug/`):

| fused attention kernel | dense fp4 | LM head | output | acceptance | tok/s |
|---|---|---|---|---|---|
| on (graphed fast path) | attn,wo_a | fp8 | **the loop** | 4.76 | 35.5 |
| not used (`DSV41_FAST=0`, un-graphed block) | attn,wo_a | fp8 | clean HTML | 4.10 | 12.0 |
| **off** (`DSV41_FUSED_ATTN=0`) | attn,wo_a | fp8 | **clean HTML** | 3.10 | 18.0 |
| on | off | bf16 | clean HTML | 3.85 | 23.9 |
| speculation off entirely | attn,wo_a | fp8 | clean HTML | -- | 3.8 |

So no single piece is broken: the kernel is clean with bf16 dense projections and a bf16 head, and
the quantized projections are clean with the torch attention. Together they cross the precision the
verify step needs, one near-tie flips (`<html` followed by `>` or by another `<!DOCTYPE`), and
greedy decoding has no way out of the loop it lands in. Speculative decoding is supposed to be
lossless -- the target verifies every drafted token -- so any divergence between decoding with and
without it is a defect by definition, whatever its size.

`DSV41_FUSED_ATTN` now defaults to 0. It was worth ~2 ms of a 119 ms step, so the shipped
configuration keeps the dense fp4 projections and the fp8 head, which are worth ~15 ms together.

What let this through: every gate used this month was teacher-forced loss, which scores text the
model is shown and never runs the decode loop, so it cannot see a verification fault. The loss of
the broken configuration is the best in the repo. `engine/test_spec_lossless.py` is the missing
gate -- it generates the same prompts greedily with and without speculation and requires the token
sequences to be identical -- and it belongs in front of any future change to the decode path.

Not fixed, only disabled: why the kernel's error is large enough to matter. Its unit test agrees
with the torch path to ~1e-4 on the attention output, and the fast path's logits have differed from
`Model.forward` by several percent since it was written (0.15 before this kernel, 0.07 after), which
was filed as bf16 noise and is too large for that. Both belong in a proper accuracy pass of the fast
path rather than a speed round.

### 2026-09-11 20:55-21:25 -- the real cause: the graphed path routed tokens to different experts

Disabling the fused attention kernel stopped one repetition loop but not the class of failure: with
it off, a chat carrying 20 tool schemas still produced 4,096 tokens of near-identical tool calls
before hitting the output cap. So the fast path was compared against `Model.forward` layer by layer
on the same block (`engine/test_fastdecode.py` with `DIAG=1`, shipped configuration):

| layer | attn input rel err | routed experts equal |
|---|---|---|
| 0 | 0.0000 | **0.89** |
| 6 | 0.1070 | 0.83 |
| 12 | 0.1361 | 0.69 |
| 18 | 0.1570 | 0.72 |
| 36 | 0.0803 | 0.69 |

Layer 0 is the finding: its inputs are bit-identical and the router still disagrees on 11 % of the
picks; by the middle layers a third of the experts differ, and the activation error grows with depth
because each layer runs a different FFN than the reference did.

The cause is one line. `Model.moe` computes the gate as `mm(y.float(), gate_w)` -- fp32 activations
against the fp32 gate weights, as the checkpoint stores them. The graphed path computed
`F.linear(y, gate_w.bfloat16())`, a bf16 GEMM, added on 2026-09-11 morning together with two changes
worth far more (torch routing instead of the per-slot Triton router, and the device slot LUT). The
gate picks 6 of 384 and its scores are dense with near-ties, so three decimal digits are not enough
to reproduce the selection. Both paths now use the fp32 GEMM; the DSpark drafter's gate too.

After the fix, same comparison:

| layer | attn input rel err | routed experts equal |
|---|---|---|
| 0 | 0.0000 | **1.00** |
| 6 | 0.0073 | 1.00 |
| 12 | 0.0087 | 1.00 |
| 18 | 0.0110 | 0.94 |
| 36 | 0.0131 | 0.94 |

Logit error against the reference 0.049 -> 0.0121, argmax agreement 1.00 on both parities, and the
HTML prompt produces a document that closes its tags and stops.

With the fused attention kernel switched back on the deep layers degrade again (error 0.048-0.093,
routed experts equal 0.72-0.92, argmax agreement 0.83 on one parity), so that kernel is a second,
smaller source of the same problem and stays off until it is replaced by a vetted implementation
(flashinfer 0.6.17 is installed on this box and has an attention-sink wrapper).

What this says about the gates used all month: teacher-forced loss never ran the decode loop, and
`engine/test_fastdecode.py` printed the per-layer table all along without anyone reading the
`route eq` column. A number that is printed is not a gate.

### 2026-09-11 21:30-23:30 -- tool calls constrained by a grammar built from the request's own schemas

The tolerant parser added at 19:55 repairs a malformed tool call after the fact. It does not stop
the model from writing one, and the repair was not the whole cost: a completion that drifts out of
the format does not stop drifting. The case that prompted this was a chat carrying 20 tool schemas
(three MCP tool servers attached at once) which produced `prompt=5899 completion=4096
finish=tool_calls` -- four thousand tokens of near-identical calls until the output cap, 161
seconds. The 20:55-21:25 entry above found one cause of that behaviour in the numerics (the graphed
path routed tokens to different experts); the format itself is the other half, and nothing in the
decode loop prevented the model from writing a block the checkpoint's own parser rejects.

DSML is defined by the checkpoint twice over: in the system prompt it writes for a tool request
(`TOOLS_TEMPLATE`) and in the parser it ships (`parse_tool_calls`). The block is

```
\n\n<｜DSML｜ calls>\n
  <｜DSML｜ invoke name="TOOL">\n
    <｜DSML｜ parameter name="P" string="true|false">VALUE</｜DSML｜ parameter>\n
  </｜DSML｜ invoke>\n
</｜DSML｜ calls>
```

and the parser is exact about every byte of it: the attribute reads `string="true"` or
`string="false"` and nothing else, a parameter name may not repeat, and no text at all may follow
the closing tag.

`server/tool_grammar.py` builds an EBNF grammar for exactly the tools of one request -- the allowed
tool names, each tool's parameter names in schema order with the required ones mandatory and the
rest optional (which makes "every required parameter present" and "no duplicate parameter"
structural rather than checked afterwards), `string="true"` for string-typed parameters,
`string="false"` and a JSON value for the others, at most eight invokes, and the block closed --
and xgrammar compiles it against the tokenizer. A gate object masks the sampling distribution while
the model is inside the block and nowhere else: it watches the decoded output for the calls marker,
creates a `GrammarMatcher` seeded with `accept_string` over the text from the marker onwards, masks
every logit row from then on, and finishes when the block closes. A complete matcher allows exactly
one token, the end of turn. That is the part that turns "keeps writing calls" into "stops".

For a request whose first tool is a search taking a string, an integer and an array of strings, the
generated rules for that tool read

```
root ::= "\n\n<｜DSML｜ calls>\n" call{1,8} "</｜DSML｜ calls>"
call ::= t0 | t1 | ... | t19
t0 ::= "<｜DSML｜ invoke name=\"web_search\">\n" t0p0 t0p1? t0p2? t0p3? "</｜DSML｜ invoke>\n"
t0p0 ::= "<｜DSML｜ parameter name=\"query\" string=\"true\">" vany "</｜DSML｜ parameter>\n"
t0p1 ::= "<｜DSML｜ parameter name=\"max_results\" string=\"false\">" jint "</｜DSML｜ parameter>\n"
t0p2 ::= "<｜DSML｜ parameter name=\"sites\" string=\"false\">" ("[" jws (jstr jws ("," jws jstr jws)*)? "]") "</｜DSML｜ parameter>\n"
t0p3 ::= "<｜DSML｜ parameter name=\"safe_search\" string=\"false\">" jbool "</｜DSML｜ parameter>\n"
```

Twenty tools of that shape come to 88 rules and 7.7 kB of EBNF. **Measured** on this box: building
the `TokenizerInfo` and the compiler costs 1.52 s once at server start, compiling that grammar
costs 28 ms and 13.1 MB of adaptive token-mask cache, and a repeat compile of the same tool list is
a dictionary lookup (2 us), which is what a chat client's second turn does.

#### Two things about xgrammar that are worth writing down

**Negated character classes are ASCII-only.** The value rule has to be "any character except
U+FF5C": every DSML token contains that character, so a value written that way can hold HTML, code,
JSON and prose and still not be able to end itself, which makes the closing tag unambiguous.
Written the obvious way, as a negated class over that one character, it compiles to a class that
matches U+FF5C as well. The library says so on stderr -- `Warning: Negative Character class
contains byte greater than 127, clamping to 127` -- and the consequence is severe: the value can
run straight over its own closing tag and swallow the rest of the block, so the grammar constrains
nothing after the first parameter opens. **Measured** with a two-rule grammar (`root ::= "<a>" v
"</a>"` over that negated class): after feeding `<a>x</a>` the matcher reports completed and still
allows 128,524 of 129,280 tokens, and `accept_string("｜")` mid-value returns true. The form that
works is a union of positive ranges, a class of `\u0000-\uFF5B` and `\uFF5D-\U0010FFFF`, which
excludes U+FF5C and keeps accented letters, CJK, halfwidth/fullwidth punctuation and the
supplementary planes; the same treatment is needed for the parameter-name and JSON-string
character classes, which also have to exclude that character. The guard against this quietly
coming back is `test_value_cannot_contain_the_dsml_bar` in `server/test_tool_grammar.py`.

**`traverse_draft_tree` fits the DSpark verify block exactly, once its convention is known.** The
convention is not written down beyond the signature, so it was established by experiment: node 0 is
the token the matcher has already accepted (its entry in `draft_tokens` is not re-accepted), row i
of the bitmask is the mask for the token that follows node i along its path, the matcher's own
state is restored before the call returns, a node whose token the grammar refuses gets an all-zero
row while its descendants are left untouched (so the bitmask must be reset to all-ones first), and
the three tensors must be int64 -- int32 raises. That is one call per decode step instead of up to
twelve fill/accept/rollback calls, and it maps onto the block `[tok, draft0..draft4]` with a chain
topology and no translation layer. The per-row walk is kept as a second implementation and
`test_gate_mask_paths_agree` requires the two to produce identical masks.

#### Where it sits in the decode loop

The gate is handed to `V41Engine.generate(grammar=...)` and the loop owes it two calls: `observe`
for every token it settles on, and `mask_rows(logits, block)` before anything reads the logits.
Both the greedy and the temperature path are masked. The temperature path draws no additional
random numbers -- masking changes the distribution it samples from, not the number of draws -- so
its RNG stream is unchanged, and the rejection-sampling residual `(p - q).clamp_min(0)` is zero
wherever the mask is, so an illegal token cannot come back through the residual either.

Speculation needs no special handling beyond the block. Row i of the six is the distribution after
`block[0..i]`, so its legal set depends on the drafts ahead of it; the gate walks the chain, and a
drafted token the grammar refuses is masked out of the row it is verified against, which makes the
verifier reject it there. Rows past that point are never read, because the accept count cannot
exceed the index of the first rejection. Masking leaves the gate's own state where it found it, so
a rollback of the cache needs no matching rollback of the grammar -- `observe` on the emitted
tokens is the only thing that advances it.

With `grammar=None` -- every request without tools, and every request at all when
`DSV41_TOOL_GRAMMAR=0` or `"tool_grammar": false` -- the loop is the code that was there before,
one `is not None` test per step, and the tolerant parser stays as the fallback for exactly those
cases and for a schema the builder cannot express.

#### Tests

`server/test_tool_grammar.py` covers the builder without any dependency (types, required versus
optional, escaping, names DSML cannot carry, the call bound) and the matcher against the
checkpoint's own parser: a well-formed block is accepted and parses back to the expected calls;
values holding HTML, quotes, braces and newlines survive a round trip; an unknown tool name, an
unknown parameter, a duplicate parameter, a missing required parameter, `string="true"` on an
integer, a quoted integer and a bare word inside an array are all refused; and the two shapes seen
in real completions are unreachable -- after `string="` only `true` and `false` continue, and after
the closing tag the mask contains exactly one token, the end of turn. `server/test_server.py` (the
mock engine, 16 tests) still passes with the three-element `build_chat_prompt` return.

#### Not done

The end-to-end run against the served model is outstanding: a 20-schema request with and without
the constraint, the completion-token counts and finish reasons, the no-tool control at temperature
0, the per-step cost of an active grammar, and `engine/test_spec_lossless.py` re-run to confirm
speculation is still lossless with no grammar in play. The machine stopped answering ssh a couple
of minutes into the warm start that was to carry those runs, with port 22 accepting the connection
and never sending a banner and port 8000 never opening, and it had not come back an hour and a half
later. That start was issued a few minutes after the previous server had gone away, without running
`./stop.sh` in between -- which is the sequence the comment at the top of `stop.sh` warns about: the
resident arena is tens of GB of page-cache-backed memory that the kernel reclaims lazily, and
starting the next warm start before that lands is how this box gets wedged.

Both test files ran on this box and passed before it stopped answering; the last edits to them --
the masked decode-loop test, the tightened mask-agreement assertions and the per-request off switch
-- came after that, and have only been exercised where they do not need a compiled grammar. Run
them first. The whole reproduction, once the box is back:

```
./stop.sh && ./start.sh                      # DSV41_STEP_TIMING=1 for the per-phase table
python3 server/test_tool_grammar.py          # CPU, needs only the tokenizer
python3 server/test_server.py                # CPU, mock engine
# the same request twice, one with "tool_grammar": false, and compare
# completion_tokens / finish_reason / x_engine_stats.tool_grammar
./stop.sh && python3 engine/test_spec_lossless.py --max-tokens 120
```

#### 2026-09-11 23:10 -- correction to the paragraph above, and what the kernel log says

The reason given above for the box going unresponsive is wrong. It was not the lazy reclaim of the
previous server's arena. The server's own first log lines say what it was:

```
21:54:31 [engine] arena 90.5 GB capped to -4.8 GB (MemAvailable 5.2 GB, keep_free 10 GB)
21:54:31 [engine] CUDA free 1.1 GB of 130.6; host MemAvailable 5.2 GB; arena 10.0 GB = 691 cb3
                  expert slots of 14.45 MB (4% of all routed experts, pinned)
```

MemAvailable was 5.2 GB and CUDA free was 1.1 GB of 130.6 GB *before the arena was filled*, because
a second engine instance with the same 90.5 GB arena (`engine/v41_engine.py --arena-gb 90.5`) was
already running on the box. Two arenas of that size do not fit in 121 GiB, and the engine's own
guard did the right thing -- it capped the arena to 10 GB -- but the box still went into swap-free
memory pressure and stopped scheduling new work, including sshd's forks, while keeping the listening
socket open. That is why port 22 accepted connections and never sent a banner.

A second attempt after the box was restarted ran into the same collision, and the kernel log of that
boot has the driver's side of it:

```
22:53:33 kernel: NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY] returned from
                 _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359     (x8, 22:53-22:55)
22:55:19 systemd-journald: Under memory pressure, flushing caches.
```

So: on this box, `./start.sh` with the shipped `ARENA_GB=90.5` is only safe when nothing else holds
the unified memory, and `free`'s MemAvailable is the number to look at before starting -- both
failures were visible in it beforehand. The engine's cap keeps the process alive; it does not keep
the machine responsive. `stop.sh`'s existing `MIN_FREE_GIB` wait is the right idea and `start.sh`
inherits it, but neither looks at what a *non-server* process on the box is holding.

### 2026-09-12 00:10 -- what actually breaks long generations: expert pruning, and two quantizations on top of it

A story prompt through the gateway degenerated into one word repeated for the whole output. The
same elimination as the HTML case, one variable per engine load (`results/htmlbug/`, each greedy):

| configuration | story | html |
|---|---|---|
| unpruned, experts streamed on a miss (hit rate 0.889, 3.33 tok/s) | **clean, and good** | -- |
| keep 31 %, dense fp4 + fp8 head | repeats | repeats |
| keep 31 %, fp8 head only | repeats | repeats |
| keep 31 %, dense fp4 only | repeats | clean |
| keep 31 %, neither | clean at 300 tokens | clean |
| keep 31 %, neither, keep-set ranked by max(coding, prose) | worse | repeats |

Three separate findings.

**Expert pruning is the root cause.** The unpruned model writes "The old man, Elara's grandfather,
had a saying: 'A story is a map, and a map is a promise.' He'd been a cartographer for the Royal
Survey..." -- there is nothing wrong with the checkpoint or the engine's arithmetic. Dropping 69 %
of the routed experts is what makes a generation fall into a repeated phrase, and RESULTS 2.4 said
so in a number nobody weighted properly: keep 31 % costs +0.19 nats on prose against +0.07 on code.
Prose leans on the tail of the router's distribution; code does not.

**The two dense quantizations make it much worse.** Each alone turns a clean 300-token generation
into a repeating one, which is why the loop appeared this evening and not this morning: the fp8 head
(0.027 relative weight error, straight onto the logits) and the fp4 attention/`wo_a` projections
(0.12). Both were accepted on a held-out teacher-forced delta of +0.0004 and +0.002 nats. Both are
now off by default.

**A "better" keep-set ranking is not a way out.** Ranking by max(coding, prose) instead of their sum
-- keeping each workload's specialists rather than what both use moderately -- was measured worse on
both prose and HTML at the same budget (`DSV41_PRUNE_RANK`, default `sum`).

> **Scoped 2026-09-12.** That sentence is true of `max` and only of `max`, which is the rule this
> entry tested. A third rule, `maxmin`, was added later and does change the outcome for a selection
> that spans several registers at once -- see the 2026-09-12 20:00 entry at the end of this file.
> It still does not create capacity.

At 288.8 GB of FP4 experts and 121 GiB of memory there is no arrangement that keeps every expert
resident. Either the experts stream on a miss (full quality, NVMe-bound) or some are dropped (fast,
and a workload the keep-set does not cover degenerates). The recipe now ships the first.

### 2026-09-12 01:05-02:45 -- the keep-set was learned from a corpus with no markup in it

The configurations that degenerated all shared one thing: the expert keep-set came from
`corpus/trace_corpus.jsonl`, 50 documents, 14 coding and 36 general, whose only content marker is
Python. No HTML, no JavaScript, no CSS, no SQL, no configuration files. The experts that write
markup never fired while the trace was taken, so they ranked cold, and every pruned configuration
dropped them. That is why a story and a Python class survived pruning while an HTML file collapsed
into `<!DOCTYPE><!DOCTYPE><!DOCTYPE>` at keep 31 %, 40 % and 44 % alike.

`corpus/trace_corpus_v2.jsonl` is 95 sequences / 18,546 tokens over the same two categories, built
from sources that cover what the server is actually asked for: two complete HTML pages, a
stylesheet, an ES module, a React component, a SQL schema, a Kubernetes manifest, a deployment
script, Python, an incident write-up and a short story. Re-traced over all 40 layers
(`results/trace-v2`, 68 s/layer, 364 of 384 experts touched at layer 39) and re-ranked. The keep-set
it produces overlaps the old one by a Jaccard of only 0.70 -- 30 % of the kept experts changed.

Generation gate (story / Python class / single-file HTML game, 300 greedy tokens each, distinct-token
ratio; <= 0.15 is degenerate):

| keep-set | story | code | html |
|---|---|---|---|
| old corpus, keep 44 % CB3 | 0.27 | 0.51 | **0.03** |
| old corpus, keep 44 % CB3 + a 12-gram repeat ban | 0.32 | 0.51 | **0.07** |
| **new corpus, keep 44 % CB3** | **0.43** | **0.50** | **0.40** |

Nothing else changed between the last two rows. The repeat ban is not needed and never fires.

Two things follow. The 3-bit expert format was never the problem -- unpruned CB3 passes the gate
(0.57 / 0.57 / 0.41), and its earlier failure was pruning plus two dense quantizations. And with a
keep-set that covers the workload, the model has enough margin to carry the fp4 attention/`wo_a`
projections and the fp8 head again: they pass the same gate (0.30 / 0.56 / 0.42).

Shipped configuration and what it measures, one request each through the server, greedy, 400 tokens:

| workload | tok/s | DSpark acceptance |
|---|---|---|
| Python class | 25.0 | 3.92 |
| story | 20.3 | 2.90 |
| single-file HTML game | 37.5 | 5.39 |

`PRUNE_KEEP=0.44 EXPERT_FORMAT=cb3 ARENA_GB=98 TRANSIENT_SLOTS=8 KEEP_FREE_GB=6`,
`TRACE_STATS=results/trace-v2/stats/coverage.json`, fp32 router, fp4 dense on attention and `wo_a`,
fp8 head, fused attention kernel off. 6,779 experts resident = 44.1 % of all routed experts, expert
hit rate 1.0, no NVMe traffic during decode. The step is ~145 ms in all three cases; the spread is
entirely how well the drafter predicts each kind of text.

### 2026-09-12 03:50-05:30 -- the keep-set has to cover every workload, and the gate has to be long enough

Two corrections to the entry above.

**The 300-token gate was too short.** Configurations that passed it failed at 900 tokens: a story
opened well and then collapsed into a semantic loop ("He saw the sea, the sea with the past. He saw
the past, the past with the keeper."). The gate now runs 900-2000 tokens per prompt and adds a
structural check, because repetition ratios do not see token corruption.

**A frequency penalty fixes prose and destroys code.** At `frequency_penalty=0.3` the 900-token
prose cases passed, and the HTML came back with `inset - 00 1 pix - 00 1 pix` in the CSS: the
penalty accumulates with a token's count, CSS repeats `px` and digits dozens of times, and the model
gets pushed off its own symbols. A presence penalty (flat per distinct token) does not corrupt code
but also does not fix prose. Both default to 0; `presence_penalty` / `frequency_penalty` are
accepted per request for clients that want them, and `DSV41_NO_REPEAT_NGRAM` is available and off.
Prose repetition is a phrase-level phenomenon and a token-level penalty is the wrong instrument.

**The keep-set has to cover every workload, and one corpus could not.** Tracing on the web/code
corpus (v2) fixed HTML and left long prose repeating; tracing on a narrative corpus (v3, six fiction
and dialogue pieces) fixed prose and broke HTML (distinct-token ratio 0.04). At 44 % of the experts
the two rankings compete for the same slots. The fix needs no third trace: an expert trace is a
per-token histogram, so `results/trace-union` is the concatenation of both traces' per-layer arrays
(190 sequences, 36,250 tokens) with the statistics rebuilt from it.

Generation gate on the union keep-set, 900-2000 greedy/sampled tokens per case, no penalties:

| case | tok/s | distinct-token ratio | structure |
|---|---|---|---|
| story (temperature 0.7) | 14.1 | 0.56 | ok |
| essay (temperature 0.7) | 14.3 | 0.50 | ok |
| Python module | 30.7 | 0.47 | ok |
| single-file HTML game | 37.6 | 0.59 | ok |
| JavaScript module | 31.5 | 0.46 | ok |

Eight-workload suite on the same configuration, one request each through the gateway: Python 24.3,
HTML 36.6, JavaScript 28.2, SQL 31.8, explanation 18.1, German 25.2, arithmetic 25.1, long prose
17.1 tok/s; thinking on 18.6; a 5,014-token prompt prefills in 14.9 s (**337 tok/s**) and decodes at
19.4; a tool call returns `finish_reason: tool_calls` with well-formed arguments. The generated HTML
game passes every structural check (doctype, balanced style and script tags, 3x3 grid, win check,
reset button, click handlers, no corrupted CSS units) and stops on its own at 982 tokens.

### 2026-09-12 09:30-10:40 -- prefill: what the routing timer was really measuring, and the memory guard

**The device slot table now covers prefill too.** `Model.moe` takes the table the fast decode path
builds (`self.slot_lut`) instead of resolving every (layer, expert) pair on the host, which turns a
Python pass over 12,288 pairs per layer per chunk into one GPU gather. `route_s` on a 7,030-token
prompt goes 13.72 s -> 0.0.

**It did not make prefill faster, and the earlier reading of that timer was wrong.** `route_s` is
nested inside `moe_s` and overlaps it, so removing it entirely leaves the wall time where it was:

| | prefill | moe_s | route_s |
|---|---|---|---|
| host routing, chunk 2048, arena 98 | 17.8 s (396 tok/s) | 15.07 | 13.72 |
| device table, chunk 2048, arena 88 | 19.0 s (369 tok/s) | 15.27 | **0.0** |

The change is kept because it is correct and removes host work from the loop, but the claim that
routing was three quarters of prefill was an artefact of reading a nested timer as if it were
additive.

**What prefill actually costs is the 3-bit unpack.** A CB3 expert cannot be used by the prefill
kernel directly, so every chunk unpacks the experts it touches back into packed FP4 and runs the FP4
kernel over them (`tools/cb3_moe.py::moe_forward_prefill`). A 2,048-token chunk touches nearly all
384 experts in all 40 layers, and the unpack repeats for every chunk, so the cost scales with chunk
*count*: at chunk 512 prefill is 291 tok/s, at 1024 it is 326, at 2048 it is 369.

**Two memory guards, both earned.** The engine now refuses to start when the arena plus warm-start
scratch plus the floor exceeds MemAvailable, and a watchdog thread samples `/proc/meminfo` twice a
second and calls `os._exit` when it stays under 2.5 GB for three seconds. The kernel does not
OOM-kill cleanly on this box: it thrashes until sshd can no longer fork, the machine answers ping
while being unreachable, and only a power cycle recovers it. Exiting immediately lets the kernel
reclaim everything at once. The watchdog fired correctly during a 7,030-token prefill at a 98 GB
arena (MemAvailable 0.2 GB) and the box stayed up. Prefill memory, not the arena, is what bounds
the arena size: 2,048-token chunks need roughly 10 GB of headroom on top of the resident experts.

Also relevant on this box: a crontab entry started a Qwen vLLM container 90 seconds after every
boot, on the same port, holding ~85 GB. It is what collided with the expert arena twice. Disabled.

### 2026-09-12 20:00 -- two faults behind the degeneration, and a third ranking rule

Everything here at `PRUNE_KEEP=0.36`, 139 of 384 experts a layer, 36-topic catalogue, same box and
checkpoint. 0.36 is not a choice: a 121 GiB GB10 pins it there, and context is not a lever out of it
-- max keep by `MAX_SEQ` is 262144 -> 0.350, 131072 -> 0.361, 65536 -> 0.366, 8192 -> 0.371.

**Fault 1: a register that was traced, never selected, and broke the generation.** The served
selection was {english, html, python, reasoning}. css sat at 0.518 and javascript at 0.564 -- both
histograms in the file all along, neither selected, so nothing on the screen was red. Asked for a
styled page the model wrote `* { }`, then `box-sizing: inline-block`, then the same `<style>` block
over and over to the token cap. Reproduced with thinking **on and off**, so it is not a think-block
fault. Selecting css, javascript and typescript under `maxmin` at the same 80.4 GB arena: 114
correct declarations, custom properties, no repetition. A coverage bar can only warn about a topic
somebody selected; the catalogue had the evidence and the selection threw it away.

**Fault 2: the think-exit was never traced by anything.** Every wrapper in `corpus/make_corpus.py`
except `wrap_think` writes `</think>` immediately after the assistant tag, closing an *empty* block:
95 sequences across trace_corpus_v2 and _v3, 85 with `</think>` adjacent to the tag, none with it
after real content. So the experts that fire on "the deliberation is finished, close it, begin the
answer" were never ranked, are not resident, and the model cannot stop deliberating. At temperature
0 it writes "I'll write the code now." and then repeats "Let me write." to the cap, answer length 0.

Refuted as fixes, each measured on the real server: temperature 0; `no_repeat_ngram=8`;
`reasoning_effort=10`, which made it *worse* (59,766 characters of thinking); frequency and presence
penalties, which broke it lexically instead. This is not a sampling problem and no sampler setting
reaches it. The fix is a corpus one: a new `think` kind in `corpus/make_corpus.py` and a hand-written
`reasoning_code` topic (8 records, 3,543 tokens, `corpus/sources/reasoning_code.txt`), traced
2026-09-12. **Its gate result is not known.** Nothing below should be read as "the think-exit is
fixed".

Two further candidates refuted in `results/htmlbug/`, one variable per engine load: speculative
decoding off (`K_nospec`) and the dense/head quantizations off (`M_nodensequant`) both degenerate
identically to the shipped stack.

**`DSV41_PRUNE_RANK=maxmin`.** Per-layer normalisation divides corpus size out, so under `sum` a
*broad* topic spreads its mass thinly, scores low on every expert, and loses slot after slot to a
peaky specialist. `maxmin` (`engine/v41_engine.py::_maxmin_counts`) hands each layer's slots out one
at a time to whichever selected topic is currently least covered, water-filling; an expert admitted
for one topic counts for every topic that routes to it, so overlap is paid for once rather than per
topic.

| | `sum` | `maxmin` |
|---|---|---|
| {english, html, python, reasoning, css, javascript} | english **0.556**, css 0.803, javascript 0.814 | 0.676 - 0.688 across all six |
| worst-served topic, 4 topics -> 18 | 0.657 -> 0.410 | the same span costs 0.06 |
| 35 of the 36 topics together | worst 0.464 (chinese) | 0.580 - 0.636 (english hardest) |

Under `maxmin` on this box about sixteen topics is where the worst-served one reaches 0.671, under
the 0.7 line. So breadth is cheap under `maxmin` and ruinous under `sum`, and neither rule creates
capacity: about 31 % of the routing mass is displaced at 0.36 whichever rule ranks the set.

`tools/budget.py` still composes and colours every selection with `sum`, so a bar read while
planning a `maxmin` run is the wrong rule's number until that is extended.

**Residual, after both faults are addressed.** A rare token corrupts (`color-scheme` ->
`color-s-s-mode`), the one-step cycle breaker breaks up runs of U+2011, and the model enters a
self-correction loop trying to repair what it just wrote. Not attributed.

Also measured today: 131,072 context prefilled and generated from, so `tools/budget.py`
`VALIDATED_MAX_SEQ` is 131,072 rather than 32,768, and the old 64k-watchdog anecdote that set the
32,768 mark is superseded.

### 2026-09-13 -- the criterion was wrong: rank on contribution, not on frequency

Full write-up, every run and both controls: `RESULTS.md` §5. Two things carry forward from it.

**Frequency and contribution are not the same measurement, and this repo used the wrong one for
four tags.** Every keep-set before today ranked experts by how often the router picked them. The
same keep-set ranked by `gate_weight(t, e) x ‖expert_e(x_t)‖₂` -- what an expert actually adds --
stops corrupting rare tokens at subword boundaries and stops failing to close think blocks, on the
identical topics at the identical 139 experts a layer (`results/keepsets/frontend/GATE.md`, 00:55
against 17:19). REAP (arXiv 2510.13999) had published the same thing on Kimi-K2, which has this
model's routing shape: LiveCodeBench 0.434 -> 0.082 at 75 % kept under frequency, 0.440 under
saliency. An expert the router reaches for constantly whose output barely moves the residual stream
tops a frequency ranking and is worth almost nothing in the arena, and that is what the budget was
being spent on. `DSV41_PRUNE_SOURCE=saliency` is the switch; the ranking rules see 384 non-negative
numbers either way and cannot tell which measurement produced them.

**Why `drop` mode failed, which is worth remembering before anyone proposes it again.** The
intuition is sound -- a substituted expert injects a signal the model never trained to receive, so
zeroing a non-resident pick should be gentler than replacing it. It is not, because with
`norm_topk_prob` on the survivors are renormalised to the routed scale. Dropping two of six picks
is therefore **top-k' routing with k' = 4 at full weight**, not an attenuation toward the shared
expert, and on roughly a token in twenty nothing survives at all and the token falls through to the
shared expert alone. Measured: 0 of 6 on Frontend at keep 0.36, including a page that passes under
`substitute` (`docs/keep-sets.md`, "Why coverage predicts whether long generations hold
together", and the `DSV41_PRUNE_MODE` block in `env.example`). The unpruned model at
`DSV41_TOPK=4` says the same thing with nothing pruned -- 1 of 3, identifiers corrupting with every
expert available (`results/keepsets/null-topk4/GATE.md`). Six experts of which two are wrong beat
four right ones here. `substitute` stays the default and `drop` stays a documented negative result.
