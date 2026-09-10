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
`~/models/DeepSeek-V4.1-Flash` with `snapshot_download(allow_patterns=...)` so a later full
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
* **tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark** (4x DGX Spark, TP4, serving 2026-09-10): the
  only measured Spark run. vLLM `dsv41-feat` on nightly 8a728663 + `_C_stable_libtorch` rebuilt for
  12.1a + FlashInfer 0.7.0rc1 (0.6.18 lacks the SM120 sparse-MLA decode kernel for V4.1's topk 1152)
  + prebuilt `mxfp8_gemm_cutlass_sm120`. Five SM12x patches (block size 64/128 for the sparse SWA
  and indexer caches, `--block-size 128`, `top_k_per_row_decode` instead of `persistent_topk`
  which needs 128 KB smem per block, GB10 has 99 KB). **Engram-on-disk patch** (`DSV41_ENGRAM_DISK=1`):
  table tensors skipped at load, rows read with `preadv` from the safetensors on NVMe/NFS by a
  32-thread pool in `prepare_inputs` before the forward (so CUDA graphs work), dequantized on CPU,
  copied to a pinned staging buffer. Measured by them: 24 serial preads ~17 ms/step on NVMe,
  parallel 3.1 ms (C1). Per rank 81.6 GiB weights (experts all resident, split 4 ways, DeepGEMM
  MXFP4 MoE backend), KV 4.84 GiB = 1.03M tokens. Numbers: 39-77 tok/s single stream
  (counting 77, code 52-57, reasoning 39, prose 23), DSpark acceptance length mean 3.56
  (1.95-5.79), TTFT 0.27-0.58 s. "TP2 does not fit either way." No expert offload of any kind.
* **SGLang**: PR #38798 open, `lmsysorg/sglang:dev-dsv41`, `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`
  (host copy of the tables, huge pages advised), `--enable-decoder-swa-bounded-replay`. Datacenter
  GPUs only. Blog: engram host offload +36% KV capacity at same decode on 4x GB300.
* **0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000** (SGLang, 4x RTX PRO 6000 + 128 GB DDR5): bounded
  64 GiB DDR5 engram row cache with exact NVMe reads on misses; b12x io_uring reader "prerequisites
  met". 200+ tok/s single stream on that hardware.
* **antirez/ds4**: no V4.1 branch, only an FYI issue (#1023). On X: "Not a fit for 128GB systems ...
  Good fit for Mac M5 Ultra 512GB ... Not really a 'local' model IMHO"; support "probably yes,
  initially as an experiment, will 2 bit quants hold up?".
* **llama.cpp**: converter-only draft PR #28696 (Engram written as row-block memmap, 508 GB at
  Q8_0 + MXFP4 experts); "the model won't load until a V4.1 runtime implementation" exists. No
  runtime, no upstream V4 runtime either (fork only).
* **exllamav3 / anemone / TabbyAPI / sparkinfer**: V4 Flash only; zero V4.1 mentions.
* **ktransformers, ik_llama.cpp, mlx-lm, transformers main**: nothing for V4.1.
* **Quants on HF (all day-0 stubs)**: GGUF (vcruz305 Q2_K uploading, apetersson MixedQ2 2.25 bpw
  experts 170 GB, engram excluded), NVFP4 (LibertAIDAI 400 GiB with engram FP8->FP4 lossy;
  msuiche 415 GB ModelOpt), MLX (Vontra 2-bit 239 GB for 256 GiB Macs at 9.5 tok/s; pipenetwork
  4/8-bit 427-477 GB; inferencerlabs Q4i 14.6 tok/s on a 512 GB M3 Ultra). No EXL3, no REAP/pruned.
* **DeepSeek's three new repos** (2026-09-10): `deepseek-recipe` = Rust + Python protocol/chat-template
  layer (Chat Completions/Responses/Messages -> V4.1 prompt, parses thinking + DSML tool calls;
  string `reasoning_effort` only: low=50, high=75, max=100; **no aarch64 wheel**, build from source
  with Rust 1.97.1 + OpenCV 4). `DeepSelect` = the top-k kernel for the sparse-attention INDEXER
  (k=512 over context positions) and the sampler, sm_100a/sm_103a only -- not expert selection.
  `DeepJIT` = header-only JIT runtime extracted from DeepGEMM, ships no kernels, no license file;
  DeepGEMM 26/09 uses it; DeepGEMM sm_121a issues open (#372, #417, #425).
* Prior art for expert caching (V4 Flash, not V4.1): ssd-moe/deepseek-v4-flash-mlx "a 32GB cache
  captures 80%+ of expert accesses" (48 GB Mac, 4.5-5 tok/s); bigs/deepseek-v4-flash-dgx-spark
  (256-slot expert arena, native packed loader, ~2 tok/s); ktransformers cpuinfer.
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

Plots: `results/trace-partial/stats/coverage.png`, `layer_hist.png`; tables `coverage.md`.

### 0.8 What the sizes alone already decide (arithmetic, no measurement needed)

Routed experts = 15,360 x 3 x 2304 x 5120 = 543.6 B weights. On disk at FP4 + UE8M0/32 =
**4.25 bits per weight = 288.8 GB**. With ~18.5 GB of non-expert weights resident and KV of a few
GB, the expert budget on this box is **~85-90 GB**, i.e. an average of **~1.3 bits per weight**
over all experts. Consequences:

* **Strategy C (everything resident, hot at FP4 + cold at 2-3 bpw) cannot fit at the quality floor.**
  Even with every cold expert at 2.0 bpw, only ~9% of experts could stay at FP4, and the whole
  set at EXL3 2.0 bpw is still 136 GB. It only fits below ~1.3 bpw average, which is below anything
  Khaled ships.
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

## Phase 1/2 -- build log (2026-09-10, Khaled: "just make it work")

Go-ahead received; the remaining 471 GB were downloaded (85 MB/s, ~95 min; Ling-3.0-flash weights
and the ling3 docker image were deleted to make room, both re-downloadable). Work is split into
Opus-5 agents with this thread orchestrating (Khaled's instruction, to stay under the Fable limit).

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
