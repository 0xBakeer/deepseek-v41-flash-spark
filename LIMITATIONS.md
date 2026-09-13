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
* **Prefill is 3.5x faster since 2026-09-10 evening but still NVMe-bound.** Decoder SWA Bounded
  Replay and 2,048-token prefill chunks took a 1,860-token prompt from 118 s to 34 s of TTFT
  (NOTES.md "Speed work"). What is left is I/O efficiency: prefill still spends ~90% of its wall
  time waiting for expert reads, at ~3.8 GB/s against the 5.5 GB/s the device gives at depth.
* **Decoder SWA Bounded Replay is an approximation, and it is on by default** (`DSV41_SWA_REPLAY=0`
  turns it off). For a prompt of at most 128 tokens it is bit-exact -- verified, `--verify-replay`
  reports a max logit delta of 0.0 -- but above that the decoder layers see a 128-token window
  instead of the whole prefix, so the prompt's own logits change: on a 448-token document the
  next-token KL was 0.995 nats and the top-1 token changed. The model was post-trained with this
  replay simulated and the tech report calls the impact negligible; greedy answers to the 1,860-token
  test prompt are character-identical with and without it. Still, this is the one place in the
  engine where output is deliberately not the reference model's.
* **The replay does not shrink prefill for prompts under ~128 tokens** (it covers the whole prompt),
  and its own pass over the last 128 tokens still touches ~300 of 384 experts per decoder layer, so
  short-prompt TTFT is unchanged.
* **The arena is sized once at load and never adapts.** No per-workload hot set (the trace shows
  coding-only and general-only top sets overlap by a Jaccard of only 0.18-0.31), no promotion of
  experts a long session keeps hitting beyond plain LRU, no prefetch on a session's first turn.

### Not measured in this tag

* **Only the `code` benchmark row exists.** `prose`, both one-shot games (`angry-birds`, `mario`)
  and the thinking-on run were stopped before they produced a number (the box was needed for
  interactive use), so
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
  SGLang both assume all experts resident across TP ranks; the closest working public code is a
  4x Spark TP4 build (engram-on-disk + SM12x fixes) which states "TP2 does not fit either way".

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

## v0.2.0-wip (2026-09-11) — what is still not done

* **Decode is bounded by the expert bytes a step has to move.** Best measured: 13.6 tok/s
  (keep 25 %, resident) / 12.9 tok/s (keep 31 %). The graphed verify step is 173 ms + 15 ms draft with
  everything resident, of which ~140 ms is the weights the step reads at the box's 273 GB/s — that
  bandwidth is the floor, not the kernels. Levers not yet taken: fewer host round-trips
  (device-side slot LUT, merged graphs), the
  `_route_kernel` (6 % of the step), fp32 GEMMs of the HC/gate path, and higher acceptance (thinking
  on, code prompts).
* **The full model stays NVMe-bound at 3.5-4 tok/s.** Only pruning changes that on this box.
* **CB3 (3-bit) kernel is 4x too slow** (54 GB/s vs 190 for FP4); the format and its quality are
  proven, the kernel needs a per-lane PTX decoder. Until then the 3-bit rows are simulation only.
* **Pruning keep-sets come from a 10k-token trace** (mixed coding/general). A different workload may
  want a different hot set; `--hot-profile` exists but was not measured after the fix.
* The routing trace and the warm-start ranking were recorded before the hc_post fix; they are
  approximate (routing agreement between the two states is high but not measured).
* Thinking-on decode, sampled-output quality, long prompts (>2k) and the container image remain unmeasured.


## v0.3.0-wip (2026-09-11) — what is still not done

* **Warm start is 183 s in the CB3 format** (GPU packing of 6,160 experts) against 19 s for FP4;
  no on-disk packed cache exists (it would be 88.8 GB).
* **Prefill in the CB3 format unpacks to FP4 on the fly**: ~1.35x the MoE time of an FP4 arena of
  the same size at 2,048-token chunks. A CB3 kernel efficient at prefill shapes is not written.
* **No two-tier arena** (hot experts at FP4, cold at CB3): at 40.8 % all-CB3 the 90.5 GB arena has
  no headroom for it, so it would trade share for precision rather than add capacity.
* **Decode is still bounded by the bytes a step moves**: after the kernel work of this tag the
  verify step is ~87 % weight traffic at the box's achievable bandwidth (routed experts, FP8 dense,
  `wo_a`, LM head). Further speed comes from fewer bytes (lower-bit cold experts, lower-bit dense
  projections), fewer of the ~5,300 small kernels per step, or higher acceptance, not from faster
  kernels for the same bytes.
* **Thinking-on, 8k+ prompts and sampled A/B are not measured in the CB3 configuration.**

## 2026-09-11 20:45 — the fast decode path's precision

* **`DSV41_FUSED_ATTN` defaults to 0.** With the dense projections in fp4 and the fp8 head, greedy
  decoding through the fused attention kernel diverges from the same decode without it and can enter
  a repetition loop (NOTES 2026-09-11 20:00-20:45). Each piece is clean on its own.
* **The graphed decode path is not numerically equal to `Model.forward`**: its logits differ by a
  few percent relative, which is far more than bf16 rounding and is not yet explained. It has been
  so since the path was written; only the combination above made it visible.
* **Teacher-forced loss cannot gate the decode path.** It never runs the loop, so a verification,
  cache or drafter fault is invisible to it. Use `engine/test_spec_lossless.py`, which requires
  greedy decoding with and without speculation to produce identical tokens.

## v0.4.0-wip (2026-09-12) — what is still not done

* **Long prose runs at 17 tok/s against 37 for markup.** The step time is the same in both cases;
  the difference is the DSpark drafter, which is accepted about 5 tokens per step on markup and 2.5
  on prose. That is a drafter-quality axis, not a bandwidth one, and nothing here addresses it.
* **The keep-set is only as good as the corpus it was ranked on.** Serving a workload that
  `results/trace-union` does not represent will degrade it the way markup was degraded before this
  tag. Re-trace with your own corpus rather than assuming this one covers you.
* **Teacher-forced loss is not a gate** and no number in this repo should be read as one. Use the
  generation gate (RESULTS 4.1).
* **The tool grammar is unverified on real weights** (`DSV41_TOOL_GRAMMAR=1`, off by default): its
  unit tests pass, the end-to-end run never happened.
* **Not measured**: sampled quality at scale, generation quality at 8k+ context, the container image
  end to end, and — added 2026-09-12 — **generation with thinking on**: every prompt of the
  generation gate in RESULTS 4.1 ran with thinking off, so nothing in this repo gates a generation
  that has to deliberate and then leave the think block.

## 2026-09-12 — `./tune.sh`, and the claim on its screen

The screen says a step reads only the experts a token activates, so choosing fewer topics buys a
smaller keep fraction rather than a faster step. The mechanism is the architecture's and the
supporting measurement is §4.3 of `RESULTS.md` — ~145 ms per step across nine workloads on one
keep-set, with the whole tok/s spread coming from drafter acceptance.

**The A/B that would settle it has not been run**: one topic against many at the same
`PRUNE_KEEP`, same prompts, comparing step time. A verify block of six tokens touches ~21
distinct experts per layer, and a keep-set matched to its workload could concentrate that.
`coverage.json` carries `block6_unique_mean` but measures it without a keep mask, so it cannot
answer the question.

The coverage numbers on the screen are measurements. The memory numbers reproduce the engine's
own pre-flight and are checked against a real load in `tools/test_budget.py`. Only the
speed sentence is an inference, and it is marked as one in `docs/tune.md`.

## 2026-09-12 14:10 — two things the end-to-end run found

**`EXPERT_TOPICS` had never worked.** `expert_topics` was read inside
`V41Engine.__init__` and passed by the engine's own CLI, but was never a parameter of it, so
every launch through `start.sh` raised `TypeError` — three minutes in, with the weights already
on the GPU. Fixed. `tools/test_engine_kwargs.py` now checks that every key the launchers can put
in `--engine-kwargs` is a parameter the constructor has; it parses the signature with `ast`, so
it needs no GPU and no torch.

**The keep 0.44 / arena 98 GB configuration in §4.3 of `RESULTS.md` does not reliably serve.**
On 2026-09-12 at 14:07 it loaded, reported ready, and the memory watchdog killed it on the first
request:

```
FATAL: host MemAvailable 0.4 GB stayed below the 2.5 GB floor for 3.0 s
```

The arithmetic says why. With `MemAvailable` at 111.0 GB once the dense weights are resident, a
98 GB arena plus 7.2 GB of drafter experts leaves 5.5 GB, and one 2,048-token prefill chunk needs
about 7.5 GB at 32k context. An 87 GB arena leaves 16.5 GB and serves. The
configuration was always inside the margin; it passed its gate on a quieter box.

The engine's own pre-flight accepts the 98 GB arena, because that check runs before the drafter
experts, the KV cache and any prefill exist. `./tune.sh` now applies the stricter test and caps
this box at 42 % of routed experts rather than 45 %.

**Long generations still degenerate, and the gate does not reach them.** At keep 0.39,
temperature 0, the HTML-game prompt is clean at 1,200 tokens (distinct-word ratio 0.527, 32.5
tok/s, acceptance 4.51, nothing read from NVMe) and collapses by 2,400 into repeated corrupted
CSS (`color: #c0.0.0.0.0;`), with penalties at 0. The generation gate runs to 2,000 tokens, so it
has never seen this. Not yet attributed: the 3-bit expert format, the FP8 LM head and the FP4
dense attention are all in the stack, and the ledger records each of them breaking generation on
its own at some point.

**Added 2026-09-12 to the paragraph above.** The failing register has since been named, and it is
not on that list of three. On the live selection {english, html, python, reasoning} at
`PRUNE_KEEP=0.36`, **css coverage was 0.278** — traced, in the file, never selected — which outranks
the 3-bit expert format, the FP8 LM head and the FP4 dense attention as an explanation, because two
of those three are now refuted directly: in `results/htmlbug/`, one variable per engine load, the
run with speculative decoding off (`K_nospec`) and the run with the dense and head quantizations off
(`M_nodensequant`) degenerate identically to the shipped stack. What is left after the selection is
fixed is a smaller, different fault: a rare token corrupts (`color-scheme` -> `color-s-s-mode`), the
one-step cycle breaker breaks up runs of U+2011, and the model loops trying to correct itself. See
RESULTS 4.5.

**Added 2026-09-12 — the think-exit was never in any trace corpus.** Every wrapper in
`corpus/make_corpus.py` except `wrap_think` writes `</think>` immediately after the assistant tag,
so it closes an empty block: 95 sequences across the two shipped corpora, 85 with `</think>`
adjacent to the tag, none with it after real content. The experts that end deliberation were
therefore never ranked and are not resident, and at temperature 0 with thinking on the server writes
"I'll write the code now." and then repeats "Let me write." to the token cap with an answer of
length zero. Refuted as workarounds, each measured: temperature 0, `no_repeat_ngram=8`,
`reasoning_effort=10` (59,766 characters of thinking, worse), and frequency/presence penalties
(lexical breakdown). The fix attempted is a new `think` corpus kind and a hand-written
`reasoning_code` topic (8 records, 3,543 tokens), traced 2026-09-12. **No gate has been run on a
keep-set containing it, so it is not known to work.**

## 2026-09-13 — what the recipe of `RESULTS.md` §5 leaves open

The seven narrow profiles were re-gated on contribution ranking at `PRUNE_KEEP=0.40` (§5.2). They
went 34 -> 39 of 61, the rare-token corruption is gone and the think blocks mostly close. Four
things it did not settle, each of which a reader should know before trusting the 39.

**The harness cannot tell a redraft from a loop.** `repeated_ngram` in `tools/gate_profile.py`
fails a run when any 12-word window appears **three** times. Sixteen of the recipe's 22 misses are
exactly that and nothing else: a fragment redrafted 3 to 10 times inside a long think block,
followed by a finished, correct answer. A model that writes a function signature four ways before
picking one and a model that has stopped making progress produce the same flag. Three was chosen so
that a four-word think-block cycle ("I keep. I write.") tiles a 12-word window exactly, which it
still does — but at effort 45 with thinking on, deliberation that revisits its own draft is normal,
and the threshold has not been re-derived for that register. Until it is, these runs are counted as
failures: the conservative direction, and it makes 39 of 61 a floor rather than a measurement.

**One Go prompt regressed and it is not explained.** Backend's `go-handler` passed round one at 115
lines (`results/keepsets/backend/GATE.md`, 01:24) and under the recipe was cut off by the server's
repeat guard after looping 12x in the think block (same file, 18:42). `java-service` moved the
other way in the same pair of runs, from cut off to 88 lines. One prompt each way in one profile is
not a pattern, and nothing in the recipe is Go-specific, so it is open rather than attributed.

**256k is unproven for long prompts.** Every profile run in §5.2 declares `max_model_len 262,144`
and no
request in any of them prefilled more than a 58-word prompt (`tools/gate_profile.py:PROMPTS`). The
memory the budget model reserves is for the long-prefill case exactly, which is why it caps this box
at keep 0.350 at that context (`RESULTS.md` §4.5) and why 0.40 is above its own ceiling. The engine
pre-flight accepts it and the watchdog did not fire in seven profiles of short prompts; that is the
whole of the evidence. `tools/budget.py:VALIDATED_MAX_SEQ` stays at 131,072 and `env.example` keeps
`PRUNE_KEEP=0.39` / `ARENA_GB=87` / `MAX_SEQ=32768` until the long-prefill test lands.

**Data and research regressed, 7 -> 5 of 11.** It is the one profile the strict count moves against
(`results/keepsets/data_and_research/GATE.md`, 04:24 against 20:26). Every one of its six misses is
a finished answer flagged for a redraft — none corrupts a token, none fails to close, none is
structurally wrong — so the profile improved in kind and lost in count, and the count is what a user
reads. Whether ten topics is simply one too many for this box at this keep fraction is untested: no
narrower variant of it has been gated.
