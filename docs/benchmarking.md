# Benchmarking

> **Work in progress. One benchmark row exists.** The `code` workload was measured on
> 2026-09-10 at the real arena size and is in [`RESULTS.md`](../RESULTS.md) §4 with its
> config; `prose`, both one-shot generations and every thinking-on run were stopped before
> they produced a number. There is no measured long generation and no thinking-mode figure
> in this repo. The early bring-up figures in [`NOTES.md`](../NOTES.md) were taken on a
> 20 GB debug arena (6.9 % of the routed experts) and measure that arena, not this recipe —
> do not quote them.

`bench/bench.py` measures one running server and writes a JSON row. It reports what the
SGLang cookbook reports — TTFT and TPOT — plus decode tok/s, so a row here is directly
comparable to a row from the ling3-flash-spark harness it was adapted from. And it records
what that harness has no equivalent of: **the engine's own `x_engine_stats`**.

That last part is the whole point. On this recipe the model is not resident — 288.8 GB of
routed experts against an arena of tens of GB — so a tok/s number without the expert hit
rate and the GB read to produce it is not a result.

## Running it

```bash
python3 bench/bench.py --workload prose  --runs 3 --out results/prose.json
python3 bench/bench.py --workload code   --runs 3 --out results/code.json
python3 bench/bench.py --workload random --isl 8192 --osl 1024 --runs 3 --out results/random8k.json
python3 bench/bench.py --workload angry-birds --label hot --out results/angry-birds.json
python3 bench/bench.py --workload mario       --label hot --out results/mario.json
python3 bench/bench.py --workload prose --thinking --effort 90 --label think90
```

Through the container dispatcher, which fills in the base URL from `.env`:

```bash
./run.sh bench --workload prose --runs 3 --out results/prose.json
```

The base URL is `http://127.0.0.1:$PORT` by default (`PORT` from `.env`, else 8000) and
can be overridden with `--base` or `$BENCH_BASE`, so a bench can never quietly measure a
closed port while the server runs somewhere else. `bench/bench.py` is stdlib-only —
running it on the host rather than inside the container keeps its own memory off the
server's pool.

| workload | prompt | default osl | for |
|---|---|---|---|
| `prose` | ~50-token essay request | 1024 | the everyday number; least favourable to a drafter |
| `code` | ~50-token "write an LRU cache" request | 1024 | what speculative decoding is best at |
| `random` | `--isl` unique random words (default 8192) | 1024 | the cookbook cell's workload; long-prefill behaviour |
| `angry-birds` | a trace-corpus one-shot prompt, verbatim | 16384 | a long real generation, and a playable artefact |
| `mario` | the other one, verbatim | 16384 | same |

The two one-shot prompts are **byte-identical** to the ones in the routing-trace corpus.
That is deliberate: the expert hit rate they produce at serving time can be laid next to
the coverage that `results/trace-*/stats/coverage.json` predicts for exactly that text.
Reword the prompt and the comparison is gone. One-shot runs default to `--runs 1
--warmup 0` and write the generated file to `results/oneshots/<label>-<workload>.html`,
with the fenced HTML unwrapped so the games open in a browser.

## What a row contains

- **TTFT** — send to first `content` or `reasoning_content` delta.
- **TPOT** — `(total − TTFT) / (completion_tokens − 1)`.
- **decode tok/s** — `(completion_tokens − 1) / (total − TTFT)`. Medians over the measured
  runs; every run's raw numbers stay in the JSON.
- **engine_median** — medians of `accept_len_mean`, `expert_hit_rate`, `nvme_gb` and
  `engram_rows`, taken from the server's `x_engine_stats`.

## Reading `x_engine_stats`

The full object is kept per run under `raw[].x_engine_stats`. Its static half is the
engine's configuration (also on `/health`), so a row always carries the conditions it was
taken under:

| field | meaning |
|---|---|
| `arena_gb`, `arena_slots` | size of the resident FP4 expert arena, in GB and in 18.8 MB slots |
| `lru_slots`, `transient_slots` | the split between the LRU region and the prefill-only ring |
| `resident_expert_pct` | `arena_slots` as a share of all 15,360 routed experts — the single number that explains most rows |
| `max_seq`, `spec`, `act_quant` | context the caches were sized for; speculative decoding on/off; whether activations were fake-quantised to FP8 |
| `kernel` | `triton-fp4` or `dequant-fallback`. A row taken on the fallback is a different measurement — check this before comparing anything |
| `trace_stats` | which `coverage.json` ranked the warm start, or `null` for index order |

And the per-run half:

| field | meaning |
|---|---|
| `prompt_tokens`, `completion_tokens` | the work actually done |
| `prefill_s`, `prefill_tok_s` | prefill wall time and rate. On a cold prompt this is dominated by expert misses, not by compute |
| `decode_s`, `decode_tok_s` | the engine's own view of decode, next to the client's |
| `steps`, `accept_len_mean` | DSpark steps and mean accepted tokens per step (1.0 = every draft rejected; the drafter proposes blocks of 5) |
| `expert_hit_rate` | fraction of `(layer, expert)` lookups served from the arena. **The number this recipe lives or dies by** |
| `expert_misses`, `prefill_expert_misses` | absolute misses, and how many of them were prefill (those go to the transient ring, not the LRU) |
| `nvme_gb`, `nvme_read_s` | GB read from the checkpoint for this request, and the seconds spent reading. `nvme_gb / nvme_read_s` against the ~5.5 GB/s device ceiling says whether the IO path or the miss rate is the problem |
| `engram_rows`, `engram_s` | 264-byte n-gram row reads and their cost. Expect this to be small |
| `attn_s`, `moe_s` | attention vs MoE wall time. On a streaming run `moe_s` swallows the NVMe wait, so a large gap here is a miss-rate story |

## What the harness refuses to get wrong

1. **Token counts come from `usage.completion_tokens`, never from chunk counts.**
   Speculative decoding emits several accepted tokens per SSE chunk; counting chunks
   under-reports decode speed by 3–5×. `stream_options.include_usage` is on for every
   request and a run without usage raises rather than guesses.
2. **Every non-one-shot run gets a fresh prompt, and its length is verified** against the
   server's own tokenizer. Repeated filler tokenises at ~6.8 chars/token instead of ~4 — a
   prompt half the intended length.
3. **Prompt lengths are measured with `/v1/debug/prompt`**, not with a one-token
   completion. Here a prefill is minutes of NVMe streaming; the fitting loop would cost
   more than the benchmark it sets up.
4. **Seeds include the label**, so two rows taken back-to-back never share prompts. On
   this engine a repeat is served not just from a prefix cache but from an expert arena the
   first run already warmed — the easiest way there is to publish a hit rate no cold client
   will ever see.
5. **Thinking is set with `chat_template_kwargs.thinking`**, the first branch of the
   server's precedence chain, so it is unambiguous.
6. **`--ignore-eos` whenever the number will be compared.** Without it a row stops wherever
   the model decided to, and two configs are then compared on two different amounts of work
   — and of expert streaming. Leave it off for the one-shot workloads, where a truncated
   HTML file is not a result.

## Reading a row honestly

Speculative decoding is lossless — the target verifies every drafted token — so these
workloads measure **speed only**, never quality.

And the first bench after a start is cold in a way the ling3 recipe never had to think
about. The arena is warm-started from the trace ranking, but everything outside that hot
set is still an 18.8 MB NVMe read on first touch. `nvme_gb` in the first run versus the
third is the size of that effect. Report both, or report neither.

More detail on the harness itself: [`bench/README.md`](../bench/README.md).
Where the numbers come from: [architecture](architecture.md).
