# bench/

`bench.py` measures one running server (`../start.sh`) and writes a JSON row.
It reports the two numbers the SGLang cookbook reports — TTFT and TPOT — plus
decode tok/s, so a row here is directly comparable to a row from the
[ling3-flash-spark](https://github.com/0xBakeer/ling3-flash-spark) harness that
this one is adapted from.

It also records what the ling3 harness has no equivalent of: **the engine's own
`x_engine_stats`**. On this recipe the model is not resident — the routed
experts are 288.8 GB against an arena of tens of GB — so a tok/s number without
the expert hit rate and the GB read to produce it is not a result, it is an
anecdote.

```bash
python3 bench/bench.py --workload prose  --runs 3 --out results/prose.json
python3 bench/bench.py --workload code   --runs 3 --out results/code.json
python3 bench/bench.py --workload random --isl 8192 --osl 1024 --runs 3 --out results/random8k.json
python3 bench/bench.py --workload angry-birds --label hot --out results/angry-birds.json
python3 bench/bench.py --workload mario       --label hot --out results/mario.json
python3 bench/bench.py --workload prose --thinking --effort 90 --label think90
```

The base URL defaults to `http://127.0.0.1:$PORT` (`PORT` from your `.env`, else
8000) and can be overridden with `--base` or `$BENCH_BASE`, so a bench can never
silently measure a closed port while the server runs somewhere else.

## Workloads

| workload | prompt | default osl | what it is for |
|---|---|---|---|
| `prose` | ~50-token essay request | 1024 | the everyday number; least favourable to a drafter |
| `code`  | ~50-token "write an LRU cache" request | 1024 | the workload speculative decoding is best at |
| `random` | `--isl` unique random words (default 8192) | 1024 | the SGLang cookbook cell's workload; long-prefill behaviour |
| `angry-birds` | `tools/make_corpus.py::ONE_SHOTS[0]`, verbatim | 16384 | a real long generation, and a playable artefact |
| `mario` | `tools/make_corpus.py::ONE_SHOTS[1]`, verbatim | 16384 | same |

The two one-shot prompts are **byte-identical** to the ones in the routing-trace
corpus (`corpus/trace_corpus.jsonl`, built by `tools/make_corpus.py`). That is
the point of them: the expert hit rate they produce at serving time can be put
next to the coverage numbers `results/trace-*/stats/coverage.json` predicts for
the same text. Reword the prompt and that comparison is gone.

One-shot runs default to `--runs 1 --warmup 0` (they are minutes each) and write
the generated file to `results/oneshots/<label>-<workload>.html` — the fenced
HTML block is unwrapped, so the games open in a browser. `.txt` is written
instead when the model produced no HTML, which is itself the result. A second
measured run appends `-run2`. `--no-save-oneshot` turns the writing off.

## Definitions

- **TTFT** — wall time from request send to the first `content` or
  `reasoning_content` delta.
- **TPOT** — `(total − TTFT) / (completion_tokens − 1)`.
- **decode tok/s** — `(completion_tokens − 1) / (total − TTFT)`. Medians over
  the measured runs; every run's raw numbers stay in the JSON.
- **engine_median** — medians of `accept_len_mean`, `expert_hit_rate`,
  `nvme_gb` and `engram_rows` from the server's `x_engine_stats`. The full
  object (prefill/decode seconds, attention vs MoE split, prefill misses, NVMe
  read seconds) is kept per run under `raw[].x_engine_stats`.

## What the harness refuses to get wrong

1. **Token counts come from `usage.completion_tokens`, never from chunk
   counts.** Speculative decoding emits several accepted tokens per SSE chunk;
   counting chunks under-reports decode speed by 3–5×. `stream_options.include_usage`
   is on for every request and a run without usage raises, it does not guess.
2. **Every non-one-shot run gets a fresh prompt, and its length is verified.**
   Repeated filler tokenises at ~6.8 chars/token instead of ~4 (a prompt half
   the intended length). `random` prompts are unique numbered words fitted by
   ratio iteration until the *server's own tokenizer* reports the target length;
   a bounded bisect saturates and silently returns a 2× prompt.
3. **Prompt lengths are measured with `/v1/debug/prompt`, not a 1-token
   completion.** That endpoint renders the chat template and tokenises without
   touching the engine. The ling3 harness used a real completion for this;
   here a prefill is minutes of NVMe streaming, so the fitting loop would cost
   more than the benchmark it is setting up.
4. **Seeds include the label.** Two rows taken back-to-back on one server must
   not share prompts — on this engine a repeat is served not just from any
   prefix cache but from an expert arena the first run already warmed, which is
   the single easiest way to publish a hit rate that no cold client will see.
5. **Thinking is set with `chat_template_kwargs.thinking`.** That is the first
   branch of the server's precedence chain (`server/README.md`), so it is
   unambiguous; `--effort` rides along as `reasoning_effort`, which sets the
   budget without flipping thinking back on by itself.

There is no `ignore_eos` here (the server does not implement one), so a `prose`
or `code` row stops at EOS and `finish_reason` is recorded per run. Compare rows
with the same `finish_reason`, or raise `--osl` until every row says `length`.

## Reading a row honestly

Speculative decoding is lossless — the target verifies every drafted token — so
these workloads measure speed only, never quality. And the first bench after a
`./start.sh` is a *cold* number in a way the ling3 recipe never had: the arena
is warm-started from the trace ranking, but everything outside the hot set is
still an 18.8 MB NVMe read on first touch. `nvme_gb` in the first run versus the
third is the size of that effect, and it is the number this whole recipe lives
or dies by.
