# `./tune.sh` — choosing what the box is good at

Only about 40 % of this model's routed experts fit in a GB10's memory at once. Which 40 % is a
real choice, and it is the one configuration decision on this machine that changes both what
the model is good at and whether it loads at all.

`./tune.sh` is that choice on one screen: pick topics, watch what they cost against the memory
the box has *right now*, and start the server from the same screen.

```
 DeepSeek-V4.1-Flash                     NVIDIA GB10 · 130.6 GB · 117.4 free · cb3 experts

 T O P I C S                                      B U D G E T
 35 available · 3 selected
 ───────────────────────────────────────────────  ──────────────────────────────────────
▌● python             ████████████████▏··· 0.91   experts resident        5,990 / 15,360
 ● html               ██████████████▎····· 0.82                                   39.0 %
 ○ rust               ██████▏············· 0.35
 ● german             █████████████▊······ 0.79   expert arena                   86.6 GB
 ○ sql                ████████▏··········· 0.46   dense weights                   7.6 GB
                                                  drafter experts                 7.2 GB
                                                  KV cache · 32k                  285 MB
                                                  ──────────────────────────────────────
                                                  resident                      101.7 GB
                                                  free after load                15.3 GB

                                                  room to launch                +13.8 GB
                                                                                  FITS
 ───────────────────────────────────────────────
 R E S I D E N T   E X P E R T S
 ◂   39 % ▸ ███████████████▌·············  max 45 % here

 C O N T E X T
 ◂   32k ▸  run to 32k here; the cache alone has room for 2.9M

 weakest selected topic  german 0.79      raise to 46 % for 0.85 on every one
 ↑↓ topic   space select   ←→ adjust   tab pane   a all   n none   / filter   m fit   r RUN
```

## Where the header's numbers come from

On a GB10 there is no separate pool to ask about: the memory is unified, and `nvidia-smi` answers
`[N/A]` for `memory.total`, `memory.used` and `memory.free`. So the only honest source is
`/proc/meminfo`, and that is what the header reads — `MemTotal` and `MemAvailable`, live. It is
also what the engine's pre-flight compares against, so the two agree by construction.

`MemAvailable` moves while you look at it. If something else on the box is holding memory the
header says so by name and the Run key refuses, because two processes each reserving an arena
this size wedge the machine past the point where a login can be opened.

## What the numbers mean

**Coverage** — the bar next to each topic — is the fraction of that topic's *measured* routing
that lands on an expert the current budget keeps resident. It is computed from the per-topic
expert histograms in a `coverage.json`, so it is a measurement, not an estimate.

A hollow bar means the topic was traced on too few tokens to rank 384 experts, and the column on
the right says how many. Treat that number as an upper bound rather than a measurement. Coverage
is computed on the same trace that chose the experts, so a topic seen for 300 tokens routes to
whatever fired during those 300 tokens and scores as if it were well served. In the 35-topic
trace the thinnest topics scored 0.75 to 0.82 while the three with real corpora behind them
scored 0.62 to 0.69. The ranking is not better on Swift than on English; the evidence is thinner.
Aim for a few thousand tokens a topic before trusting a bar.

It is also the number that predicts whether long generations hold together. An expert that is
not resident is not routable, so a topic the keep-set does not cover routes to its second
choice on every token, and the output degenerates into repetition. That failure looked for
days like a quantization bug; it was a corpus that contained no markup, and the markup topic's
coverage was 0.03. Raising it to 0.40 fixed it. Coverage below about 0.7 is where that starts.

**Room to launch** reproduces the engine's own pre-flight, which refuses to start when the
arena plus its packing scratch plus the free-memory floor exceeds `MemAvailable`
(`engine/v41_engine.py`). Finding that out by loading costs three minutes; this costs nothing.

**Free after load** is what is left for a prefill chunk once everything resident is resident.
The KV cache is the smallest term on the screen — 285 MB at 32k, 3.4 GB at 1M — so it is not
what bounds the context window. Prefill is, and the tool says how far the box has actually been
run rather than predicting a ceiling it has not reached.

The panel lists everything that holds memory for the whole run. The one thing it leaves out is
the Engram row cache, which is capped at 200,000 rows of 264 bytes per table and two tables, so
53 MB each at most. Engram is also the only thing still read from NVMe once a keep-set is fully
resident: 24 rows per token per table, about 13 KB a token, against zero bytes of expert
weights.

## Fewer topics are not faster. They are cheaper.

A decode step reads the experts the token activates — six of 384 per layer — and that count does
not depend on how the keep-set was chosen. The measured behaviour agrees: across nine workloads
on one keep-set the step is ~145 ms in every case, and the 17-to-37 tok/s spread between them is
entirely the drafter's acceptance length (`RESULTS.md` §4.3). **So selecting fewer topics should
not be expected to make a step faster.**

What it does is reach a given coverage at a *smaller* budget, and the budget is the arena:

| selection | keep fraction for 0.85 coverage | arena |
|---|---|---|
| one topic | 32 % | 71 GB |
| two topics | 51 % | more than this box holds |

That is the trade the screen is built around. Press `m` to snap the keep fraction to the
smallest one that serves every selected topic, and read the arena off the panel. The 27 GB
between those two rows is context window and prefill room.

> **Not measured yet.** There is one path by which topic choice could touch step time after all.
> Speculative decoding verifies a block of six tokens, and that block touches about 21 *distinct*
> experts per layer rather than six. A keep-set matched to the workload may concentrate routing
> and lower that count. `block6_unique_mean` in `coverage.json` is measured without a keep mask,
> so it cannot answer this — only an A/B at a fixed keep fraction with one topic against many
> can, and it has not been run. Until it is, treat the paragraph above as the mechanism, not a
> measurement.

## Without a terminal

```bash
./tune.sh --list                          # the topics this keep-set carries, with coverage
./tune.sh --topics python,html --print    # the environment that selection implies
./tune.sh --topics python,html --write    # write those settings into .env
```

`--print` exits non-zero when the selection will not load, so it works as a check in a script.
The interactive `r` writes the same settings and then runs `./start.sh`; `w` writes them and
stops. `.env` is only touched for the keys the tool manages (`EXPERT_TOPICS`, `PRUNE_KEEP`,
`MAX_SEQ`, `ARENA_GB`, `EXPERT_FORMAT`, `TRACE_STATS`) and the previous file is kept as
`.env.bak`.

## Where topics come from

A topic is one per-layer expert histogram, measured by routing a corpus of that topic alone
through the model. They are stored inside `coverage.json` next to the mixed histogram, so
composing a keep-set out of several of them is arithmetic on numbers already in the checkout —
it needs no GPU and no new trace.

To add one, tag a corpus and trace it:

```bash
python3 corpus/make_corpus.py --topic rust:code:~/src/some-rust-project \
                              --topic german:prose:~/texts/de --out corpus/trace.jsonl
python3 tools/engram_rows.py  --corpus corpus/trace.jsonl --out engram_rows
python3 tools/expert_trace.py --corpus corpus/trace.jsonl --engram-dir engram_rows \
                              --out results/trace-mine --layers 0-39
python3 tools/expert_stats.py --trace results/trace-mine
```

The trace is one pass over the corpus per layer and costs about the same whether the corpus
carries five topics or thirty-five, so it is worth tagging generously.

Selecting no topics is not an error: the engine then ranks on every topic in the file, which is
what the shipped profiles in `results/keepsets/` do.

## Checks

```bash
python3 tools/test_budget.py
```

Cross-checks the slot sizes against the kernel's own constant, the KV formula against two
measured lengths, and the launch gate against an arena the box accepted and one it did not.
