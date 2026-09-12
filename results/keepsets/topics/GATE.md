# Profile `topics` — 36 topics, PARTIALLY GATED

Built 2026-09-12. This is the keep-set `EXPERT_TOPICS` and `./tune.sh` are meant to be used with:
it carries a per-layer expert histogram for each of 35 topics, so any subset of them can be
composed into a keep-set without tracing again.

| | |
|---|---|
| corpus | `corpus/trace_topics_v2.jsonl`, 430 sequences, 115,898 tokens |
| topics | 16 programming languages, 11 natural languages, 8 domain registers |
| per topic | 3,035 to 3,819 tokens, deliberately even |
| sources | code from local and public repositories; languages and registers from Wikipedia (`corpus/fetch_topics.py`) |
| trace | 40 layers, every layer touching all 384 experts |

**No generation gate has been run on this keep-set.** Coverage is a measurement; whether a given
selection of these topics produces sound long output is not, until it is gated. Use the shipped
`general` profile for anything you cannot check yourself.

## Why the corpus was rebuilt, and what it changed

The first attempt at these 35 topics averaged a few hundred tokens each, with Chinese at 150. That
is not enough to rank 384 experts per layer, and the failure is not merely noisy, it is
*optimistic*: coverage is computed on the same trace that chose the experts, so a thinly traced
topic scores as though it were well served.

The two traces are directly comparable, and the contamination is visible:

| | coverage range | tokens per topic | correlation of tokens with coverage |
|---|---|---|---|
| first attempt | 0.59 – 0.82 | 221 – 2,778 | **−0.36** |
| this one | 0.49 – 0.77 | 3,035 – 3,819 | **−0.00** |

A negative correlation means the better-sampled topics scored *lower*, which is the artefact
rather than a property of those topics. With the evidence levelled it disappears, and coverage
measures the topic instead of the sample behind it.

The keep-sets the two produce overlap by **68 %** at keep 39 %, so roughly a third of the resident
experts changed. That is not attributable to sample size alone: the correlation between a topic's
old token count and how much its keep-set moved is only 0.29, and the eight thinnest topics moved
barely more than the eight best-sampled (66 % against 70 % overlap). The sources changed as well as
the size, and this measurement cannot separate the two.

## What the spread says

At keep 39 % with all 35 selected, coverage runs 0.49 to 0.77.

* Hardest to serve: `chinese`, `html`, `latex`, `python`, `java`.
* Easiest: `italian`, `french`, `portuguese`, `ruby`, `translation`.

The Romance languages clustering at the top is the expected shape — they share experts, so serving
one serves the others cheaply. No selection of all 35 reaches 0.85 on every topic within this
box's memory; `./tune.sh` reports how many clear the bar at the largest arena that fits.

## 2026-09-12 — topic 36: `reasoning`

Thinking mode was degenerating on this model: an arithmetic word problem produced 10,937
characters of reasoning circling the same subtraction and never answered, and two coding tasks
dissolved into repeated fragments. The same prompts with thinking off answered correctly, so the
failing thing was the register, not the task. None of the 35 topics contained a line of
deliberative text, so the experts that write it were never observed firing and ranked cold.

| | |
|---|---|
| corpus | `corpus/trace_reasoning.jsonl`, 11 sequences, 3,490 tokens |
| source | `reasoning_all.txt` — 27 of 32 server generations (thinking OFF) + authored deliberative prose |
| shapes | arithmetic, algebra, debugging, planning, weighing options, checking an argument, tracing code, estimating |
| trace | 40 layers, 36 minutes, `results/trace-reasoning` |
| merged | `counts_reasoning` added per layer; the other 35 histograms are byte-identical |
| original | kept as `coverage.json.before-reasoning` |

Coverage at keep 38 %: **0.51** with all 36 topics selected — second lowest after `chinese`,
which is what a genuinely distinct register looks like. **0.74** when the selection names it
(`reasoning,python,html,english`).

### The corpus could not simply be generated

Asking the server for ~900 tokens of deliberative text returned degenerate samples every time:
distinct-token ratios of 0.10–0.24 against 0.48–0.54 for the prose already in `corpus/sources/`.
An arithmetic sample read `7,340 - 1,285 = 6,340? No, that's wrong. 7,340 - 1,285 = 6,340?`
forty times. Thinking-off generation is only healthy in registers the keep-set already covers —
asked for this one it collapses the same way, `<think>` block or not. Shortening generations to
~300 tokens fixed most of it. Two filters were applied: the whole-sample 0.25 floor, and a tail
check (last 100 tokens below 0.35), because a sample can clear 0.25 overall and still be looping
at the end — `arith-3` scored 0.268 with a visibly looping tail. 5 of 32 were dropped, 3 on ratio
and 2 on tail. The arithmetic and algebra shapes the model could not write cleanly were supplied
as authored deliberative prose.

### Gate: PARTIAL

`EXPERT_TOPICS=english,html,python,reasoning`, `PRUNE_KEEP=0.38`, `MAX_SEQ=32768`, `ARENA_GB=85`,
thinking on, effort 45, greedy, 1,600-token cap. Baseline is the same three prompts on
`keepsets/general` at keep 0.38.

| prompt | before (general) | after (with `reasoning`) | |
|---|---|---|---|
| arithmetic word problem | 4,267 c reasoning, ratio 0.096, no answer, hit the cap | 305 c reasoning, ratio 0.731, answers, stops on its own | **fixed** |
| Python function | 1,319 c reasoning, aborted on repetition | 5,534 c reasoning, ratio 0.327, loops to the cap, no answer | **still fails** |
| single-file HTML | reasoning ratio 0.605, produced valid HTML | loops on `background-color: green-blue-white` to the cap | **still fails** |

The two coding failures are not a code problem. On this same keep-set with thinking **off** all
three prompts are healthy (ratios 0.612 / 0.628 / 0.268, Python and arithmetic terminate on their
own), so what still breaks is deliberation *about code*, inside the thinking block.

That is a gap in this corpus, not in the method. Its 3,490 tokens are prose deliberation; almost
none of it interleaves reasoning with code or markup fragments, which is exactly the register
that still loops — `We can include >>> merge_sorted_lists([1,2,3], [1,2,3])` repeated, and
` ```css body { background-color: green-blue-white; } ``` ` repeated. A second topic whose corpus
is deliberation carrying inline code, CSS and shell fragments is the indicated next step.

Two further notes. The arithmetic case now terminates but gets the sum wrong (1,284 + 967 given
as 2,281; it is 2,251) — and it is wrong with thinking off too, so that is an accuracy problem
this keep-set neither caused nor fixed. And at the all-36 ranking `reasoning` sits at 0.51, below
the ~0.7 that `docs/tune.md` names as where degeneration starts, so the topic only helps when a
selection names it explicitly.
