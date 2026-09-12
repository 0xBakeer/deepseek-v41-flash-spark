# Profile `topics` — 35 topics, NOT YET GATED

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

The first attempt at these 35 topics averaged a few hundred tokens each, the thinnest at 221. That
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
