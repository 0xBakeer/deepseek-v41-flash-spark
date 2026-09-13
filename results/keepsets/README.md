# Expert profiles

The resident expert set is a fixed budget: `PRUNE_KEEP` of the 384 experts in every layer, and the
router may only choose among them. Every domain that budget covers competes for the same slots, so a
keep-set ranked on web code and a keep-set ranked on narrative prose are different sets, and at
44 % keep their overlap is only about 0.70 by Jaccard.

A **profile** is one ranking of that budget, built from a trace corpus of the workloads it is meant
to serve. Select one with `EXPERT_PROFILE=<name>` (ignored if `TRACE_STATS` is set explicitly).
Switching profiles changes which experts are resident, so it takes a restart.

```
EXPERT_PROFILE=general ./start.sh
```

## What a profile can and cannot do

A specialised profile is **better than the general one on its own domains**, because it spends the
whole budget there. It is **worse outside them**, and the failure is not graceful: a keep-set that
never saw a domain produces degenerate output in it, not merely weaker output. Measured on this
repo's own history, a keep-set with no markup in its corpus wrote correct Python and could not
produce a valid HTML file at any keep fraction; one with no Arabic produced a collage of Polish,
Portuguese and Romanian fragments when asked for Arabic.

**Each profile therefore ships a `GATE.md`, and the first line of it says whether a gate was ever
run.** Of the keep-set files below, `general` is the only gated one: `code`, `prose` and `topics`
each say NOT YET GATED at the top of their own `GATE.md`, and what is written under that heading is
inference from how their keep-sets were built, not a generation run on them. Read it before
choosing. If your traffic is mixed, or you cannot predict it, or you are not going to run a gate
yourself, use `general`.

## The named profiles

These are the bundles `./tune.sh` opens on (`tools/tune.py:PROFILES`). Each is a selection of
topics out of the `topics` keep-set, not a file of its own, and each is gated by running
`tools/gate_profile.py --profile <name>` on the prompts its topics carry.

The result column is the run of **2026-09-13 evening** on the recipe of `RESULTS.md` §5 —
`DSV41_PRUNE_SOURCE=saliency`, `DSV41_PRUNE_RANK=maxmin`, `PRUNE_KEEP=0.40`, thinking on, 256k. It
is the profile's score on its own prompts, and a miss is usually a finished answer the harness
failed for redrafting a fragment; the `GATE.md` row says which.

| profile | topics | on the recipe (2026-09-13) | gate record |
|---|---|---|---|
| Frontend | 9 | 7 of 10 | `frontend/GATE.md` |
| Backend | 9 | 5 of 10 | `backend/GATE.md` |
| Systems programming | 10 | not yet gated on the recipe | — |
| Chat and explanation | 7 | 6 of 8 | `chat_and_explanation/GATE.md` |
| Medicine | 6 | 6 of 7 | `medicine/GATE.md` |
| Law and finance | 6 | 5 of 7 | `law_and_finance/GATE.md` |
| Data and research | 10 | 5 of 11 | `data_and_research/GATE.md` |
| European languages | 9 | not yet gated on the recipe | — |
| World languages | 9 | not yet gated on the recipe | — |
| Writing | 7 | 5 of 8 | `writing/GATE.md` |

The three ungated ones are new, and they exist because their predecessors were gated and failed.
"Programming, broadly" carried fifteen topics and passed 4 of 16; "Many languages" carried fourteen
and passed 2 of 15, every natural-language prompt failing; "Everything", all 37 topics, passed 4 of
39 (`programming__broadly/GATE.md`, `many_languages/GATE.md`, `everything/GATE.md`, all
2026-09-13). Six to ten topics serve on a 121 GiB box and fourteen and up do not, so the programming
bundle became Frontend plus Systems programming, the languages bundle became European languages
plus World languages, and Everything is no longer offered — the topic screen still lets you select
all 37 by hand and shows what that costs. Those three records are kept as the reason.

## The keep-set files

| keep-set | trace corpus | gated | topics inside it | intended for |
|---|---|---|---|---|
| `general` | web + code + configuration + technical prose + narrative | yes | `coding`, `general` | mixed traffic; the default |
| `code` | HTML, CSS, JavaScript, React, SQL, YAML, shell, Python, technical prose | no | none | programming and markup only |
| `prose` | narrative fiction, dialogue, essays | no | none | long-form English writing only |
| `topics` | the 37-topic catalogue | no | 37 | `EXPERT_TOPICS`, `./tune.sh` and every named profile above |

Each directory holds `coverage.json` (the per-layer expert histograms the engine ranks from) and
`GATE.md`. The raw per-layer trace arrays are not shipped; `coverage.json` carries the per-category
histograms, so a keep-set can be rebuilt from it alone.

The `topics inside it` column is what `EXPERT_TOPICS` can name. **`code` and `prose` carry only the
mixed `counts` histogram and no per-topic ones at all**, so there is nothing in those files to
select from: `EXPERT_TOPICS` cannot be used with them, and `./tune.sh` shows their budget panel with
an empty topic list. Composing a selection needs `topics` (37 histograms) or `general` (two).

## Composing one from topics

A profile is a fixed ranking. `EXPERT_TOPICS` is the same budget spent on a set of topics you
name, where a topic is one per-layer expert histogram measured on a corpus of that topic alone.
Those histograms live inside `coverage.json` next to the mixed one, so composing a keep-set from
any subset of them is arithmetic on numbers already in the checkout — no GPU and no new trace.

```
EXPERT_TOPICS=python,html,german ./start.sh
```

`./tune.sh` is the same choice with the consequences on screen: how much of each topic's measured
routing the current budget keeps resident, what that budget costs against the memory the box has
free, and how many tokens each topic was traced on. See [`docs/tune.md`](../../docs/tune.md).

A selection worth keeping becomes a named profile on the tool's first screen. `profiles.json` in
this directory is one of the two files it reads them from, and the one to use for a profile that
should travel with the checkout; the other is
`$XDG_CONFIG_HOME/deepseek-v41-flash-spark/profiles.json`, which is where `s` on the topic screen
saves and which survives a fresh clone. Neither is shipped. The format and the rules are in
[`docs/tune-reference.md`](../../docs/tune-reference.md#profiles-from-a-file).

```json
{
  "profiles": [
    {"name": "Arabic desk",
     "description": "Arabic and English prose, for a bilingual assistant",
     "topics": ["arabic", "english", "translation"]}
  ]
}
```

A profile from a file is never gated, and the screen says `untested` for it, because the gate is a
generation run on a keep-set rather than a property of a name and a list of topics.

Selecting nothing is not an error — the engine then ranks on every topic in the file, which is
what the profiles below do.

## Building your own

A profile is only as good as the corpus it was ranked on, and that is the whole lesson of this
directory. To cover a workload, put that workload in the corpus:

`corpus/fetch_topics.py` collects the sources for a 35-topic catalogue and prints the flags for
the next step; `--list` shows what it covers.

```bash
python3 corpus/make_corpus.py --tokenizer $MODEL_DIR \
    --code your_sources/*.ts your_sources/*.go --prose your_docs/*.md \
    --target 9000 --out corpus/trace_mine.jsonl
python3 tools/engram_rows.py --model-dir $MODEL_DIR --corpus corpus/trace_mine.jsonl \
    --out engram_rows_mine --local-shard $MODEL_DIR/model-00047-of-00048.safetensors \
    --local-shard $MODEL_DIR/model-00048-of-00048.safetensors
python3 tools/expert_trace.py --model-dir $MODEL_DIR --corpus corpus/trace_mine.jsonl \
    --engram-dir engram_rows_mine --out results/trace-mine --layers 0-39
python3 tools/expert_stats.py --trace results/trace-mine --out results/keepsets/mine
```

The trace is the expensive step (about a minute per layer). Then run the generation gate on your
domains **and on domains outside your corpus**, and write down both results.
