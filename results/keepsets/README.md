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

**Each profile therefore ships a `GATE.md` recording which domains were measured, which passed and
which failed.** Read it before choosing. If your traffic is mixed or you cannot predict it, use
`general`.

## Profiles

| profile | trace corpus | intended for |
|---|---|---|
| `general` | web + code + configuration + technical prose + narrative | mixed traffic; the default |
| `code` | HTML, CSS, JavaScript, React, SQL, YAML, shell, Python, technical prose | programming and markup only |
| `prose` | narrative fiction, dialogue, essays | long-form English writing only |

Each directory holds `coverage.json` (the per-layer expert histograms the engine ranks from) and
`GATE.md`. The raw per-layer trace arrays are not shipped; `coverage.json` carries the per-category
histograms, so a keep-set can be rebuilt from it alone.

## Building your own

A profile is only as good as the corpus it was ranked on, and that is the whole lesson of this
directory. To cover a workload, put that workload in the corpus:

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
