# Profile `prose` — NOT YET GATED

Trace corpus: `trace_corpus_v3` only (six narrative fiction and dialogue pieces, an essay; 95
sequences, 17,704 tokens).

**This profile has not been run through the generation gate on its own.** What is known comes from
building the keep-set it is derived from: ranked on this corpus alone, long English prose improves
markedly (distinct-token ratio 0.54 against 0.12 for the code-only corpus) and **HTML collapses**
(0.04). Expect it to be better than `general` for long-form English and unusable for markup.

The measurements below belong to `general` and are kept only as the comparison baseline until this
profile is gated.

---

# (baseline: profile `general`, measured 2026-09-12)

Trace corpus: `trace_corpus_v2` (two HTML pages, a stylesheet, an ES module, a React component, a
SQL schema, a Kubernetes manifest, a deployment script, Python, an incident write-up) +
`trace_corpus_v3` (six narrative fiction and dialogue pieces, an essay). 190 sequences, 36,250
tokens. Configuration: `PRUNE_KEEP=0.39 EXPERT_FORMAT=cb3 ARENA_GB=88`.

Gate: greedy or sampled generation, 900-2,000 tokens per prompt, judged on repetition (most-repeated
line < 30 % of lines, distinct-token ratio > 0.25) and, where the generation finished on its own,
structural validity.

## Passes

| domain | tok/s | distinct-token ratio |
|---|---|---|
| single-file HTML game | 37.3 | 0.59 |
| JavaScript module | 32.3 | 0.45 |
| SQL schema and query | 31.8 | 0.51 |
| Rust module | 29.8 | 0.38 |
| Python module | 28.6 | 0.59 |
| Dockerfile | 23.2 | 0.66 |
| German essay (600 words) | 14.3 | 0.61 |
| LaTeX document | 22.8 | 0.67 |
| mathematical proof | 24.3 | 0.38 |
| English story / essay | 14.2 | 0.50 |
| poetry (rhymed, 12 stanzas) | 11.9 | 0.61 |

## Fails — do not use this profile for these

| domain | what happens |
|---|---|
| **Arabic** | not Arabic: a collage of Polish, Portuguese and Romanian fragments. Distinct-token ratio 0.08 |
| **Translation into French or Spanish** | answers in Italian and repeats a phrase until the cap |
| **Formal legal register** | degenerates within a few hundred tokens (`LICENSE CLAINT`, ratio 0.06) |

The corpus contains no Arabic, no Romance-language text and no legal register, so the experts those
domains need were ranked cold and dropped. This is the expected failure mode of a pruned keep-set,
not a defect in the format or the kernels: the same budget ranked on a corpus containing those
domains restores them.
