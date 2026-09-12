# Profile `prose` — NOT YET GATED

Trace corpus: `trace_corpus_v3` only (six narrative fiction and dialogue pieces, an essay; 95
sequences, 17,704 tokens).

**This profile has not been run through the generation gate on its own.** What is known comes from
building the keep-set it is derived from: ranked on this corpus alone, long English prose improves
markedly (distinct-token ratio 0.54 against 0.12 for the code-only corpus) and **HTML collapses**
(0.04). Expect it to be better than `general` for long-form English and unusable for markup.

The pass/fail table that stood below this line was a verbatim copy of `general`'s, not a
measurement of `prose`, so on 2026-09-12 it was deleted rather than superseded: read
[`../general/GATE.md`](../general/GATE.md) for the baseline it was copying, and take nothing in it
as a result for this profile.
