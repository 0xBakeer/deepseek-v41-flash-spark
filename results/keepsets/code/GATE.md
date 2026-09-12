# Profile `code` — NOT YET GATED

Trace corpus: `trace_corpus_v2` only (HTML, CSS, JavaScript, React, SQL, YAML, shell, Python,
technical prose; 95 sequences, 18,546 tokens).

**This profile has not been run through the generation gate on its own.** What is known comes from
building the keep-set it is derived from: ranked on this corpus alone, HTML and code pass and long
English prose degenerates into repeated phrasing (distinct-token ratio 0.12 against 0.50 for the
general profile). Expect it to be better than `general` on markup and code and worse on everything
else. Do not use it for prose, other languages or legal register.

The pass/fail table that stood below this line was a verbatim copy of `general`'s, not a
measurement of `code`, so on 2026-09-12 it was deleted rather than superseded: read
[`../general/GATE.md`](../general/GATE.md) for the baseline it was copying, and take nothing in it
as a result for this profile.
