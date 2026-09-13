# Generation gate — 2026-09-13 22:05

| | |
|---|---|
| profile | European languages |
| topics | english, german, french, spanish, italian, portuguese, translation, reasoning, reasoning_code |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `en-explain` | on | stop | 5,743 | 1,204 | 116 | PASS | 2 paragraphs, 7 sentences |
| `en-note` | on | stop | 2,998 | 925 | 61 | PASS | 3 paragraphs, 7 sentences |
| `fr-essay` | on | length | 57,414 | 0 | 754 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 164x on '« coérence » ? je vais utiliser « coérence » ? je' |
| `de-essay` | on | length | 11,378 | 139 | 223 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on '= "trotz" maybe. m. "trotzdem" = "trotz" = "trotz" maybe' |
| `it-essay` | on | stop | 23,751 | 1,472 | 433 | **FAIL** | reasoning loops 5x on 'fallimento paga il costo del lookup in cache più quello ' |
| `pt-essay` | on | stop | 6,127 | 1,635 | 143 | PASS | 2 paragraphs, 9 sentences, 4 markers |
| `es-essay` | on | stop | 3,670 | 1,802 | 104 | **FAIL** | only 3 of 7 language markers (para, porque, cuando) |
| `xl-en-fr` | on | stop | 1,483 | 1,560 | 49 | PASS | 4 paragraphs, 10 sentences, 4 markers |
| `reason-bat-ball` | on | stop | 589 | 514 | 23 | PASS | says 0.05 |
| `reason-machines` | on | stop | 779 | 422 | 19 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `fr-essay` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 164x on '« coérence » ? je vais utiliser « coérence » ? je'; `de-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on '= "trotz" maybe. m. "trotzdem" = "trotz" = "trotz" maybe'; `it-essay` (on) reasoning loops 5x on 'fallimento paga il costo del lookup in cache più quello '; `es-essay` (on) only 3 of 7 language markers (para, porque, cuando)

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 23:07

| | |
|---|---|
| profile | European languages |
| topics | english, german, french, spanish, italian, portuguese, translation, reasoning, reasoning_code |
| prompts | 4 runs over 4 prompts |
| thinking | off |
| reasoning effort | 45 |
| max tokens | 4,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| only | `fr-essay,de-essay,es-essay,it-essay` — a filtered re-run, not a full gate |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `fr-essay` | off | stop | 0 | 1,627 | 36 | **FAIL** | only 1 of 7 language markers (qui) |
| `de-essay` | off | stop | 0 | 1,766 | 39 | PASS | 2 paragraphs, 8 sentences, 5 markers |
| `it-essay` | off | stop | 0 | 1,535 | 36 | PASS | 2 paragraphs, 8 sentences, 5 markers |
| `es-essay` | off | stop | 0 | 1,901 | 37 | **FAIL** | only 3 of 7 language markers (pero, porque, entre) |

**Verdict: FAIL** — 2 of 4 runs failed: `fr-essay` (off) only 1 of 7 language markers (qui); `es-essay` (off) only 3 of 7 language markers (pero, porque, entre)

2 of 4 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 2, repeat 0.

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-14 01:54

| | |
|---|---|
| profile | European languages |
| topics | english, german, french, spanish, italian, portuguese, translation, reasoning, reasoning_code, reasoning_lang |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code, reasoning_lang — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `en-explain` | on | stop | 24,649 | 1,172 | 355 | **FAIL** | repeat: reasoning loops 50x on 'a cache is not just a fast table; it is an extra' — the answer itself is sound (2 paragraphs, 13 sentences) |
| `en-note` | on | stop | 5,883 | 1,067 | 105 | PASS | 3 paragraphs, 10 sentences |
| `fr-essay` | on | length | 12,265 | 139 | 196 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 5x on 'cache très efficace peut ralentir un système parce que s' |
| `de-essay` | on | length | 59,326 | 0 | 539 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 154x on 'absatz 1: ein cache mit hoher trefferrate wird oft als b' |
| `it-essay` | on | length | 8,927 | 139 | 166 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 18x on "e e e e e the cache's own e e e e" |
| `pt-essay` | on | length | 2,779 | 139 | 65 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself |
| `es-essay` | on | length | 7,043 | 139 | 148 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself |
| `xl-en-fr` | on | length | 3,095 | 139 | 49 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 11x on '"nous l\'avons appenu d\'un client" no. "nous l\'avons appe' |
| `reason-bat-ball` | on | stop | 1,907 | 374 | 39 | PASS | says 0.05 |
| `reason-machines` | on | stop | 522 | 484 | 15 | PASS | says 5 minutes |

**Verdict: FAIL** — 7 of 10 runs failed: `en-explain` (on) repeat: reasoning loops 50x on 'a cache is not just a fast table; it is an extra' — the answer itself is sound (2 paragraphs, 13 sentences); `fr-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 5x on 'cache très efficace peut ralentir un système parce que s'; `de-essay` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 154x on 'absatz 1: ein cache mit hoher trefferrate wird oft als b'; `it-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 18x on "e e e e e the cache's own e e e e"; `pt-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; `es-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 11x on '"nous l\'avons appenu d\'un client" no. "nous l\'avons appe'

4 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 1, guard 5, corrupt 0, content 0, repeat 1.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_lang carry no prompt, so a pass says nothing about them.
