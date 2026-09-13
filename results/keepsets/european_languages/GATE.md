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
