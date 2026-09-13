# Generation gate — 2026-09-13 00:55

| | |
|---|---|
| profile | Frontend |
| topics | html, css, javascript, typescript, english, technical, config, reasoning, reasoning_code |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 1,561 | 2,001 | 56 | PASS | 32 keys, anchored |
| `css-card` | on | stop | 257 | 3,157 | 46 | **FAIL** | missing prefers-color-scheme |
| `en-explain` | on | length | 67,513 | 0 | 738 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 68x on 'with no cache at all, every request goes to origin; if o' |
| `en-note` | on | stop | 9,451 | 1,293 | 145 | PASS | 3 paragraphs, 7 sentences |
| `html-page` | on | stop | 22,188 | 7,307 | 419 | PASS | 93 declarations, 24 functions, 0 empty rules |
| `js-debounce` | on | stop | 36,653 | 1,066 | 571 | **FAIL** | reasoning loops 9x on 'wait) { let timer = null; let lastargs = null; let lastt' |
| `tech-explain` | on | stop | 9,046 | 1,422 | 176 | PASS | 2 paragraphs, 8 sentences |
| `ts-groupby` | on | length | 59,295 | 0 | 911 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 15x on 'must be a property of the element type whose value is a' |
| `reason-bat-ball` | on | stop | 1,141 | 548 | 32 | PASS | says 0.05 |
| `reason-machines` | on | stop | 751 | 381 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 4 of 10 runs failed: `css-card` (on) missing prefers-color-scheme; `en-explain` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 68x on 'with no cache at all, every request goes to origin; if o'; `js-debounce` (on) reasoning loops 9x on 'wait) { let timer = null; let lastargs = null; let lastt'; `ts-groupby` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 15x on 'must be a property of the element type whose value is a'

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

## 2026-09-13 11:00 — DSV41_PRUNE_MODE=drop, same nine topics, keep 0.36, 256k, thinking on

A displaced pick dropped (weight 0, survivors renormalised) instead of substituted. Six prompts run
before the gate was stopped; every one failed, and `html-page` — which passes under `substitute` on
this profile and on the eight-topic set — broke down into a repeated syllable. Drop is top-*k′*
routing with about four survivors plus a shared-expert-only fallthrough on roughly one token in
twenty; this model is worse served that way than by six experts of which two are wrong. Recorded
as a negative result; the default stays `substitute`.

```
yaml-anchors       on    stop      34,860   1,300    527 FAIL reasoning loops 13x on '"write a docker-compose file with three services that sh'
css-card           on    length       377   4,947     85 FAIL finish_reason 'length'; answer loops 6x on '- `:root` variables. - `:root` variables. - 
en-explain         on    length    20,516     139    288 FAIL finish_reason 'length'; server cut the generation off for repeating itself; reasoning lo
en-note            on    stop      11,084     831    172 FAIL reasoning loops 4x on '"before it goes out again, it must be true that the fix'
html-page          on    length    14,237     139    294 FAIL finish_reason 'length'; server cut the generation off for repeating itself; reasoning lo
js-debounce        on    stop       7,570  22,234    442 FAIL reasoning loops 4x on 'timerid = settimeout(() => { timerid = null; if (pending'; answer
```
(stopped after six of twelve — the remaining rows would not have changed the verdict)
