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

---

# Generation gate — 2026-09-13 17:19

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
| `yaml-anchors` | on | stop | 19,876 | 1,027 | 260 | **FAIL** | reasoning loops 5x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i' |
| `css-card` | on | stop | 21,820 | 2,450 | 333 | **FAIL** | reasoning loops 4x on 'dark) { :root { color-scheme: dark; --card-surface: #181' |
| `en-explain` | on | stop | 5,957 | 1,564 | 125 | PASS | 2 paragraphs, 13 sentences |
| `en-note` | on | stop | 27,870 | 1,171 | 345 | **FAIL** | reasoning loops 17x on 'fix must be proven to eliminate the symptom under the sa' |
| `html-page` | on | stop | 15,136 | 4,226 | 252 | PASS | 58 declarations, 16 functions, 0 empty rules |
| `js-debounce` | on | stop | 28,590 | 1,687 | 401 | **FAIL** | reasoning loops 6x on 'function debounce(fn, wait) { let timerid = null; let la' |
| `tech-explain` | on | stop | 5,725 | 1,462 | 112 | PASS | 2 paragraphs, 9 sentences |
| `ts-groupby` | on | stop | 57,644 | 1,023 | 833 | **FAIL** | reasoning loops 20x on 'function groupby<t, k extends groupablekey<t>>( items: r' |
| `reason-bat-ball` | on | stop | 860 | 528 | 26 | PASS | says 0.05 |
| `reason-machines` | on | stop | 636 | 445 | 16 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 10 runs failed: `yaml-anchors` (on) reasoning loops 5x on 'test: ["cmd", "curl", "-f", "http://localhost/health"] i'; `css-card` (on) reasoning loops 4x on 'dark) { :root { color-scheme: dark; --card-surface: #181'; `en-note` (on) reasoning loops 17x on 'fix must be proven to eliminate the symptom under the sa'; `js-debounce` (on) reasoning loops 6x on 'function debounce(fn, wait) { let timerid = null; let la'; `ts-groupby` (on) reasoning loops 20x on 'function groupby<t, k extends groupablekey<t>>( items: r'

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 18:14

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
| `yaml-anchors` | on | stop | 21,544 | 961 | 271 | **FAIL** | reasoning loops 6x on 'service_role: shared x-common-healthcheck: &common-healt' |
| `css-card` | on | stop | 4,694 | 1,792 | 87 | PASS | 39 declarations, 0 empty rules |
| `en-explain` | on | stop | 12,960 | 1,367 | 223 | PASS | 2 paragraphs, 10 sentences |
| `en-note` | on | stop | 6,851 | 1,044 | 118 | PASS | 3 paragraphs, 7 sentences |
| `html-page` | on | stop | 4,201 | 3,613 | 103 | PASS | 48 declarations, 10 functions, 0 empty rules |
| `js-debounce` | on | stop | 10,788 | 1,756 | 169 | **FAIL** | reasoning loops 3x on 'function debounced(...args) { lastargs = args; lastthis ' |
| `tech-explain` | on | stop | 9,447 | 1,509 | 176 | PASS | 2 paragraphs, 8 sentences |
| `ts-groupby` | on | length | 60,064 | 0 | 843 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 17x on 'must be a property of the element type whose value is a' |
| `reason-bat-ball` | on | stop | 870 | 528 | 26 | PASS | says 0.05 |
| `reason-machines` | on | stop | 630 | 452 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 10 runs failed: `yaml-anchors` (on) reasoning loops 6x on 'service_role: shared x-common-healthcheck: &common-healt'; `js-debounce` (on) reasoning loops 3x on 'function debounced(...args) { lastargs = args; lastthis '; `ts-groupby` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 17x on 'must be a property of the element type whose value is a'

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-14 01:22

| | |
|---|---|
| profile | Frontend |
| topics | html, css, javascript, typescript, english, technical, config, reasoning, reasoning_code, reasoning_design |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code, reasoning_design — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 5,919 | 2,341 | 105 | PASS | 31 keys, anchored |
| `css-card` | on | stop | 12,190 | 1,946 | 192 | PASS | 45 declarations, 0 empty rules |
| `en-explain` | on | stop | 5,099 | 1,035 | 100 | PASS | 2 paragraphs, 11 sentences |
| `en-note` | on | stop | 12,893 | 1,165 | 171 | **FAIL** | repeat: reasoning loops 4x on 'symptom was, and what has to be true before it goes out' — the answer itself is sound (3 paragraphs, 7 sentences) |
| `html-page` | on | stop | 6,982 | 4,152 | 141 | PASS | 62 declarations, 15 functions, 0 empty rules |
| `js-debounce` | on | stop | 15,284 | 1,820 | 230 | **FAIL** | repeat: reasoning loops 3x on 'function debounce(fn, wait) { let timerid = null; let la' — the answer itself is sound (11 callables) |
| `tech-explain` | on | stop | 8,143 | 1,532 | 163 | PASS | 2 paragraphs, 10 sentences |
| `ts-groupby` | on | stop | 11,772 | 1,292 | 185 | **FAIL** | reasoning loops 4x on 'record<string, t[]> = {}; for (const item of items) { co' |
| `reason-bat-ball` | on | stop | 949 | 450 | 26 | PASS | says 0.05 |
| `reason-machines` | on | stop | 956 | 323 | 23 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 10 runs failed: `en-note` (on) repeat: reasoning loops 4x on 'symptom was, and what has to be true before it goes out' — the answer itself is sound (3 paragraphs, 7 sentences); `js-debounce` (on) repeat: reasoning loops 3x on 'function debounce(fn, wait) { let timerid = null; let la' — the answer itself is sound (11 callables); `ts-groupby` (on) reasoning loops 4x on 'record<string, t[]> = {}; for (const item of items) { co'

9 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 1, repeat 2.

This run gated 7 of the profile's 10 topics; reasoning, reasoning_code, reasoning_design carry no prompt, so a pass says nothing about them.
