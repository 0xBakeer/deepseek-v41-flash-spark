# Generation gate — 2026-09-13 22:33

| | |
|---|---|
| profile | World languages |
| topics | english, arabic, chinese, japanese, russian, turkish, translation, reasoning, reasoning_code |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `ar-essay` | on | stop | 12,380 | 921 | 280 | **FAIL** | reasoning loops 5x on 'قد تؤدي ذاكرة تخزين مؤقت ذات نسبة إصابة عالية إلى إبطاء ' |
| `zh-essay` | on | stop | 4,367 | 667 | 154 | **FAIL** | only 34% of the letters are han |
| `en-explain` | on | length | 13,172 | 139 | 214 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 4x on '"ev old entries" no. use "ev old entries"? h. "ev old en' |
| `en-note` | on | stop | 4,108 | 1,297 | 82 | PASS | 3 paragraphs, 8 sentences |
| `ja-essay` | on | length | 5,059 | 139 | 128 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself |
| `ru-essay` | on | stop | 9,513 | 1,708 | 191 | **FAIL** | reasoning loops 3x on 'дороже прямого чтения из локального источника. если кэш ' |
| `xl-en-fr` | on | length | 3,303 | 139 | 56 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 5x on '"the outage began at 14:05" = "elincance? "elincance? le' |
| `tr-essay` | on | stop | 9,619 | 974 | 202 | **FAIL** | reasoning loops 4x on 'haline gelmesidir. yüksek isabet oranı, trafiğin büyük k' |
| `reason-bat-ball` | on | stop | 1,674 | 529 | 40 | PASS | says 0.05 |
| `reason-machines` | on | stop | 1,191 | 421 | 24 | PASS | says 5 minutes |

**Verdict: FAIL** — 7 of 10 runs failed: `ar-essay` (on) reasoning loops 5x on 'قد تؤدي ذاكرة تخزين مؤقت ذات نسبة إصابة عالية إلى إبطاء '; `zh-essay` (on) only 34% of the letters are han; `en-explain` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 4x on '"ev old entries" no. use "ev old entries"? h. "ev old en'; `ja-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; `ru-essay` (on) reasoning loops 3x on 'дороже прямого чтения из локального источника. если кэш '; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 5x on '"the outage began at 14:05" = "elincance? "elincance? le'; `tr-essay` (on) reasoning loops 4x on 'haline gelmesidir. yüksek isabet oranı, trafiğin büyük k'

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 23:14

| | |
|---|---|
| profile | World languages |
| topics | english, arabic, chinese, japanese, russian, turkish, translation, reasoning, reasoning_code |
| prompts | 4 runs over 4 prompts |
| thinking | off |
| reasoning effort | 45 |
| max tokens | 4,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| only | `zh-essay,ja-essay,ar-essay,ru-essay` — a filtered re-run, not a full gate |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `ar-essay` | off | stop | 0 | 1,420 | 43 | PASS | 100% arabic, 2 paragraphs, 6 sentences |
| `zh-essay` | off | stop | 0 | 401 | 25 | PASS | 95% han, 2 paragraphs, 4 sentences |
| `ja-essay` | off | stop | 0 | 736 | 41 | PASS | 69% kana, 2 paragraphs, 10 sentences |
| `ru-essay` | off | stop | 0 | 1,637 | 37 | PASS | 96% cyrillic, 2 paragraphs, 7 sentences |

**Verdict: PASS** — all 4 runs produced sound output.

4 of 4 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 0, repeat 0.

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
