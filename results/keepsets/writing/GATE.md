# Generation gate — 2026-09-13 06:07

| | |
|---|---|
| profile | Writing |
| topics | english, journalism, marketing, academic, translation, reasoning, reasoning_code |
| prompts | 8 runs over 8 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 2,104 | 2,231 | 64 | PASS | 4 paragraphs, 11 sentences |
| `en-explain` | on | stop | 13,108 | 1,233 | 221 | **FAIL** | reasoning loops 5x on '95% hit rate can leave a system slower than no cache at' |
| `en-note` | on | stop | 16,842 | 1,246 | 236 | **FAIL** | reasoning loops 5x on 'be reviewed for backward compatibility and tested agains' |
| `news-lede` | on | length | 63,833 | 0 | 595 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 255x on 'i can say "the hospital said the outage lasted for a day' |
| `copy-landing` | on | stop | 12,345 | 758 | 195 | **FAIL** | reasoning loops 3x on 'reports indexes for which no recorded query plan used th' |
| `xl-en-fr` | on | length | 1,301 | 139 | 21 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 9x on 'let\'s think. "les non?" let\'s do: "le? " no. the outage ' |
| `reason-bat-ball` | on | stop | 897 | 518 | 27 | PASS | says 0.05 |
| `reason-machines` | on | stop | 864 | 495 | 22 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 8 runs failed: `en-explain` (on) reasoning loops 5x on '95% hit rate can leave a system slower than no cache at'; `en-note` (on) reasoning loops 5x on 'be reviewed for backward compatibility and tested agains'; `news-lede` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 255x on 'i can say "the hospital said the outage lasted for a day'; `copy-landing` (on) reasoning loops 3x on 'reports indexes for which no recorded query plan used th'; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 9x on 'let\'s think. "les non?" let\'s do: "le? " no. the outage '

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 20:45

| | |
|---|---|
| profile | Writing |
| topics | english, journalism, marketing, academic, translation, reasoning, reasoning_code |
| prompts | 8 runs over 8 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 14,620 | 2,362 | 214 | PASS | 4 paragraphs, 13 sentences |
| `en-explain` | on | length | 8,298 | 139 | 138 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 4x on 'use "evies" -> "evies" no. i\'ll write "evies" -> "evies"' |
| `en-note` | on | stop | 3,893 | 1,145 | 74 | PASS | 3 paragraphs, 7 sentences |
| `news-lede` | on | stop | 12,583 | 1,185 | 188 | **FAIL** | reasoning loops 3x on 'to close its elective surgery list for a day, hospital o' |
| `copy-landing` | on | stop | 5,293 | 936 | 96 | PASS | 4 paragraphs, 10 sentences |
| `xl-en-fr` | on | length | 11,397 | 139 | 154 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 7x on '"because the health check tested the process and not the' |
| `reason-bat-ball` | on | stop | 1,155 | 601 | 33 | PASS | says 0.05 |
| `reason-machines` | on | stop | 616 | 384 | 17 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 8 runs failed: `en-explain` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 4x on 'use "evies" -> "evies" no. i\'ll write "evies" -> "evies"'; `news-lede` (on) reasoning loops 3x on 'to close its elective surgery list for a day, hospital o'; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 7x on '"because the health check tested the process and not the'

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
