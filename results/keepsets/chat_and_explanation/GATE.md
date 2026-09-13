# Generation gate — 2026-09-13 03:12

| | |
|---|---|
| profile | Chat and explanation |
| topics | english, technical, academic, journalism, translation, reasoning, reasoning_code |
| prompts | 8 runs over 8 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 1,943 | 2,437 | 62 | PASS | 4 paragraphs, 18 sentences |
| `en-explain` | on | stop | 22,684 | 1,411 | 349 | **FAIL** | reasoning loops 10x on 'cache with a 95 % hit rate can leave a system slower' |
| `en-note` | on | stop | 4,531 | 959 | 80 | PASS | 3 paragraphs, 7 sentences |
| `news-lede` | on | stop | 13,294 | 1,327 | 202 | PASS | 4 paragraphs, 12 sentences |
| `tech-explain` | on | stop | 15,363 | 1,556 | 302 | **FAIL** | reasoning loops 3x on 'if the bottleneck for this flow is the upgraded link and' |
| `xl-en-fr` | on | length | 11,574 | 139 | 171 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 10x on 'let\'s use "avout"? hmm. i need to produce translation. l' |
| `reason-bat-ball` | on | stop | 1,202 | 470 | 33 | PASS | says 0.05 |
| `reason-machines` | on | stop | 1,464 | 361 | 27 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 8 runs failed: `en-explain` (on) reasoning loops 10x on 'cache with a 95 % hit rate can leave a system slower'; `tech-explain` (on) reasoning loops 3x on 'if the bottleneck for this flow is the upgraded link and'; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 10x on 'let\'s use "avout"? hmm. i need to produce translation. l'

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 19:08

| | |
|---|---|
| profile | Chat and explanation |
| topics | english, technical, academic, journalism, translation, reasoning, reasoning_code |
| prompts | 8 runs over 8 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 10,342 | 1,882 | 166 | **FAIL** | reasoning loops 3x on 'hypothesis that code review latency (hours from review r' |
| `en-explain` | on | stop | 3,659 | 1,070 | 72 | PASS | 2 paragraphs, 10 sentences |
| `en-note` | on | stop | 2,416 | 1,003 | 52 | PASS | 3 paragraphs, 6 sentences |
| `news-lede` | on | stop | 8,292 | 1,228 | 131 | PASS | 4 paragraphs, 8 sentences |
| `tech-explain` | on | stop | 12,673 | 1,885 | 239 | PASS | 2 paragraphs, 13 sentences |
| `xl-en-fr` | on | length | 49,081 | 0 | 554 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 120x on 'let me just think: "outage" = "inccident" in french? act' |
| `reason-bat-ball` | on | stop | 1,221 | 610 | 35 | PASS | says 0.05 |
| `reason-machines` | on | stop | 699 | 360 | 18 | PASS | says 5 minutes |

**Verdict: FAIL** — 2 of 8 runs failed: `acad-abstract` (on) reasoning loops 3x on 'hypothesis that code review latency (hours from review r'; `xl-en-fr` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 120x on 'let me just think: "outage" = "inccident" in french? act'

This run gated 5 of the profile's 7 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
