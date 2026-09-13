# Generation gate — 2026-09-13 03:25

| | |
|---|---|
| profile | Medicine |
| topics | medical, academic, technical, english, reasoning, reasoning_code |
| prompts | 7 runs over 7 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 1,987 | 1,546 | 52 | PASS | 4 paragraphs, 8 sentences |
| `en-explain` | on | stop | 8,079 | 1,011 | 143 | **FAIL** | reasoning loops 3x on 'with 95% hit rate can leave system slower than no cache ' |
| `en-note` | on | stop | 4,128 | 857 | 74 | PASS | 3 paragraphs, 6 sentences |
| `med-explain` | on | length | 3,352 | 139 | 56 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 14x on '"acatic"? the "acatic" is "acatic"? the "acatic" is "aca' |
| `tech-explain` | on | stop | 5,300 | 950 | 117 | PASS | 2 paragraphs, 6 sentences |
| `reason-bat-ball` | on | stop | 1,269 | 488 | 30 | PASS | says 0.05 |
| `reason-machines` | on | stop | 1,111 | 259 | 20 | PASS | says 5 minutes |

**Verdict: FAIL** — 2 of 7 runs failed: `en-explain` (on) reasoning loops 3x on 'with 95% hit rate can leave system slower than no cache '; `med-explain` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 14x on '"acatic"? the "acatic" is "acatic"? the "acatic" is "aca'

This run gated 4 of the profile's 6 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
