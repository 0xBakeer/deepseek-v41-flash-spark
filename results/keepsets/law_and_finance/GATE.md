# Generation gate — 2026-09-13 03:48

| | |
|---|---|
| profile | Law and finance |
| topics | legal, finance, academic, english, reasoning, reasoning_code |
| prompts | 7 runs over 7 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 15,758 | 1,746 | 236 | **FAIL** | reasoning loops 3x on 'hypothesis was that longer code review latency would pos'; reasoning has a corrupted run 's...3' in 'Should I include "sample" in methods paragraph? I have "1,82' |
| `en-explain` | on | stop | 15,864 | 1,408 | 268 | **FAIL** | reasoning loops 3x on '95% hit rate can leave a system slower than no cache at' |
| `en-note` | on | stop | 6,949 | 1,241 | 121 | PASS | 3 paragraphs, 10 sentences |
| `fin-explain` | on | stop | 7,818 | 1,073 | 137 | PASS | 2 paragraphs, 8 sentences |
| `legal-clause` | on | length | 17,753 | 139 | 278 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 34x on 'i i. i i. i i. i i. i i. i i.' |
| `reason-bat-ball` | on | stop | 1,063 | 530 | 28 | PASS | says 0.05 |
| `reason-machines` | on | stop | 1,761 | 583 | 36 | PASS | says 5 minutes |

**Verdict: FAIL** — 3 of 7 runs failed: `acad-abstract` (on) reasoning loops 3x on 'hypothesis was that longer code review latency would pos'; reasoning has a corrupted run 's...3' in 'Should I include "sample" in methods paragraph? I have "1,82'; `en-explain` (on) reasoning loops 3x on '95% hit rate can leave a system slower than no cache at'; `legal-clause` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 34x on 'i i. i i. i i. i i. i i. i i.'

This run gated 4 of the profile's 6 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-13 19:44

| | |
|---|---|
| profile | Law and finance |
| topics | legal, finance, academic, english, reasoning, reasoning_code |
| prompts | 7 runs over 7 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `acad-abstract` | on | stop | 6,938 | 1,639 | 108 | **FAIL** | reasoning loops 3x on 'as confirmed bug issues per 1,000 lines of changed code ' |
| `en-explain` | on | stop | 5,944 | 1,330 | 116 | PASS | 2 paragraphs, 16 sentences |
| `en-note` | on | stop | 4,126 | 1,077 | 73 | PASS | 3 paragraphs, 5 sentences |
| `fin-explain` | on | stop | 10,034 | 1,372 | 157 | PASS | 2 paragraphs, 8 sentences |
| `legal-clause` | on | stop | 19,658 | 2,693 | 281 | **FAIL** | reasoning loops 5x on 'paid by customer to supplier under this agreement during' |
| `reason-bat-ball` | on | stop | 1,221 | 526 | 31 | PASS | says 0.05 |
| `reason-machines` | on | stop | 857 | 439 | 20 | PASS | says 5 minutes |

**Verdict: FAIL** — 2 of 7 runs failed: `acad-abstract` (on) reasoning loops 3x on 'as confirmed bug issues per 1,000 lines of changed code '; `legal-clause` (on) reasoning loops 5x on 'paid by customer to supplier under this agreement during'

This run gated 4 of the profile's 6 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
