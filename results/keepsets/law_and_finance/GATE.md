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
