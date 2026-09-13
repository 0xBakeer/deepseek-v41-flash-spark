# Generation gate — 2026-09-13 01:24

| | |
|---|---|
| profile | Backend |
| topics | python, go, java, sql, config, technical, english, reasoning, reasoning_code |
| prompts | 10 runs over 10 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 484 | 13,291 | 230 | **FAIL** | answer loops 3x on 'test: ["cmd", "/healthcheck.sh"] interval: 10s timeout: '; answer has a corrupted run 'L...H' in 'Maybe I output: The main compose with `SERVICE_ROL...Hif sin' |
| `en-explain` | on | length | 4,156 | 139 | 71 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 4x on 'hit rate can leave a system slower than no cache at all.' |
| `en-note` | on | length | 10,730 | 139 | 145 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on '"what has to be true before it goes out again" maybe "wh' |
| `go-handler` | on | stop | 11,169 | 4,123 | 220 | PASS | 115 lines |
| `java-service` | on | length | 17,748 | 139 | 287 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 5x on 'hman hman. hman hman hman. hman hman hman. hman hman hma' |
| `py-walk` | on | stop | 6,640 | 2,197 | 112 | **FAIL** | reasoning loops 3x on 'followlinks=false): for name in filnames: path = os.path' |
| `sql-window` | on | stop | 15,319 | 559 | 201 | **FAIL** | reasoning loops 7x on "customer's three most recent orders with a running total" |
| `tech-explain` | on | stop | 6,024 | 1,622 | 134 | PASS | 2 paragraphs, 11 sentences |
| `reason-bat-ball` | on | stop | 1,395 | 556 | 37 | PASS | says 0.05 |
| `reason-machines` | on | stop | 475 | 480 | 15 | PASS | says 5 minutes |

**Verdict: FAIL** — 6 of 10 runs failed: `yaml-anchors` (on) answer loops 3x on 'test: ["cmd", "/healthcheck.sh"] interval: 10s timeout: '; answer has a corrupted run 'L...H' in 'Maybe I output: The main compose with `SERVICE_ROL...Hif sin'; `en-explain` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 4x on 'hit rate can leave a system slower than no cache at all.'; `en-note` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on '"what has to be true before it goes out again" maybe "wh'; `java-service` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 5x on 'hman hman. hman hman hman. hman hman hman. hman hman hma'; `py-walk` (on) reasoning loops 3x on 'followlinks=false): for name in filnames: path = os.path'; `sql-window` (on) reasoning loops 7x on "customer's three most recent orders with a running total"

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
