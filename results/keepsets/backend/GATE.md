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

---

# Generation gate — 2026-09-13 18:42

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
| `yaml-anchors` | on | stop | 19,756 | 889 | 258 | **FAIL** | reasoning loops 7x on 'block of environment variables and one healthcheck defin' |
| `en-explain` | on | stop | 15,673 | 1,849 | 278 | **FAIL** | reasoning loops 3x on '"why a cache with a 95 % hit rate can leave a' |
| `en-note` | on | stop | 1,081 | 1,089 | 36 | PASS | 3 paragraphs, 6 sentences |
| `go-handler` | on | length | 17,261 | 139 | 237 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on "i'll write `timeout time.dduration`? no. i'll write `tim" |
| `java-service` | on | stop | 11,757 | 4,336 | 213 | PASS | 88 lines |
| `py-walk` | on | length | 6,931 | 139 | 94 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 19x on 'let me be careful. let me be careful. let me be careful.' |
| `sql-window` | on | stop | 6,658 | 923 | 98 | **FAIL** | reasoning loops 3x on "each customer's three most recent orders with a running " |
| `tech-explain` | on | stop | 5,403 | 1,805 | 119 | PASS | 2 paragraphs, 11 sentences |
| `reason-bat-ball` | on | stop | 732 | 533 | 26 | PASS | says 0.05 |
| `reason-machines` | on | stop | 544 | 273 | 12 | PASS | says 5 minutes |

**Verdict: FAIL** — 5 of 10 runs failed: `yaml-anchors` (on) reasoning loops 7x on 'block of environment variables and one healthcheck defin'; `en-explain` (on) reasoning loops 3x on '"why a cache with a 95 % hit rate can leave a'; `go-handler` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on "i'll write `timeout time.dduration`? no. i'll write `tim"; `py-walk` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 19x on 'let me be careful. let me be careful. let me be careful.'; `sql-window` (on) reasoning loops 3x on "each customer's three most recent orders with a running "

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.

---

# Generation gate — 2026-09-14 03:36

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
| `yaml-anchors` | on | stop | 19,348 | 1,186 | 258 | **FAIL** | repeat: reasoning loops 4x on 'one block of environment variables and one healthcheck d' — the answer itself is sound (30 keys, anchored) |
| `en-explain` | on | stop | 8,982 | 1,400 | 145 | **FAIL** | repeat: reasoning loops 3x on '95% hit rate can leave a system slower than no cache at' — the answer itself is sound (2 paragraphs, 7 sentences) |
| `en-note` | on | stop | 12,368 | 961 | 182 | **FAIL** | repeat: reasoning loops 4x on 'i’m rolling back the release that went out this afternoo' — the answer itself is sound (4 paragraphs, 5 sentences) |
| `go-handler` | on | stop | 18,474 | 2,083 | 252 | **FAIL** | repeat: reasoning loops 3x on 'http.statusunprocessableentity, "validation_failed", err' — the answer itself is sound (77 lines) |
| `java-service` | on | stop | 26,964 | 1,259 | 402 | **FAIL** | repeat: reasoning loops 5x on '} private static final class entry<v> { final v value; f' — the answer itself is sound (39 lines) |
| `py-walk` | on | stop | 38,016 | 1,290 | 547 | **FAIL** | repeat: reasoning loops 3x on 'for dirpath, dirnames, filenames in os.wwalk(root, onerr' — the answer itself is sound (2 defs) |
| `sql-window` | on | stop | 18,734 | 623 | 255 | **FAIL** | repeat: reasoning loops 8x on "customer's three most recent orders with a running total" — the answer itself is sound (select, join, window) |
| `tech-explain` | on | stop | 9,284 | 1,198 | 166 | PASS | 2 paragraphs, 6 sentences |
| `reason-bat-ball` | on | stop | 590 | 496 | 21 | PASS | says 0.05 |
| `reason-machines` | on | stop | 657 | 413 | 16 | PASS | says 5 minutes |

**Verdict: FAIL** — 7 of 10 runs failed: `yaml-anchors` (on) repeat: reasoning loops 4x on 'one block of environment variables and one healthcheck d' — the answer itself is sound (30 keys, anchored); `en-explain` (on) repeat: reasoning loops 3x on '95% hit rate can leave a system slower than no cache at' — the answer itself is sound (2 paragraphs, 7 sentences); `en-note` (on) repeat: reasoning loops 4x on 'i’m rolling back the release that went out this afternoo' — the answer itself is sound (4 paragraphs, 5 sentences); `go-handler` (on) repeat: reasoning loops 3x on 'http.statusunprocessableentity, "validation_failed", err' — the answer itself is sound (77 lines); `java-service` (on) repeat: reasoning loops 5x on '} private static final class entry<v> { final v value; f' — the answer itself is sound (39 lines); `py-walk` (on) repeat: reasoning loops 3x on 'for dirpath, dirnames, filenames in os.wwalk(root, onerr' — the answer itself is sound (2 defs); `sql-window` (on) repeat: reasoning loops 8x on "customer's three most recent orders with a running total" — the answer itself is sound (select, join, window)

10 of 10 finished a correct answer (strict passes plus repeat-only misses); misses by kind: think-exit 0, guard 0, corrupt 0, content 0, repeat 7.

This run gated 7 of the profile's 9 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
