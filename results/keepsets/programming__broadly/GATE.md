# Generation gate — 2026-09-13 02:48

| | |
|---|---|
| profile | Programming, broadly |
| topics | python, javascript, typescript, html, css, go, rust, cpp, java, sql, config, technical, english, reasoning, reasoning_code |
| prompts | 16 runs over 16 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `yaml-anchors` | on | stop | 2,476 | 1,666 | 66 | PASS | 29 keys, anchored |
| `cpp-ringbuffer` | on | stop | 28,779 | 3,710 | 498 | **FAIL** | reasoning loops 3x on '``` bool push(const t& value) { ... } bool push(t&& valu'; reasoning has a corrupted run '6...1' in 'H p 1..5, overflow 6...10 but only if returns false each 6 e' |
| `css-card` | on | stop | 4,659 | 1,498 | 99 | **FAIL** | missing prefers-color-scheme |
| `en-explain` | on | stop | 19,048 | 802 | 304 | **FAIL** | reasoning loops 22x on '95% hit rate can leave a system slower than no cache at' |
| `en-note` | on | length | 5,588 | 139 | 79 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 15x on '"hi teammate". or maybe "hi teammate". or maybe "hi team' |
| `go-handler` | on | length | 79,744 | 0 | 555 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 80x on 'i need to know that the user wants a go http handler.' |
| `html-page` | on | stop | 11,335 | 26,228 | 529 | **FAIL** | answer loops 5x on 'return winninglines.s.s(function(line) { return line.e.e' |
| `java-service` | on | length | 16,369 | 17,937 | 463 | **FAIL** | finish_reason 'length'; reasoning loops 5x on 'c c c c" "c c c c c" "c c c'; answer loops 31x on 'final. i. final. i. final. i. final. i. final. i. final.' |
| `js-debounce` | on | stop | 10,635 | 1,639 | 166 | **FAIL** | reasoning loops 6x on '= () => { if (timer !== null) cleartimeout(timer); timer' |
| `py-walk` | on | stop | 26,400 | 2,227 | 454 | **FAIL** | reasoning loops 6x on 'for unit in units: if size < 1024.0: return f"{size:.1f}' |
| `rust-parse` | on | length | 137 | 29,811 | 373 | **FAIL** | finish_reason 'length'; answer loops 5x on 'while parsing `key=value` lines. #[derive(debug, partial' |
| `sql-window` | on | length | 26,909 | 139 | 423 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on 'as ( select c.id as customer_id, c.name, o.id as order_i' |
| `tech-explain` | on | stop | 9,112 | 1,290 | 184 | PASS | 2 paragraphs, 11 sentences |
| `ts-groupby` | on | stop | 30,825 | 551 | 500 | **FAIL** | reasoning loops 16x on '``` function groupby<t, k extends keyof t>( items: reado' |
| `reason-bat-ball` | on | stop | 4,525 | 692 | 98 | PASS | says 0.05 |
| `reason-machines` | on | stop | 381 | 478 | 14 | PASS | says 5 minutes |

**Verdict: FAIL** — 12 of 16 runs failed: `cpp-ringbuffer` (on) reasoning loops 3x on '``` bool push(const t& value) { ... } bool push(t&& valu'; reasoning has a corrupted run '6...1' in 'H p 1..5, overflow 6...10 but only if returns false each 6 e'; `css-card` (on) missing prefers-color-scheme; `en-explain` (on) reasoning loops 22x on '95% hit rate can leave a system slower than no cache at'; `en-note` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 15x on '"hi teammate". or maybe "hi teammate". or maybe "hi team'; `go-handler` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 80x on 'i need to know that the user wants a go http handler.'; `html-page` (on) answer loops 5x on 'return winninglines.s.s(function(line) { return line.e.e'; `java-service` (on) finish_reason 'length'; reasoning loops 5x on 'c c c c" "c c c c c" "c c c'; answer loops 31x on 'final. i. final. i. final. i. final. i. final. i. final.'; `js-debounce` (on) reasoning loops 6x on '= () => { if (timer !== null) cleartimeout(timer); timer'; `py-walk` (on) reasoning loops 6x on 'for unit in units: if size < 1024.0: return f"{size:.1f}'; `rust-parse` (on) finish_reason 'length'; answer loops 5x on 'while parsing `key=value` lines. #[derive(debug, partial'; `sql-window` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 12x on 'as ( select c.id as customer_id, c.name, o.id as order_i'; `ts-groupby` (on) reasoning loops 16x on '``` function groupby<t, k extends keyof t>( items: reado'

This run gated 13 of the profile's 15 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
