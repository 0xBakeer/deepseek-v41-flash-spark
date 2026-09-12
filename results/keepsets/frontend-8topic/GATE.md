# Generation gate — 2026-09-12 23:59

| | |
|---|---|
| profile | ad-hoc |
| topics | english, html, python, reasoning, reasoning_code, css, javascript, typescript |
| prompts | 9 runs over 9 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `css-card` | on | stop | 14,111 | 2,180 | 245 | **FAIL** | missing prefers-color-scheme |
| `en-explain` | on | stop | 18,234 | 1,179 | 265 | **FAIL** | reasoning loops 17x on '95% hit rate can leave a system slower than no cache at' |
| `en-note` | on | stop | 7,958 | 943 | 124 | **FAIL** | reasoning loops 3x on '"before it goes out again, the following has to be true:' |
| `html-page` | on | stop | 5,154 | 5,473 | 150 | PASS | 79 declarations, 12 functions, 0 empty rules |
| `js-debounce` | on | length | 24,486 | 15,003 | 526 | **FAIL** | finish_reason 'length'; reasoning loops 12x on '= () => { if (timerid !== null) { cleartimeout(ttimerid)'; answer loops 36x on '= () => { if (timerid !== null) { cleartimeout(ttimerid)' |
| `py-walk` | on | stop | 49,094 | 1,477 | 723 | **FAIL** | reasoning loops 10x on 'except oenerror as err: if onerror is not none: onerror(' |
| `ts-groupby` | on | stop | 14,289 | 871 | 223 | **FAIL** | reasoning loops 6x on 'key must be a property of the element type whose value i' |
| `reason-bat-ball` | on | stop | 871 | 500 | 27 | PASS | says 0.05 |
| `reason-machines` | on | stop | 912 | 550 | 25 | PASS | says 5 minutes |

**Verdict: FAIL** — 6 of 9 runs failed: `css-card` (on) missing prefers-color-scheme; `en-explain` (on) reasoning loops 17x on '95% hit rate can leave a system slower than no cache at'; `en-note` (on) reasoning loops 3x on '"before it goes out again, the following has to be true:'; `js-debounce` (on) finish_reason 'length'; reasoning loops 12x on '= () => { if (timerid !== null) { cleartimeout(ttimerid)'; answer loops 36x on '= () => { if (timerid !== null) { cleartimeout(ttimerid)'; `py-walk` (on) reasoning loops 10x on 'except oenerror as err: if onerror is not none: onerror('; `ts-groupby` (on) reasoning loops 6x on 'key must be a property of the element type whose value i'

This run gated 6 of the profile's 8 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
