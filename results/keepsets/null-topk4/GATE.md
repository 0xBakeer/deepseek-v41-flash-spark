# Generation gate — 2026-09-13 12:22

| | |
|---|---|
| profile | ad-hoc |
| topics | english, javascript, html |
| prompts | 3 runs over 3 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 8,192 |
| only | `en-explain,js-debounce,html-page` — a filtered re-run, not a full gate |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `en-explain` | on | stop | 8,553 | 1,131 | 799 | **FAIL** | reasoning loops 3x on 'with 95% hit rate can leave a system slower than no cach' |
| `html-page` | on | stop | 5,180 | 4,107 | 859 | PASS | 60 declarations, 13 functions, 0 empty rules |
| `js-debounce` | on | stop | 12,080 | 1,972 | 1173 | **FAIL** | reasoning loops 5x on '{ const args = lastargs; const ctx = lastthis; lastargs ' |

**Verdict: FAIL** — 2 of 3 runs failed: `en-explain` (on) reasoning loops 3x on 'with 95% hit rate can leave a system slower than no cach'; `js-debounce` (on) reasoning loops 5x on '{ const args = lastargs; const ctx = lastthis; lastargs '
