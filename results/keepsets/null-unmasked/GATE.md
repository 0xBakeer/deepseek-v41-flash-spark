# Generation gate — 2026-09-13 10:38

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
| `en-explain` | on | stop | 6,425 | 1,603 | 1290 | PASS | 2 paragraphs, 9 sentences |
| `html-page` | on | stop | 4,284 | 5,574 | 1549 | PASS | 71 declarations, 15 functions, 0 empty rules |
| `js-debounce` | on | stop | 3,971 | 1,782 | 1008 | PASS | 12 callables |

**Verdict: PASS** — all 3 runs produced sound output.
