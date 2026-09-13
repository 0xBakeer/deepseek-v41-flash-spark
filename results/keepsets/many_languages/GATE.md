# Generation gate — 2026-09-13 05:40

| | |
|---|---|
| profile | Many languages |
| topics | english, german, french, spanish, italian, portuguese, arabic, chinese, japanese, russian, turkish, translation, reasoning, reasoning_code |
| prompts | 15 runs over 15 prompts |
| thinking | on |
| reasoning effort | 45 |
| max tokens | 16,000 |
| server | `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, max_model_len 262,144 |
| no prompts for | reasoning, reasoning_code — these topics were NOT gated |

| prompt | thinking | finish | reasoning | answer | s | | why |
|---|---|---|---|---|---|---|---|
| `ar-essay` | on | length | 9,874 | 139 | 153 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 16x on 'the buffer pool is high. the buffer pool is high. the bu' |
| `zh-essay` | on | length | 5,085 | 139 | 91 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on '"cache hit rate high" but "cache hit rate high" but "sys' |
| `en-explain` | on | stop | 20,740 | 1,165 | 338 | **FAIL** | reasoning loops 12x on '95% hit rate can leave a system slower than no cache at' |
| `en-note` | on | length | 70,070 | 0 | 518 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 539x on 'i need to explain the checks that must hold before re sh' |
| `fr-essay` | on | length | 30,051 | 0 | 1238 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning has a corrupted run 'k...p' in 'HslI m udra p; ("N°I R2?? ox v a sh Svpk...pa" sy' |
| `de-essay` | on | length | 12,821 | 139 | 238 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 3x on 'h. h? cache locks? h. h? at high hit rate, many requests' |
| `it-essay` | on | length | 4,002 | 139 | 73 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on 'se l\'hit ratio alto può causare "cache miss? se l\'hit ra' |
| `ja-essay` | on | length | 7,363 | 139 | 157 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 3x on '"キャー" "キャー" "キャー" "キャー". h. i can. but maybe "キャー" "キャー"' |
| `pt-essay` | on | length | 17,240 | 139 | 184 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself |
| `ru-essay` | on | length | 48,835 | 0 | 520 | **FAIL** | finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 334x on 'высокий hit ratio может быть из-лишним, но сам кэш может' |
| `es-essay` | on | length | 11,878 | 139 | 240 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 28x on 'c c? h. la c c? h. la c c? h. la' |
| `xl-en-fr` | on | length | 17,580 | 139 | 277 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 19x on '"parce que le contrôle de santé testé le processus et no' |
| `tr-essay` | on | length | 9,864 | 139 | 187 | **FAIL** | finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 9x on 'küçük görünse de toplu halde cpu, bellek, ağ ve kilit çe' |
| `reason-bat-ball` | on | stop | 639 | 468 | 22 | PASS | says 0.05 |
| `reason-machines` | on | stop | 1,861 | 434 | 36 | PASS | says 5 minutes |

**Verdict: FAIL** — 13 of 15 runs failed: `ar-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 16x on 'the buffer pool is high. the buffer pool is high. the bu'; `zh-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on '"cache hit rate high" but "cache hit rate high" but "sys'; `en-explain` (on) reasoning loops 12x on '95% hit rate can leave a system slower than no cache at'; `en-note` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 539x on 'i need to explain the checks that must hold before re sh'; `fr-essay` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning has a corrupted run 'k...p' in 'HslI m udra p; ("N°I R2?? ox v a sh Svpk...pa" sy'; `de-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 3x on 'h. h? cache locks? h. h? at high hit rate, many requests'; `it-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 8x on 'se l\'hit ratio alto può causare "cache miss? se l\'hit ra'; `ja-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 3x on '"キャー" "キャー" "キャー" "キャー". h. i can. but maybe "キャー" "キャー"'; `pt-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; `ru-essay` (on) finish_reason 'length'; think-exit: reasoned and then produced no answer; reasoning loops 334x on 'высокий hit ratio может быть из-лишним, но сам кэш может'; `es-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 28x on 'c c? h. la c c? h. la c c? h. la'; `xl-en-fr` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 19x on '"parce que le contrôle de santé testé le processus et no'; `tr-essay` (on) finish_reason 'length'; server cut the generation off for repeating itself; reasoning loops 9x on 'küçük görünse de toplu halde cpu, bellek, ağ ve kilit çe'

This run gated 12 of the profile's 14 topics; reasoning, reasoning_code carry no prompt, so a pass says nothing about them.
