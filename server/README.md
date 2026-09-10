# server/ -- OpenAI-compatible HTTP layer

Standard-library-only (`http.server` + `json`) front end for the DeepSeek-V4.1-Flash engine.
Requests are served **one at a time** (the engine is single-sequence); concurrent requests wait
on a lock, nothing is rejected.

```bash
python3 server/app.py --model-dir ./models/DeepSeek-V4.1-Flash --engine mock      # HTTP layer only
python3 server/app.py --model-dir ./models/DeepSeek-V4.1-Flash --engine v41       # real engine
python3 server/test_server.py                                                     # 15 e2e tests vs the mock
```

| flag | default | meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8000` | bind address |
| `--model-dir` | required | directory holding `tokenizer.json` and `encoding/encoding.py` |
| `--served-model-name` | `deepseek-v4.1-flash` | id reported by `/v1/models` |
| `--default-thinking` | `off` | `on`/`off` when a request says nothing about thinking |
| `--default-effort` | `75` | reasoning effort 1-100 when thinking is on and no effort is given |
| `--engine` | `mock` | `mock` (canned replies) or `v41` (`engine/v41_engine.py:V41Engine`, imported lazily) |
| `--engine-kwargs` | `""` | JSON object forwarded to `V41Engine(model_dir=..., **kwargs)`; **wins over the four v41 flags below** |
| `--max-seq` | `32768` | v41: context/KV length the caches are allocated for (becomes `engine.max_context`) |
| `--arena-gb` | auto | v41: GB of resident FP4 expert arena; omitted = sized from free GPU memory at load |
| `--trace-stats` | none | v41: `coverage.json` from `tools/expert_stats.py`, ranks the warm-start hot set |
| `--no-spec` | off | v41: disable MTP speculative decoding |
| `--max-context`, `--mock-delay` | `32768`, `0` | mock-engine knobs |

Dependencies: `tokenizers` (preferred) or `transformers` for `tokenizer.json`; nothing else.
The tests need a directory with `tokenizer.json` + `encoding/encoding.py` (the checkpoint's
metadata files are enough -- no weights): `$V41_MODEL_DIR`, `~/models/DeepSeek-V4.1-Flash`
or `<repo>/models/DeepSeek-V4.1-Flash`, first match wins.

## Endpoints

| route | notes |
|---|---|
| `GET /health` | `{"status":"ok","engine":..,"busy":bool,"max_context":..,"engine_config":{..}}`. `engine_config` is the v41 engine's static configuration -- `arena_gb`, `arena_slots`, `lru_slots`, `transient_slots`, `resident_expert_pct`, `max_seq`, `spec`, `trace_stats`, `kernel`, `act_quant` -- so a benchmark never has to be told what the server was started with. Absent for `--engine mock`. |
| `GET /v1/models`, `GET /v1/models/{id}` | one model card |
| `POST /v1/chat/completions` | `messages`, `tools`, `tool_choice` (`none` drops the schemas), `response_format` (`json_schema` is rendered into the system prompt), `stream`, `stream_options.include_usage`, `max_tokens`/`max_completion_tokens` (4096), `temperature` (1.0), `top_p` (0.95), `stop`, `seed`, `ignore_eos`, thinking controls below. `n>1`, images and `logprobs` are rejected with 400. |
| `POST /v1/completions` | raw `prompt` (string, or a list of token ids). Strings get BOS prepended unless `"add_bos": false`. No thinking/tool parsing: the text comes back verbatim. Stream and non-stream. |
| `POST /v1/debug/prompt` | same body as chat; returns the rendered prompt string, its token ids and the resolved thinking/effort. Handy for the engine author. |

Errors are OpenAI-shaped: `{"error": {"message", "type", "param", "code"}}` with HTTP 400/404/500.
Mid-stream failures are sent as a `data: {"error": ...}` event followed by `data: [DONE]`.

### Response extras

* `usage.completion_tokens_details.reasoning_tokens` = tokens generated before `</think>`.
* `x_engine_stats` = whatever `Engine.stats()` returned (tok/s, acceptance rate, cache hit rate ...)
  plus `server_completion_tok_per_s`. Present on non-stream responses and on the final SSE chunk
  (the one carrying `finish_reason`, which also carries `usage`); `stream_options.include_usage`
  adds the usual trailing empty-choices usage chunk as well.
* Chat deltas use `reasoning_content` for text before `</think>` and `content` after it; the marker
  itself is never emitted. Tool calls arrive in one delta at the end (`tool_calls[].index/id/type/function`),
  `finish_reason` is then `tool_calls`.

## Thinking and reasoning effort

Resolved per request, first match wins:

1. `chat_template_kwargs.thinking` (bool)
2. `enable_thinking` (bool; top-level, then inside `chat_template_kwargs`)
3. `reasoning.effort` / `reasoning_effort` (top-level, then `chat_template_kwargs.reasoning_effort`):

   | value | thinking | effort passed to the encoder |
   |---|---|---|
   | `none` | off | server default |
   | `low` | off | 50 |
   | `medium` | on | 60 |
   | `high` | on | 75 |
   | `xhigh` | on | 90 |
   | `max` | on | 100 |
   | integer 1-100 (or a numeric string) | on | the integer |

4. `--default-thinking` / `--default-effort`.

A boolean from steps 1-2 decides thinking; an effort from step 3 still sets the budget
(`enable_thinking: true` + `reasoning_effort: "low"` = thinking on at 50). The effort prefix
(`Reasoning Effort: N (range 1-100, ...)`) is only rendered by the encoder in thinking mode, so an
effort with thinking off is harmless.

Prompt building calls `encoding.encode_messages(messages, thinking_mode="thinking"|"chat",
reasoning_effort=<int>, return_multi_modal_data=True)` from `--model-dir/encoding/encoding.py`,
with `tools` (and a `json_schema` response format) attached to the first message (an empty system
message is inserted if the conversation does not start with one). The prompt string is tokenized
with `add_special_tokens=False` -- it already contains BOS and every role/think token. The
generation starts right after `<｜Assistant｜><think>` (thinking) or `<｜Assistant｜></think>` (chat).

## Output handling

* Bursts of token ids are detokenized incrementally; text is released only when the decode is
  stable (no trailing U+FFFD), so multi-token UTF-8 sequences are never split across deltas.
* Stop conditions: EOS id (`<｜end▁of▁sentence｜>` = 1), request `stop` strings (matched across
  reasoning and content; text after the match is dropped), `max_tokens` (`finish_reason: length`).
* `ignore_eos: true` empties the stop-id set on both sides -- the engine keeps decoding past EOS
  and the server stops truncating at it -- so the completion is exactly `max_tokens` tokens long.
  Benchmarks need it: without it a "512-token" run really ends wherever the model decided to stop,
  and two configs are then compared on two different amounts of work. Request `stop` strings still
  apply.
  After EOS/length the engine generator is drained (so it can finish its own bookkeeping); on a
  stop string or client disconnect it is closed (`GeneratorExit` inside the engine). `max_tokens` is
  clamped so that `prompt + max_tokens + 8 <= max_context` (the engine's DSpark headroom).
* The `\n\n<｜DSML｜ calls` lead-in switches the router to tool mode: nothing after it is streamed as
  content. When generation ends, the full completion (with the EOS string appended) goes through
  `encoding.parse_message_from_completion_text`, whose DSML tool calls become OpenAI `tool_calls`
  (arguments as a JSON string; namespaced tools are returned as `namespace::name`). If that parse
  fails (malformed or truncated tool block) the raw tail is returned as `content` instead.
