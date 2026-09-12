#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for the DeepSeek-V4.1-Flash engine.

Standard library only (http.server + json). One request is served at a time
because the engine is single-sequence; other requests queue on a lock.

Endpoints
    GET  /health
    GET  /v1/models            GET /v1/models/{id}
    POST /v1/chat/completions  (stream and non-stream, tools, reasoning)
    POST /v1/completions       (raw prompt, stream and non-stream)
    POST /v1/debug/prompt      (render a chat request to prompt text + ids)

See README.md next to this file for the thinking/effort mapping.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (HERE, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine_api import Engine, MockEngine  # noqa: E402
from tool_grammar import TOOL_CALLS_MARKER, make_factory  # noqa: E402

log = logging.getLogger("dsv41.server")

THINK_END = "</think>"
EFFORT_ALIASES = {"low": 50, "medium": 60, "high": 75, "xhigh": 90, "max": 100}
NO_THINKING_EFFORTS = {"none", "low"}
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Raised for client-visible errors; rendered in OpenAI's error shape."""

    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error",
                 param: Optional[str] = None, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.status, self.message, self.err_type, self.param, self.code = status, message, err_type, param, code

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.err_type,
                          "param": self.param, "code": self.code}}


# ---------------------------------------------------------------------------
# Tokenizer + encoding module
# ---------------------------------------------------------------------------

class Tok:
    """Thin wrapper over ``tokenizers`` (preferred) or ``transformers``."""

    def __init__(self, model_dir: str) -> None:
        path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        self.backend = None
        try:
            from tokenizers import Tokenizer  # type: ignore
            self._tk = Tokenizer.from_file(path)
            self.backend = "tokenizers"
        except ImportError:
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Neither `tokenizers` nor `transformers` is installed; "
                    "run `pip install --user tokenizers`.") from e
            self._tk = AutoTokenizer.from_pretrained(model_dir)
            self.backend = "transformers"

    def encode(self, text: str) -> List[int]:
        if self.backend == "tokenizers":
            return self._tk.encode(text, add_special_tokens=False).ids
        return self._tk.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self._tk.decode(ids, skip_special_tokens=False)

    def token_to_id(self, token: str) -> Optional[int]:
        if self.backend == "tokenizers":
            return self._tk.token_to_id(token)
        tid = self._tk.convert_tokens_to_ids(token)
        return None if tid == getattr(self._tk, "unk_token_id", None) else tid


def load_encoding_module(model_dir: str):
    path = os.path.join(model_dir, "encoding", "encoding.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"encoding/encoding.py not found in {model_dir}")
    spec = importlib.util.spec_from_file_location("dsv41_encoding", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------

def _num(body: dict, key: str, default: float, lo: float, hi: float) -> float:
    v = body.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise APIError(400, f"`{key}` must be a number", param=key)
    if not (lo <= v <= hi):
        raise APIError(400, f"`{key}` must be in [{lo}, {hi}]", param=key)
    return float(v)


def parse_sampling(body: dict) -> dict:
    mt = body.get("max_completion_tokens", body.get("max_tokens", DEFAULT_MAX_TOKENS))
    if mt is None:
        mt = DEFAULT_MAX_TOKENS
    if isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
        raise APIError(400, "`max_tokens` must be a positive integer", param="max_tokens")
    stop = body.get("stop")
    if stop is None:
        stops: List[str] = []
    elif isinstance(stop, str):
        stops = [stop]
    elif isinstance(stop, list) and all(isinstance(s, str) for s in stop):
        stops = list(stop)
    else:
        raise APIError(400, "`stop` must be a string or a list of strings", param="stop")
    stops = [s for s in stops if s]
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise APIError(400, "`seed` must be an integer", param="seed")
    n = body.get("n", 1)
    if n not in (None, 1):
        raise APIError(400, "only n=1 is supported", param="n")
    ignore_eos = body.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool):
        raise APIError(400, "`ignore_eos` must be a boolean", param="ignore_eos")
    # Off switch for the tool-call grammar, per request. It exists for the A/B (is a difference the
    # constraint's doing?) and as an escape hatch for a tool schema the builder gets wrong.
    tg = body.get("tool_grammar", True)
    if not isinstance(tg, bool):
        raise APIError(400, "`tool_grammar` must be a boolean", param="tool_grammar")
    return {
        "max_tokens": mt,
        "temperature": _num(body, "temperature", DEFAULT_TEMPERATURE, 0.0, 2.0),
        "top_p": _num(body, "top_p", DEFAULT_TOP_P, 0.0, 1.0),
        # OpenAI's two repetition controls, plus the no-repeat-ngram guard. Defaults come from the
        # environment so an operator can set them per deployment: long prose on a pruned expert set
        # drifts into repeated rhetorical structure, which is what a frequency penalty is for, while
        # code is legitimately repetitive and wants them at 0 (NOTES 2026-09-12).
        "presence_penalty": _num(body, "presence_penalty", float(os.environ.get("DSV41_PRESENCE_PENALTY", "0")), -2.0, 2.0),
        "frequency_penalty": _num(body, "frequency_penalty", float(os.environ.get("DSV41_FREQUENCY_PENALTY", "0")), -2.0, 2.0),
        "stop": stops,
        "seed": seed,
        "ignore_eos": ignore_eos,
        "tool_grammar": tg,
    }


def resolve_thinking(body: dict, default_thinking: bool, default_effort: int) -> Tuple[bool, int]:
    """Return (thinking_on, effort_int) per the precedence documented in README."""
    ctk = body.get("chat_template_kwargs") or {}
    if not isinstance(ctk, dict):
        raise APIError(400, "`chat_template_kwargs` must be an object", param="chat_template_kwargs")

    thinking: Optional[bool] = None
    for src, key in ((ctk, "thinking"), (body, "enable_thinking"), (ctk, "enable_thinking")):
        v = src.get(key)
        if v is not None:
            if not isinstance(v, bool):
                raise APIError(400, f"`{key}` must be a boolean", param=key)
            thinking = v
            break

    raw = None
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        raw = reasoning["effort"]
    elif body.get("reasoning_effort") is not None:
        raw = body["reasoning_effort"]
    elif ctk.get("reasoning_effort") is not None:
        raw = ctk["reasoning_effort"]

    effort: Optional[int] = None
    effort_thinking: Optional[bool] = None
    if raw is not None:
        if isinstance(raw, bool):
            raise APIError(400, "reasoning effort must be a string or an integer 1-100", param="reasoning_effort")
        if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
            raw = int(raw.strip())
        if isinstance(raw, int):
            if not 1 <= raw <= 100:
                raise APIError(400, "integer reasoning effort must be within 1-100", param="reasoning_effort")
            effort, effort_thinking = raw, True
        elif isinstance(raw, str):
            name = raw.strip().lower()
            if name == "none":
                effort_thinking = False
            elif name in EFFORT_ALIASES:
                effort = EFFORT_ALIASES[name]
                effort_thinking = name not in NO_THINKING_EFFORTS
            else:
                raise APIError(400, f"unknown reasoning effort {raw!r}; use none/low/medium/high/xhigh/max or 1-100",
                               param="reasoning_effort")
        else:
            raise APIError(400, "reasoning effort must be a string or an integer 1-100", param="reasoning_effort")

    if thinking is None:
        thinking = effort_thinking if effort_thinking is not None else default_thinking
    if effort is None:
        effort = default_effort
    return thinking, effort


_IMAGE_BLOCK_TYPES = {"image", "image_url", "input_image"}


def _reject_images(messages: List[dict]) -> None:
    for i, m in enumerate(messages):
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
                    raise APIError(400, "image inputs are not supported yet", param=f"messages[{i}].content")


def build_chat_prompt(body: dict, enc, tok: Tok, thinking: bool,
                      effort: int) -> Tuple[str, List[int], Optional[List[dict]]]:
    """Render a chat request. The third element is the tool list as the *model* sees it
    (namespaces folded into the names), which is what a tool grammar has to be built from."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise APIError(400, "`messages` must be a non-empty list", param="messages")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise APIError(400, f"messages[{i}] must be an object with a `role`", param=f"messages[{i}]")
    messages = copy.deepcopy(messages)
    _reject_images(messages)

    # Tools and response_format ride on the first message; the encoder only
    # renders them for a system message, so add an empty one if needed.
    tools = body.get("tools")
    if body.get("tool_choice") == "none":
        tools = None
    if tools is not None and not isinstance(tools, list):
        raise APIError(400, "`tools` must be a list", param="tools")
    schema = None
    rf = body.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        schema = (rf.get("json_schema") or {}).get("schema")
    if tools or schema:
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if tools:
            messages[0]["tools"] = tools
        if schema:
            messages[0]["response_format"] = schema

    try:
        prompt, media = enc.encode_messages(
            messages,
            thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=effort,
            return_multi_modal_data=True,
        )
    except (AssertionError, ValueError, KeyError, TypeError, NotImplementedError) as e:
        raise APIError(400, f"cannot encode messages: {e}", param="messages")
    if media.get("images"):
        raise APIError(400, "image inputs are not supported yet", param="messages")
    rendered = None
    if tools:
        try:
            rendered = enc.tools_from_openai_format(copy.deepcopy(tools))
        except Exception as e:  # noqa: BLE001 - the encoder already accepted these
            log.warning("cannot normalise the tool list for the grammar: %s", e)
    return prompt, tok.encode(prompt), rendered


# ---------------------------------------------------------------------------
# Incremental detokenization and output routing
# ---------------------------------------------------------------------------

class IncrementalDetokenizer:
    """Turn token bursts into text without emitting half a UTF-8 character.

    Text is only released once decoding a window of recent tokens is stable
    (does not end in U+FFFD); the window is re-decoded from ``prefix`` so that
    byte-fallback tokens that need 2-4 pieces are joined correctly.
    """

    def __init__(self, tok: Tok) -> None:
        self.tok = tok
        self.ids: List[int] = []
        self.prefix = 0   # start of the re-decode window
        self.read = 0     # end of the text already released

    def push(self, new_ids: List[int]) -> str:
        self.ids.extend(new_ids)
        full = self.tok.decode(self.ids[self.prefix:])
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        if len(full) > len(prev) and not full.endswith("�"):
            out = full[len(prev):]
            self.prefix, self.read = self.read, len(self.ids)
            return out
        return ""

    def flush(self) -> str:
        full = self.tok.decode(self.ids[self.prefix:])
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        self.prefix = self.read = len(self.ids)
        return full[len(prev):] if len(full) > len(prev) else ""


def _held_suffix_len(text: str, markers: List[str]) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of a marker."""
    best = 0
    for m in markers:
        for k in range(min(len(m) - 1, len(text)), 0, -1):
            if text.endswith(m[:k]):
                best = max(best, k)
                break
    return best


class OutputRouter:
    """Route streamed text into reasoning / content, detect stop strings and tool calls."""

    def __init__(self, thinking: bool, stop_strings: List[str], detect_tool_calls: bool) -> None:
        self.phase = "reasoning" if thinking else "content"
        self.pending = ""
        self.stop_strings = stop_strings
        self.detect_tool_calls = detect_tool_calls
        self.reasoning = ""
        self.content = ""
        self.tool_text = ""
        self.stopped = False       # a stop string was hit
        self.saw_think_end = thinking is False  # in chat mode the prompt already closed <think>

    def _markers(self) -> List[str]:
        if self.phase == "reasoning":
            return [THINK_END] + self.stop_strings
        if self.phase == "content":
            return ([TOOL_CALLS_MARKER] if self.detect_tool_calls else []) + self.stop_strings
        return []

    def _emit(self, text: str) -> Optional[Tuple[str, str]]:
        if not text:
            return None
        if self.phase == "reasoning":
            self.reasoning += text
        else:
            self.content += text
        return (self.phase, text)

    def feed(self, text: str) -> List[Tuple[str, str]]:
        """Return a list of (kind, text) events; kind is "reasoning" or "content"."""
        events: List[Tuple[str, str]] = []
        self.pending += text
        while self.pending and not self.stopped:
            if self.phase == "tool":
                self.tool_text += self.pending
                self.pending = ""
                break
            markers = self._markers()
            hit, pos = None, len(self.pending)
            for m in markers:
                p = self.pending.find(m)
                if p != -1 and p < pos:
                    hit, pos = m, p
            if hit is None:
                keep = _held_suffix_len(self.pending, markers)
                ev = self._emit(self.pending[:len(self.pending) - keep])
                if ev:
                    events.append(ev)
                self.pending = self.pending[len(self.pending) - keep:]
                break
            ev = self._emit(self.pending[:pos])
            if ev:
                events.append(ev)
            if hit == THINK_END and self.phase == "reasoning":
                self.phase = "content"
                self.saw_think_end = True
                self.pending = self.pending[pos + len(hit):]
            elif hit == TOOL_CALLS_MARKER and self.phase == "content":
                self.phase = "tool"
                self.pending = self.pending[pos:]
            else:  # stop string
                self.stopped = True
                self.pending = ""
        return events

    def finish(self) -> List[Tuple[str, str]]:
        """Release whatever is still held back (generation is over)."""
        events: List[Tuple[str, str]] = []
        if self.phase == "tool":
            self.tool_text += self.pending
        elif not self.stopped:
            ev = self._emit(self.pending)
            if ev:
                events.append(ev)
        self.pending = ""
        return events


# ---------------------------------------------------------------------------
# Generation driver
# ---------------------------------------------------------------------------

class GenerationResult:
    def __init__(self) -> None:
        self.gen_ids: List[int] = []
        self.finish_reason = "stop"
        self.reasoning_tokens = 0
        self.tool_calls: List[dict] = []
        self.router: Optional[OutputRouter] = None
        self.stats: dict = {}


class State:
    """Everything the handlers share."""

    def __init__(self, args: argparse.Namespace, tok: Tok, enc, engine: Engine) -> None:
        self.args = args
        self.tok = tok
        self.enc = enc
        self.engine = engine
        self.lock = threading.Lock()
        self.model_name = args.served_model_name
        self.bos_id = tok.token_to_id(enc.bos_token)
        self.eos_id = tok.token_to_id(enc.eos_token)
        self.think_end_id = tok.token_to_id(THINK_END)
        self.started = int(time.time())
        if self.eos_id != engine.eos_token_id:
            log.warning("engine.eos_token_id=%s differs from tokenizer EOS id %s", engine.eos_token_id, self.eos_id)
        # Constrained tool calls: only with an engine that can mask its own sampling, and only
        # when xgrammar is installed. Without it everything below is a no-op and tool calls are
        # parsed out of the text as before.
        self.grammars = None
        if getattr(engine, "supports_grammar", False):
            self.grammars = make_factory(
                tok, engine.eos_token_id if self.eos_id is None else self.eos_id,
                enabled=os.environ.get("DSV41_TOOL_GRAMMAR", "0") == "1")  # off until the end-to-end gates on real weights have run; see NOTES 2026-09-11

    def stop_ids(self) -> Set[int]:
        ids = {self.engine.eos_token_id}
        if self.eos_id is not None:
            ids.add(self.eos_id)
        return ids

    def generate(self, prompt_ids: List[int], sampling: dict, *, thinking: bool,
                 detect_tool_calls: bool, result: GenerationResult,
                 tools: Optional[List[dict]] = None) -> Iterator[Tuple[str, str]]:
        """Drive the engine; yield (kind, text) events; fill ``result`` at the end.

        The caller must hold ``self.lock``.
        """
        max_ctx = self.engine.max_context
        # Headroom the engine needs beyond prompt + max_tokens (DSpark verifies a
        # 6-token block, the V4.1 engine asserts an 8-token margin).
        margin = int(getattr(self.engine, "context_margin", 8))
        if len(prompt_ids) + margin >= max_ctx:
            raise APIError(400, f"prompt has {len(prompt_ids)} tokens, engine context is {max_ctx}",
                           code="context_length_exceeded")
        max_tokens = min(sampling["max_tokens"], max_ctx - len(prompt_ids) - margin)
        # ignore_eos: benchmarks need a fixed output length, so neither the engine nor this loop
        # may stop early. The stop set is emptied on both sides -- passing it to the engine and
        # then truncating here would give a short run anyway.
        ignore_eos = bool(sampling.get("ignore_eos"))
        stop_ids: Set[int] = set() if ignore_eos else self.stop_ids()
        detok = IncrementalDetokenizer(self.tok)
        router = OutputRouter(thinking, sampling["stop"], detect_tool_calls)
        result.router = router
        t0 = time.perf_counter()

        gen_kwargs = dict(max_tokens=max_tokens, temperature=sampling["temperature"],
                          top_p=sampling["top_p"], stop_token_ids=stop_ids, seed=sampling["seed"])
        pen = None
        if getattr(self.engine, "supports_penalties", False):
            from engine.v41_engine import Penalties
            pen = Penalties(presence=sampling["presence_penalty"], frequency=sampling["frequency_penalty"])
            if pen.active:
                gen_kwargs["penalties"] = pen
        # The gate constrains nothing until the model opens a tool-calls block, so it costs a
        # dictionary lookup per step on a request that never calls a tool.
        gate = None
        if tools and detect_tool_calls and self.grammars is not None and sampling["tool_grammar"]:
            gate = self.grammars.for_tools(tools)
        if gate is not None:
            gen_kwargs["grammar"] = gate
        if ignore_eos:
            try:
                gen = self.engine.generate(prompt_ids, ignore_eos=True, **gen_kwargs)
            except TypeError:  # an engine without the kwarg (MockEngine): empty stop set is enough
                gen = self.engine.generate(prompt_ids, **gen_kwargs)
        else:
            gen = self.engine.generate(prompt_ids, **gen_kwargs)
        hit_eos = False
        try:
            for burst in gen:
                burst = list(burst)
                for i, t in enumerate(burst):
                    if t in stop_ids:
                        burst, hit_eos = burst[:i], True
                        break
                room = max_tokens - len(result.gen_ids)
                if len(burst) > room:
                    burst = burst[:room]
                result.gen_ids.extend(burst)
                for ev in router.feed(detok.push(burst)):
                    yield ev
                if hit_eos or router.stopped or len(result.gen_ids) >= max_tokens:
                    break
            if not router.stopped:
                # EOS or length: the engine's own loop ends right after this
                # yield, so let it run its epilogue (stats) instead of closing
                # it mid-flight. Anything it still emits is ignored; an engine
                # that keeps going is cut off after a few bursts.
                for _ in range(4):
                    if next(gen, None) is None:
                        break
        finally:
            gen.close()

        for ev in router.feed(detok.flush()):
            yield ev
        for ev in router.finish():
            yield ev

        if hit_eos or router.stopped:
            result.finish_reason = "stop"
        elif len(result.gen_ids) >= max_tokens:
            result.finish_reason = "length"
        if thinking:
            try:
                result.reasoning_tokens = result.gen_ids.index(self.think_end_id)
            except ValueError:
                result.reasoning_tokens = len(result.gen_ids)

        if detect_tool_calls:
            result.tool_calls = self._parse_tool_calls(router, thinking)
            if result.tool_calls:
                result.finish_reason = "tool_calls"

        try:
            result.stats = dict(self.engine.stats() or {})
        except Exception as e:  # engine stats must never break a response
            log.warning("engine.stats() failed: %s", e)
            result.stats = {}
        if gate is not None:
            st = dict(gate.stats)
            st["mask_ms_per_call"] = round(st["mask_s"] / st["mask_calls"] * 1e3, 3) if st["mask_calls"] else None
            st["mask_s"] = round(st["mask_s"], 4)
            result.stats["tool_grammar"] = st
        dt = time.perf_counter() - t0
        result.stats.setdefault("server_completion_tok_per_s", round(len(result.gen_ids) / dt, 1) if dt > 0 else None)
        log.info("generation done: prompt=%d completion=%d reasoning=%d finish=%s %.2fs",
                 len(prompt_ids), len(result.gen_ids), result.reasoning_tokens, result.finish_reason, dt)

    # A completion that calls tools is DSML, and the checkpoint's own parser
    # (corpus/sources/dsv41_encoding.py::parse_message_from_completion_text) is strict about it: a
    # parameter must read `name="x" string="true|false">value<`, and nothing may follow the calls
    # block. Real completions deviated in two ways, and both were costing the whole tool call --
    # the client then saw only the sentence before it and a `stop` finish:
    #   * the value is put in the `string` attribute (`name="query" string="a b c"`), with no
    #     separate value body;
    #   * ordinary prose follows the closing tag of the calls block.
    # Both are now unreachable when a tool grammar is in force (server/tool_grammar.py): the mask
    # will not let the model write them. This tolerant pass is the fallback for the cases where
    # there is no grammar -- xgrammar missing, DSV41_TOOL_GRAMMAR=0, a schema the builder could not
    # express, an engine that cannot mask -- and it runs only after the strict parser has raised.
    _RE_INVOKE = re.compile(r'<｜DSML｜ invoke name="(?P<name>[^"]*)"\s*>?\n?(?P<body>.*?)(?=<｜DSML｜ invoke |</｜DSML｜ calls>|\Z)', re.DOTALL)
    _RE_PARAM_SPEC = re.compile(r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<s>true|false)"\s*>(?P<v>.*?)<', re.DOTALL)
    _RE_PARAM_ATTR = re.compile(r'<｜DSML｜ parameter name="(?P<k>[^"]*)" string="(?P<v>.*?)"\s*>', re.DOTALL)

    @classmethod
    def _parse_tool_calls_tolerant(cls, text: str) -> List[dict]:
        """Best-effort DSML tool calls. Returns [] when nothing parses."""
        out: List[dict] = []
        for m in cls._RE_INVOKE.finditer(text):
            name, body = m.group("name"), m.group("body")
            if not name:
                continue
            args: Dict[str, Any] = {}
            for pm in cls._RE_PARAM_SPEC.finditer(body):
                k, is_str, v = pm.group("k"), pm.group("s"), pm.group("v")
                if is_str == "true":
                    args[k] = v
                else:
                    try:
                        args[k] = json.loads(v)
                    except Exception:  # noqa: BLE001 - a malformed literal is still worth sending as text
                        args[k] = v
            consumed = {pm.group("k") for pm in cls._RE_PARAM_SPEC.finditer(body)}
            for pm in cls._RE_PARAM_ATTR.finditer(body):
                k, v = pm.group("k"), pm.group("v")
                if k and k not in consumed and k not in args:
                    args[k] = v
            out.append({"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
        return out

    def _parse_tool_calls(self, router: OutputRouter, thinking: bool) -> List[dict]:
        if not router.tool_text:
            return []
        text = ""
        if thinking:
            text += router.reasoning + THINK_END
        text += router.content + router.tool_text
        if not text.endswith(self.enc.eos_token):
            text += self.enc.eos_token
        try:
            parsed = self.enc.parse_message_from_completion_text(
                text, thinking_mode="thinking" if thinking else "chat")
        except Exception as e:
            recovered = self._parse_tool_calls_tolerant(text)
            if recovered:
                log.warning("tool-call strict parse failed (%s); recovered %d call(s) tolerantly",
                            e, len(recovered))
                parsed = {"tool_calls": recovered}
            else:
                log.warning("tool-call parse failed (%s); returning raw text as content", e)
                if os.environ.get("DSV41_LOG_TOOL_TEXT") == "1":
                    log.warning("unparsed tool text: %r", router.tool_text[:2000])
                router.content += router.tool_text
                router.tool_text = ""
                return []
        calls = []
        for tc in parsed.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            if tc.get("namespace"):
                name = f"{tc['namespace']}::{name}"
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": args}})
        return calls


# ---------------------------------------------------------------------------
# Response shaping
# ---------------------------------------------------------------------------

def usage_dict(prompt_tokens: int, result: GenerationResult) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(result.gen_ids),
        "total_tokens": prompt_tokens + len(result.gen_ids),
        "completion_tokens_details": {"reasoning_tokens": result.reasoning_tokens},
    }


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dsv41-server/0.1"

    @property
    def state(self) -> State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s %s", self.address_string(), fmt % args)

    # -- plumbing ----------------------------------------------------------
    def _send_json(self, status: int, obj: Any) -> None:
        data = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, err: APIError) -> None:
        self._send_json(err.status, err.body())

    def _read_json(self) -> dict:
        length = self.headers.get("Content-Length")
        if length is None:
            raise APIError(411, "Content-Length required")
        try:
            n = int(length)
        except ValueError:
            raise APIError(400, "bad Content-Length")
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise APIError(400, f"request body is not valid JSON: {e}")
        if not isinstance(body, dict):
            raise APIError(400, "request body must be a JSON object")
        if os.environ.get("DSV41_LOG_BODIES"):
            log.info("request body: %s", json.dumps(body)[:4000])
        return body

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse(self, obj: Any) -> None:
        payload = b"data: " + (obj if isinstance(obj, bytes) else _json_bytes(obj)) + b"\n\n"
        self.wfile.write(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")
        self.wfile.flush()

    def _end_sse(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # -- routing -----------------------------------------------------------
    def do_OPTIONS(self) -> None:  # CORS preflight for browser clients
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        st = self.state
        try:
            if path == "/health":
                body = {"status": "ok", "model": st.model_name, "engine": st.args.engine,
                        "busy": st.lock.locked(), "max_context": st.engine.max_context,
                        "uptime_s": int(time.time()) - st.started}
                # Static engine configuration (arena GB/slots, max_seq, spec, trace stats, kernel):
                # every measured number in RESULTS.md has to be quoted with the config that produced
                # it, and a bench run should not have to be told what the server was started with.
                cfg = getattr(st.engine, "config", None)
                if callable(cfg):
                    try:
                        body["engine_config"] = cfg()
                    except Exception as e:  # never let introspection break the health probe
                        log.warning("engine.config() failed: %s", e)
                self._send_json(200, body)
            elif path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [self._model_card()]})
            elif path.startswith("/v1/models/"):
                if path[len("/v1/models/"):] != st.model_name:
                    raise APIError(404, "model not found", code="model_not_found", param="model")
                self._send_json(200, self._model_card())
            else:
                raise APIError(404, f"no route for GET {path}")
        except APIError as e:
            self._send_error(e)

    def _model_card(self) -> dict:
        st = self.state
        return {"id": st.model_name, "object": "model", "created": st.started,
                "owned_by": "deepseek-v41-flash-spark", "max_model_len": st.engine.max_context}

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            body = self._read_json()
            if path == "/v1/chat/completions":
                self._chat(body)
            elif path == "/v1/completions":
                self._completions(body)
            elif path == "/v1/debug/prompt":
                self._debug_prompt(body)
            else:
                raise APIError(404, f"no route for POST {path}")
        except APIError as e:
            self._send_error(e)
        except (BrokenPipeError, ConnectionResetError):
            log.info("client disconnected")
        except Exception as e:  # anything else is a server bug
            log.exception("unhandled error")
            try:
                self._send_error(APIError(500, f"internal error: {e}", err_type="server_error"))
            except Exception:
                pass

    # -- endpoints ---------------------------------------------------------
    def _debug_prompt(self, body: dict) -> None:
        st = self.state
        thinking, effort = resolve_thinking(body, st.args.default_thinking, st.args.default_effort)
        prompt, ids, tools = build_chat_prompt(body, st.enc, st.tok, thinking, effort)
        body_out = {"thinking": thinking, "reasoning_effort": effort, "prompt": prompt,
                    "prompt_ids": ids, "prompt_tokens": len(ids)}
        if tools and st.grammars is not None:
            from tool_grammar import build_tool_grammar  # noqa: PLC0415 - debug endpoint only
            body_out["tool_grammar"] = build_tool_grammar(tools)
        self._send_json(200, body_out)

    def _chat(self, body: dict) -> None:
        st = self.state
        sampling = parse_sampling(body)
        thinking, effort = resolve_thinking(body, st.args.default_thinking, st.args.default_effort)
        _, prompt_ids, tools = build_chat_prompt(body, st.enc, st.tok, thinking, effort)
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        result = GenerationResult()

        def chunk(delta: dict, finish: Optional[str] = None, extra: Optional[dict] = None) -> dict:
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": st.model_name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}
            if extra:
                obj.update(extra)
            return obj

        with st.lock:
            if not stream:
                for _ in st.generate(prompt_ids, sampling, thinking=thinking, detect_tool_calls=True,
                                     result=result, tools=tools):
                    pass
                router = result.router
                message: Dict[str, Any] = {"role": "assistant", "content": router.content}
                if thinking:
                    message["reasoning_content"] = router.reasoning
                if result.tool_calls:
                    message["tool_calls"] = result.tool_calls
                    if not router.content:
                        message["content"] = None
                self._send_json(200, {
                    "id": rid, "object": "chat.completion", "created": created, "model": st.model_name,
                    "choices": [{"index": 0, "message": message, "finish_reason": result.finish_reason,
                                 "logprobs": None}],
                    "usage": usage_dict(len(prompt_ids), result),
                    "x_engine_stats": result.stats,
                })
                return

            self._start_sse()
            try:
                self._sse(chunk({"role": "assistant", "content": ""}))
                for kind, text in st.generate(prompt_ids, sampling, thinking=thinking,
                                              detect_tool_calls=True, result=result, tools=tools):
                    self._sse(chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text}))
                if result.tool_calls:
                    self._sse(chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(result.tool_calls)]}))
                final_extra = {"usage": usage_dict(len(prompt_ids), result), "x_engine_stats": result.stats}
                self._sse(chunk({}, result.finish_reason, final_extra))
                if include_usage:
                    self._sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                               "model": st.model_name, "choices": [], **final_extra})
            except APIError as e:
                self._sse(e.body())
            except (BrokenPipeError, ConnectionResetError):
                log.info("client disconnected mid-stream after %d tokens", len(result.gen_ids))
                return
            except Exception as e:
                log.exception("engine error mid-stream")
                self._sse({"error": {"message": f"engine error: {e}", "type": "server_error",
                                     "param": None, "code": None}})
            self._sse(b"[DONE]")
            self._end_sse()

    def _completions(self, body: dict) -> None:
        st = self.state
        sampling = parse_sampling(body)
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]
        if isinstance(prompt, str):
            prompt_ids = st.tok.encode(prompt)
            if body.get("add_bos", True) and st.bos_id is not None and (not prompt_ids or prompt_ids[0] != st.bos_id):
                prompt_ids = [st.bos_id] + prompt_ids
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            prompt_ids = list(prompt)
        else:
            raise APIError(400, "`prompt` must be a string or a list of token ids (batching is not supported)",
                           param="prompt")
        if body.get("echo"):
            raise APIError(400, "`echo` is not supported", param="echo")
        stream = bool(body.get("stream", False))
        rid = "cmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        result = GenerationResult()

        def obj(text: str, finish: Optional[str], extra: Optional[dict] = None) -> dict:
            o = {"id": rid, "object": "text_completion", "created": created, "model": st.model_name,
                 "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]}
            if extra:
                o.update(extra)
            return o

        with st.lock:
            if not stream:
                for _ in st.generate(prompt_ids, sampling, thinking=False, detect_tool_calls=False, result=result):
                    pass
                self._send_json(200, obj(result.router.content, result.finish_reason,
                                         {"usage": usage_dict(len(prompt_ids), result),
                                          "x_engine_stats": result.stats}))
                return
            self._start_sse()
            try:
                for _, text in st.generate(prompt_ids, sampling, thinking=False, detect_tool_calls=False, result=result):
                    self._sse(obj(text, None))
                self._sse(obj("", result.finish_reason,
                              {"usage": usage_dict(len(prompt_ids), result), "x_engine_stats": result.stats}))
            except APIError as e:
                self._sse(e.body())
            except (BrokenPipeError, ConnectionResetError):
                log.info("client disconnected mid-stream after %d tokens", len(result.gen_ids))
                return
            except Exception as e:
                log.exception("engine error mid-stream")
                self._sse({"error": {"message": f"engine error: {e}", "type": "server_error",
                                     "param": None, "code": None}})
            self._sse(b"[DONE]")
            self._end_sse()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def make_engine(args: argparse.Namespace, tok: Tok, enc) -> Engine:
    if args.engine == "mock":
        return MockEngine(tok.encode, tok.decode,
                          eos_token_id=tok.token_to_id(enc.eos_token),
                          think_start_id=tok.token_to_id(enc.thinking_start_token),
                          think_end_id=tok.token_to_id(enc.thinking_end_token),
                          max_context=args.max_context,
                          burst_delay_s=args.mock_delay)
    if args.engine == "v41":
        try:
            from engine.v41_engine import V41Engine  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "--engine v41 needs engine/v41_engine.py (class V41Engine, an engine_api.Engine subclass) "
                f"at the repo root ({REPO_ROOT}); it is not importable: {e}. "
                "Use --engine mock to run the HTTP layer alone.")
        kwargs: Dict[str, Any] = {
            "max_seq": args.max_seq,
            "arena_gb": args.arena_gb,
            "trace_stats": args.trace_stats,
            "spec": not args.no_spec,
        }
        # --engine-kwargs is the escape hatch for anything without its own flag,
        # and it wins over the flags so a JSON override is never silently ignored.
        kwargs.update(json.loads(args.engine_kwargs) if args.engine_kwargs else {})
        return V41Engine(model_dir=args.model_dir, **kwargs)
    raise SystemExit(f"unknown engine {args.engine!r}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible server for DeepSeek-V4.1-Flash")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-dir", required=True, help="directory with tokenizer.json and encoding/encoding.py")
    p.add_argument("--served-model-name", default="deepseek-v4.1-flash")
    p.add_argument("--default-thinking", choices=["off", "on"], default="off")
    p.add_argument("--default-effort", type=int, default=75, help="reasoning effort 1-100 when thinking is on")
    p.add_argument("--engine", choices=["mock", "v41"], default="mock")
    p.add_argument("--engine-kwargs", default="",
                   help="JSON object passed to V41Engine(model_dir=..., **kwargs); overrides the flags below")
    p.add_argument("--max-seq", type=int, default=32768,
                   help="v41 engine: KV/context length to allocate caches for (engine.max_context)")
    p.add_argument("--arena-gb", type=float, default=None,
                   help="v41 engine: GB of resident FP4 expert arena; omit to size it from free GPU memory")
    p.add_argument("--trace-stats", default=None,
                   help="v41 engine: coverage.json from tools/expert_stats.py, ranks the warm-start hot set")
    p.add_argument("--no-spec", action="store_true", help="v41 engine: disable MTP speculative decoding")
    p.add_argument("--max-context", type=int, default=32768, help="context limit for the mock engine")
    p.add_argument("--mock-delay", type=float, default=0.0, help="seconds to sleep per mock burst")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    if not 1 <= args.default_effort <= 100:
        p.error("--default-effort must be within 1-100")
    args.default_thinking = args.default_thinking == "on"
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tok = Tok(args.model_dir)
    enc = load_encoding_module(args.model_dir)
    engine = make_engine(args, tok, enc)
    state = State(args, tok, enc, engine)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.state = state  # type: ignore[attr-defined]
    log.info("serving %s (engine=%s, tokenizer=%s, thinking default=%s effort=%d) on http://%s:%d",
             args.served_model_name, args.engine, tok.backend, "on" if args.default_thinking else "off",
             args.default_effort, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        engine.close()


if __name__ == "__main__":
    main()
