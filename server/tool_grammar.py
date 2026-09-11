#!/usr/bin/env python3
"""Grammar-constrained DSML tool calls.

The checkpoint tells the model, in its own system prompt (``TOOLS_TEMPLATE`` in
``encoding/encoding.py``), to emit tool calls as a DSML block::

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="$TOOL">
    <｜DSML｜ parameter name="$P" string="true|false">$VALUE</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

and its own parser (``parse_tool_calls``) is strict about every byte of that:
``name="x" string="true">value<`` exactly, no duplicate parameter names, and
nothing at all after the closing tag of the block. With ~20 tool schemas in the
prompt the model drifts out of that format -- the value ends up inside the
``string`` attribute, or prose continues after the block -- and a drifting model
does not stop: it re-emits near-identical calls until the output cap.

This module removes the possibility. For the tool list of one request it builds
an EBNF grammar that describes exactly the legal calls block for *those* tools,
compiles it with xgrammar, and hands the engine a gate object that masks the
sampling distribution while (and only while) the model is inside the block.

Three things make the grammar small and unambiguous:

* Every DSML token contains U+FF5C (the fullwidth vertical bar). A parameter
  value is therefore "any run of characters that does not contain U+FF5C":
  HTML, code, JSON and prose stay legal inside a value, while the closing tag
  can never be confused with value text.
* Parameters are emitted in schema order, required ones mandatory and optional
  ones wrapped in ``?``. That makes "all required present" and "no duplicates"
  structural rather than a post-hoc check; the only freedom lost is the order,
  which the parser ignores anyway.
* The block is closed by construction, and the grammar ends there. Once the
  closing tag is emitted the matcher is complete, and a complete xgrammar
  matcher allows exactly one thing: the end-of-turn token. That is what stops
  the spiral.

Nothing here is engine-specific: the engine sees an object with ``observe`` and
``mask_rows`` (see ``server/engine_api.py``), and when a request has no tools it
gets ``None`` and nothing changes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("dsv41.tool_grammar")

# --------------------------------------------------------------------------
# DSML syntax (mirrors the constants near line 424 of encoding/encoding.py)
# --------------------------------------------------------------------------

DSML = "｜DSML｜"                      # every DSML token contains U+FF5C
CALLS_BLOCK = " calls"
INVOKE_TAG = " invoke"
PARAM_TAG = " parameter"

#: what ``server/app.py``'s OutputRouter watches for, and where the grammar starts.
TOOL_CALLS_MARKER = f"\n\n<{DSML}{CALLS_BLOCK}"

BLOCK_OPEN = f"\n\n<{DSML}{CALLS_BLOCK}>\n"
BLOCK_CLOSE = f"</{DSML}{CALLS_BLOCK}>"
INVOKE_OPEN = f"<{DSML}{INVOKE_TAG} name="        # + "name">\n
INVOKE_CLOSE = f"</{DSML}{INVOKE_TAG}>\n"
PARAM_OPEN = f"<{DSML}{PARAM_TAG} name="          # + "name" string="true">
PARAM_CLOSE = f"</{DSML}{PARAM_TAG}>\n"

# "Any character except U+FF5C" has to be written as a union of positive ranges.
# xgrammar's negated character classes are ASCII-only: a codepoint above 127 inside
# [^...] is dropped with a warning ("Negative Character class contains byte greater
# than 127, clamping to 127"), so `[^｜]` silently compiles to a class that matches
# U+FF5C as well -- which would let a parameter value swallow the rest of the block.
# test_value_cannot_contain_the_dsml_bar guards this.
VALUE_CHAR = r"[\u0000-\uFF5B\uFF5D-\U0010FFFF]"
#: a parameter name: no quote, no control character, no U+FF5C
NAME_CHAR = r"[\x20-\x21\x23-\uFF5B\uFF5D-\U0010FFFF]"
#: a character inside a JSON string: no quote, no backslash, no control character, no U+FF5C
JSON_STRING_CHAR = r"[\x20-\x21\x23-\x5b\x5d-\uFF5B\uFF5D-\U0010FFFF]"

DEFAULT_MAX_CALLS = 8

Q = '"'   # f-strings cannot nest the quote they are written with


# --------------------------------------------------------------------------
# EBNF helpers
# --------------------------------------------------------------------------

def _lit(s: str) -> str:
    """One EBNF string literal for ``s``."""
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append("\\x%02x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def dsml_safe(name: str) -> bool:
    """Can ``name`` appear in a DSML ``name="..."`` attribute unambiguously?

    The checkpoint's parser reads attributes with ``name="(.*?)"``, so a quote
    inside the name truncates it; U+FF5C would collide with the tags themselves;
    control characters break the line structure of the block.
    """
    if not name or not isinstance(name, str):
        return False
    return not any(c == '"' or c == "｜" or ord(c) < 0x20 or ord(c) == 0x7F for c in name)


# JSON value rules, emitted only when a non-string parameter needs them. Strings
# exclude U+FF5C for the same reason parameter values do; whitespace is bounded
# so that "pretty-print forever" is not a legal continuation.
_JSON_RULES = r"""
jws ::= [ \n\r\t]{0,4}
jnull ::= "null"
jbool ::= "true" | "false"
jint ::= "-"? ("0" | [1-9] [0-9]*)
jnum ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
jhex ::= [0-9a-fA-F]
jesc ::= [\"\\/bfnrt] | "u" jhex jhex jhex jhex
jchar ::= JSON_STRING_CHAR | "\\" jesc
jstr ::= "\"" jchar* "\""
jval ::= jstr | jnum | jbool | jnull | jarr | jobj
jarr ::= "[" jws (jval jws ("," jws jval jws)*)? "]"
jobj ::= "{" jws (jstr jws ":" jws jval jws ("," jws jstr jws ":" jws jval jws)*)? "}"
""".replace("JSON_STRING_CHAR", JSON_STRING_CHAR)

_SCALAR_RULE = {"integer": "jint", "number": "jnum", "boolean": "jbool", "null": "jnull",
                "string": "jstr"}


def _param_value(schema: Dict[str, Any]) -> Optional[str]:
    """EBNF fragment for a non-string parameter's JSON value, or None if unknown.

    Arrays and objects get a *generic* JSON rule (any well-formed array/object),
    with the element rule narrowed only when the declared item type is a scalar.
    Narrowing further -- object properties, oneOf, additionalProperties -- would
    risk making a call the client meant to send unrepresentable, which is worse
    than accepting a well-formed value of the wrong shape.
    """
    t = schema.get("type")
    if isinstance(t, str):
        t = t.strip()
    if t in ("integer", "number", "boolean", "null"):
        return _SCALAR_RULE[t]
    if t == "object":
        return "jobj"
    if t == "array":
        item = schema.get("items")
        rule = None
        if isinstance(item, dict):
            it = item.get("type")
            if isinstance(it, str):
                rule = _SCALAR_RULE.get(it.strip())
        if rule is None:
            return "jarr"
        return f'("[" jws ({rule} jws ("," jws {rule} jws)*)? "]")'
    return None


def _string_value(schema: Dict[str, Any]) -> str:
    """EBNF fragment for a string-typed parameter's raw value."""
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and all(
            isinstance(e, str) and "｜" not in e for e in enum):
        return "(" + " | ".join(_lit(e) for e in enum) + ")"
    return "vany"


def _functions(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept both OpenAI ``{"function": {...}}`` entries and flat function dicts."""
    out = []
    for t in tools or ():
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            out.append(fn)
    return out


def build_tool_grammar(tools: Sequence[Dict[str, Any]], *,
                       max_calls: int = DEFAULT_MAX_CALLS) -> Optional[str]:
    """EBNF for the DSML calls block of exactly ``tools``.

    ``tools`` is either the request's OpenAI-format list or the flattened list
    the encoder builds from it (``tools_from_openai_format``) -- pass the latter
    when namespaces are in play, so that the names in the grammar are the names
    the prompt told the model to use.

    Returns None when nothing usable is left (no tools, or every name is one
    DSML cannot represent), in which case decoding stays unconstrained.
    """
    fns = _functions(tools)
    calls: List[str] = []
    rules: List[str] = []
    need_json = False
    need_vany = False
    need_pname = False

    for i, fn in enumerate(fns):
        name = fn.get("name")
        if not dsml_safe(name):
            log.warning("tool %r cannot be expressed in DSML; left out of the grammar", name)
            continue
        params = fn.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        required = params.get("required") if isinstance(params, dict) else None
        required = set(required) if isinstance(required, list) else set()

        rid = f"t{i}"
        seq: List[str] = []
        if isinstance(props, dict):
            # `properties` present, even empty, is the declared set of parameters
            for j, (pname, pschema) in enumerate(props.items()):
                if not dsml_safe(pname):
                    log.warning("parameter %r of tool %r cannot be expressed in DSML; left out",
                                pname, name)
                    continue
                if not isinstance(pschema, dict):
                    pschema = {}
                prid = f"{rid}p{j}"
                head = f'{PARAM_OPEN}{Q}{pname}{Q} string='
                jrule = _param_value(pschema)
                is_str = (isinstance(pschema.get("type"), str)
                          and pschema["type"].strip() == "string")
                val = _string_value(pschema)
                if jrule is not None:
                    need_json = True
                    body = f'{_lit(head + Q + "false" + Q + ">")} {jrule}'
                elif is_str or val != "vany":
                    # a string type, or an enum of strings with the type left out
                    need_vany = need_vany or val == "vany"
                    body = f'{_lit(head + Q + "true" + Q + ">")} {val}'
                else:
                    # no usable type: let the model pick the flag, value unconstrained
                    need_vany = True
                    body = f'{_lit(head + Q)} ("true" | "false") {_lit(Q + ">")} vany'
                rules.append(f'{prid} ::= {body} {_lit(PARAM_CLOSE)}')
                seq.append(prid if pname in required else f"{prid}?")
        elif isinstance(params, dict) and params.get("type") == "object":
            # an object schema that declares no `properties` at all accepts
            # anything: allow any number of parameters with any name DSML can
            # carry, rather than making a legitimate call unrepresentable.
            need_vany = True
            prid = f"{rid}pany"
            rules.append(
                f'{prid} ::= {_lit(PARAM_OPEN + Q)} pname {_lit(Q + " string=" + Q)} '
                f'("true" | "false") {_lit(Q + ">")} vany {_lit(PARAM_CLOSE)}')
            need_pname = True
            seq.append(f"{prid}{{0,16}}")

        open_tag = _lit(f'{INVOKE_OPEN}{Q}{name}{Q}>\n')
        body = (" " + " ".join(seq)) if seq else ""
        rules.append(f'{rid} ::= {open_tag}{body} {_lit(INVOKE_CLOSE)}')
        calls.append(rid)

    if not calls:
        return None

    parts = [
        f"# DSML tool calls for {len(calls)} tool(s); generated by server/tool_grammar.py",
        f'root ::= {_lit(BLOCK_OPEN)} call{{1,{max(1, int(max_calls))}}} {_lit(BLOCK_CLOSE)}',
        "call ::= " + " | ".join(calls),
    ]
    if need_vany:
        parts.append(f"vany ::= {VALUE_CHAR}*")
    if need_pname:
        parts.append(f"pname ::= {NAME_CHAR}*")
    parts.extend(rules)
    if need_json:
        parts.append(_JSON_RULES.strip())
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# The gate the engine sees
# --------------------------------------------------------------------------

class ToolCallGrammar:
    """Masks sampling inside the DSML calls block, and nowhere else.

    Lifecycle, driven entirely by ``observe``:

    * **idle** -- no constraint at all. Every generated token is decoded into a
      short rolling window and searched for ``TOOL_CALLS_MARKER``.
    * **active** -- the marker has been emitted. A ``GrammarMatcher`` is created
      and seeded with ``accept_string`` over the text from the marker onwards
      (which may be more than the marker: a burst can carry the marker and the
      first tag together), and from then on every mask is the matcher's.
    * **done** -- the matcher is complete (the block closed) or terminated. A
      complete matcher allows only the end-of-turn token, which is what ends
      generation; ``observe`` of that token terminates the matcher.

    ``mask_rows`` never advances the matcher: the draft walk is undone before it
    returns, so ``observe`` stays the single place where state moves forward.
    """

    def __init__(self, compiled, *, xgr, torch, eos_id: int, vocab_size: int,
                 decode, marker: str = TOOL_CALLS_MARKER, window: int = 64,
                 max_rollback: int = 16, use_traverse: bool = True) -> None:
        self._compiled = compiled
        self._xgr = xgr
        self._torch = torch
        self._eos_id = eos_id
        self._vocab = vocab_size
        self._decode = decode
        self._marker = marker
        self._max_rollback = max_rollback
        self._use_traverse = use_traverse
        self._recent: deque = deque(maxlen=window)
        self._matcher = None
        self._failed = False
        # mask staging, allocated on first use
        self._bitmask = None
        self._bitmask_dev = None
        self._tree = {}
        self._block_cpu = None
        self.stats = {"activated": False, "mask_calls": 0, "mask_s": 0.0,
                      "masked_rows": 0, "accept_fail": 0}

    # -- state ----------------------------------------------------------
    @property
    def active(self) -> bool:
        """Is the constraint in force right now?

        False before the marker, after a failure, and after the end-of-turn token
        has been accepted -- a terminated matcher has no next-token set to give.
        """
        return (self._matcher is not None and not self._failed
                and not self._matcher.is_terminated())

    @property
    def completed(self) -> bool:
        return self.active and self._matcher.is_completed()

    def _activate(self, seed: str) -> None:
        xgr = self._xgr
        m = xgr.GrammarMatcher(self._compiled, override_stop_tokens=[self._eos_id],
                               max_rollback_tokens=self._max_rollback)
        if not m.accept_string(seed):
            # Only reachable if the marker was emitted in a shape the grammar
            # cannot start from. Staying unconstrained is always safe: the
            # tolerant parser in app.py is still behind us.
            log.warning("tool grammar could not seed on %r; decoding stays unconstrained", seed[:120])
            self._failed = True
            return
        self._matcher = m
        self.stats["activated"] = True
        log.info("tool grammar active (seeded with %d chars)", len(seed))

    def observe(self, token_ids: Sequence[int]) -> None:
        """Take the tokens that are now final, in order."""
        if self._failed or not token_ids:
            return
        if self._matcher is None:
            self._recent.extend(token_ids)
            try:
                tail = self._decode(list(self._recent))
            except Exception as e:  # a detokenizer hiccup must not break decoding
                log.warning("tool grammar detokenize failed: %s", e)
                self._failed = True
                return
            pos = tail.find(self._marker)
            if pos >= 0:
                self._activate(tail[pos:])
            return
        for t in token_ids:
            if self._matcher.is_terminated():
                return
            if not self._matcher.accept_token(int(t)):
                # The mask should have made this impossible; if it happens the
                # grammar is out of step with the stream and the honest move is
                # to get out of the way rather than wedge the request.
                self.stats["accept_fail"] += 1
                log.warning("tool grammar rejected an emitted token (%d); constraint dropped", int(t))
                self._failed = True
                return

    # -- masking --------------------------------------------------------
    def _stage(self, rows: int, device):
        torch = self._torch
        if self._bitmask is None or self._bitmask.shape[0] < rows:
            self._bitmask = self._xgr.allocate_token_bitmask(rows, self._vocab)
            self._bitmask_dev = None
        if device.type != "cpu" and (self._bitmask_dev is None
                                     or self._bitmask_dev.shape[0] < rows
                                     or self._bitmask_dev.device != device):
            self._bitmask_dev = torch.empty(self._bitmask.shape, dtype=self._bitmask.dtype,
                                            device=device)
        return self._bitmask

    def _tree_of(self, rows: int):
        """Chain topology for ``traverse_draft_tree``: node i is block[i]."""
        t = self._tree.get(rows)
        if t is None:
            torch = self._torch
            nxt = torch.tensor([i + 1 for i in range(rows - 1)] + [-1], dtype=torch.int64)
            sib = torch.full((rows,), -1, dtype=torch.int64)
            t = self._tree[rows] = (nxt, sib)
        return t

    def mask_rows(self, logits, block_ids=None) -> int:
        """Mask ``logits`` in place; return the number of rows actually masked.

        ``logits`` is [R, V] (one row per verify position) and ``block_ids`` the
        R tokens the rows follow -- ``[tok, draft0 .. draft4]`` for a DSpark
        block -- so row i is the distribution after block[0..i]. Row 0 needs no
        block at all; pass ``block_ids=None`` for a plain single-token step.

        Rows after the first illegal draft are left untouched: that draft is
        masked out of its own row, so it is rejected there, and the verifier
        never reads past the rejection point.
        """
        if not self.active:
            return 0
        try:
            return self._mask_rows(logits, block_ids)
        except Exception as e:  # noqa: BLE001
            # The mask sits on the decode loop's critical path. Whatever goes wrong in here --
            # a kernel that will not build, a tensor that is not the shape this thinks it is --
            # losing the constraint is recoverable (the tolerant parser is behind it) and
            # killing the request is not.
            self._failed = True
            self.stats["error"] = f"{type(e).__name__}: {e}"
            log.warning("tool grammar masking failed (%s); constraint dropped for this request", e)
            return 0

    def _mask_rows(self, logits, block_ids) -> int:
        t0 = time.perf_counter()
        xgr, torch = self._xgr, self._torch
        rows = int(logits.shape[0]) if logits.dim() > 1 else 1
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        bitmask = self._stage(rows, logits.device)

        if rows == 1 or block_ids is None:
            rows = 1
            self._matcher.fill_next_token_bitmask(bitmask, 0)
        else:
            ids = block_ids
            if hasattr(ids, "to"):
                if self._block_cpu is None or self._block_cpu.numel() < rows:
                    self._block_cpu = torch.empty(rows, dtype=torch.int64)
                self._block_cpu[:rows].copy_(ids[:rows].reshape(-1))
                ids = self._block_cpu[:rows]
            else:
                ids = torch.tensor(list(ids)[:rows], dtype=torch.int64)
            if self._use_traverse:
                xgr.reset_token_bitmask(bitmask)
                nxt, sib = self._tree_of(rows)
                self._matcher.traverse_draft_tree(nxt, sib, ids, bitmask)
            else:
                # Reference walk: identical masks, one Python call per row. Row r+1 needs
                # block[r+1] accepted, so the walk stops at the first draft the grammar
                # refuses -- and at one it accepts that ends the turn, because a matcher
                # that has taken the stop token has no next-token set at all.
                accepted = filled = 0
                for r in range(rows):
                    self._matcher.fill_next_token_bitmask(bitmask, r)
                    filled = r + 1
                    if r + 1 >= rows or not self._matcher.accept_token(int(ids[r + 1])):
                        break
                    accepted += 1
                    if self._matcher.is_terminated():
                        break
                rows = filled
                if accepted:
                    self._matcher.rollback(accepted)

        dev_mask = bitmask
        if logits.device.type != "cpu":
            self._bitmask_dev[:rows].copy_(bitmask[:rows])
            dev_mask = self._bitmask_dev
        xgr.apply_token_bitmask_inplace(logits[:rows], dev_mask[:rows], vocab_size=self._vocab)
        self.stats["mask_calls"] += 1
        self.stats["masked_rows"] += rows
        self.stats["mask_s"] += time.perf_counter() - t0
        return rows


# --------------------------------------------------------------------------
# Factory: tokenizer info + compiled-grammar cache
# --------------------------------------------------------------------------

class ToolGrammarFactory:
    """Builds one :class:`ToolCallGrammar` per request, caching compiled grammars.

    The expensive part is compiling an EBNF against a 129k-token vocabulary, and
    a chat client sends the same tool list on every turn of a conversation, so
    the compiled grammar is cached under the grammar text. A compiled grammar of
    twenty tools holds about 13 MB of adaptive token-mask cache, and this box has
    little host memory to spare, so the cache is small.
    """

    def __init__(self, tok, *, eos_id: int, max_calls: int = DEFAULT_MAX_CALLS,
                 cache_size: int = 4, use_traverse: bool = True) -> None:
        import torch  # noqa: PLC0415 - optional, and only needed with a real engine
        import xgrammar as xgr  # noqa: PLC0415

        self._torch = torch
        self._xgr = xgr
        self._tok = tok
        self._eos_id = eos_id
        self._max_calls = max_calls
        self._use_traverse = use_traverse
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._cache_size = cache_size

        vocab = self._encoded_vocab(tok)
        self._vocab_size = len(vocab)
        self._info = xgr.TokenizerInfo(vocab, xgr.VocabType.BYTE_LEVEL,
                                       vocab_size=self._vocab_size, stop_token_ids=[eos_id])
        self._compiler = xgr.GrammarCompiler(self._info, max_threads=8)
        self.compile_s = 0.0

    @staticmethod
    def _encoded_vocab(tok) -> List[str]:
        """The vocabulary in id order, as the tokenizer stores it."""
        inner = getattr(tok, "_tk", tok)
        if hasattr(inner, "get_vocab"):
            vocab = inner.get_vocab(with_added_tokens=True) if getattr(tok, "backend", "") == "tokenizers" \
                else inner.get_vocab()
        else:
            raise TypeError(f"cannot read a vocabulary out of {type(tok).__name__}")
        size = max(vocab.values()) + 1
        get_size = getattr(inner, "get_vocab_size", None)
        if callable(get_size):
            size = max(size, get_size(with_added_tokens=True))
        out = [""] * size
        for piece, idx in vocab.items():
            if 0 <= idx < size:
                out[idx] = piece
        return out

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def compile(self, ebnf: str):
        cg = self._cache.get(ebnf)
        if cg is not None:
            self._cache.move_to_end(ebnf)
            return cg
        t0 = time.perf_counter()
        cg = self._compiler.compile_grammar(self._xgr.Grammar.from_ebnf(ebnf))
        self.compile_s = time.perf_counter() - t0
        self._cache[ebnf] = cg
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        log.info("compiled a tool grammar for %d rule(s) in %.0f ms",
                 ebnf.count("::="), self.compile_s * 1e3)
        return cg

    def for_tools(self, tools: Sequence[Dict[str, Any]]) -> Optional[ToolCallGrammar]:
        """A gate for this request's tools, or None to leave decoding alone."""
        try:
            ebnf = build_tool_grammar(tools, max_calls=self._max_calls)
        except Exception as e:  # noqa: BLE001 - a bad schema must not fail the request
            log.warning("tool grammar build failed: %s", e)
            return None
        if not ebnf:
            return None
        try:
            cg = self.compile(ebnf)
        except Exception as e:  # noqa: BLE001
            log.warning("tool grammar compile failed: %s", e)
            if os.environ.get("DSV41_LOG_TOOL_GRAMMAR") == "1":
                log.warning("grammar was:\n%s", ebnf)
            return None
        if os.environ.get("DSV41_LOG_TOOL_GRAMMAR") == "1":
            log.info("tool grammar:\n%s", ebnf)
        return ToolCallGrammar(cg, xgr=self._xgr, torch=self._torch, eos_id=self._eos_id,
                               vocab_size=self._vocab_size, decode=self._tok.decode,
                               use_traverse=self._use_traverse)


def make_factory(tok, eos_id: int, *, enabled: bool = True, **kw) -> Optional[ToolGrammarFactory]:
    """A factory, or None (with one log line) when constrained decoding is off."""
    if not enabled:
        log.info("tool-call grammar disabled (DSV41_TOOL_GRAMMAR=0)")
        return None
    try:
        f = ToolGrammarFactory(tok, eos_id=eos_id, **kw)
    except ImportError as e:
        log.warning("tool-call grammar unavailable (%s); tool calls fall back to text parsing", e)
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("tool-call grammar could not be set up (%s); falling back to text parsing", e)
        return None
    log.info("tool-call grammar ready (xgrammar, vocab %d)", f.vocab_size)
    return f


if __name__ == "__main__":  # print the grammar for a tool list on stdin or a demo
    import sys
    if not sys.stdin.isatty():
        payload = json.load(sys.stdin)
        tools = payload.get("tools", payload) if isinstance(payload, dict) else payload
    else:
        tools = [{"type": "function", "function": {
            "name": "web_search",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"]}}}]
    print(build_tool_grammar(tools) or "(no grammar)")
