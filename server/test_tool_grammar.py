#!/usr/bin/env python3
"""Tests for the DSML tool-call grammar (server/tool_grammar.py).

Two halves:

* the EBNF **builder**, which needs nothing but the standard library;
* the **matcher**, which needs xgrammar and the checkpoint's tokenizer. Those
  tests are skipped (and say so) where either is missing -- they run on the box,
  on CPU: no model weights are involved.

The matcher tests are written against the checkpoint's own parser: what the
grammar allows, ``parse_message_from_completion_text`` must accept, and the two
malformations seen in real completions must be unreachable.

Run: python3 server/test_tool_grammar.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tool_grammar import (  # noqa: E402
    BLOCK_CLOSE, TOOL_CALLS_MARKER, ToolCallGrammar, ToolGrammarFactory,
    build_tool_grammar, dsml_safe,
)

D = "｜DSML｜"

CANDIDATE_MODEL_DIRS = [
    os.environ.get("V41_MODEL_DIR", ""),
    os.path.expanduser("~/models/DeepSeek-V4.1-Flash"),
    os.path.join(os.path.dirname(HERE), "models", "DeepSeek-V4.1-Flash"),
]

# The three shapes a real tool list mixes: a search tool with a string parameter,
# a counter with an integer, a filter with an array, plus an object and a boolean.
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer"},
            "sites": {"type": "array", "items": {"type": "string"}},
            "opts": {"type": "object"},
            "safe": {"type": "boolean"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["metric", "imperial"]},
        }, "required": ["city"]}}},
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}},
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def model_dir():
    for d in CANDIDATE_MODEL_DIRS:
        if d and os.path.exists(os.path.join(d, "tokenizer.json")):
            return d
    return None


_ENV = {}


def env():
    """xgrammar + tokenizer + compiled grammar for TOOLS, or None."""
    if "v" in _ENV:
        return _ENV["v"]
    _ENV["v"] = None
    md = model_dir()
    if md is None:
        print("  (skipped: no model dir with tokenizer.json; set V41_MODEL_DIR)")
        return None
    try:
        import xgrammar  # noqa: F401,PLC0415
        import torch  # noqa: F401,PLC0415
    except ImportError as e:
        print(f"  (skipped: {e})")
        return None
    import app as A  # noqa: PLC0415

    tok = A.Tok(md)
    enc = A.load_encoding_module(md) if os.path.exists(
        os.path.join(md, "encoding", "encoding.py")) else None
    eos = tok.token_to_id("<｜end▁of▁sentence｜>")
    factory = ToolGrammarFactory(tok, eos_id=eos)
    _ENV["v"] = (tok, enc, eos, factory)
    return _ENV["v"]


def matcher_for(tools=None, *, use_traverse=True):
    tok, enc, eos, factory = env()
    gate = factory.for_tools(tools if tools is not None else TOOLS)
    assert gate is not None
    return tok, enc, eos, gate


def new_matcher(factory, tools, eos):
    import xgrammar as xgr
    cg = factory.compile(build_tool_grammar(tools))
    return xgr.GrammarMatcher(cg, override_stop_tokens=[eos], max_rollback_tokens=16)


def feed(m, tok, text):
    """Accept ``text`` token by token as the tokenizer segments it.

    Returns (ok, n_accepted, first_rejected_text).
    """
    ids = tok.encode(text)
    for i, t in enumerate(ids):
        if not m.accept_token(int(t)):
            return False, i, tok.decode([t])
    return True, len(ids), None


def mask_allows(m, factory, tid):
    import xgrammar as xgr
    bm = xgr.allocate_token_bitmask(1, factory.vocab_size)
    m.fill_next_token_bitmask(bm, 0)
    return bool((int(bm[0, tid // 32].item()) >> (tid % 32)) & 1)


def allowed_ids(m, factory):
    import xgrammar as xgr
    bm = xgr.allocate_token_bitmask(1, factory.vocab_size)
    m.fill_next_token_bitmask(bm, 0)
    out = []
    for i in range(factory.vocab_size):
        if (int(bm[0, i // 32].item()) >> (i % 32)) & 1:
            out.append(i)
    return out


def block(*invokes):
    return TOOL_CALLS_MARKER + ">\n" + "".join(invokes) + BLOCK_CLOSE


def invoke(name, *params):
    return f'<{D} invoke name="{name}">\n' + "".join(params) + f"</{D} invoke>\n"


def param(name, value, is_str=True):
    return f'<{D} parameter name="{name}" string="{str(is_str).lower()}">{value}</{D} parameter>\n'


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def test_builder_types():
    g = build_tool_grammar(TOOLS)
    assert 'name=\\"query\\" string=\\"true\\"' in g, g
    assert 'name=\\"max_results\\" string=\\"false\\">" jint' in g, g
    assert 'name=\\"safe\\" string=\\"false\\">" jbool' in g, g
    assert 'name=\\"opts\\" string=\\"false\\">" jobj' in g, g
    assert 'name=\\"sites\\" string=\\"false\\">" ("[" jws (jstr' in g, g
    assert '("metric" | "imperial")' in g, g          # enum becomes a literal choice
    assert "jval ::=" in g and "vany ::= [\\u0000-\\uFF5B\\uFF5D-\\U0010FFFF]*" in g, g


def test_builder_required_and_optional():
    g = build_tool_grammar(TOOLS)
    line = next(l for l in g.splitlines() if l.startswith("t0 ::="))
    assert " t0p0 " in line and "t0p0?" not in line, line     # query is required
    assert "t0p1?" in line and "t0p2?" in line, line          # the rest are not
    line = next(l for l in g.splitlines() if l.startswith("t2 ::="))
    assert "t2p" not in line, line                            # `now` takes nothing


def test_builder_escaping_and_rejection():
    tools = [
        {"type": "function", "function": {"name": "weird\\name", "parameters": {
            "type": "object", "properties": {"a\\b": {"type": "string"}, 'q"x': {"type": "string"}},
            "required": ["a\\b"]}}},
        {"type": "function", "function": {"name": 'bad"name', "parameters": {}}},
        {"type": "function", "function": {"name": f"bad{D}name", "parameters": {}}},
    ]
    g = build_tool_grammar(tools)
    assert '"<｜DSML｜ invoke name=\\"weird\\\\name\\">\\n"' in g, g
    assert 'name=\\"a\\\\b\\"' in g, g
    assert 'q\\"x' not in g, g            # a quote in a parameter name is not representable
    assert "bad" not in g.replace("weird\\name", ""), g
    assert dsml_safe("ok-name_1") and not dsml_safe('a"b') and not dsml_safe(f"a{D}b")
    assert not dsml_safe("a\nb") and not dsml_safe("")


def test_builder_empty():
    assert build_tool_grammar([]) is None
    assert build_tool_grammar([{"type": "function", "function": {"name": 'x"y'}}]) is None
    assert build_tool_grammar(None) is None


def test_builder_max_calls():
    assert "call{1,3}" in build_tool_grammar(TOOLS, max_calls=3)
    assert "call{1,1}" in build_tool_grammar(TOOLS, max_calls=0)


def test_builder_flat_and_openai_shapes():
    flat = [{"name": "web_search", "parameters": {"type": "object",
                                                  "properties": {"query": {"type": "string"}},
                                                  "required": ["query"]}}]
    wrapped = [{"type": "function", "function": flat[0]}]
    assert build_tool_grammar(flat) == build_tool_grammar(wrapped)


# ---------------------------------------------------------------------------
# matcher: what the grammar allows, the checkpoint's parser accepts
# ---------------------------------------------------------------------------

def test_matcher_accepts_a_well_formed_block():
    if env() is None:
        return
    tok, enc, eos, factory = env()
    m = new_matcher(factory, TOOLS, eos)
    text = block(
        invoke("web_search",
               param("query", "nvidia DGX Spark specs"),
               param("max_results", "5", is_str=False),
               param("sites", '["nvidia.com", "anandtech.com"]', is_str=False)),
        invoke("get_weather", param("city", "Berlin"), param("units", "metric")),
        invoke("now"),
    )
    ok, n, bad = feed(m, tok, text)
    assert ok, f"rejected token {n} ({bad!r}) of a well-formed block"
    assert m.is_completed(), "the block closed but the matcher is not complete"
    if enc is not None:
        parsed = enc.parse_message_from_completion_text(text + enc.eos_token, thinking_mode="chat")
        names = [c["function"]["name"] for c in parsed["tool_calls"]]
        assert names == ["web_search", "get_weather", "now"], parsed
        args = json.loads(parsed["tool_calls"][0]["function"]["arguments"])
        assert args == {"query": "nvidia DGX Spark specs", "max_results": 5,
                        "sites": ["nvidia.com", "anandtech.com"]}, args


def test_matcher_allows_awkward_values():
    """A value may hold anything but U+FF5C -- HTML, quotes, braces, newlines."""
    if env() is None:
        return
    tok, enc, eos, factory = env()
    for value in ['<div class="x">a & b</div>',
                  'she said "hi"\nand left',
                  '{"not": "json, just text"} </ almost a tag >',
                  "line1\nline2\n\n<｜Assistant｜".replace("｜", "|"),
                  ""]:
        m = new_matcher(factory, TOOLS, eos)
        text = block(invoke("web_search", param("query", value)))
        ok, n, bad = feed(m, tok, text)
        assert ok, f"value {value!r}: rejected token {n} ({bad!r})"
        assert m.is_completed()
        if enc is not None:
            parsed = enc.parse_message_from_completion_text(text + enc.eos_token, thinking_mode="chat")
            got = json.loads(parsed["tool_calls"][0]["function"]["arguments"])["query"]
            assert got == value, (got, value)


def test_malformation_value_in_the_string_attribute():
    """`string="nvidia DGX Spark specs"` -- logged in a real completion -- is unreachable."""
    if env() is None:
        return
    tok, enc, eos, factory = env()
    m = new_matcher(factory, TOOLS, eos)
    head = TOOL_CALLS_MARKER + ">\n" + f'<{D} invoke name="web_search">\n' \
        + f'<{D} parameter name="query" string="'
    ok, n, bad = feed(m, tok, head)
    assert ok, f"the legal prefix was rejected at {n} ({bad!r})"
    # only `true` and `false` can continue, and both only as the whole word
    for bad_start in ["nvidia DGX Spark specs", "a", "tru3", "TRUE", "1", '"']:
        mm = new_matcher(factory, TOOLS, eos)
        assert feed(mm, tok, head)[0]
        assert not feed(mm, tok, bad_start)[0], f"string=\"{bad_start}\" was accepted"
    # `query` is a string parameter, so only string="true" continues; the integer
    # next to it is the mirror image.
    mm = new_matcher(factory, TOOLS, eos)
    assert feed(mm, tok, head)[0]
    assert feed(mm, tok, 'true">')[0]
    mm = new_matcher(factory, TOOLS, eos)
    assert feed(mm, tok, head)[0]
    assert not feed(mm, tok, 'false">')[0], 'string="false" on a string parameter'
    int_head = (TOOL_CALLS_MARKER + ">\n" + f'<{D} invoke name="web_search">\n'
                + param("query", "a") + f'<{D} parameter name="max_results" string="')
    mm = new_matcher(factory, TOOLS, eos)
    assert feed(mm, tok, int_head)[0]
    assert feed(mm, tok, 'false">')[0]
    mm = new_matcher(factory, TOOLS, eos)
    assert feed(mm, tok, int_head)[0]
    assert not feed(mm, tok, 'true">')[0], 'string="true" on an integer parameter'
    # and the checkpoint's parser agrees the malformed shape is not parseable
    if enc is not None:
        malformed = (head + 'nvidia DGX Spark specs">\n' + f"</{D} invoke>\n" + BLOCK_CLOSE
                     + enc.eos_token)
        try:
            enc.parse_message_from_completion_text(malformed, thinking_mode="chat")
            raise AssertionError("the strict parser accepted the malformed attribute form")
        except (ValueError, AssertionError) as e:
            assert "AssertionError" not in type(e).__name__ or "strict parser" not in str(e), e


def test_value_cannot_contain_the_dsml_bar():
    """The terminator is only unambiguous because U+FF5C cannot occur in a value.

    xgrammar's negated character classes are ASCII-only and drop a non-ASCII
    codepoint with a warning, so ``[^｜]`` would compile to something that lets a
    value run over the closing tag and swallow the rest of the block. This is the
    test that catches that.
    """
    if env() is None:
        return
    tok, enc, eos, factory = env()
    head = TOOL_CALLS_MARKER + ">\n" + f'<{D} invoke name="web_search">\n' \
        + f'<{D} parameter name="query" string="true">abc'
    m = new_matcher(factory, TOOLS, eos)
    assert feed(m, tok, head)[0]
    assert not m.accept_string("｜"), "U+FF5C is legal inside a value; the terminator is ambiguous"
    assert not m.accept_string("｜DSML｜"), "a DSML token is legal inside a value"
    assert m.accept_string("é中\n\"<>&"), "a value should take any other character"
    # ... and the tool name and parameter name are just as closed
    m2 = new_matcher(factory, TOOLS, eos)
    assert feed(m2, tok, TOOL_CALLS_MARKER + ">\n" + f'<{D} invoke name="web')[0]
    assert not m2.accept_string("｜")


def test_malformation_prose_after_the_block():
    """Nothing but the end-of-turn token may follow `</｜DSML｜ calls>`."""
    if env() is None:
        return
    tok, enc, eos, factory = env()
    m = new_matcher(factory, TOOLS, eos)
    assert feed(m, tok, block(invoke("web_search", param("query", "spark"))))[0]
    assert m.is_completed()
    ids = allowed_ids(m, factory)
    assert ids == [eos], f"after the closing tag the mask allows {len(ids)} tokens, not just EOS"
    for prose in ["\n", "I", " I", "Let", "\n\nI will", f"<{D} invoke"]:
        mm = new_matcher(factory, TOOLS, eos)
        assert feed(mm, tok, block(invoke("web_search", param("query", "spark"))))[0]
        assert not feed(mm, tok, prose)[0], f"prose {prose!r} was accepted after the block"
    assert m.accept_token(eos) and m.is_terminated()


def test_matcher_rejects_unknown_names_and_shapes():
    if env() is None:
        return
    tok, enc, eos, factory = env()
    cases = {
        "unknown tool": block(invoke("rm_rf", param("path", "/"))),
        "unknown parameter": block(invoke("web_search", param("qeury", "x"))),
        "duplicate parameter": block(invoke("web_search", param("query", "a"), param("query", "b"))),
        "missing required": block(invoke("web_search", param("max_results", "3", is_str=False))),
        "string flag on an integer": block(invoke("web_search", param("query", "a"),
                                                  param("max_results", "3", is_str=True))),
        "quoted integer": block(invoke("web_search", param("query", "a"),
                                       param("max_results", '"3"', is_str=False))),
        "bare word in an array": block(invoke("web_search", param("query", "a"),
                                              param("sites", "[nvidia.com]", is_str=False))),
        "no closing tag": TOOL_CALLS_MARKER + ">\n" + invoke("web_search", param("query", "a")),
    }
    for name, text in cases.items():
        m = new_matcher(factory, TOOLS, eos)
        ok, n, bad = feed(m, tok, text)
        if name == "no closing tag":
            assert ok and not m.is_completed(), name    # legal prefix, but not a finished block
        else:
            assert not ok, f"{name}: the grammar accepted it"


def test_matcher_bounds_the_number_of_calls():
    """The spiral that ran to the output cap cannot be a legal continuation."""
    if env() is None:
        return
    tok, enc, eos, factory = env()
    import xgrammar as xgr
    cg = factory.compile(build_tool_grammar(TOOLS, max_calls=2))
    m = xgr.GrammarMatcher(cg, override_stop_tokens=[eos], max_rollback_tokens=16)
    one = invoke("web_search", param("query", "a"))
    assert feed(m, tok, TOOL_CALLS_MARKER + ">\n" + one + one)[0]
    assert not feed(m, tok, one)[0], "a third call was accepted with max_calls=2"


# ---------------------------------------------------------------------------
# the gate: activation, state tracking, mask equivalence
# ---------------------------------------------------------------------------

def test_gate_stays_out_of_prose():
    if env() is None:
        return
    tok, enc, eos, factory = env()
    gate = factory.for_tools(TOOLS)
    prose = "Sure. Let me look that up for you; I will call the search tool now."
    for t in tok.encode(prose):
        gate.observe([t])
        assert not gate.active, "the grammar engaged on ordinary prose"
    assert gate.mask_rows(_FakeLogits(), None) == 0


def test_gate_activates_on_the_marker_and_tracks_the_block():
    if env() is None:
        return
    tok, enc, eos, factory = env()
    text = "I will search." + block(invoke("web_search", param("query", "dgx spark")))
    ids = tok.encode(text)
    marker_seen = False
    gate = factory.for_tools(TOOLS)
    # feed in bursts of 6, the way a DSpark step emits
    for i in range(0, len(ids), 6):
        gate.observe(ids[i:i + 6])
        marker_seen = marker_seen or gate.active
    assert gate.active and gate.completed, (gate.active, gate.stats)
    assert gate.stats["accept_fail"] == 0


def test_gate_mask_paths_agree():
    """traverse_draft_tree and the per-row walk must produce the same masks."""
    if env() is None:
        return
    import torch
    tok, enc, eos, factory = env()
    prefix = "ok." + TOOL_CALLS_MARKER + ">\n" + f'<{D} invoke name="web_'
    rest = tok.encode('search">\n<｜DSML｜ parameter name="query" string="true">a')
    a = factory.for_tools(TOOLS)
    b = factory.for_tools(TOOLS)
    b._use_traverse = False
    for g in (a, b):
        g.observe(tok.encode(prefix))
        assert g.active
    # a six-token block: the real token, four legal drafts, one illegal draft
    drafts = rest[:4] + [tok.encode("ZZZ never legal here")[0]]
    blk = torch.tensor([tok.encode(prefix)[-1]] + drafts, dtype=torch.int64)
    la = torch.zeros(6, factory.vocab_size)
    lb = torch.zeros(6, factory.vocab_size)
    ra = a.mask_rows(la, blk)
    rb = b.mask_rows(lb, blk)
    assert ra >= 5 and rb >= 5, (ra, rb)
    n = min(ra, rb)
    assert torch.equal(la[:n].isinf(), lb[:n].isinf()), "traverse and the row walk disagree"
    # masking must not move the matcher on
    assert not a.completed and not b.completed
    assert a.stats["mask_calls"] == 1 and b.stats["mask_calls"] == 1
    # the illegal draft is masked out of the row it is verified against (row 4 holds the
    # distribution drafts[4] is compared with), so the verifier cannot accept it
    bad = drafts[-1]
    assert bool(la[4][bad].isinf()) and bool(lb[4][bad].isinf()), "an illegal draft was left reachable"
    # and a legal draft is not masked there
    assert not bool(la[3][drafts[3]].isinf())


def _masked_greedy_loop(gate, tok, eos, target, vocab, steps=400):
    """Run the engine's speculative accept/reject arithmetic against the gate.

    The arithmetic is the one in ``engine/v41_engine.py``'s lean path -- argmax of
    the six rows, cumprod of the draft comparisons, bonus at index ``a`` -- with
    the model replaced by logits that want ``target`` but want the end-of-turn
    token even more, and a drafter that proposes the right token on some
    positions and the end-of-turn token on the others. Masking is the only thing
    that can keep the loop on ``target``: without it the first step emits EOS.
    """
    import torch
    out = []
    logits = torch.zeros(6, vocab)
    for step in range(steps):
        i = len(out)
        if i >= len(target):
            break
        want = [target[min(i + r, len(target) - 1)] for r in range(6)]
        logits.zero_()
        logits[:, eos] = 20.0                       # the trap
        for r, t in enumerate(want):
            logits[r, t] = 10.0
        if step % 2 == 0:
            drafts = torch.tensor(want[:5], dtype=torch.int64)
        else:
            drafts = torch.tensor([want[0], eos, want[2], eos, want[4]], dtype=torch.int64)
        block = torch.cat([torch.tensor([out[-1] if out else target[0]], dtype=torch.int64), drafts])
        gate.mask_rows(logits, block)
        am = logits.argmax(-1)
        acc = am[:5].eq(drafts).to(torch.int32).cumprod(0)
        a = int(acc.sum())
        cand = am.tolist()
        new, bonus = cand[:a], cand[a]
        for j, t in enumerate(new):
            if t == eos:
                a, new, bonus = j + 1, new[:j + 1], None
                break
        emitted = list(new) + ([bonus] if bonus is not None else [])
        gate.observe(emitted)
        out += emitted
        if eos in emitted:
            break
    return out


def test_masked_decode_loop_reproduces_a_valid_block():
    """The full speculative loop, gate included, on CPU and without the model."""
    if env() is None:
        return
    tok, enc, eos, factory = env()
    text = block(
        invoke("web_search", param("query", "dgx spark memory bandwidth"),
               param("max_results", "3", is_str=False)),
        invoke("get_weather", param("city", "Berlin")),
    )
    target = tok.encode(text) + [eos]
    for traverse in (True, False):
        gate = factory.for_tools(TOOLS)
        gate._use_traverse = traverse
        # the model has already written the marker; the gate engages on it
        marker_at = text.index(TOOL_CALLS_MARKER)
        gate.observe(tok.encode("Let me look that up." + text[:marker_at + len(TOOL_CALLS_MARKER)]))
        assert gate.active, "the gate did not engage on the marker"
        already = len(tok.encode(text[:marker_at + len(TOOL_CALLS_MARKER)]))
        out = _masked_greedy_loop(gate, tok, eos, target[already:], factory.vocab_size)
        got = tok.decode(out)
        assert out[-1] == eos, f"traverse={traverse}: the loop did not stop at the end of turn"
        assert got == text[len(tok.decode(target[:already])):] + tok.decode([eos]), \
            f"traverse={traverse}: {got[:200]!r}"
        # the end-of-turn token terminates the matcher, so the gate is no longer in force;
        # what must hold is that nothing the loop emitted was ever refused
        assert gate.stats["accept_fail"] == 0, gate.stats
        assert not gate.active, "the gate is still constraining after the end of turn"
        parsed = enc.parse_message_from_completion_text("x" + text + enc.eos_token,
                                                        thinking_mode="chat")
        assert [c["function"]["name"] for c in parsed["tool_calls"]] == ["web_search", "get_weather"]


class _FakeLogits:
    """Enough of a tensor for the inactive early-out."""
    shape = (1, 1)

    def dim(self):
        return 2


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
