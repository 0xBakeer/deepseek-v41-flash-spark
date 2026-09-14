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
    BLOCK_CLOSE, NEG_INF, TOOL_CALLS_MARKER, PlainTextGate, ToolCallGrammar,
    ToolGrammarFactory, _ValueTracker, build_tool_grammar, dsml_safe,
    markup_unbalanced,
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
# the `</` boundary: markup balance, the value tracker, the plain-text gate
# ---------------------------------------------------------------------------

PAGE = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<title>Lumen</title>\n</head>\n<body>\n<img src="a.png">\n'
        '<p>Hamburg</p>\n</body>\n</html>\n')


def test_markup_balanced_page_is_balanced():
    assert markup_unbalanced(PAGE) is False
    # ... and every prefix that still owes a closing tag is not
    assert markup_unbalanced(PAGE[:PAGE.index("Lumen") + len("Lumen")]) is True


def test_markup_void_and_self_closing_elements_do_not_count():
    assert markup_unbalanced('<div><br><img src="x"><hr></div>') is False
    assert markup_unbalanced('<svg><circle r="1"/><rect x="0"/></svg>') is False
    assert markup_unbalanced("<div><br><span>") is True


def test_markup_comments_are_not_tags():
    assert markup_unbalanced("<html><body><!-- <div><div><div> --></body></html>") is False


def test_markup_needs_to_look_like_markup_at_all():
    for value in ["",
                  "just a sentence about a < b and c > d",
                  '{"a": 1, "b": "<3", "c": [1, 2]}',
                  "def f(x):\n    return x < 3 and x > 1\n",
                  "<p>one paragraph</p>",
                  "<div>one tag name is not a document"]:
        assert markup_unbalanced(value) is False, value
    # a single tag name is enough when it is the document element itself
    assert markup_unbalanced("<html>and then nothing") is True


def test_value_tracker_follows_the_value():
    t = _ValueTracker()
    t.feed(TOOL_CALLS_MARKER + f'>\n<{D} invoke name="w">\n<{D} parameter name="q" string="true">')
    assert t.in_value and t.text() == ""
    t.feed("<title>Lumen</")
    assert t.text() == "<title>Lumen</"
    mark = t.mark()                      # a speculative walk over drafts ...
    t.feed("title></head>")
    assert t.text() == "<title>Lumen</title></head>"
    t.restore(mark)                      # ... and back where it started
    assert t.in_value and t.text() == "<title>Lumen</"
    t.feed(f"{D} parameter>\n")          # the bar can only be the closing tag
    assert not t.in_value and t.text() == ""
    t.feed(f'<{D} parameter name="n" string="false">42')
    assert t.in_value and t.text() == "42"


class _FakeLogits:
    """A [R, V] logits stand-in that records the assignments made to it."""

    def __init__(self, rows: int = 1, cols: int = 256, dims: int = 2) -> None:
        self.shape = (rows, cols) if dims == 2 else (cols,)
        self._dims = dims
        self.assigned = []

    def dim(self):
        return self._dims

    def __setitem__(self, key, value):
        self.assigned.append((key, value))


def test_plain_text_gate_masks_the_bar_in_every_row():
    gate = PlainTextGate(128825)
    assert gate.active is False
    gate.observe([1, 2, 3])                       # a no-op, and must stay one
    rows = _FakeLogits(6)
    assert gate.mask_rows(rows, None) == 6
    assert rows.assigned == [((slice(None), 128825), NEG_INF)]
    one = _FakeLogits(1, dims=1)
    assert gate.mask_rows(one) == 1
    assert one.assigned == [(128825, NEG_INF)]
    assert gate.stats["masked_rows"] == 7 and gate.stats["mask_calls"] == 2
    assert gate.stats["mask_s"] >= 0.0


def test_plain_text_gate_drops_itself_rather_than_the_request():
    class Boom(_FakeLogits):
        def __setitem__(self, key, value):
            raise RuntimeError("this tensor does not take assignments")

    gate = PlainTextGate(7)
    assert gate.mask_rows(Boom(2)) == 0
    assert "error" in gate.stats
    assert gate.mask_rows(_FakeLogits(2)) == 0    # and stays off for the rest of the request


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
    """The grammar itself never engages on prose -- only rule 2's single column."""
    if env() is None:
        return
    import torch
    tok, enc, eos, factory = env()
    gate = factory.for_tools(TOOLS)
    prose = "Sure. Let me look that up for you; I will call the search tool now."
    for t in tok.encode(prose):
        gate.observe([t])
        assert not gate.active, "the grammar engaged on ordinary prose"
    logits = torch.zeros(1, factory.vocab_size)
    gate.mask_rows(logits, None)
    assert int(logits.isinf().sum()) == 1, "an inactive gate masked more than the DSML bar"
    assert bool(logits[0, factory.bar_id].isinf())


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


# ---------------------------------------------------------------------------
# the `</` boundary, with the real tokenizer behind it
# ---------------------------------------------------------------------------

VALUE_PAGE = ('<!DOCTYPE html><html><head><title>Lumen</title></head>'
              '<body><p>Hamburg</p></body></html>\n')


def _gate_in_value(factory, tok, value):
    """A gate that has watched the block opened and `value` written into a parameter."""
    gate = factory.for_tools(TOOLS)
    head = (TOOL_CALLS_MARKER + ">\n" + f'<{D} invoke name="web_search">\n'
            + f'<{D} parameter name="query" string="true">' + value)
    gate.observe(tok.encode(head))
    assert gate.active, "the gate did not engage on the marker"
    return gate


def test_the_three_ids_are_single_tokens():
    if env() is None:
        return
    tok, enc, eos, factory = env()
    assert factory.bar_id is not None, "the DSML bar is not one token; the rules are off"
    assert factory.close_prefix_id is not None, "`</` is not one token; the rules are off"
    assert factory.lt_id in factory.lt_ids and factory.lt_ids, factory.lt_ids
    assert tok.decode([factory.bar_id]) == D
    assert tok.decode([factory.close_prefix_id]) == "</"


def test_guard_masks_the_bar_after_an_unclosed_tag():
    """`<title>Lumen</` may only continue as an HTML close tag."""
    if env() is None:
        return
    import torch
    tok, enc, eos, factory = env()
    bar, close = factory.bar_id, factory.close_prefix_id
    title = tok.encode("title")
    gate = _gate_in_value(factory, tok, "<!DOCTYPE html><html><head><title>Lumen")
    gate.observe([close])
    logits = torch.zeros(1, factory.vocab_size)
    gate.mask_rows(logits, None)
    assert bool(logits[0, bar].isinf()), "the parameter close was reachable mid-document"
    assert not bool(logits[0, title[0]].isinf()), "the HTML close tag was masked as well"
    assert gate.stats["value_guard_masked"] == 1, gate.stats

    # the same inside a draft block: row i is guarded against block_ids[i]
    gate = _gate_in_value(factory, tok, "<!DOCTYPE html><html><head><title>Lumen")
    ids = torch.tensor([tok.encode("Lumen")[-1], close, title[0]], dtype=torch.int64)
    logits = torch.zeros(3, factory.vocab_size)
    gate.mask_rows(logits, ids)
    assert bool(logits[1, bar].isinf()), "the draft row after `</` was left reachable"
    assert not bool(logits[1, title[0]].isinf())
    assert gate.stats["value_guard_masked"] == 1, gate.stats
    # masking left the gate's own state where it found it
    assert gate._value.text().endswith("<title>Lumen"), gate._value.text()


def test_guard_lets_the_close_through_after_a_newline():
    """A value that has finished the file closes on the next line, and may."""
    if env() is None:
        return
    import torch
    tok, enc, eos, factory = env()
    gate = _gate_in_value(factory, tok, VALUE_PAGE)
    gate.observe([factory.close_prefix_id])
    logits = torch.zeros(1, factory.vocab_size)
    gate.mask_rows(logits, None)
    assert not bool(logits[0, factory.bar_id].isinf()), "a legitimate parameter close was trapped"
    assert gate.stats["value_guard_masked"] == 0, gate.stats


def test_idle_bar_mask_allows_only_the_open_angle():
    """Rule 2: outside the block the bar may only follow `<`."""
    if env() is None:
        return
    import torch
    tok, enc, eos, factory = env()
    bar = factory.bar_id
    gate = factory.for_tools(TOOLS)
    gate.observe(tok.encode("Here is the plan."))
    logits = torch.zeros(1, factory.vocab_size)
    assert gate.mask_rows(logits, None) == 1
    assert int(logits.isinf().sum()) == 1 and bool(logits[0, bar].isinf())
    gate.observe([factory.lt_id])
    logits = torch.zeros(1, factory.vocab_size)
    assert gate.mask_rows(logits, None) == 0
    assert int(logits.isinf().sum()) == 0, "the block could not be opened"

    # a draft block, through both the host list and the device tensor path
    ids = [tok.encode(" plan")[0], factory.lt_id, tok.encode("x")[0]]
    a, b = (torch.zeros(3, factory.vocab_size), torch.zeros(3, factory.vocab_size))
    ga, gb = factory.for_tools(TOOLS), factory.for_tools(TOOLS)
    for g in (ga, gb):
        g.observe(tok.encode("Here is the plan."))
    assert ga.mask_rows(a, ids) == 3
    assert gb.mask_rows(b, torch.tensor(ids, dtype=torch.int64)) == 3
    assert torch.equal(a.isinf(), b.isinf()), "the list and tensor paths disagree"
    assert int(a.isinf().sum()) == 2
    assert bool(a[0, bar].isinf()) and not bool(a[1, bar].isinf()) and bool(a[2, bar].isinf())


def test_masked_decode_loop_writes_a_whole_page_into_a_value():
    """The speculative loop writes a full HTML page as a parameter value, guard included."""
    if env() is None:
        return
    tok, enc, eos, factory = env()
    page = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<title>Lumen</title>\n</head>\n<body>\n<h1>Lumen</h1>\n'
            '<p>Hamburg</p>\n</body>\n</html>\n')
    text = block(invoke("web_search", param("query", page)))
    target = tok.encode(text) + [eos]
    seed = text[:text.index(TOOL_CALLS_MARKER) + len(TOOL_CALLS_MARKER)]
    gate = factory.for_tools(TOOLS)
    gate.observe(tok.encode(seed))
    assert gate.active, "the gate did not engage on the marker"
    already = len(tok.encode(seed))
    out = _masked_greedy_loop(gate, tok, eos, target[already:], factory.vocab_size, steps=2000)
    assert out and out[-1] == eos, "the loop did not stop at the end of turn"
    assert tok.decode(out) == tok.decode(target[already:]), tok.decode(out)[:200]
    assert gate.stats["accept_fail"] == 0, gate.stats
    # the guard fired inside the page, and the parameter still closed at the end
    assert gate.stats["value_guard_masked"] >= 1, gate.stats
    if enc is not None:
        parsed = enc.parse_message_from_completion_text(text + enc.eos_token, thinking_mode="chat")
        got = json.loads(parsed["tool_calls"][0]["function"]["arguments"])["query"]
        assert got == page, got[:200]


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
