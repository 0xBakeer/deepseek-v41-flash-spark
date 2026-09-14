#!/usr/bin/env python3
"""gate_profile.py -- make a profile write, and judge what it writes.

A keep-set is chosen on coverage, and coverage is measured teacher-forced: the
trace replays text that already exists, so the next token is the right one
whether or not the model could have found it. Perplexity on that trace cannot
see the way a starved keep-set actually fails, because the failure only happens
while the model is writing its OWN continuation and nothing is there to pull it
back. Observed on one single-file HTML prompt, 2026-09-12, all on keep-sets
whose coverage bars looked fine:

  * empty rules -- `* { }` -- and then the whole `<style>` block again, and
    again, to the token cap
  * inside the think block: "I keep. / I write. / I keep. / I write."
  * at temperature 0, after "I'll write the code now.": "Let me write."
    forever, and an answer of length ZERO -- a 200 with an empty message
  * token corruption the model then loops trying to repair: `color-scheme` ->
    `color-s-s-mode`, a title `Tic-Tac-Toe` written with runs of U+2011,
    `'Seg UI'` for `'Segoe UI'`
  * a 5,000-character answer whose last paragraph is one sentence, verbatim,
    over and over

Four of those five produce MORE text than a good answer. So length is never
evidence here: every judgement below is structural, and a prompt passes only
when the thing it asked for is actually in the output.

The gate therefore runs FREE generation, with thinking on, on prompts drawn
from the profile's own topics -- the register a profile claims is exactly the
register it has to be tested in. Prompts are keyed by TOPIC, not by profile, so
a profile's suite is the union of its topics' prompts and a new profile needs no
new prompts.

  python3 tools/gate_profile.py --profile frontend
  python3 tools/gate_profile.py --topics python,sql,english --dry-run
  python3 tools/gate_profile.py --profile chat --thinking both --effort 60

The verdict carries two numbers, because they answer different questions. The
strict one -- N of M runs passed -- is the gate, and it is unforgiving on
purpose: a 12-word fragment redrafted three times fails the row wherever it
sits, the think block included. The second says how many runs finished the
answer the prompt asked for -- the strict passes plus the misses whose only
fault was that repeat. A keep-set that writes correct pages while redrafting a
line behind the scenes and one that writes `* { }` to the cap both read as
"6 of 11", and they are not the same result, so every miss is named as well:
think-exit, guard, corrupt, content, repeat.

Exit 0 only if every prompt passed. Stdlib only: no torch, no model load, and
the checks import nothing, so tools/test_gate_profile.py runs them anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from urllib import error as urlerror
from urllib import request as urlrequest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# checks
# =============================================================================
# One dict of pure functions of the output text. Every one returns (ok, why),
# and `why` is written even when ok is True -- the report shows what a pass was
# worth, which is the difference between "20 declarations" and "barely legal".
#
# The second argument is the prompt's own `want` field, which is what that check
# needs and nothing more: a list of substrings the code must contain, a
# (min, max) sentence range, a script name, the acceptable forms of a number.
# It stays a pure function of (text, want) so a test can call it directly.


def _fenced(text: str) -> str:
    """The code out of the fenced blocks, or the whole text when there are none.

    A model asked for a file answers with a ```html fence about half the time,
    and an unterminated fence is itself a symptom worth surviving: a generation
    cut off mid-file has an opening fence and no closing one, and the check
    should judge what it wrote rather than fall over.
    """
    blocks = re.findall(r"```[A-Za-z0-9_+#-]*\n(.*?)```", text, re.S)
    if blocks:
        return "\n".join(blocks)
    m = re.search(r"```[A-Za-z0-9_+#-]*\n(.*)\Z", text, re.S)
    return m.group(1) if m else text


def _decls(css: str) -> int:
    """`property: value;` pairs. Not `{`-counting: the failure shape is a rule
    set with the right braces and nothing between them."""
    return len(re.findall(r"[-a-zA-Z_][-a-zA-Z0-9_]*\s*:\s*[^;{}]+;", css))


def _empty_rules(css: str) -> int:
    return len(re.findall(r"\{\s*\}", css))


# `if (x) {` is indistinguishable from a method definition to a regex, and a
# page whose script is nothing but control flow would otherwise sail through a
# "4 functions" check.
_NOT_A_DEF = {"if", "for", "while", "switch", "catch", "do", "else", "return", "typeof", "with"}


def _js_defs(js: str) -> int:
    n = len(re.findall(r"\bfunction\s*[A-Za-z_$][\w$]*\s*\(", js))
    n += len(re.findall(r"\bfunction\s*\(", js))
    n += len(re.findall(r"=>", js))
    for m in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*\{", js):
        if m.group(1) not in _NOT_A_DEF:
            n += 1
    return n


def _named_js(js: str) -> bool:
    return bool(re.search(r"\bfunction\s+[A-Za-z_$][\w$]*|\bclass\s+[A-Za-z_$][\w$]*"
                          r"|\b(?:const|let|var)\s+[A-Za-z_$][\w$]*\s*=\s*(?:async\s*)?"
                          r"(?:\([^()]*\)|[A-Za-z_$][\w$]*)\s*=>", js))


def _missing(text: str, want) -> list:
    """The `want` substrings that are not in the text. Case-insensitive: a
    keyword's case is the language's business, not the gate's."""
    low = text.lower()
    return [w for w in (want or []) if w.lower() not in low]


_SENT_END = ".!?\u3002\uff01\uff1f"


def _sentences(text: str) -> list:
    return [s.strip() for s in re.findall(rf"[^{re.escape(_SENT_END)}\n]+[{re.escape(_SENT_END)}]+", text)
            if s.strip()]


def _paragraphs(text: str) -> list:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _prose_shape(text: str, rng=None) -> tuple:
    """The structure every prose answer must have, whatever language it is in.

    The repeated-sentence rule is here and not only in the universal n-gram
    check because a short sentence repeated twice -- "The cache is warm." at the
    end of five paragraphs -- is under the 12-word window and is still the same
    failure.
    """
    lo, hi = rng or (4, 150)
    paras, sents = _paragraphs(text), _sentences(text)
    if len(paras) < 2:
        return False, "one paragraph; the answer must be at least two"
    if not lo <= len(sents) <= hi:
        return False, f"{len(sents)} sentences, wanted {lo}-{hi}"
    dupes = [s for s, n in Counter(s for s in sents if len(s.split()) >= 5).items() if n > 1]
    if dupes:
        return False, f"sentence repeated verbatim: {dupes[0][:60]!r}"
    return True, f"{len(paras)} paragraphs, {len(sents)} sentences"


# Letters, by script. A natural-language topic that comes back in the wrong
# script is the loudest keep-set failure there is, and it has happened here: the
# `general` profile answered an Arabic prompt with a collage of Polish,
# Portuguese and Romanian (results/keepsets/code/GATE.md).
_SCRIPTS = {
    "arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F]"),
    "cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "han": re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]"),
    "kana": re.compile(r"[\u3040-\u30FF]"),
}


def html_page(text, want=None):
    """A single self-contained page: one style block, one script block, real
    CSS in the first and real functions in the second. This is the prompt every
    failure in the docstring came from, so it is the strictest check here."""
    code = _fenced(text)
    if not re.search(r"<!doctype html", code, re.I):
        return False, "no <!doctype html>"
    styles, scripts = re.findall(r"<style\b", code, re.I), re.findall(r"<script\b", code, re.I)
    if len(styles) != 1:
        return False, f"{len(styles)} <style> blocks, wanted exactly 1"
    if len(scripts) != 1:
        return False, f"{len(scripts)} <script> blocks, wanted exactly 1"
    css = "".join(re.findall(r"<style\b[^>]*>(.*?)</style>", code, re.I | re.S))
    js = "".join(re.findall(r"<script\b[^>]*>(.*?)</script>", code, re.I | re.S))
    nd, ne, nf = _decls(css), _empty_rules(css), _js_defs(js)
    if nd < 20:
        return False, f"{nd} CSS declarations, wanted 20"
    if ne > 2:
        return False, f"{ne} empty rules such as `* {{ }}`"
    if nf < 4:
        return False, f"{nf} JS functions, wanted 4"
    miss = _missing(code, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{nd} declarations, {nf} functions, {ne} empty rules"


def css_block(text, want=None):
    css = _fenced(text)
    nd, ne = _decls(css), _empty_rules(css)
    if nd < 15:
        return False, f"{nd} declarations, wanted 15"
    if ne > 1:
        return False, f"{ne} empty rules"
    miss = _missing(css, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{nd} declarations, {ne} empty rules"


def js_function(text, want=None):
    js = _fenced(text)
    n = _js_defs(js)
    if n < 3:
        return False, f"{n} functions or arrows, wanted 3"
    if not _named_js(js):
        return False, "no named function, class or assigned arrow"
    miss = _missing(js, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{n} callables"


def ts_generic(text, want=None):
    ts = _fenced(text)
    if not re.search(r"<\s*[A-Z]\w*(?:\s*,\s*[A-Z]\w*)*\s+extends\s", ts):
        return False, "no generic parameter with an `extends` constraint"
    if not re.search(r"\)\s*:\s*[A-Za-z_{\[]", ts):
        return False, "no annotated return type"
    if re.search(r":\s*any\b", ts):
        return False, "`any` in a signature"
    miss = _missing(ts, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, "generic constrained, return typed"


def python_script(text, want=None):
    py = _fenced(text)
    defs = len(re.findall(r"^\s*(?:async\s+)?def\s+\w+", py, re.M))
    if defs < 2:
        return False, f"{defs} defs, wanted 2"
    if not re.search(r"^\s*(?:import\s+\w|from\s+\w+.*\simport\s)", py, re.M):
        return False, "no import"
    # `...` is what a model writes instead of the body it was asked for, and it
    # is syntactically valid Python, so nothing downstream would complain.
    if re.search(r"^\s*\.\.\.\s*$", py, re.M):
        return False, "`...` placeholder instead of a body"
    miss = _missing(py, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{defs} defs"


def sql_query(text, want=None):
    sql = _fenced(text)
    for kw in ("SELECT", "JOIN", "OVER"):
        if not re.search(rf"\b{kw}\b", sql, re.I):
            return False, f"no {kw}"
    miss = _missing(sql, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, "select, join, window"


def yaml_doc(text, want=None):
    y = _fenced(text)
    keys = len(re.findall(r"^\s*[-\w.]+\s*:", y, re.M))
    if keys < 8:
        return False, f"{keys} keys, wanted 8"
    if not re.search(r"(?<![\w&])&[A-Za-z_][\w-]*", y):
        return False, "no YAML anchor"
    if not re.search(r"(?:<<\s*)?\*[A-Za-z_][\w-]*", y):
        return False, "no alias referring to an anchor"
    # A tab in the indentation is not a style opinion: YAML forbids it, and
    # every parser rejects the file outright.
    if re.search(r"^\t| \t", y, re.M):
        return False, "tab in the indentation -- YAML forbids it"
    miss = _missing(y, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{keys} keys, anchored"


def shell_pipeline(text, want=None):
    sh = _fenced(text)
    stages = [ln for ln in sh.splitlines() if len(re.findall(r"(?<!\|)\|(?!\|)", ln)) >= 1]
    if not stages:
        return False, "no pipeline"
    # The prompt asks for quoting because an unquoted expansion is the bug this
    # register exists to avoid, and a model that has lost the register writes
    # $f where it means "$f".
    if not re.search(r'"\$', sh):
        return False, "no quoted expansion (\"$...\")"
    miss = _missing(sh, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{len(stages)} pipelines"


def latex_doc(text, want=None):
    tex = _fenced(text)
    for cmd in (r"\documentclass", r"\begin{document}", r"\end{document}"):
        if cmd not in tex:
            return False, f"no {cmd}"
    envs = set(re.findall(r"\\begin\{(\w+\*?)\}", tex)) - {"document"}
    if len(envs) < 2:
        return False, f"{len(envs)} environments besides document, wanted 2"
    miss = _missing(tex, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{len(envs)} environments"


def code_block(text, want=None):
    """The generic one, for the languages that do not have a check of their own.
    `want` carries the constructs that language cannot be written without, which
    is where the bite is -- Go without `err != nil` is not Go."""
    code = _fenced(text)
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if len(lines) < 12:
        return False, f"{len(lines)} non-blank lines, wanted 12"
    if code.count("{") != code.count("}"):
        return False, f"unbalanced braces ({code.count('{')} open, {code.count('}')} close)"
    miss = _missing(code, want)
    if miss:
        return False, f"missing {', '.join(miss)}"
    return True, f"{len(lines)} lines"


def prose(text, want=None):
    """`want` is an optional (min, max) sentence range."""
    return _prose_shape(text, want)


def prose_markers(text, want=None):
    """Prose that has to be in a particular language. `want` is a list of words
    common in it; four distinct ones have to appear.

    Asked for French, a starved keep-set has answered in Italian and then
    repeated one phrase to the cap. Counting markers is what separates "wrote
    French" from "wrote a Romance language", which no structural check can do.
    """
    ok, why = _prose_shape(text)
    if not ok:
        return False, why
    low = text.lower()
    hit = [w for w in (want or []) if re.search(rf"(?<!\w){re.escape(w.lower())}(?!\w)", low)]
    if len(hit) < 4:
        return False, f"only {len(hit)} of {len(want or [])} language markers ({', '.join(hit) or 'none'})"
    return True, f"{why}, {len(hit)} markers"


def prose_script(text, want=None):
    """Prose that has to be in a particular writing system: `want` is one of
    arabic, cyrillic, han, kana."""
    rx = _SCRIPTS.get(want or "")
    if rx is None:
        return False, f"unknown script {want!r}"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, "no letters at all"
    frac = sum(1 for c in letters if rx.match(c)) / len(letters)
    if frac < 0.6:
        return False, f"only {frac:.0%} of the letters are {want}"
    # Japanese without kana is Chinese: the Han share alone cannot tell them
    # apart, and answering a Japanese prompt in Chinese is exactly the kind of
    # neighbour-language drift a thin keep-set produces.
    if want == "kana" and len(_SCRIPTS["kana"].findall(text)) < 20:
        return False, "han without kana -- that is Chinese, not Japanese"
    ok, why = _prose_shape(text)
    return (ok, f"{frac:.0%} {want}, {why}") if ok else (False, why)


def numeric_answer(text, want=None):
    """A question with one right number. `want` lists the acceptable ways of
    writing it; any one of them has to appear in the ANSWER, not the reasoning.

    These prompts are traps on purpose. The obvious arithmetic gives a different
    number, so an answer that merely looks confident still fails.
    """
    low = text.lower()
    for form in (want or []):
        if form.lower() in low:
            return True, f"says {form}"
    return False, f"none of {', '.join(want or [])} in the answer"


CHECKS = {
    "html_page": html_page,
    "css_block": css_block,
    "js_function": js_function,
    "ts_generic": ts_generic,
    "python_script": python_script,
    "sql_query": sql_query,
    "yaml_doc": yaml_doc,
    "shell_pipeline": shell_pipeline,
    "latex_doc": latex_doc,
    "code_block": code_block,
    "prose": prose,
    "prose_markers": prose_markers,
    "prose_script": prose_script,
    "numeric_answer": numeric_answer,
}


# =============================================================================
# the universal checks
# =============================================================================
# These run on every output whatever it was asked for, and each one is a failure
# on its own. They are the ones that catch the shapes in the docstring, and they
# are deliberately domain-blind: a keep-set does not degenerate in a way that
# respects the subject.

def repeated_ngram(text: str, n: int = 12, times: int = 3):
    """The most-repeated n-word window, when it repeats at least `times`.

    Twelve words, because in code that spans several lines and in prose it is
    most of a sentence -- long enough that three ordinary event handlers or
    three list items do not collide, short enough that a four-word think-block
    cycle ("I keep. I write.") tiles it exactly.
    """
    words = text.lower().split()
    if len(words) < n:
        return None
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    return (" ".join(gram), count) if count >= times else None


_RUN = re.compile(r"(\S)\1\1+")


def _is_rule_line(line: str, ch: str) -> bool:
    """A deliberate separator -- `------`, `======`, a box-drawing rule -- is a
    line that is nearly all one character. Corruption is not: it sits inside a
    line of ordinary text."""
    s = line.strip()
    return bool(s) and s.count(ch) / len(s) >= 0.6


_TOOL_MARK = "\uff5cDSML\uff5c"   # the fullwidth bars of the model's tool-call markup


def corrupt_run(text: str):
    """A run of three or more identical non-alphanumeric characters that no
    writer meant.

    Two shapes, because a blanket rule is unusable: `===`, `---`, `...` and a
    docstring fence are ordinary code and ordinary markdown. What is never
    ordinary is
    * a repeated character outside ASCII -- `Tic‑‑‑‑Tac‑‑‑‑Toe` is U+2011 four
      times, and the model then loops trying to repair the word;
    * a repeated ASCII punctuation mark welded INSIDE a word (`Tic---Tac`),
      where a separator or an operator cannot be.
    Whitespace is excluded outright: three blank lines are just three blank
    lines.

    A third shape is not a run at all but is the same fault: the model's own
    tool-call markup (`<｜DSML｜…>`) inside prose or a page. It leaks at a `</`
    boundary in place of an HTML close tag -- `</｜DSML｜ parameter>` where
    `</title>` belonged -- and no writer means it either.
    """
    k = text.find(_TOOL_MARK)
    if k >= 0:
        line = text[text.rfind("\n", 0, k) + 1: (text.find("\n", k) + 1 or len(text) + 1) - 1]
        return _TOOL_MARK, line.strip()[:60]
    for m in _RUN.finditer(text):
        ch, i, j = m.group(1), m.start(), m.end()
        if ch.isalnum() or ch.isspace():
            continue
        line = text[text.rfind("\n", 0, i) + 1: (text.find("\n", j) + 1 or len(text) + 1) - 1]
        if _is_rule_line(line, ch):
            continue
        if ord(ch) > 127:
            return m.group(0), line.strip()[:60]
        before = text[i - 1] if i else " "
        after = text[j] if j < len(text) else " "
        # An ellipsis between words or numbers is ordinary prose -- `wait...no`, `6...10` -- and a
        # numeric range tripped this on a C++ ring-buffer answer. Three or four dots welded in are
        # a writer's ellipsis; five or more are a run.
        if ch == "." and j - i < 5:
            continue
        if before.isalnum() and after.isalnum():
            return text[i - 1:j + 1], line.strip()[:60]
    return None


SERVER_NOTE = "[stopped:"   # server/app.py writes this when it cut a loop off


def universal(reasoning: str, answer: str, finish_reason: str, thinking: bool) -> list:
    """Everything wrong with this output that has nothing to do with its domain.
    Returns a list of reasons; empty means nothing universal was wrong."""
    bad = []
    if finish_reason != "stop":
        bad.append(f"finish_reason {finish_reason!r}")
    if answer.lstrip().startswith(SERVER_NOTE):
        bad.append("server cut the generation off for repeating itself")
    if thinking and not reasoning.strip():
        bad.append("thinking was on and the reasoning is empty")
    if not answer.strip():
        # The failure that reads as success: a 200, a well-formed message, and
        # nothing in it. An agent calling this gets no error and no content, so
        # it stops without saying anything. It has its own name in the report.
        bad.append("think-exit: reasoned and then produced no answer"
                   if (thinking and reasoning.strip()) else "empty answer")
    for where, text in (("reasoning", reasoning), ("answer", answer)):
        hit = repeated_ngram(text)
        if hit:
            bad.append(f"{where} loops {hit[1]}x on {hit[0][:56]!r}")
        run = corrupt_run(text)
        if run:
            bad.append(f"{where} has a corrupted run {run[0]!r} in {run[1]!r}")
    return bad


# The five shapes a failing run can have. `repeat` is last because it is the
# narrowest, not because it is the least important: it is the one the strict
# count cannot distinguish from wreckage.
KINDS = ("think-exit", "guard", "corrupt", "content", "repeat")


def classify(reasoning: str, answer: str, finish_reason: str, thinking: bool,
             check_ok: bool) -> str:
    """Which one of the five shapes a failing run has. One kind per row.

      think-exit  reasoned and then wrote nothing -- a 200 with no content
      guard       the server cut a loop off: its `[stopped:` note, or the
                  `length` finish it sets when the degeneration window trips
                  on text that is visibly repeating
      corrupt     a character run no writer meant, `Tic‑‑‑‑Tac‑‑‑‑Toe`
      content     the answer is not the thing the prompt asked for
      repeat      the answer IS the thing the prompt asked for, the model chose
                  to stop, and the only count against the row is the n-gram rule

    `repeat` is tested last and can never take a row `content` would have had:
    it requires the structural check to have PASSED and `content` requires it to
    have failed. Everything the four named shapes do not describe -- a run
    truncated at max_tokens, a request that never came back -- lands in
    `content`, which is the honest reading of it: whatever else went wrong, the
    thing that was asked for is not there.
    """
    if thinking and reasoning.strip() and not answer.strip():
        return "think-exit"
    if answer.lstrip().startswith(SERVER_NOTE):
        return "guard"
    # A `length` finish on its own is a run that hit max_tokens. A `length`
    # finish on text that loops is the server's degeneration window, which cuts
    # the generation and leaves no note when there is already content.
    if finish_reason == "length" and (repeated_ngram(reasoning) or repeated_ngram(answer)):
        return "guard"
    if corrupt_run(reasoning) or corrupt_run(answer):
        return "corrupt"
    if (check_ok and finish_reason == "stop" and answer.strip()
            and (reasoning.strip() or not thinking)
            and (repeated_ngram(reasoning) or repeated_ngram(answer))):
        return "repeat"
    return "content"


# =============================================================================
# the prompt suites
# =============================================================================
# Keyed by TOPIC, because that is what a keep-set is built from: a profile is a
# set of topic names (tools/tune.py PROFILES), so a profile's suite is the union
# of its topics' prompts and a new profile costs no new prompts.
#
# Every prompt is written to make the register it belongs to unavoidable. A
# prompt a good general model answers from habit proves nothing about whether
# that topic's experts are resident; what proves it is a prompt whose answer is
# wrong-shaped without them, which is why the checks demand structure the
# register carries -- custom properties in CSS, a window function in SQL,
# anchors in YAML, `err != nil` in Go.

PROMPTS = {
    # --- code ---------------------------------------------------------------
    "html": [
        {"name": "html-page", "check": "html_page",
         "want": ["<title", "grid"],
         "prompt": "Write a complete single-file HTML page for a tic-tac-toe game: one <style> "
                   "block and one <script> block in the same file, a 3x3 grid, win and draw "
                   "detection, a status line and a reset button. Output the file and nothing "
                   "else."},
    ],
    "css": [
        {"name": "css-card", "check": "css_block",
         "want": ["--", "prefers-color-scheme", ":focus-visible"],
         "prompt": "Write the CSS for a `.card` component: custom properties on :root for its "
                   "colours, spacing and radius, a hover state, a :focus-visible ring, and a "
                   "dark-mode override through prefers-color-scheme. CSS only, no HTML."},
    ],
    "javascript": [
        {"name": "js-debounce", "check": "js_function",
         "want": ["cancel", "flush", "clearTimeout"],
         "prompt": "Write `debounce(fn, wait)` in modern JavaScript. The returned function must "
                   "preserve `this` and its arguments and carry `.cancel()` and `.flush()` "
                   "methods, and a timer that fires after a cancel must do nothing. Add a short "
                   "usage example."},
    ],
    "typescript": [
        {"name": "ts-groupby", "check": "ts_generic",
         "want": ["Record<", "keyof"],
         "prompt": "Write a TypeScript `groupBy` that takes a readonly array and a key selector, "
                   "generically constrained so the key must be a property of the element type "
                   "whose value is a string or a number, and returns a Record. No `any`."},
    ],
    "python": [
        {"name": "py-walk", "check": "python_script",
         "want": ["argparse", "except"],
         "prompt": "Write a Python script that walks a directory tree and prints the ten largest "
                   "files with their sizes. It takes the root and the count as arguments, and it "
                   "must survive permission errors and broken symlinks rather than stopping on "
                   "them."},
    ],
    "sql": [
        {"name": "sql-window", "check": "sql_query",
         "want": ["customers", "orders"],
         "prompt": "Write one SQL query over `orders(id, customer_id, placed_at, total)` and "
                   "`customers(id, name)` returning each customer's three most recent orders with "
                   "a running total of their order value. Use a join and a window function."},
    ],
    "config": [
        {"name": "yaml-anchors", "check": "yaml_doc",
         "want": ["healthcheck", "environment"],
         "prompt": "Write a docker-compose file with three services that share one block of "
                   "environment variables and one healthcheck definition through a YAML anchor "
                   "and aliases, each service overriding a single field of its own."},
    ],
    "go": [
        {"name": "go-handler", "check": "code_block",
         "want": ["package ", "func ", "err != nil", "http."],
         "prompt": "Write a Go HTTP handler that reads a JSON body, validates two fields, calls a "
                   "store interface with a context deadline and writes a JSON error envelope on "
                   "failure. Include the struct definitions."},
    ],
    "rust": [
        {"name": "rust-parse", "check": "code_block",
         "want": ["fn ", "Result<", "match "],
         "prompt": "Write a Rust function that parses `key=value` lines from a &str into a "
                   "HashMap, returning a Result with a custom error enum that implements Display. "
                   "Include the enum and one unit test."},
    ],
    "java": [
        {"name": "java-service", "check": "code_block",
         "want": ["class ", "public ", "private "],
         "prompt": "Write a Java class that caches lookups behind a ConcurrentHashMap with a "
                   "per-entry expiry, exposing get(key) and invalidate(key). Include the "
                   "constructor and the imports."},
    ],
    "cpp": [
        {"name": "cpp-ringbuffer", "check": "code_block",
         "want": ["#include", "class ", "::"],
         "prompt": "Write a fixed-capacity ring buffer in modern C++ as a class template, with "
                   "push, pop, size and a move constructor, and a short main() that exercises it."},
    ],
    "php": [
        {"name": "php-router", "check": "code_block",
         "want": ["<?php", "function ", "$"],
         "prompt": "Write a small PHP request router: register routes with a method and a path "
                   "pattern carrying named parameters, match an incoming request against them and "
                   "return a 404 when nothing matches."},
    ],
    "ruby": [
        {"name": "ruby-retry", "check": "code_block",
         "want": ["def ", "end", "rescue"],
         "prompt": "Write a Ruby module with a `with_retries` method taking a block, an attempt "
                   "count and an exponential backoff, re-raising the last error when the attempts "
                   "run out. Include a usage example."},
    ],
    "swift": [
        {"name": "swift-loader", "check": "code_block",
         "want": ["func ", "guard ", "->"],
         "prompt": "Write a Swift type that loads and decodes a Codable value from a URL with "
                   "async/await, caching the decoded result in memory and throwing a typed error "
                   "on a non-200 response."},
    ],
    "rlang": [
        {"name": "r-summary", "check": "code_block",
         "want": ["<-", "function(", "data"],
         "prompt": "Write R code that reads a CSV of measurements, drops rows with missing values, "
                   "computes the mean and the 95 % confidence interval per group, and returns a "
                   "data frame. Write it as a function plus a small example."},
    ],
    "latex": [
        {"name": "latex-note", "check": "latex_doc",
         "want": ["\\section", "\\label"],
         "prompt": "Write a complete LaTeX article: title, two sections, one numbered equation "
                   "that is referenced in the text, and a two-column table with a caption. "
                   "Compilable as it stands."},
    ],
    # `shell` is not in the trace catalogue yet (corpus/fetch_topics.py has no
    # shell glob). The suite is keyed by topic, so the day the corpus gains one
    # this prompt is already the gate for it.
    "shell": [
        {"name": "sh-pipeline", "check": "shell_pipeline",
         "want": ["set -", "IFS", "find"],
         "prompt": "Write a bash script that finds every file over 100 MB under a directory given "
                   "as $1, sorts them by size and prints the ten largest with human-readable "
                   "sizes. It must be correct for paths containing spaces and newlines."},
    ],
    # --- natural languages ---------------------------------------------------
    "english": [
        {"name": "en-explain", "check": "prose",
         "prompt": "Explain, in exactly two paragraphs of plain English and no bullet points, why "
                   "a cache with a 95 % hit rate can leave a system slower than no cache at all."},
        {"name": "en-note", "check": "prose",
         "prompt": "Write a short note to a colleague explaining why you are rolling back this "
                   "afternoon's release, what the symptom was, and what has to be true before it "
                   "goes out again. Three paragraphs, no lists."},
    ],
    "german": [
        {"name": "de-essay", "check": "prose_markers",
         "want": ["und", "nicht", "werden", "zwischen", "auch", "dass", "können"],
         "prompt": "Schreibe zwei Absätze darüber, warum ein Zwischenspeicher mit hoher Trefferrate "
                   "ein System trotzdem verlangsamen kann. Fließtext, keine Aufzählungen."},
    ],
    "french": [
        {"name": "fr-essay", "check": "prose_markers",
         "want": ["qui", "pour", "dans", "avec", "nous", "cette", "parce"],
         "prompt": "Rédigez deux paragraphes expliquant pourquoi un cache très efficace peut "
                   "malgré tout ralentir un système. Texte suivi, sans listes."},
    ],
    "spanish": [
        {"name": "es-essay", "check": "prose_markers",
         "want": ["para", "pero", "porque", "también", "entre", "cuando", "este"],
         "prompt": "Escribe dos párrafos explicando por qué una caché con una tasa de aciertos muy "
                   "alta puede aun así hacer que un sistema sea más lento. Texto corrido, sin "
                   "listas."},
    ],
    "italian": [
        {"name": "it-essay", "check": "prose_markers",
         "want": ["perché", "anche", "questo", "della", "sono", "quando", "molto"],
         "prompt": "Scrivi due paragrafi che spiegano perché una cache con un alto tasso di "
                   "successo può comunque rendere un sistema più lento. Testo discorsivo, senza "
                   "elenchi."},
    ],
    "portuguese": [
        {"name": "pt-essay", "check": "prose_markers",
         "want": ["não", "também", "porque", "está", "entre", "quando", "muito"],
         "prompt": "Escreva dois parágrafos explicando por que um cache com uma taxa de acertos "
                   "muito alta ainda assim pode deixar um sistema mais lento. Texto corrido, sem "
                   "listas."},
    ],
    "turkish": [
        {"name": "tr-essay", "check": "prose_markers",
         "want": ["için", "ile", "bir", "ancak", "olarak", "daha", "gibi"],
         "prompt": "Yüksek isabet oranına sahip bir önbelleğin bir sistemi neden yine de "
                   "yavaşlatabileceğini iki paragrafta açıkla. Düz metin, madde işareti kullanma."},
    ],
    "arabic": [
        {"name": "ar-essay", "check": "prose_script", "want": "arabic",
         "prompt": "اكتب فقرتين تشرح فيهما لماذا قد تؤدي ذاكرة تخزين مؤقت ذات نسبة إصابة عالية إلى "
                   "إبطاء النظام رغم ذلك. نص متصل بدون قوائم."},
    ],
    "chinese": [
        {"name": "zh-essay", "check": "prose_script", "want": "han",
         "prompt": "用两段话解释：为什么命中率很高的缓存仍然可能让系统变慢。请写成连贯的段落，不要使用列表。"},
    ],
    "japanese": [
        {"name": "ja-essay", "check": "prose_script", "want": "kana",
         "prompt": "ヒット率が高いキャッシュでもシステムが遅くなることがある理由を、二つの段落で説明してください。"
                   "箇条書きは使わず、文章で書いてください。"},
    ],
    "russian": [
        {"name": "ru-essay", "check": "prose_script", "want": "cyrillic",
         "prompt": "Напишите два абзаца о том, почему кэш с очень высокой долей попаданий всё равно "
                   "может замедлить систему. Связный текст, без списков."},
    ],
    # --- domain registers ----------------------------------------------------
    "technical": [
        {"name": "tech-explain", "check": "prose",
         "prompt": "Explain in two paragraphs, for an engineer who knows sockets but not TCP "
                   "internals, what congestion control is doing during a slow transfer and why "
                   "adding bandwidth may not help."},
    ],
    "academic": [
        {"name": "acad-abstract", "check": "prose",
         "prompt": "Write the abstract and the first paragraph of the methods section of a paper "
                   "measuring whether code review latency predicts defect density, including the "
                   "design, the sample and the pre-registered hypothesis."},
    ],
    "journalism": [
        {"name": "news-lede", "check": "prose",
         "prompt": "Write the first four paragraphs of a news report on a regional power outage "
                   "that closed a hospital's elective list for a day. Inverted pyramid, attributed "
                   "quotes, no invented statistics."},
    ],
    "marketing": [
        {"name": "copy-landing", "check": "prose",
         "prompt": "Write the opening copy for the landing page of a tool that finds unused "
                   "database indexes: a headline, two paragraphs of body copy and a closing line. "
                   "No bullet lists, no superlatives you cannot support."},
    ],
    "medical": [
        {"name": "med-explain", "check": "prose",
         "prompt": "Explain to a first-year resident, in two paragraphs, the difference between a "
                   "type 1 and a type 2 myocardial infarction and what that distinction changes "
                   "about immediate management."},
    ],
    "legal": [
        {"name": "legal-clause", "check": "prose",
         "prompt": "Draft a limitation-of-liability clause for a software support agreement, in "
                   "formal contractual register, then add one paragraph in plain language saying "
                   "what it does not cover."},
    ],
    "finance": [
        {"name": "fin-explain", "check": "prose",
         "prompt": "Explain in two paragraphs, to a founder reading their first set of accounts, "
                   "how a profitable company runs out of cash, and which line of the cash-flow "
                   "statement shows it happening first."},
    ],
    "translation": [
        {"name": "xl-en-fr", "check": "prose_markers",
         "want": ["qui", "pour", "dans", "avec", "cette", "nous", "mais"],
         "prompt": "Translate into French, keeping the register: \"The outage began at 14:05, when "
                   "a routine certificate rotation removed the key the payment gateway still "
                   "depended on. Nothing alerted, because the health check tested the process and "
                   "not the transaction. We learned about it from a customer.\" Then add a short "
                   "paragraph in French on what you changed and why."},
    ],
}

# Run on every profile, whatever its topics. The catalogue has no topic for the
# register thinking mode writes IN -- deliberation, self-correction, arithmetic
# talked through -- and that gap is invisible to coverage, because coverage can
# only report on topics that exist (tools/tune.py, the note above PROFILES: the
# Chat profile scored 0.85 or better on all five of its topics and still
# reasoned in circles on a two-train arithmetic question). Both of these are
# traps: the obvious arithmetic gives a different number, so an answer that only
# sounds confident still fails.
ALWAYS = [
    {"name": "reason-bat-ball", "topic": "reasoning", "check": "numeric_answer",
     "want": ["0.05", "5 cents", "five cents", "$.05"],
     "prompt": "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How "
               "much does the ball cost? Give the amount and show why the obvious answer is wrong."},
    {"name": "reason-machines", "topic": "reasoning", "check": "numeric_answer",
     "want": ["5 minutes", "five minutes"],
     "prompt": "If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to "
               "make 100 widgets? State the time and explain the rate you used."},
]


def suite(topics, extra: dict | None = None) -> tuple:
    """The prompts for a set of topics, and the topics that contributed none.

    A selected topic with no prompt is worth saying out loud: the profile is
    being gated on less than it claims, and silence there is how a profile comes
    out green on four topics out of five.
    """
    table = dict(PROMPTS)
    for topic, prompts in (extra or {}).items():
        table[topic] = prompts          # a file replaces a topic's suite outright
    out, silent = [], []
    for topic in sorted(set(topics)):
        got = table.get(topic) or []
        if not got:
            silent.append(topic)
        for p in got:
            out.append(dict(p, topic=topic))
    return out + [dict(p) for p in ALWAYS], silent


def read_prompts_file(path: str) -> dict:
    """{topic: [prompt, ...]}, validated. Same shape as PROMPTS, so a suite kept
    in a file and the built-in one are the same thing."""
    raw = json.load(open(os.path.expanduser(path)))
    if isinstance(raw, dict) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: expected {{"topic": [prompt, ...]}}')
    for topic, prompts in raw.items():
        if not isinstance(prompts, list):
            raise ValueError(f"{path}: {topic!r} must hold a list of prompts")
        for p in prompts:
            if not isinstance(p, dict) or not p.get("name") or not p.get("prompt"):
                raise ValueError(f"{path}: {topic!r} has a prompt without a name or a prompt")
            if p.get("check") not in CHECKS:
                raise ValueError(f"{path}: {p['name']}: unknown check {p.get('check')!r} "
                                 f"(have: {', '.join(sorted(CHECKS))})")
    return raw


# =============================================================================
# profiles
# =============================================================================

def load_profiles() -> list:
    """The profiles tune.py resolves, shipped and user, as (name, blurb, topics).

    Imported lazily and forgivingly: tune.py pulls in curses and the coverage
    file, and none of that is needed to run a check or to gate an explicit
    --topics list. A missing tune.py should cost the --profile flag, not the
    tool.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import tune                                          # noqa: PLC0415
    user, _problems = tune.load_profiles(tune.profiles_files())
    return [(p[0], p[1], p[2]) for p in tune.merge_profiles(tune.PROFILES, user)]


def slug(name: str) -> str:
    """A profile name as the directory its record lives in. Underscores, because
    that is what the directories in results/keepsets/ are named: `Chat and
    explanation` is `chat_and_explanation`, and a hyphen here would have sent
    its next gate run to a second, empty directory instead of appending to the
    record that is already there."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_profile(want: str):
    """(name, topics) for a profile named on the command line. `Everything` has
    no topic list of its own -- it means the whole catalogue, which here is every
    topic the prompt table carries."""
    profiles = load_profiles()
    hits = [p for p in profiles if want.lower() in (p[0].lower(), slug(p[0]))]
    if not hits:
        hits = [p for p in profiles if slug(p[0]).startswith(slug(want))]
    if not hits:
        raise SystemExit(f"no profile {want!r}. Have: "
                         + ", ".join(slug(p[0]) for p in profiles))
    if len(hits) > 1:
        raise SystemExit(f"{want!r} matches {', '.join(slug(p[0]) for p in hits)}")
    name, _blurb, topics = hits[0]
    return name, list(topics) if topics else sorted(PROMPTS)


# =============================================================================
# the server
# =============================================================================
# Never streamed. server/app.py appends its degeneration note AFTER the last SSE
# chunk of content, and the finish_reason that says the model was cut off rather
# than finished arrives in the final chunk too -- a client that reads the stream
# and stops at the last piece of text sees a plausible answer and no sign that
# anything went wrong. The whole point of this tool is to see that sign.

def _get(url: str, api_key: str | None, timeout: int = 30) -> dict:
    req = urlrequest.Request(url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    with urlrequest.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_card(base: str, api_key: str | None) -> dict:
    """id and max_model_len, recorded in the report: a gate run against a 32k
    context and one against 256k are not the same measurement, and the arena
    that bought the context is what changed the keep-set."""
    data = _get(base + "/models", api_key).get("data") or []
    return data[0] if data else {}


def generate(base: str, model: str, prompt: str, thinking: bool, effort: int,
             max_tokens: int, api_key: str | None, timeout: int,
             temperature: float | None = None, no_repeat_ngram: int | None = None,
             presence_penalty: float | None = None) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "chat_template_kwargs": {"thinking": thinking},
        "reasoning_effort": effort,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    # Both are per-request on this server. The n-gram guard exists for exactly the failure the
    # saliency keep-set leaves: a fragment redrafted many times inside a long think block.
    if no_repeat_ngram:
        body["no_repeat_ngram"] = no_repeat_ngram
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    req = urlrequest.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    t0 = time.perf_counter()
    with urlrequest.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return {"reasoning": msg.get("reasoning_content") or "",
            "answer": msg.get("content") or "",
            "finish": choice.get("finish_reason") or "",
            "seconds": time.perf_counter() - t0}


# =============================================================================
# the run
# =============================================================================

HEADER = ("prompt", "think", "finish", "reason", "answer", "s", "", "why")


def _row(cells) -> str:
    """One line of the stdout table, printed as each prompt finishes rather than
    at the end: a full suite is an hour of generation and the row that matters
    is usually the first failure."""
    name, think, finish, nr, na, secs, verdict, why = cells
    return (f"{name:<18.18} {think:<5} {finish:<8.8} {nr:>7} {na:>7} {secs:>6} "
            f"{verdict:<4} {why}")


def judge(p: dict, got: dict, thinking: bool) -> tuple:
    """(passed, why, kind). The universal failures come first and they are
    final: an output that loops or stops early is not saved by containing the
    right number of CSS declarations somewhere in the wreckage.

    The domain check runs on a failing row all the same -- not to rescue it, but
    because whether the answer was the asked-for thing is the whole difference
    between a redraft in the think block and a page of `* { }`. `kind` is "" on
    a pass and one of KINDS on a failure.
    """
    bad = universal(got["reasoning"], got["answer"], got["finish"], thinking)
    fn = CHECKS[p["check"]]
    ok, why = fn(got["answer"], p.get("want"))
    if not bad:
        return ok, why, "" if ok else "content"
    kind = classify(got["reasoning"], got["answer"], got["finish"], thinking, ok)
    reasons = "; ".join(bad)
    # `repeat:` leads the line so these rows can be counted with grep, and what
    # the check made of the answer is carried along: that is the evidence that
    # the miss was a redraft and not wreckage.
    if kind == "repeat":
        return False, f"repeat: {reasons} — the answer itself is sound ({why})", kind
    return False, reasons, kind


def run(args, prompts, card) -> list:
    modes = [True, False] if args.thinking == "both" else [args.thinking == "on"]
    print(_row(HEADER), flush=True)
    print("-" * 96, flush=True)
    rows = []
    for p in prompts:
        for thinking in modes:
            try:
                got = generate(args.url, card.get("id") or args.model, p["prompt"], thinking,
                               args.effort, args.max_tokens, args.api_key, args.timeout,
                               args.temperature, args.no_repeat_ngram, args.presence_penalty)
                ok, why, kind = judge(p, got, thinking)
            except (urlerror.URLError, OSError, ValueError, KeyError) as e:
                got = {"reasoning": "", "answer": "", "finish": "error", "seconds": 0.0}
                ok, why, kind = False, f"request failed: {e}", "content"
            row = {"name": p["name"], "topic": p["topic"], "check": p["check"],
                   "thinking": "on" if thinking else "off", "finish": got["finish"],
                   "reasoning_chars": len(got["reasoning"]), "answer_chars": len(got["answer"]),
                   "seconds": got["seconds"], "ok": ok, "why": why, "kind": kind}
            rows.append(row)
            print(_row((row["name"], row["thinking"], row["finish"], f"{row['reasoning_chars']:,}",
                        f"{row['answer_chars']:,}", f"{row['seconds']:.0f}",
                        "PASS" if ok else "FAIL", why)), flush=True)
    return rows


def tally(rows) -> tuple:
    """(finished, {kind: n}). `finished` counts the runs that produced the
    answer the prompt asked for: the strict passes, plus the misses whose only
    fault was the n-gram rule. It is not a second gate and it never moves the
    exit code -- it is the number that says whether the misses were wreckage."""
    by = {k: 0 for k in KINDS}
    for r in rows:
        if not r["ok"]:
            by[r.get("kind") or "content"] += 1
    return sum(1 for r in rows if r["ok"]) + by["repeat"], by


def finished_line(rows) -> str:
    """The second sentence of the verdict, in stdout and in GATE.md alike."""
    finished, by = tally(rows)
    return (f"{finished} of {len(rows)} finished a correct answer (strict passes plus "
            f"repeat-only misses); misses by kind: "
            + ", ".join(f"{k} {by[k]}" for k in KINDS) + ".")


def report(rows, name, topics, silent, args, card) -> str:
    """The GATE.md section. Appended, never overwritten: a gate result is a dated
    measurement of one configuration, and the previous one is the comparison."""
    failed = [r for r in rows if not r["ok"]]
    ctx = card.get("max_model_len")
    ctx = f"{ctx:,}" if isinstance(ctx, int) else "unknown"
    out = [f"# Generation gate \u2014 {datetime.now():%Y-%m-%d %H:%M}", ""]
    out += ["| | |", "|---|---|",
            f"| profile | {name} |",
            f"| topics | {', '.join(topics)} |",
            f"| prompts | {len(rows)} runs over {len({r['name'] for r in rows})} prompts |",
            f"| thinking | {args.thinking} |",
            f"| reasoning effort | {args.effort} |",
            f"| max tokens | {args.max_tokens:,} |",
            f"| server | `{args.url}`, model `{card.get('id', '?')}`, "
            f"max_model_len {ctx} |"]
    if args.only:
        # A filtered re-run is a useful thing to record and a dangerous thing to
        # read as a gate: it says the named prompts pass, and nothing about the
        # ones that were not run.
        out.append(f"| only | `{args.only}` — a filtered re-run, not a full gate |")
    if silent:
        out.append(f"| no prompts for | {', '.join(silent)} — these topics were NOT gated |")
    # WHICH keep-set this was. Without it a gate result is a count with no
    # configuration attached, and the counts move with the configuration: the
    # same Backend prompts went 5 of 10 at keep 0.40 and 3 of 10 strict with 10
    # of 10 finished at 0.36. Read from the environment the run was launched
    # with, which is the same .env the engine read.
    cfg = ", ".join(f"{k}={os.environ[k]}" for k in
                    ("PRUNE_KEEP", "DSV41_PRUNE_RANK", "DSV41_PRUNE_SOURCE")
                    if os.environ.get(k, "").strip())
    out.append(f"| keep-set | {cfg} |" if cfg else
               "| keep-set | not recorded — PRUNE_KEEP and the ranking pair were not in the "
               "environment of this run |")
    out += ["",
            "| prompt | thinking | finish | reasoning | answer | s | | why |",
            "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| `{r['name']}` | {r['thinking']} | {r['finish']} | {r['reasoning_chars']:,} | "
                   f"{r['answer_chars']:,} | {r['seconds']:.0f} | "
                   f"{'PASS' if r['ok'] else '**FAIL**'} | {r['why']} |")
    out += [""]
    if failed:
        out.append(f"**Verdict: FAIL** — {len(failed)} of {len(rows)} runs failed: "
                   + "; ".join(f"`{r['name']}` ({r['thinking']}) {r['why']}" for r in failed))
    else:
        out.append(f"**Verdict: PASS** — all {len(rows)} runs produced sound output.")
    out.append("")
    out.append(finished_line(rows))
    if silent:
        out.append("")
        out.append(f"This run gated {len(topics) - len(silent)} of the profile's {len(topics)} "
                   f"topics; {', '.join(silent)} carry no prompt, so a pass says nothing about "
                   f"them.")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="a profile from tools/tune.py, by name or slug")
    ap.add_argument("--topics", help="comma-separated topic names, instead of a profile")
    ap.add_argument("--url", default=os.environ.get("DSV41_URL", "http://127.0.0.1:8000/v1"),
                    help="the OpenAI-compatible base (default %(default)s)")
    ap.add_argument("--api-key", default=os.environ.get("DSV41_API_KEY"))
    ap.add_argument("--model", default="default", help="only used if /v1/models cannot be read")
    ap.add_argument("--thinking", choices=("on", "off", "both"), default="on",
                    help="thinking mode; `both` runs every prompt twice (default %(default)s)")
    ap.add_argument("--effort", type=int, default=45, help="reasoning effort, 1-100")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--no-repeat-ngram", type=int, default=None,
                    help="ban repeating an n-gram already in the output; matches the harness's 12-word rule at 12")
    ap.add_argument("--presence-penalty", type=float, default=None, help="OpenAI presence penalty, per request")
    ap.add_argument("--temperature", type=float, default=None,
                    help="passed through when given; the temperature-0 loop is reproducible here")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per request")
    ap.add_argument("--out", help="GATE.md to append to "
                                  "(default results/keepsets/<profile>/GATE.md)")
    ap.add_argument("--prompts-file", help="JSON {topic: [prompt, ...]}, replacing or adding topics")
    ap.add_argument("--only", help="comma-separated prompt names, for re-running one failure")
    ap.add_argument("--dry-run", action="store_true", help="print the suite and stop, no HTTP")
    a = ap.parse_args()

    if not a.profile and not a.topics:
        ap.error("one of --profile or --topics is required")
    if a.profile:
        name, topics = find_profile(a.profile)
    else:
        name, topics = "ad-hoc", [t.strip() for t in a.topics.split(",") if t.strip()]

    extra = read_prompts_file(a.prompts_file) if a.prompts_file else None
    prompts, silent = suite(topics, extra)
    if a.only:
        want = {n.strip() for n in a.only.split(",") if n.strip()}
        unknown = want - {p["name"] for p in prompts}
        if unknown:
            raise SystemExit(f"--only names prompts this suite does not have: {', '.join(sorted(unknown))}")
        prompts = [p for p in prompts if p["name"] in want]

    a.url = a.url.rstrip("/")
    if not a.url.endswith("/v1"):
        a.url += "/v1"

    print(f"profile {name}: {len(topics)} topics, {len(prompts)} prompts, thinking {a.thinking}")
    print(f"topics: {', '.join(topics)}")
    if silent:
        print(f"NOT GATED (no prompts): {', '.join(silent)}")
    if a.dry_run:
        for p in prompts:
            print(f"  {p['name']:<18} {p['topic']:<12} {p['check']:<14} "
                  f"{' '.join(p['prompt'].split())[:60]}…")
        return 0

    try:
        card = model_card(a.url, a.api_key)
    except (urlerror.URLError, OSError, ValueError) as e:
        raise SystemExit(f"cannot reach {a.url}/models: {e}")
    print(f"server: {card.get('id', '?')}, max_model_len {card.get('max_model_len', '?')}\n")

    rows = run(a, prompts, card)
    failed = [r for r in rows if not r["ok"]]
    text = report(rows, name, topics, silent, a, card)

    out = a.out or (os.path.join(ROOT, "results", "keepsets", slug(name), "GATE.md")
                    if a.profile else None)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        had = os.path.exists(out) and os.path.getsize(out) > 0
        with open(out, "a") as f:
            f.write(("\n---\n\n" if had else "") + text)
        print(f"\nappended to {os.path.relpath(out, ROOT) if out.startswith(ROOT) else out}")
    else:
        print("\nno --out: an ad-hoc topic list does not own a GATE.md, so nothing was written")

    print(f"{len(rows) - len(failed)} of {len(rows)} runs passed"
          + (f" — FAILED: {', '.join(sorted({r['name'] for r in failed}))}" if failed else ""))
    print(finished_line(rows))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
