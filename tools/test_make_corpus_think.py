"""The `think` corpus kind must close a NON-EMPTY think block, with its code intact.

Every other wrapper in corpus/make_corpus.py writes `</think>` immediately after the assistant
tag, so it closes an empty block: across corpus/trace_corpus_v2.jsonl and _v3.jsonl, 85 of 95
sequences have `</think>` adjacent to the assistant tag and none has it after real content. The
experts that fire on "the deliberation is finished, close it, begin the answer" were therefore
never ranked and are not resident, and a pruned server cannot stop deliberating -- at temperature 0
it writes "I'll write the code now." and then repeats "Let me write." to the token cap with an
answer of length zero.

This checks the two properties that make the kind worth having, and that the shipped source file
still has them. make_corpus.py imports transformers at run time, so the parser is lifted out with
`ast` the way tools/test_engine_kwargs.py reads the engine signature.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "corpus/make_corpus.py")
CORPUS = os.path.join(ROOT, "corpus/sources/reasoning_code.txt")

ns = {"re": re}
for node in ast.parse(open(SRC).read()).body:
    keep = (isinstance(node, ast.FunctionDef) and node.name in ("think_records", "wrap_think")) or \
           (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("THINK_SECTIONS", "BOS", "USER", "ASSISTANT", "EOS"))
    if keep:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<make_corpus>", "exec"), ns)
think_records, wrap_think = ns["think_records"], ns["wrap_think"]

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


SAMPLE = """=== PROMPT
Centre a card.
=== THINK
Flexbox or grid? Grid is one property:

```css
.wrap { display: grid; place-items: center; }
```

`100vh` is wrong on mobile --
  min-height: 100dvh;
is what the visible viewport measures. That settles it, writing it out.
=== ANSWER
```css
.wrap { display: grid; place-items: center; min-height: 100dvh; }
```
=== END
"""

r = think_records(SAMPLE)
check("one record parsed", len(r) == 1, f"got {len(r)}")
check("fenced code survives the parser", "```css" in r[0]["THINK"])
check("indentation survives the parser", "\n  min-height: 100dvh;" in r[0]["THINK"])

wrapped = wrap_think(r[0]["PROMPT"], r[0]["THINK"], r[0]["ANSWER"])
close = wrapped.index("</think>")
between = wrapped[wrapped.index("<think>") + len("<think>"):close]
check("the think block is not empty", len(between) > 200, f"{len(between)} chars before </think>")
check("`</think>` is not adjacent to the assistant tag",
      not wrapped[:close].endswith(ns["ASSISTANT"]))
check("an answer follows the close", len(wrapped[close + len("</think>"):]) > 20)

# the shipped corpus, which is what the trace actually measured
if not os.path.exists(CORPUS):
    check("corpus/sources/reasoning_code.txt present", False)
else:
    recs = think_records(open(CORPUS).read())
    check("shipped corpus parses", len(recs) >= 8, f"{len(recs)} records")
    thin = [i for i, x in enumerate(recs)
            if "```" not in x["THINK"] and x["THINK"].count("`") < 12]
    check("every deliberation carries code", not thin, f"records without code: {thin}")
    # the exact corruption the pruned server emits: a non-breaking hyphen run. Never teach it.
    bad = [i for i, x in enumerate(recs)
           if any(c in x["THINK"] + x["ANSWER"] for c in "‑‐–—‘’“”")]
    check("no non-ascii punctuation in the corpus", not bad, f"records: {bad}")
    stub = [i for i, x in enumerate(recs) if len(x["THINK"].rstrip().rsplit("\n", 1)[-1]) < 12]
    check("every deliberation closes with a real sentence", not stub, f"records: {stub}")
    over = [i for i, x in enumerate(recs)
            if len(x["PROMPT"]) + len(x["THINK"]) + len(x["ANSWER"]) > 1900]
    check("records stay inside the 512-token budget", not over, f"over 1900 chars: {over}")

print()
print(f"{len(fails)} failed: {', '.join(fails)}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
