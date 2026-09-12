#!/usr/bin/env python3
"""
make_corpus.py -- build the teacher-forced trace corpus for tools/expert_trace.py.

Two categories, ~5k tokens each, every sequence <= --max-len tokens (512 by default,
the exactness bound of the pure-torch port), each wrapped in the V4.1 chat format from
encoding/README.md so the tokens sit where they would in real use:

  coding  : user asks for a program, assistant answers with real code (teacher-forced).
            Sources: files passed with --code (public, permissively licensed) plus the
            two one-shot prompts this repo benchmarks (Angry Birds HTML, Mario) as user turns.
  general : prose. Sources: files passed with --prose, cut into paragraphs, used both as
            user-turn documents ("summarize this") and as assistant answers to
            "explain ..." questions, plus a handful of short QA prompts (math/science/
            reasoning style) as user turns.

Every text is tokenized with the model tokenizer and split so that the wrapped sequence
fits --max-len. Output: JSONL {"id", "category", "source", "text"}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
EOS = "<｜end▁of▁sentence｜>"

ONE_SHOTS = [
    ("angry-birds", "Write a complete Angry Birds style game as a single self-contained HTML file: canvas rendering, "
                    "a slingshot with drag-to-aim, projectile physics with gravity, destructible block structures, "
                    "pigs as targets, a score counter, and a restart button. No external libraries or assets."),
    ("mario", "Write a complete side-scrolling Mario style platformer as a single HTML file with inline JavaScript: "
              "keyboard controls, jumping with gravity, moving enemies you can stomp, coins, a scrolling level, "
              "and a win condition. No external libraries."),
]

QA_PROMPTS = [
    "A train leaves city A at 60 km/h and another leaves city B, 300 km away, at 90 km/h toward it. When do they meet?",
    "Explain why the sky is blue and sunsets are red, using Rayleigh scattering.",
    "What is the time complexity of building a heap from n elements, and why is it not O(n log n)?",
    "Compare TCP and QUIC head-of-line blocking behaviour in one paragraph.",
    "If p is prime and a is not divisible by p, what does Fermat's little theorem say about a^(p-1)?",
    "Give three arguments for and against using a monorepo for a 30-person engineering team.",
    "What are the security implications of enabling HTTP request smuggling defences at a reverse proxy?",
    "Summarize the difference between MoE expert parallelism and tensor parallelism.",
]


def wrap_code(prompt: str, code: str) -> str:
    return f"{BOS}{USER}{prompt}{ASSISTANT}</think>{code}{EOS}"


def wrap_user_doc(doc: str) -> str:
    return f"{BOS}{USER}Summarize the following text in three sentences.\n\n{doc}{ASSISTANT}</think>"


def wrap_answer(question: str, answer: str) -> str:
    return f"{BOS}{USER}{question}{ASSISTANT}</think>{answer}{EOS}"


def wrap_qa(question: str) -> str:
    return f"{BOS}{USER}{question}{ASSISTANT}<think>"


def wrap_think(prompt: str, think: str, answer: str) -> str:
    """A turn that deliberates and then answers -- the only wrapper that closes a real think block.

    Every other wrapper here emits `</think>` immediately after `<｜Assistant｜>`, so it closes an
    EMPTY block: across trace_corpus_v2 and _v3, 85 of 95 sequences have `</think>` adjacent to the
    assistant tag and none has it after real content. The experts that fire on "the deliberation is
    finished, close it, begin the answer" are therefore never ranked, never resident, and a pruned
    server cannot stop deliberating: the observed failure is finish=length with every token inside
    the think block and an answer of length zero.
    """
    return f"{BOS}{USER}{prompt}{ASSISTANT}<think>{think}</think>{answer}{EOS}"


THINK_SECTIONS = ("PROMPT", "THINK", "ANSWER")


def think_records(text: str):
    """Parse `=== PROMPT / === THINK / === ANSWER / === END` records, whitespace intact.

    Indentation and blank lines are the register here -- a fenced fragment inside deliberation is
    the whole point -- so this does NOT go through paragraphs(), which collapses every run of
    whitespace to a single space and would flatten the code it is meant to measure.
    """
    cur, field, out = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^===\s+([A-Z]+)\s*$", line)
        if m:
            tag = m.group(1)
            if tag == "END":
                if all(k in cur for k in THINK_SECTIONS):
                    out.append({k: cur[k].strip("\n") for k in THINK_SECTIONS})
                cur, field = {}, None
            elif tag in THINK_SECTIONS:
                field = tag
                cur[field] = ""
            else:
                field = None
            continue
        if field is not None:
            cur[field] += line + "\n"
    return out


def chunks_by_tokens(tok, text: str, budget: int):
    """Split text on line boundaries into pieces of <= budget tokens."""
    lines = text.split("\n")
    out, cur, cur_n = [], [], 0
    for ln in lines:
        n = len(tok.encode(ln + "\n", add_special_tokens=False))
        if n > budget:  # pathological long line: hard cut
            ids = tok.encode(ln, add_special_tokens=False)
            for i in range(0, len(ids), budget):
                out.append(tok.decode(ids[i:i + budget]))
            continue
        if cur_n + n > budget and cur:
            out.append("\n".join(cur))
            cur, cur_n = [], 0
        cur.append(ln)
        cur_n += n
    if cur:
        out.append("\n".join(cur))
    return out


def paragraphs(text: str):
    text = re.sub(r"[ \t]+\n", "\n", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [re.sub(r"\s+", " ", p) for p in paras if len(p) > 200 and not re.match(r"^[\d\s.|:—-]+$", p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.json")
    ap.add_argument("--code", nargs="*", default=[], help="code files (path[:lang])")
    ap.add_argument("--prose", nargs="*", default=[], help="prose files")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--target", type=int, default=5000, help="tokens per category")
    ap.add_argument("--topic", action="append", default=[], metavar="NAME:KIND:PATH[,PATH...]",
                    help="a labelled group of sources whose label becomes the sequence category, so ONE trace "
                         "over a multi-topic corpus yields a per-topic expert histogram for each and a keep-set "
                         "can then be composed from any subset without tracing again. KIND is code|prose|think; "
                         "`think` reads `=== PROMPT/THINK/ANSWER/END` records and is the only kind that closes a "
                         "non-empty think block. "
                         "Repeatable: --topic html:code:a.html,b.html --topic arabic:prose:ar.txt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    n = lambda s: len(tok.encode(s, add_special_tokens=False))

    seqs = []
    budget_code = a.max_len - 80  # header + prompt

    def emit_code(paths, topic, lang_default=""):
        t = 0
        for spec in paths:
            path, _, lang = spec.partition(":")
            path = os.path.expanduser(path)
            src = open(path).read(); name = os.path.basename(path)
            for i, piece in enumerate(chunks_by_tokens(tok, src, budget_code)):
                text = wrap_code(f"Show me the implementation of `{name}` (part {i + 1}).",
                                 f"```{lang or lang_default}\n{piece}\n```")
                if n(text) > a.max_len:
                    continue
                seqs.append({"id": f"{topic}-{hashlib.sha1(text.encode()).hexdigest()[:8]}", "category": topic,
                             "source": f"{name}#{i + 1}", "text": text})
                t += n(text)
                if t >= a.target: return t
        return t

    def emit_prose(paths, topic):
        t = 0; k = 0
        for path in paths:
            path = os.path.expanduser(path)
            name = os.path.basename(path)
            for para in paragraphs(open(path).read()):
                if n(para) > a.max_len - 60:
                    continue
                k += 1
                text = wrap_user_doc(para) if k % 2 else wrap_answer(
                    "Explain the following topic in a detailed paragraph: " + para[:60].rsplit(" ", 1)[0] + " ...", para)
                if n(text) > a.max_len:
                    continue
                seqs.append({"id": f"{topic}-{hashlib.sha1(text.encode()).hexdigest()[:8]}", "category": topic,
                             "source": f"{name}#{k}", "text": text})
                t += n(para)
                if t >= a.target: return t
        return t

    def emit_think(paths, topic):
        t = 0
        for path in paths:
            path = os.path.expanduser(path)
            name = os.path.basename(path)
            recs = think_records(open(path).read())
            if not recs:
                raise SystemExit(f"{path}: no `=== PROMPT/THINK/ANSWER/END` records found")
            for k, r in enumerate(recs, 1):
                text = wrap_think(r["PROMPT"], r["THINK"], r["ANSWER"])
                if n(text) > a.max_len:
                    # loud, not silent: a dropped record is a register that never reaches the trace
                    print(f"  skip {name}#{k}: {n(text)} tokens > --max-len {a.max_len}")
                    continue
                seqs.append({"id": f"{topic}-{hashlib.sha1(text.encode()).hexdigest()[:8]}", "category": topic,
                             "source": f"{name}#{k}", "text": text})
                t += n(text)
                if t >= a.target: return t
        return t

    KINDS = {"code": emit_code, "prose": emit_prose, "think": emit_think}
    for spec in a.topic:
        name_, _, rest = spec.partition(":")
        kind, _, paths = rest.partition(":")
        files = [x for x in paths.split(",") if x]
        if kind not in KINDS:
            raise SystemExit(f"--topic {spec}: unknown kind {kind!r} (code | prose | think)")
        got = KINDS[kind](files, name_)
        print(f"topic {name_:14s} {kind:5s} {len(files)} files -> {got} tokens")
    if a.topic and not (a.code or a.prose):
        with open(a.out, "w") as f:
            for s_ in seqs:
                f.write(json.dumps(s_, ensure_ascii=False) + "\n")
        import collections as _c
        by = _c.Counter(x["category"] for x in seqs)
        print(json.dumps(dict(by)), "->", a.out)
        return
    # coding
    total = 0
    for key, prompt in ONE_SHOTS:
        seqs.append({"id": f"code-oneshot-{key}", "category": "coding", "source": "one-shot prompt", "text": wrap_qa(prompt)})
    for spec in a.code:
        path, _, lang = spec.partition(":")
        src = open(path).read()
        name = os.path.basename(path)
        for i, piece in enumerate(chunks_by_tokens(tok, src, budget_code)):
            prompt = f"Show me the implementation of `{name}` (part {i + 1})."
            text = wrap_code(prompt, f"```{lang or ''}\n{piece}\n```")
            if n(text) > a.max_len:
                continue
            seqs.append({"id": f"code-{hashlib.sha1(text.encode()).hexdigest()[:8]}", "category": "coding",
                         "source": f"{name}#{i + 1}", "text": text})
            total += n(text)
            if total >= a.target:
                break
        if total >= a.target:
            break
    # general
    total = 0
    for q in QA_PROMPTS:
        seqs.append({"id": f"gen-qa-{hashlib.sha1(q.encode()).hexdigest()[:8]}", "category": "general", "source": "qa prompt",
                     "text": wrap_qa(q)})
    k = 0
    for path in a.prose:
        name = os.path.basename(path)
        for p in paragraphs(open(path).read()):
            if n(p) > a.max_len - 60:
                continue
            k += 1
            if k % 2:
                text = wrap_user_doc(p)
            else:
                text = wrap_answer("Explain the following topic in a detailed paragraph: " + p[:60].rsplit(" ", 1)[0] + " ...", p)
            if n(text) > a.max_len:
                continue
            seqs.append({"id": f"gen-{hashlib.sha1(text.encode()).hexdigest()[:8]}", "category": "general",
                         "source": f"{name}#{k}", "text": text})
            total += n(text)
            if total >= a.target:
                break
        if total >= a.target:
            break

    with open(a.out, "w") as f:
        for s in seqs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    by = {}
    for s in seqs:
        c = by.setdefault(s["category"], [0, 0])
        c[0] += 1
        c[1] += n(s["text"])
    print(json.dumps(by), "->", a.out)


if __name__ == "__main__":
    main()
