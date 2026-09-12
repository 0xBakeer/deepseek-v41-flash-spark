#!/usr/bin/env python3
"""fetch_topics.py -- gather source text for a multi-topic trace corpus.

A keep-set is only as good as the corpus it was ranked on, and a topic traced
on a few hundred tokens ranks noisily: its per-layer histogram spreads 6 picks
per token over 384 experts, so a few hundred tokens leave most experts with a
count of one or two. Worse, the coverage such a topic reports is biased
*upward*, because it is measured on the same trace that chose the experts. Aim
for a few thousand tokens per topic, which is what this script collects.

Three kinds of source:

  code    files matching a glob under --code-root, concatenated
  lang    random article introductions from that language's Wikipedia
  domain  a fixed set of Wikipedia articles that carry a domain's register

Then feed the result to make_corpus.py, one --topic per file:

  python3 corpus/fetch_topics.py --out topics --code-root ~/src
  python3 corpus/make_corpus.py --tokenizer $MODEL_DIR --target 3000 \
      --out corpus/trace_topics.jsonl \
      $(python3 corpus/fetch_topics.py --out topics --print-topic-flags)

Wikipedia asks for one request at a time from an identified client. This
script obeys that: 20 article introductions per request, one request every two
seconds, and it backs off when told to. A burst of parallel requests earns an
IP-level 429 that lasts long enough to matter.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Wikipedia language codes. Random article introductions: encyclopedic prose in
# that language, which is what a routing trace needs -- not parallel sentences.
LANGS = {"arabic": "ar", "chinese": "zh", "japanese": "ja", "russian": "ru", "portuguese": "pt",
         "italian": "it", "turkish": "tr", "french": "fr", "german": "de", "spanish": "es",
         "english": "en"}

# A register is not a language: these are English articles chosen so the
# vocabulary and sentence shape of a field are present in the trace.
DOMAINS = {
    "academic": ["Scientific method", "Peer review", "Academic publishing", "Research design",
                 "Statistical hypothesis testing", "Systematic review", "Citation", "Reproducibility"],
    "finance": ["Financial statement", "Capital asset pricing model", "Bond (finance)", "Derivative (finance)",
                "Monetary policy", "Discounted cash flow", "Hedge fund", "Balance sheet"],
    "journalism": ["Journalism", "Inverted pyramid (journalism)", "News agency", "Investigative journalism",
                   "Freedom of the press", "Editorial", "Fact-checking", "Associated Press"],
    "marketing": ["Marketing", "Brand", "Market segmentation", "Advertising", "Consumer behaviour",
                  "Marketing mix", "Search engine optimization", "Customer relationship management"],
    "medical": ["Myocardial infarction", "Diabetes mellitus", "Pharmacology", "Immune system",
                "Antibiotic", "Clinical trial", "Hypertension", "Anesthesia"],
    "legal": ["Contract", "Tort", "Constitutional law", "Criminal procedure", "Intellectual property",
              "Civil procedure", "Precedent", "Due process"],
    "technical": ["Computer network", "Operating system", "Database", "Compiler", "Cryptography",
                  "Distributed computing", "Transmission Control Protocol", "Virtual memory"],
    "translation": ["Translation", "Machine translation", "Linguistics", "Semantics", "Language interpretation",
                    "Translation studies", "Syntax", "Morphology (linguistics)"],
}

# topic -> (glob, the fence label make_corpus.py should use)
CODE = {
    "go": ("**/*.go", "go"), "java": ("**/*.java", "java"), "cpp": ("**/*.cpp", "cpp"),
    "typescript": ("**/*.ts", "typescript"), "php": ("**/*.php", "php"), "ruby": ("**/*.rb", "ruby"),
    "swift": ("**/*.swift", "swift"), "python": ("**/*.py", "python"), "sql": ("**/*.sql", "sql"),
    "css": ("**/*.css", "css"), "rust": ("**/*.rs", "rust"), "javascript": ("**/*.js", "javascript"),
    "html": ("**/*.html", "html"), "rlang": ("**/*.R", "r"), "latex": ("**/*.tex", "latex"),
    "config": ("**/*.y*ml", "yaml"),
}
SKIP_DIRS = ("/.git/", "/node_modules/", "/dist/", "/build/", "/.venv/", "/__pycache__/")

UA = "deepseek-v41-flash-spark corpus builder (routing trace; one request at a time)"
REQUEST_GAP = 2.0
_last = [0.0]


def wiki(lang: str, params: dict, tries: int = 6):
    """One Wikipedia API call, rate limited and backing off on 429/503."""
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(tries):
        gap = REQUEST_GAP - (time.time() - _last[0])
        if gap > 0:
            time.sleep(gap)
        _last[0] = time.time()
        try:
            return json.loads(urllib.request.urlopen(req, timeout=45).read())
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or i == tries - 1:
                raise
            time.sleep(5 * (i + 1) + random.random())
    return {}


def wiki_random_intros(lang: str, target: int, max_requests: int = 14) -> list:
    """Introductions of random articles. exlimit allows 20 per request as long
    as only the intro is asked for -- a full extract is one page per request."""
    out, used = [], 0
    for _ in range(max_requests):
        if used >= target:
            break
        d = wiki(lang, {"action": "query", "generator": "random", "grnnamespace": "0",
                        "grnlimit": "20", "prop": "extracts", "exintro": "1",
                        "explaintext": "1", "exlimit": "20"})
        pages = d.get("query", {}).get("pages", [])
        if not pages:
            break
        for p in pages:
            t = p.get("extract") or ""
            if len(t) > 250:
                out.append(t)
                used += len(t)
    return out


def wiki_articles(lang: str, titles: list, cap: int = 24_000) -> list:
    """Full text of named articles, one request each, each capped so a single
    very long article cannot become the whole topic."""
    out = []
    for title in titles:
        d = wiki(lang, {"action": "query", "prop": "extracts", "explaintext": "1",
                        "exlimit": "1", "titles": title})
        pages = d.get("query", {}).get("pages", [])
        t = (pages[0].get("extract") or "") if pages else ""
        if len(t) > 800:
            out.append(t[:cap])
    return out


def code_files(root: str, pattern: str, target: int, seed: int = 11) -> list:
    files = [f for f in glob.glob(os.path.join(os.path.expanduser(root), pattern), recursive=True)
             if not any(s in f for s in SKIP_DIRS) and 1500 < os.path.getsize(f) < 80_000]
    random.Random(seed).shuffle(files)
    out, used = [], 0
    for f in files:
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if len(t) < 800:
            continue
        out.append(t)
        used += len(t)
        if used >= target:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="topics", help="directory to write <kind>/<topic>.txt into")
    ap.add_argument("--code-root", default=None, help="tree to take code from; without it the code topics are skipped")
    ap.add_argument("--chars", type=int, default=40_000, help="characters to gather per topic")
    ap.add_argument("--only", default="", help="comma-separated topics, instead of the whole catalogue")
    ap.add_argument("--list", action="store_true", help="print the catalogue and exit")
    ap.add_argument("--print-topic-flags", action="store_true",
                    help="print the --topic flags for make_corpus.py for whatever is already in --out")
    a = ap.parse_args()

    kinds = {**{t: "code" for t in CODE}, **{t: "lang" for t in LANGS}, **{t: "domain" for t in DOMAINS}}
    if a.list:
        for kind in ("code", "lang", "domain"):
            print(f"{kind:7s} {' '.join(sorted(t for t, k in kinds.items() if k == kind))}")
        return 0

    if a.print_topic_flags:
        flags = []
        for topic, kind in sorted(kinds.items()):
            path = os.path.join(a.out, kind, f"{topic}.txt")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                flags.append(f"--topic {topic}:{'code' if kind == 'code' else 'prose'}:{path}"
                             + (f":{CODE[topic][1]}" if kind == "code" else ""))
        print(" ".join(flags))
        return 0

    want = [t.strip() for t in a.only.split(",") if t.strip()] or sorted(kinds)
    unknown = [t for t in want if t not in kinds]
    if unknown:
        print(f"unknown topics: {', '.join(unknown)} (--list shows the catalogue)", file=sys.stderr)
        return 2

    gaps = []
    for topic in want:
        kind = kinds[topic]
        path = os.path.join(a.out, kind, f"{topic}.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) >= a.chars * 0.6:
            print(f"ok   {topic:<12} {os.path.getsize(path):>7,} chars (already there)")
            continue
        if kind == "code":
            if not a.code_root:
                continue
            parts = code_files(a.code_root, CODE[topic][0], a.chars)
        elif kind == "lang":
            parts = wiki_random_intros(LANGS[topic], a.chars)
        else:
            parts = wiki_articles("en", DOMAINS[topic])
        used = sum(len(p) for p in parts)
        # CJK text is far denser per character, so a smaller file is still a
        # large number of tokens; the token count make_corpus.py prints is the
        # one that matters.
        thin = used < a.chars * 0.4
        print(f"{'THIN' if thin else 'ok  '} {topic:<12} {used:>7,} chars from {len(parts)} sources")
        if thin:
            gaps.append(topic)
        if used:
            with open(path, "w") as f:
                f.write("\n\n".join(parts))

    if gaps:
        print(f"\nthin: {', '.join(gaps)} -- point --code-root at a tree that has them, "
              f"or drop them from the corpus", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
