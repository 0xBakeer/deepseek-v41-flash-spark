"""The generated task brief: that it comes from the loaded keep-set, and that
the commands in it are right.

Run: python3 tools/test_tune_brief.py

A brief whose commands do not run is worse than no brief, so every command it
emits is taken apart and checked against the argument parser of the script it
names: the script exists, every flag it passes is one that script defines, and
every flag that script requires is there. The Python snippet is run for real
against two keep-sets in the checkout, because "per-layer counts can be
concatenated" is the claim the whole no-re-trace section rests on.
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402
import tune as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))
    if not ok:
        fails.append(name)


TMP = tempfile.mkdtemp(prefix="tune-brief-")
os.environ["XDG_CONFIG_HOME"] = TMP          # never read the box's own profiles

TOPICS = os.path.join(ROOT, "results/keepsets/topics/coverage.json")   # 35 topics
PLAIN = os.path.join(ROOT, "results/keepsets/general/coverage.json")   # two mixed ones
NONE = os.path.join(ROOT, "results/keepsets/code/coverage.json")       # no topics at all
host = B.Host("gb10-test", 130.6e9, 118.6e9, True)


def state(stats, sel=()):
    idx = B.TopicIndex(stats) if stats and os.path.exists(stats) else None
    return T.State(host, idx, stats, 0.39, 32768, "cb3", sel, profiles_path=os.path.join(TMP, "p.json"))


rich = T.brief(state(TOPICS))
plain = T.brief(state(PLAIN))
none = T.brief(state(NONE))
empty = T.brief(T.State(host, None, None, 0.39, 32768, "cb3", ()))

# --- it is written out of the file that is loaded, not out of a template -----
idx = B.TopicIndex(TOPICS)
check("the brief names the keep-set it was written from",
      "results/keepsets/topics/coverage.json" in rich, True)
check("  every topic in it", all(f"`{t}`" in rich for t in idx.topics), True)
check("  with the tokens each was traced on",
      all(f"{idx.tokens[t]:,}" in rich for t in idx.topics), True)
check("  and a different keep-set gives a different table",
      ("| `chinese` |" in rich, "| `chinese` |" in plain, "| `coding` |" in plain),
      (True, False, True))
check("the sections a reader needs are all there",
      [h for h in ("## Why this is a task at all", "## What this keep-set already carries",
                   "## Where the catalogue is thin", "## The commands",
                   "## How much text a new topic needs",
                   "## What a new topic costs the topics already here",
                   "## Nothing already traced has to be traced again",
                   "## Gate it before trusting it") if h not in rich], [])
check("nothing in it is wider than the docs are", [ln for ln in rich.splitlines() if len(ln) > 99], [])

# --- the catalogue comparison is computed, not asserted ----------------------
cat = T.catalogue()
check("the catalogue is read from corpus/fetch_topics.py", sorted(cat), ["code", "domain", "lang"])
check("  all 35 of its topics are in the 35-topic keep-set",
      re.search(r"\| code \| 16 \| 16 \| 0 \|", rich) is not None, True)
check("  and none of them in a keep-set that carries two mixed ones",
      re.search(r"\| code \| 16 \| 0 \| 16 \|", plain) is not None, True)
check("  which are absent is spelled out", "absent from this keep-set, code: `config`" in plain, True)
check("  and a topic the catalogue does not know is named as such",
      "carried here but not in the catalogue" in plain, True)

# --- the evidence for how much text a topic needs ---------------------------
check("the measured correlation is quoted with where it came from",
      all(s in rich for s in ("**-0.36**", "**-0.00**", "results/keepsets/topics/GATE.md")), True)
check("  and recomputed on the file in hand", "The same correlation on the file loaded here" in rich, True)
check("  with the target it implies", "3,000 tokens" in rich, True)
check("the thin threshold comes from the tool", f"{B.TopicIndex.THIN:,}-token" in rich, True)

# --- the budget warning is this keep-set's own numbers ----------------------
check("the cost of one more topic is measured, not asserted",
      re.search(r"raises the coverage of the other 34 by \*\*\+0\.\d\d\d\*\*", rich) is not None, True)
check("  in keep fraction as well as coverage", "of the experts with all 35" in rich, True)
check("  and one keep step is priced", re.search(r"is 320 more resident experts, 4\.\d GB", rich) is not None, True)
check("  a narrow selection pays more than a broad one",
      float(re.search(r"other 2 by \*\*\+(0\.\d+)\*\*", T.brief(state(TOPICS, ("html", "css", "english")))).group(1))
      > float(re.search(r"other 34 by \*\*\+(0\.\d+)\*\*", rich).group(1)), True)

# --- a keep-set with no per-topic histograms at all -------------------------
for label, text in (("a keep-set that carries none", none), ("no keep-set at all", empty)):
    check(f"the brief is still written for {label}", text.startswith("# Task: add a topic"), True)
    check("  and says so rather than printing an empty table",
          "carries no per-topic histograms" in text or "no keep-set" in text, True)
    check("  the commands are there anyway", "## The commands" in text, True)
check("with no keep-set the sections that need one are left out",
      ("What a new topic costs" in empty, "| topic | traced tokens" in empty,
       "correlation on the file loaded here" in none), (True, False, False))
check("  and the catalogue table then reads absent all the way down",
      re.search(r"\| domain \| 8 \| 0 \| 8 \|", none) is not None, True)

# --- every command in it is checked against the parser it will meet ---------
def parser(script):
    """The flags a script defines, and the ones it requires. Read out of the
    source: importing these needs torch and a checkpoint."""
    src = open(os.path.join(ROOT, script)).read()
    flags, required = set(), set()
    for chunk in src.split("add_argument(")[1:]:
        m = re.match(r'\s*"(--[a-z0-9-]+)"', chunk)
        if not m:
            continue
        flags.add(m.group(1))
        if re.search(r"required=True", chunk.split("add_argument")[0]):
            required.add(m.group(1))
    return flags, required


def commands(text):
    """The shell commands out of the ```bash blocks, continuations joined."""
    out, inside, buf = [], False, ""
    for ln in text.splitlines():
        if ln.startswith("```"):
            inside = ln.startswith("```bash")
            continue
        if not inside or not ln.strip() or ln.lstrip().startswith("#"):
            continue
        buf += ln.rstrip("\\ ") + " "
        if not ln.rstrip().endswith("\\"):
            out.append(buf.strip())
            buf = ""
    return out


SCRIPT = {"tune.sh": "tools/tune.py", "./tune.sh": "tools/tune.py"}
seen = 0
for cmd in commands(rich):
    argv = shlex.split(cmd)
    script = SCRIPT.get(argv[0]) or (argv[1] if argv[0] in ("python3", "python") else None)
    check(f"a command names a script that exists: {' '.join(argv[:2])}",
          bool(script) and os.path.exists(os.path.join(ROOT, script)), True)
    if not script or not os.path.exists(os.path.join(ROOT, script)):
        continue
    flags, required = parser(script)
    used = {a for a in argv if a.startswith("--")}
    check(f"  every flag exists: {os.path.basename(script)}", sorted(used - flags), [])
    check("  and every required flag is given", sorted(required - used), [])
    seen += 1
check("every command block was checked", seen >= 6, True)

# --- the snippet that concatenates two traces, run for real -----------------
snippet = rich.split("```python")[1].split("```")[0]
compile(snippet, "<brief>", "exec")
work = os.path.join(TMP, "merge")
os.makedirs(os.path.join(work, "results/keepsets/topics"), exist_ok=True)
os.makedirs(os.path.join(work, "results/keepsets/reasoning"), exist_ok=True)
shutil.copyfile(TOPICS, os.path.join(work, "results/keepsets/topics/coverage.json"))
shutil.copyfile(PLAIN, os.path.join(work, "results/keepsets/reasoning/coverage.json"))
here = os.getcwd()
try:
    os.chdir(work)
    exec(compile(snippet, "<brief>", "exec"), {"__name__": "__main__"})
finally:
    os.chdir(here)
out = os.path.join(work, "results/keepsets/reasoning-merged/coverage.json")
check("the merge snippet in the brief runs", os.path.exists(out), True)
merged = B.TopicIndex(out)
check("  and the two sets of topics are both in the result",
      (len(merged.topics), "chinese" in merged.topics, "coding" in merged.topics),
      (len(idx.topics) + 2, True, True))
check("  with the counts intact", merged.tokens["chinese"], idx.tokens["chinese"])

# --- the ways a user reaches it ---------------------------------------------
r = subprocess.run([sys.executable, os.path.join(ROOT, "tools/tune.py"), "--stats", TOPICS,
                    "--topics", "", "--brief"], capture_output=True, text=True,
                   env={**os.environ, "EXPERT_TOPICS": ""})
check("--brief prints Markdown and exits 0", (r.returncode, r.stdout.startswith("# Task")),
      (0, True))
check("  and it is the same brief", "## The commands" in r.stdout, True)

root_was = T.ROOT
try:
    T.ROOT = TMP                      # so the key under test does not write into the checkout
    msg = T.write_brief(state(TOPICS))
finally:
    T.ROOT = root_was
check("the profile screen's key says where it wrote it", T.BRIEF_FILENAME in msg, True)
check("  and there is something there", os.path.getsize(os.path.join(TMP, T.BRIEF_FILENAME)) > 4000,
      True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
