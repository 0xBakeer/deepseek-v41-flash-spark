"""Profiles read from a file, and keeping the current selection as one.

Run: python3 tools/test_tune_profiles.py

The failure paths are most of this on purpose. A profiles file is written by
hand, so it will be malformed and it will name topics that are not there, and
neither may stop the tool from starting on its built-in profiles or drop a
topic without saying which. No terminal, no GPU, no checkpoint.
"""
import json
import os
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


TMP = tempfile.mkdtemp(prefix="tune-profiles-")
# Nothing here may depend on what is in the config directory of the box this
# runs on, so point the user's file at the temporary directory as well.
os.environ["XDG_CONFIG_HOME"] = TMP

STATS = os.path.join(ROOT, "results/keepsets/topics/coverage.json")
index = B.TopicIndex(STATS) if os.path.exists(STATS) else None
host = B.Host("gb10-test", 130.6e9, 118.6e9, True)


def write(name, obj):
    p = os.path.join(TMP, name)
    open(p, "w").write(obj if isinstance(obj, str) else json.dumps(obj))
    return p


def state(profiles, idx=index, sel=()):
    return T.State(host, idx, STATS, 0.39, 32768, "cb3", sel,
                   user_profiles=profiles, profiles_path=os.path.join(TMP, "saved.json"))


def run(args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools/tune.py")] + args,
                          capture_output=True, text=True,
                          env={**os.environ, "EXPERT_TOPICS": "", "PRUNE_KEEP": "0.39"})


# --- where the files are ----------------------------------------------------
check("the user's file lives under XDG_CONFIG_HOME", T.user_profiles_path(),
      os.path.join(TMP, "deepseek-v41-flash-spark", "profiles.json"))
check("the checkout's file is read first", T.profiles_files()[0],
      os.path.join(ROOT, "results", "keepsets", "profiles.json"))
check("  and the user's last, so it wins", T.profiles_files()[-1], T.user_profiles_path())
check("an explicit path replaces both", T.profiles_files("/tmp/x.json"), ["/tmp/x.json"])
check("a file that is not there is not a problem", T.read_profiles(os.path.join(TMP, "no.json")),
      ([], []))

# --- a file that is right ---------------------------------------------------
good = write("good.json", {"profiles": [
    {"name": "Arabic desk", "description": "Arabic and English, for a bilingual assistant",
     "topics": ["arabic", "english", "translation"]},
    {"name": "Frontend", "description": "markup only, on purpose", "topics": ["html", "css"],
     "gated": True},
]})
profs, problems = T.read_profiles(good)
check("a good file loads every profile in it", len(profs), 2)
check("  with nothing to report", problems, [])
check("  the name", profs[0][0], "Arabic desk")
check("  the description", profs[0][1], "Arabic and English, for a bilingual assistant")
check("  the topics", profs[0][2], ["arabic", "english", "translation"])
check("  never gated, whatever the file says", [p[3] for p in profs], [False, False])
check("  and where it came from", profs[0][4], T.short_path(good))
check("a bare list of profiles is accepted too",
      [p[0] for p in T.read_profiles(write("list.json", [
          {"name": "Plain list", "description": "", "topics": ["python"]}]))[0]], ["Plain list"])

# --- built-ins stay, a same-named user profile replaces one -----------------
merged = T.merge_profiles(T.PROFILES, profs)
check("the built-in profiles stay", len(merged), len(T.PROFILES) + 1)
check("  a user profile with a built-in's name replaces it, once",
      sum(1 for m in merged if m[0].lower() == "frontend"), 1)
check("  in place, so the order does not jump around", [m[0] for m in merged][:2],
      ["Frontend", "Backend"])
check("  and it is the user's one that survives",
      [m[2] for m in merged if m[0] == "Frontend"], [["html", "css"]])
check("  a new name is added at the end", merged[-1][0], "Arabic desk")

by = {p["name"]: p for p in state(profs).profiles()}
check("the screen shows a user profile", by["Arabic desk"]["topics"],
      ["arabic", "english", "translation"])
check("  as untested, like every other one", "untested" in by["Arabic desk"]["status"], True)
check("  marked as the user's", by["Arabic desk"]["mine"], True)
check("  and a shipped one not", by["Backend"]["mine"], False)
check("  with a budget of its own", by["Arabic desk"]["plan"].verdict != "over", True)

# --- a malformed file -------------------------------------------------------
bad = write("bad.json", '{"profiles": [')
got, problems = T.read_profiles(bad)
check("a malformed file yields no profiles", got, [])
check("  and one problem", len(problems), 1)
check("  that names the file", T.short_path(bad) in problems[0], True)
check("  and says what is wrong with it", "not valid JSON" in problems[0], True)
check("a file of the wrong shape is caught too",
      "expected" in T.read_profiles(write("shape.json", {"profiles": {"name": "x"}}))[1][0], True)

r = run(["--stats", STATS, "--topics", "", "--profiles-file", bad, "--profiles"])
check("the tool still starts on the built-in profiles", r.returncode, 0)
check("  every one of them", all(n in r.stdout for n, *_ in T.PROFILES), True)
check("  and the problem is on stderr, not swallowed", "not valid JSON" in r.stderr, True)

messy = write("messy.json", {"profiles": [
    {"name": "", "topics": ["python"]},
    ["not", "an", "object"],
    {"name": "No topics"},
    {"name": "Bad topics", "topics": "python"},
    {"name": "Bad text", "description": 7, "topics": ["python"]},
    {"name": "Fine", "description": "the one good entry", "topics": ["python", "html"]},
]})
got, problems = T.read_profiles(messy)
check("one bad entry does not cost the good ones", [p[0] for p in got], ["Fine"])
check("  every bad entry is reported", len(problems), 5)
check("  by name where it has one", any("'Bad topics'" in m for m in problems), True)
check("  and by position where it does not", any("profile 2" in m for m in problems), True)

# --- a topic name that is not in this keep-set ------------------------------
typo = write("typo.json", {"profiles": [
    {"name": "Typo set", "description": "one real topic and two typos",
     "topics": ["python", "hmtl", "englsh"]}]})
got, problems = T.read_profiles(typo)
check("a typo is not a file problem", problems, [])
msgs = T.unknown_topics(got, index)
check("an unknown topic name is reported", len(msgs), 1)
check("  naming both of them", all(t in msgs[0] for t in ("hmtl", "englsh")), True)
check("  and the profile they are in", "Typo set" in msgs[0], True)
pr = [p for p in state(got).profiles() if p["name"] == "Typo set"][0]
check("  the profile still applies, with the topics that exist", pr["topics"], ["python"])
check("  and carries the ones it could not use", pr["missing"], ["hmtl", "englsh"])
check("nothing survives silently against a keep-set with no topics",
      len(T.unknown_topics(got, None)), 1)

# --- saving the current selection -------------------------------------------
target = os.path.join(TMP, "saved.json")
T.save_profile(target, "Mine", "three topics", ["html", "css", "english"])
got, problems = T.read_profiles(target)
check("a saved profile reads back", got[0][2], ["css", "english", "html"])
check("  with its description", got[0][1], "three topics")
check("  and no problems", problems, [])
T.save_profile(target, "Other", "another", ["python"])
T.save_profile(target, "mine", "replaced", ["go"])
got, _ = T.read_profiles(target)
check("saving a name twice replaces it", len(got), 2)
check("  keeping the other entries", sorted(p[0] for p in got), ["Other", "mine"])
check("  and taking the new topics", [p[2] for p in got if p[0] == "mine"], [["go"]])
check("a saved profile describes itself by its topics",
      T.describe_selection(["html", "css"]), "css, html")
try:
    T.save_profile(bad, "Nope", "", ["python"])
    check("saving refuses to write over a file it cannot read", "it wrote", "it refused")
except T.ProfileError as e:
    check("saving refuses to write over a file it cannot read", "fix or move" in str(e), True)
check("  and leaves that file exactly as it was", open(bad).read(), '{"profiles": [')

st = state([], sel=["html", "css"])
msg = T.save_current(st, "From the screen")
check("the screen's save reports where it went", T.short_path(target) in msg, True)
check("  and the profile is on the screen immediately",
      "From the screen" in [p["name"] for p in st.profiles()], True)
check("  saving with nothing selected says so",
      T.save_current(state([]), "Empty"), "nothing selected, so there is nothing to save")
check("  and so does saving without a name", T.save_current(st, "  "), "a profile needs a name")

# --- the same thing from the command line -----------------------------------
cli = os.path.join(TMP, "cli.json")
r = run(["--stats", STATS, "--topics", "html,css", "--profiles-file", cli,
         "--save-profile", "From the CLI"])
check("--save-profile writes the file", r.returncode, 0)
check("  and it loads again", [p[0] for p in T.read_profiles(cli)[0]], ["From the CLI"])
r = run(["--stats", STATS, "--topics", "", "--profiles-file", cli, "--save-profile", "Empty"])
check("--save-profile with nothing selected exits 2", r.returncode, 2)
r = run(["--stats", STATS, "--topics", "python,nope", "--profiles-file", cli,
         "--save-profile", "Typo"])
check("an unknown topic on the command line still exits 2", r.returncode, 2)
r = run(["--stats", STATS, "--topics", "", "--profiles-file", cli, "--profiles"])
check("a saved profile shows up in --profiles", "From the CLI" in r.stdout, True)
check("  marked as the user's", "(yours)" in r.stdout, True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
