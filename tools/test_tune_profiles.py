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


def state(profiles, idx=index, sel=(), rank="sum"):
    return T.State(host, idx, STATS, 0.39, 32768, "cb3", sel, rank=rank,
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
check("  owning no gate record, whatever the file says", [p[3] for p in profs], [None, None])
check("  and where it came from", profs[0][4], T.short_path(good))
check("  naming no ranking rule, so applying it leaves the one in force",
      [p[5] for p in profs], [None, None])
odd = write("odd.json", {"profiles": [
    {"name": "  Spaced   out ", "description": "a description\nover two lines",
     "topics": ["python", "python", "html"]}]})
got, problems = T.read_profiles(odd)
check("a name is read as one line of text", got[0][0], "Spaced out")
check("  and so is the description", got[0][1], "a description over two lines")
check("  a topic named twice counts once", got[0][2], ["python", "html"])
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
check("  every shipped profile is budgeted maxmin",
      sorted({m[5] for m in merged if m[4] == T.BUILT_IN}), ["maxmin"])
check("  and the user's carries no rule of its own", merged[-1][5], None)

by_user = {p["name"]: p for p in state(profs).profiles()}
by = by_user
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

# --- the ranking rule the run will use --------------------------------------
# Written out with the rest, because the coverage on the screen was read off a
# keep-set built with it: a .env that reproduces the keep fraction but not the
# rule reproduces a different keep-set.
r = run(["--stats", STATS, "--topics", "english,html,css", "--keep", "0.36",
         "--rank", "maxmin", "--print"])
check("--print emits the rule", "DSV41_PRUNE_RANK=maxmin" in r.stdout, True)
check("  and it is one of the keys the tool manages", "DSV41_PRUNE_RANK" in T.MANAGED, True)
sums = run(["--stats", STATS, "--topics", "english,html,css", "--keep", "0.36",
            "--rank", "sum", "--print"])
check("  sum says so instead", "DSV41_PRUNE_RANK=sum" in sums.stdout, True)
check("a bad rule is refused rather than served as the default",
      run(["--stats", STATS, "--topics", "", "--rank", "mxmn", "--print"]).returncode, 2)
env_rank = subprocess.run(
    [sys.executable, os.path.join(ROOT, "tools/tune.py"), "--stats", STATS,
     "--topics", "english,html,css", "--keep", "0.36", "--print"],
    capture_output=True, text=True,
    env={**os.environ, "EXPERT_TOPICS": "", "PRUNE_KEEP": "0.39", "DSV41_PRUNE_RANK": "maxmin"})
check("the rule defaults from the environment the engine reads",
      "DSV41_PRUNE_RANK=maxmin" in env_rank.stdout, True)
# and the two rules have to be telling the screen different things, or none of
# the above is worth writing down
mm = run(["--stats", STATS, "--topics", "english,html,css", "--keep", "0.36",
          "--rank", "maxmin", "--list"]).stdout
sm = run(["--stats", STATS, "--topics", "english,html,css", "--keep", "0.36",
          "--rank", "sum", "--list"]).stdout
check("--list ranks with the rule it was given", ("rank maxmin" in mm, "rank sum" in sm),
      (True, True))
check("  and the coverage it prints moves with it", mm.split("\n")[1:] != sm.split("\n")[1:], True)

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

# --- the generation gate, read back off the records --------------------------
# Every count the profile screen shows comes out of a GATE.md, and the index
# beside them says which run is which profile's record and what keep-set that
# run measured. Neither can be checked by reading the screen, so it is checked
# here: the record exists, it is the newest full run on exactly those topics,
# and the numbers parse to what the file says in words.
import gate_profile as G  # noqa: E402

KEEPSETS = os.path.join(ROOT, "results", "keepsets")
gates_index = T.read_gates_index()
check("the gate index parses", len(gates_index) > 0, True)

for name, _blurb, topics, record, _rank in T.PROFILES:
    where = os.path.join(KEEPSETS, record, T.GATE_BASENAME)
    check(f"{name}: its record is in the checkout", os.path.exists(where), True)
    # gate_profile.py writes there by default, so the two have to agree or the
    # next run of the gate starts a second, empty directory beside this one
    check(f"{name}: the gate tool writes to that directory", G.slug(name), record)
    g = T.gate_for(record, topics, gates_index)
    check(f"{name}: it has a gate record", bool(g), True)
    if not g:
        continue
    check(f"{name}: on exactly the topics it ships", sorted(g["topics"]), sorted(topics))
    check(f"{name}: with a keep fraction", isinstance(g["keep"], float), True)
    check(f"{name}: and the pair it was ranked with", (g["rank"], g["source"]),
          ("maxmin", "saliency"))
    check(f"{name}: strict is within the run count", 0 <= g["strict"] <= g["runs"], True)
    # nothing newer in the same file describes the same bundle: a later full run
    # on these topics would be the record, and this one would be history
    later = [x for x in T.read_gate(where)
             if not x["filtered"] and sorted(x["topics"]) == sorted(topics)]
    check(f"{name}: nothing newer supersedes it", later[-1]["run"], g["run"])

# the numbers, against what the file says in words
backend = T.gate_for("backend", dict(zip([p[0] for p in T.PROFILES],
                                         [p[2] for p in T.PROFILES]))["Backend"], gates_index)
check("Backend's record is the 2026-09-14 run", backend["run"], "2026-09-14 03:36")
check("  7 of 10 failed, so 3 passed strictly", (backend["strict"], backend["runs"]), (3, 10))
check("  and all ten finished a correct answer", backend["finished"], 10)
check("  at the keep fraction that holds a filled 256k", backend["keep"], T.SAFE_256K_KEEP)

# A filtered re-run says the prompts it ran pass and nothing about the ones it
# did not, so it can never be a profile's record.
euro = os.path.join(KEEPSETS, "european_languages", T.GATE_BASENAME)
check("a filtered re-run is in the file", any(x["filtered"] for x in T.read_gate(euro)), True)
check("  and is not the record", T.gate_for("european_languages",
      dict((p[0], p[2]) for p in T.PROFILES)["European languages"], gates_index)["run"],
      "2026-09-13 22:05")
# ... and a run on a different bundle is a different measurement: European
# languages was gated WITH reasoning_lang on 2026-09-14 and ships without it.
check("a run on other topics is not the record either",
      T.gate_for("european_languages", ["english", "german"], gates_index), None)
check("no record directory, no gate", T.gate_for(None, ["english"], gates_index), None)
check("a file that is not a gate log yields nothing",
      T.read_gate(os.path.join(KEEPSETS, "general", T.GATE_BASENAME)), [])
check("the smallest keep fraction anything was gated at", T.gate_floor(gates_index),
      T.SAFE_256K_KEEP)
check("every index entry names a run that is in its file",
      [k for k in gates_index
       if k[1] not in {x["run"] for x in
                       T.read_gate(os.path.join(KEEPSETS, k[0], T.GATE_BASENAME))}], [])

# --- what the screen does with it -------------------------------------------
# The keep fraction a shipped profile is budgeted at is the one its gate ran at,
# not the one the coverage target implies. Under `saliency` those are three
# times apart: every topic in the catalogue is above 0.85 at keep 0.12, which is
# below anything that has ever been asked to generate a sentence.
sal = B.TopicIndex(STATS, "saliency")
st_sal = T.State(host, sal, STATS, 0.36, 32768, "cb3", [], rank="maxmin", source="saliency")
by = {p["name"]: p for p in st_sal.profiles()}
check("a shipped profile is budgeted at the keep it was gated at", by["Frontend"]["keep"], 0.36)
check("  which the coverage target alone would not have chosen",
      sal.keep_for(tuple(sorted(by["Frontend"]["topics"])), 0.85, rank="maxmin") < 0.2, True)
check("  and the screen says both counts", by["Frontend"]["status"],
      "7 of 10 strict · 9 finished")
check("  with the strict count alone where nothing counted the rest",
      by["Writing"]["status"], "5 of 8 strict")
check("  a good finished count is not called a warning",
      (by["Backend"]["status"], by["Backend"]["tone"]),
      ("3 of 10 strict · 10 finished", "good"))
line = T.gate_line(by["Frontend"], st_sal, 96)
check("the gate line dates it", "gated 2026-09-14" in line, True)
check("  and names the keep fraction", "keep 36 %" in line, True)
check("  and does not repeat the pair when it is the screen's",
      "maxmin/saliency" in line, False)
other = T.State(host, B.TopicIndex(STATS), STATS, 0.36, 32768, "cb3", [], rank="maxmin")
mismatch = {p["name"]: p for p in other.profiles()}["Frontend"]
check("a screen ranking something else says so on every gate line",
      all("maxmin/saliency" in T.gate_line(mismatch, other, w) for w in (60, 80, 96, 140)), True)

# A profile whose topics this keep-set does not all carry is not the profile
# that was gated. It still applies, with the topics that are there, but the
# counts from a run of the whole bundle would be the most confident wrong number
# on the screen, so it shows none.
shipped = list(T.PROFILES)
try:
    T.PROFILES.append(("Frontend plus one", "the Frontend bundle and a topic that is not here",
                       shipped[0][2] + ["not-a-topic"], shipped[0][3], "maxmin"))
    partial = {p["name"]: p for p in
               T.State(host, sal, STATS, 0.36, 32768, "cb3", [],
                       rank="maxmin", source="saliency").profiles()}
finally:
    T.PROFILES[:] = shipped
short = partial["Frontend plus one"]
check("a profile short of a topic still applies", len(short["topics"]), len(shipped[0][2]))
check("  naming what it could not use", short["missing"], ["not-a-topic"])
check("  and shows no gate, because that is not what was gated", short["gate"], None)
check("  so it reads untested", "untested" in short["status"], True)
check("a profile from a file is untested too", "untested" in by_user["Arabic desk"]["status"], True)
check("  and is budgeted from the coverage target, having no measured keep",
      by_user["Arabic desk"]["gate"], None)
check("a keep-set carrying none of a profile's topics says that instead",
      {p["name"]: p for p in T.State(host, B.TopicIndex(
          os.path.join(KEEPSETS, "general", "coverage.json")), STATS, 0.36, 32768,
          "cb3", []).profiles()}["Frontend"]["status"], "not in this keep-set")

# --- --print emits the keep-set the profile was gated on ---------------------
# The keep fraction alone does not name a set of experts: the ranking rule and
# the histogram family pick which ones a fraction holds. A run reproduced with
# the gated keep and the screen's default family is a different keep-set from
# the one the counts on the screen were measured on, so applying a profile with
# a record takes its whole pair.
def run_clean(args, **env):
    """Without whatever DSV41_* the shell running the tests happens to carry."""
    e = {k: v for k, v in os.environ.items() if not k.startswith("DSV41_")}
    e.update({"EXPERT_TOPICS": "", "PRUNE_KEEP": "0.39"}, **env)
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools/tune.py")] + args,
                          capture_output=True, text=True, env=e)


r = run_clean(["--stats", STATS, "--profile", "backend", "--print"])
check("--profile prints the keep fraction its gate ran at", "PRUNE_KEEP=0.36" in r.stdout, True)
check("  the rule that record was measured with", "DSV41_PRUNE_RANK=maxmin" in r.stdout, True)
check("  and the histogram family it was measured on", "DSV41_PRUNE_SOURCE=saliency" in r.stdout,
      True)
check("  none of which came from the environment",
      ("DSV41_PRUNE_RANK" in r.stderr or "DSV41_PRUNE_SOURCE" in r.stderr), False)
check("  so the settings load", r.returncode, 0)
# --source on the command line is overridden the same way --rank already was
check("an explicit --source is overridden by the record",
      "DSV41_PRUNE_SOURCE=saliency" in
      run_clean(["--stats", STATS, "--source", "counts", "--profile", "backend",
                 "--print"]).stdout, True)

# in the screen, the same thing: applying takes the pair, and everything derived
# from the histograms is re-read off the family that record used
st_counts = T.State(host, B.TopicIndex(STATS, "counts"), STATS, 0.39, 32768, "cb3", [],
                    rank="sum", source="counts")
front = {p["name"]: p for p in st_counts.profiles()}["Frontend"]
check("before applying, the screen names the pair that differs",
      "on maxmin/saliency" in T.gate_line(front, st_counts, 96), True)
st_counts.apply_profile(front)
check("applying a gated profile takes its histogram family", st_counts.source, "saliency")
check("  and its ranking rule", st_counts.rank, "maxmin")
check("  and the index is re-read off that family", st_counts.index.source, "saliency")
check("  so .env says what was measured",
      {k: v for k, v in T.env_for(st_counts).items() if k.startswith("DSV41_")},
      {"DSV41_PRUNE_RANK": "maxmin", "DSV41_PRUNE_SOURCE": "saliency"})
check("  and the line no longer has a pair to warn about",
      "maxmin/saliency" in T.gate_line({p["name"]: p for p in st_counts.profiles()}["Frontend"],
                                       st_counts, 96), False)

# a profile from a file names no pair and must leave both where they are
st_file = T.State(host, B.TopicIndex(STATS, "counts"), STATS, 0.39, 32768, "cb3", [],
                  rank="sum", source="counts",
                  user_profiles=[("From a file", "no gate, no pair", ["html", "css"], None,
                                  "results/keepsets/profiles.json", None)])
st_file.apply_profile({p["name"]: p for p in st_file.profiles()}["From a file"])
check("a profile from a file leaves the family alone", st_file.source, "counts")
check("  and the rule", st_file.rank, "sum")

# A keep-set that cannot be ranked the way the record was: nothing is switched,
# because writing a family the file has no histograms for is a configuration the
# engine refuses three minutes into a load. It is said out loud instead.
thin = os.path.join(TMP, "counts-only")
os.makedirs(thin, exist_ok=True)
_cov = json.load(open(STATS))
for _rows in _cov["per_layer"].values():
    for _k in [k for k in _rows if k.startswith("saliency_")]:
        del _rows[_k]
json.dump(_cov, open(os.path.join(thin, "coverage.json"), "w"))
r = run_clean(["--stats", os.path.join(thin, "coverage.json"), "--profile", "backend", "--print"])
check("a keep-set with no saliency histograms is not written as saliency",
      "DSV41_PRUNE_SOURCE=counts" in r.stdout, True)
check("  and the keep fraction still comes from the record", "PRUNE_KEEP=0.36" in r.stdout, True)
check("  with the mismatch on stderr, not left to be noticed later",
      "is not the keep-set that was gated" in r.stderr, True)
st_thin = T.State(host, B.TopicIndex(os.path.join(thin, "coverage.json"), "counts"),
                  os.path.join(thin, "coverage.json"), 0.39, 32768, "cb3", [],
                  rank="sum", source="counts")
back = {p["name"]: p for p in st_thin.profiles()}["Backend"]
st_thin.apply_profile(back)
check("  the screen keeps the family it can serve", st_thin.source, "counts")
check("  and goes on naming the pair the record used",
      all("maxmin/saliency" in T.gate_line(back, st_thin, w) for w in (60, 80, 96, 140)), True)

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
