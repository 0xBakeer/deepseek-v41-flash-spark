"""Draw the tune screen at many sizes and states, with a fake curses window.

A TUI fails by writing outside the window or over itself, and neither shows up
in a unit test of the arithmetic. This renders the real `draw()` into a
character grid and checks three things: it never raises, it never writes out of
bounds, and the rows that must stay legible are not overwritten by something
else. No terminal, no pyte, no GPU.

If this disagrees with what you see on screen, suspect a stale bytecode cache
before you suspect the test. Some interpreters set `sys.pycache_prefix`, which
puts the .pyc somewhere other than the package's own `__pycache__` -- on macOS
the system Python uses `~/Library/Caches/com.apple.python/<abs path>` -- so
deleting `tools/__pycache__` clears nothing. `python3 -c "import sys;
print(sys.pycache_prefix)"` says where to look.
"""
import curses
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402
import tune as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


# the window-shaped object the tool already uses for --render
FakeWin = T._Grid


def render(st, h, w):
    win = FakeWin(h, w)
    T.draw(win, st)
    return win


def advanced(win, h, w):
    """(ok, detail) for one drawn topic screen. Every state of that screen is
    put through this, so a row added to it is covered by all of them."""
    rows = [win.row(y) for y in range(h)]
    problems = []
    if not any("BUDGET" in r.replace(" ", "") or "B U D G E T" in r for r in rows):
        problems.append("no budget panel")
    # nothing in the left pane may reach the right pane's column
    rx = max(46, int(w * 0.54)) + 3
    for r in rows[6:h - 9]:
        left = r[:rx - 1].rstrip()
        if len(left) >= rx - 1:
            problems.append(f"left pane reaches the budget column: {left[-28:]!r}")
            break
    if not any("RESIDENTEXPERTS" in r.replace(" ", "") for r in rows):
        problems.append("no keep slider")
    if not any("CONTEXT" in r.replace(" ", "") for r in rows):
        problems.append("no context row")
    # Both sliders must survive. A collision does not leave two rows
    # overlapping -- the later write wins and the earlier row disappears --
    # so the test is that BOTH adjustable rows are still on screen, on
    # different lines, and neither is the footer.
    # Told apart by what the arrows are wrapped around -- `◂ 36 % ▸` against
    # `◂ 256k ▸` -- and not by whether a per cent sign appears anywhere on the
    # row: the message beside the context arrows carries one too.
    arrows = [y for y, r in enumerate(rows) if r.lstrip().startswith("◂")]
    pct = [y for y in arrows if re.match(r"◂\s*[\d.]+ %", rows[y].lstrip())]
    ctx = [y for y in arrows if y not in pct]
    foot = [y for y, r in enumerate(rows) if "weakest" in r or "no topic selected" in r]
    if len(pct) != 1:
        problems.append(f"keep slider rows: {len(pct)}")
    if len(ctx) != 1:
        problems.append(f"context slider rows: {len(ctx)}")
    if pct and ctx and pct[0] == ctx[0]:
        problems.append("the two sliders are on one row")
    if foot and (set(pct) | set(ctx)) & set(foot):
        problems.append("a slider row collides with the footer")
    if any(len(r) > w for r in rows):
        problems.append("a row is wider than the window")
    return not problems, "; ".join(problems)


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail and not ok else ''}")
    if not ok:
        fails.append(name)


# draw() reads colours out of a module dict that init_colors() fills from a real
# terminal; plain attributes render the same layout.
T.C.update({k: 0 for k in ("accent", "good", "warn", "bad", "muted", "bright", "on")})
curses.doupdate = lambda: None

stats = os.path.join(ROOT, "results/keepsets/general/coverage.json")
index = B.TopicIndex(stats) if os.path.exists(stats) else None
host = B.Host("gb10-test", 130.6e9, 118.6e9, True)

SIZES = [(24, 80), (30, 100), (32, 104), (40, 140), (24, 70), (60, 200), (20, 70)]
STATES = [
    ("nothing selected", set(), 0.39, 32768, 0),
    ("one topic", {index.topics[0]} if index and index.topics else set(), 0.32, 32768, 0),
    ("all topics", set(index.topics) if index else set(), 0.44, 131072, 1),
    ("over budget", set(index.topics) if index else set(), 0.60, 262144, 2),
    ("tiny keep", set(), 0.06, 4096, 1),
]
# the state nothing exercised: a keep-set with no per-topic histograms at all,
# where the left pane is help text instead of a list
NO_INDEX = ("no topic histograms", set(), 0.39, 32768, 0)

for label, sel, keep, seq, pane in STATES + [NO_INDEX]:
    use_index = None if label == NO_INDEX[0] else index
    for h, w in SIZES:
        st = T.State(host, use_index, stats, keep, seq, "cb3", sorted(sel))
        st.view = "advanced"
        st.pane = pane
        st.msg = "a message that has to fit" if pane == 2 else ""
        try:
            win = render(st, h, w)
        except curses.error:
            check(f"{label} at {w}x{h}", False, "wrote out of bounds")
            continue
        except Exception as e:  # noqa: BLE001
            check(f"{label} at {w}x{h}", False, f"{type(e).__name__}: {e}")
            continue
        if h < T.MIN_H or w < T.MIN_W:
            check(f"{label} at {w}x{h} says it is too small", "at least" in win.row(0))
            continue
        check(f"{label} at {w}x{h}", *advanced(win, h, w))

# --- the rows the topic screen grew later -----------------------------------
# `s` opens a prompt on the key line, and a profiles file that will not parse
# takes the warning row under the header. Both are drawn over rows that already
# carried something, so they go through the same collision checks.
EXTRAS = [
    ("saving a profile",
     lambda st: setattr(st, "asking", {"label": "save these 2 topics as:",
                                       "buf": "a name that is being typed"})),
    ("a profiles file that will not parse",
     lambda st: setattr(st, "problem", "results/keepsets/profiles.json: not valid JSON "
                                       "(Expecting value: line 2 column 1 (char 15))")),
]
for label, mutate in EXTRAS:
    for h, w in SIZES:
        st = T.State(host, index, stats, 0.39, 32768, "cb3", sorted(index.topics) if index else [])
        st.view = "advanced"
        mutate(st)
        try:
            win = render(st, h, w)
        except curses.error:
            check(f"{label} at {w}x{h}", False, "wrote out of bounds")
            continue
        except Exception as e:  # noqa: BLE001
            check(f"{label} at {w}x{h}", False, f"{type(e).__name__}: {e}")
            continue
        if h < T.MIN_H or w < T.MIN_W:
            continue
        ok, detail = advanced(win, h, w)
        rows = [win.row(y) for y in range(h)]
        if st.asking and "save these 2 topics as:" not in rows[h - 1]:
            ok, detail = False, f"the prompt is not on the key line: {rows[h - 1]!r}"
        if st.problem and not any("not valid JSON" in r for r in rows):
            ok, detail = False, "the profiles problem is not on the screen"
        check(f"{label} at {w}x{h}", ok, detail)

# the two keys added later have to be offered, not just accepted
st = T.State(host, index, stats, 0.39, 32768, "cb3", [])
st.view = "advanced"
check("the topic screen offers the save key", "s save" in render(st, 60, 200).row(59))
st.view = "easy"
check("the profile screen offers the brief key", "b brief" in render(st, 60, 200).row(59))

# `a` opens Weight Atlas on both screens, so select-all is `A` -- a key that moved has to be
# offered under its new name, or the screen teaches the old one. The label is the credit as
# well as the key, so it is compared exactly. tools/test_tune_atlas.py covers the rest of it.
st = T.State(host, index, stats, 0.39, 32768, "cb3", [])
st.view = "advanced"
line = render(st, 60, 200).row(59)
check("the topic screen offers the atlas key", T.ATLAS_KEY in line, repr(line))
check("  and select-all under its new name", "A all" in line, repr(line))
st.view = "easy"
line = render(st, 60, 200).row(59)
check("the profile screen offers the atlas key", T.ATLAS_KEY in line, repr(line))

# the title bar names the checkout and the handle, at every size and on both screens
for view in ("easy", "advanced"):
    for h, w in SIZES:
        st = T.State(host, index, stats, 0.39, 32768, "cb3", [])
        st.view = view
        head = render(st, h, w).row(0)
        if h < T.MIN_H or w < T.MIN_W:
            continue
        check(f"the {view} title bar at {w}x{h}", "deepseek-v41-flash-spark · 0xbakeer" in head,
              repr(head))

# --- the easy view, at every size and on every profile ----------------------
for h, w in SIZES:
    for cursor in range(len(T.PROFILES)):
        st = T.State(host, index, stats, 0.39, 32768, "cb3", [])
        st.pcursor = cursor
        try:
            win = render(st, h, w)
        except curses.error:
            check(f"profiles at {w}x{h} row {cursor}", False, True)
            continue
        except Exception as e:  # noqa: BLE001
            check(f"profiles at {w}x{h} row {cursor}", False, True)
            print(f"     {type(e).__name__}: {e}")
            continue
        if h < T.MIN_H or w < T.MIN_W:
            continue
        rows = [win.row(y) for y in range(h)]
        bad = []
        if not any("WHATSHOULDTHISBOX" in r.replace(" ", "") for r in rows):
            bad.append("no heading")
        # the highlighted profile must always be on screen, whatever the scroll
        if not any(T.PROFILES[cursor][0] in r for r in rows):
            bad.append(f"selected profile {T.PROFILES[cursor][0]!r} scrolled off")
        if not any(r.startswith("▌") for r in rows):
            bad.append("no cursor marker")
        if any(len(r) > w for r in rows):
            bad.append("a row is wider than the window")
        if cursor == 0:
            check(f"profiles at {w}x{h}", not bad, "; ".join(bad))
        elif bad:
            check(f"profiles at {w}x{h} row {cursor}", False, "; ".join(bad))

# --- a profile that came out of a file --------------------------------------
# It is drawn like the shipped ones, plus two things they never show: where it
# came from, and which of its topics this keep-set does not carry.
# None for the rank: a file does not name one, so such a profile is budgeted
# with whatever rule the screen is already on.
USER = [("Mine", "a profile read from a file, naming one topic that is not here",
         [index.topics[0] if index and index.topics else "python", "not-a-topic"], False,
         "results/keepsets/profiles.json", None)]
for h, w in SIZES:
    st = T.State(host, index, stats, 0.39, 32768, "cb3", [], user_profiles=USER)
    st.pcursor = len(T.PROFILES)          # a new name is appended after the built-ins
    st.msg = "wrote the topic brief to tune-brief.md"
    try:
        win = render(st, h, w)
    except curses.error:
        check(f"a user profile at {w}x{h}", False, "wrote out of bounds")
        continue
    except Exception as e:  # noqa: BLE001
        check(f"a user profile at {w}x{h}", False, f"{type(e).__name__}: {e}")
        continue
    if h < T.MIN_H or w < T.MIN_W:
        continue
    rows = [win.row(y) for y in range(h)]
    bad = []
    if not any("Mine" in r for r in rows):
        bad.append("the user's profile is not on screen")
    if not any("yours" in r for r in rows):
        bad.append("it is not marked as the user's")
    if not any("not in this keep-set: not-a-topic" in r for r in rows):
        bad.append("the topic it could not use is not named")
    if "wrote the topic brief" not in rows[h - 1]:
        bad.append(f"the message is not on the key line: {rows[h - 1]!r}")
    if any(len(r) > w for r in rows):
        bad.append("a row is wider than the window")
    check(f"a user profile at {w}x{h}", not bad, "; ".join(bad))

# applying a profile selects its topics, a keep fraction that fits, and the
# ranking rule it was budgeted with -- the last one because the keep fraction
# shown for it was computed under that rule and means nothing under another.
# The two-topic `general` keep-set above carries none of a built-in profile's topics, so this
# block needs the topic catalogue -- since "Everything" was retired (2026-09-13) no profile
# resolves to "all topics in the file" any more.
topic_stats = os.path.join(ROOT, "results/keepsets/topics/coverage.json")
topic_index = B.TopicIndex(topic_stats)
st = T.State(host, topic_index, topic_stats, 0.39, 32768, "cb3", [], rank="sum")
pr = next(p for p in st.profiles() if p["topics"])
st.apply_profile(pr)
check("applying a profile selects its topics", sorted(st.sel) == sorted(pr["topics"]),
      f"{sorted(st.sel)} != {sorted(pr['topics'])}")
check("  and a keep fraction that loads", st.plan().verdict != "over", True)
check("  and the rule it was budgeted with", st.rank == pr["rank"] == "maxmin", st.rank)
check("  every profile resolves", all(p["status"] for p in st.profiles()), True)
# A profile out of a file names no rule, so applying it must not change the one
# in force -- it would silently re-budget the selection.
st = T.State(host, index, stats, 0.39, 32768, "cb3", [], rank="sum", user_profiles=USER)
mine = next(p for p in st.profiles() if p["name"] == "Mine")
check("a user profile is budgeted with the rule in force", mine["rank"], "sum")
st.apply_profile(mine)
check("  and applying it leaves that rule alone", st.rank, "sum")

# --- the rule has to be on the screen, both views ---------------------------
# The same keep fraction under two rules is two different keep-sets, so a
# coverage bar without the rule next to it cannot be checked against anything.
for rank in ("sum", "maxmin"):
    # the catalogue, not the two-topic general set: a profile shows its rule only once it resolves
    st = T.State(host, topic_index, topic_stats, 0.39, 32768, "cb3", sorted(topic_index.topics)[:3],
                 rank=rank)
    st.view = "advanced"
    rows = [render(st, 40, 140).row(y) for y in range(40)]
    check(f"the topic screen names the rule ({rank})", any(f"rank {rank}" in r for r in rows),
          "not on screen")
    # The profile screen shows each profile's OWN rule beside its keep fraction,
    # which for every shipped profile is maxmin whatever the screen is set to.
    st.view = "easy"
    rows = [render(st, 40, 140).row(y) for y in range(40)]
    check(f"the profile screen names each profile's rule ({rank})",
          any("% of experts · maxmin" in r for r in rows), "not on screen")
    # and it must not have pushed anything off the edge
    check(f"  without overflowing a row ({rank})", all(len(r) <= 140 for r in rows))

# every topic row must carry its traced-token count
if index and index.topics:
    st = T.State(host, index, stats, 0.39, 32768, "cb3", index.topics[:1])
    st.view = "advanced"
    win = render(st, 32, 104)
    rows = [win.row(y) for y in range(32)]
    hit = [r for r in rows if index.topics[0] in r]
    check("a topic row shows its sample size", bool(hit) and "tokens" not in hit[0] and any(
        c.isdigit() for c in hit[0].split()[-1]), hit[0] if hit else "row not found")

# --- the gate verdict, on the screen ----------------------------------------
# A user choosing "Backend" is choosing a keep-set that has been measured, and
# the measurement has to be on the row with the name. These checks are on the
# topic catalogue, because a profile only resolves against a keep-set that
# carries its topics.
for h, w in SIZES:
    if h < T.MIN_H or w < T.MIN_W:
        continue
    st = T.State(host, topic_index, topic_stats, 0.39, 32768, "cb3", [],
                 rank="maxmin", source="saliency")
    rows = [render(st, h, w).row(y) for y in range(h)]
    body = "\n".join(rows)
    bad = []
    if "strict" not in body:
        bad.append("no gate verdict on any row")
    if "gated 2026-09-" not in body:
        bad.append("no gate date on any row")
    if "keep 36 %" not in body and "keep 40 %" not in body:
        bad.append("no gated keep fraction on any row")
    if any(len(r) > w for r in rows):
        bad.append("a row is wider than the window")
    check(f"the profile screen carries the gate at {w}x{h}", not bad, "; ".join(bad))
    # The sub-line and the scroll marker share row 3. The marker is written at
    # a fixed column, so the sub-line has to stop before it rather than be
    # overwritten mid-word: `…the topic-by-topic vi5 more below` is the failure
    # this catches, and the check is that the column in front of the marker is
    # blank.
    sub = rows[3].ljust(w)
    if "more below" in sub:
        check(f"  the sub-line stops before the scroll marker at {w}x{h}",
              sub[w - 15] == " ", repr(sub))

# which keep fraction holds a filled 256k is a fact, and the screen has to carry
# it where the context length is being chosen -- on both screens
for view, seq in (("easy", 262144), ("easy", 32768), ("advanced", 262144)):
    st = T.State(host, topic_index, topic_stats, 0.36, seq, "cb3",
                 sorted(topic_index.topics)[:3], rank="maxmin", source="saliency")
    st.view = view
    body = "\n".join(render(st, 40, 140).row(y) for y in range(40))
    check(f"the {view} screen names the 256k keep fraction at {seq // 1024}k",
          "36 %" in body and "256k" in body, body[:0])

# a profile from a file has no gate, and must not borrow the row above's
NOGATE = [("No gate", "topics that were never gated together",
           sorted(topic_index.topics)[:3], None, "results/keepsets/profiles.json", None)]
st = T.State(host, topic_index, topic_stats, 0.39, 32768, "cb3", [], user_profiles=NOGATE,
             rank="maxmin", source="saliency")
st.pcursor = len(T.PROFILES)
rows = [render(st, 40, 140).row(y) for y in range(40)]
at = next(i for i, r in enumerate(rows) if "No gate" in r)
check("a profile with no gate reads untested", "untested" in rows[at])
check("  and gets no gate line of its own", "gated" not in rows[at + 2], repr(rows[at + 2]))

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
