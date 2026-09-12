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

for label, sel, keep, seq, pane in STATES:
    for h, w in SIZES:
        st = T.State(host, index, stats, keep, seq, "cb3", sorted(sel))
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
        rows = [win.row(y) for y in range(h)]
        problems = []
        if not any("BUDGET" in r.replace(" ", "") or "B U D G E T" in r for r in rows):
            problems.append("no budget panel")
        if not any("RESIDENTEXPERTS" in r.replace(" ", "") for r in rows):
            problems.append("no keep slider")
        if not any("CONTEXT" in r.replace(" ", "") for r in rows):
            problems.append("no context row")
        # Both sliders must survive. A collision does not leave two rows
        # overlapping -- the later write wins and the earlier row disappears --
        # so the test is that BOTH adjustable rows are still on screen, on
        # different lines, and neither is the footer.
        arrows = [y for y, r in enumerate(rows) if r.lstrip().startswith("◂")]
        pct = [y for y in arrows if "%" in rows[y]]
        ctx = [y for y in arrows if "%" not in rows[y]]
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
        check(f"{label} at {w}x{h}", not problems, "; ".join(problems))

# every topic row must carry its traced-token count
if index and index.topics:
    st = T.State(host, index, stats, 0.39, 32768, "cb3", index.topics[:1])
    win = render(st, 32, 104)
    rows = [win.row(y) for y in range(32)]
    hit = [r for r in rows if index.topics[0] in r]
    check("a topic row shows its sample size", bool(hit) and "tokens" not in hit[0] and any(
        c.isdigit() for c in hit[0].split()[-1]), hit[0] if hit else "row not found")

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
sys.exit(1 if fails else 0)
