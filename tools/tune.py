"""tune.py -- choose what this box should be good at, see what it costs, run it.

A keep-set is a cache policy: per layer only the top-N routed experts stay in
memory, and which N they are is decided by a measured routing trace. Picking
that set is the one configuration choice on this machine that changes both what
the model is good at and whether it fits, and until now it was made by editing
two numbers in a file and waiting three minutes to find out.

This is that choice, made visible. Select topics; the coverage bars say how much
of each topic's measured routing survives the current budget, the panel on the
right says what the budget costs against the memory this box has right now, and
the footer says which topic is worst served and what it would take to fix.

  ./tune.sh                       interactive
  ./tune.sh --list                the topics this keep-set carries
  ./tune.sh --topics a,b --print  the environment a selection implies
"""

from __future__ import annotations

import argparse
import curses
import glob
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCKS = " ▏▎▍▌▋▊▉█"
KEEP_STEPS = [round(0.02 * i, 2) for i in range(3, 31)]          # 6 % .. 60 %
CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
COVERAGE_TARGET = 0.85


def bar(frac: float, width: int) -> str:
    """A meter with eighth-block resolution, so small differences are visible."""
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    rem = int((frac * width - full) * 8)
    s = "█" * full + (BLOCKS[rem] if rem and full < width else "")
    return s.ljust(width, "·")


def find_stats(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    prof = os.environ.get("EXPERT_PROFILE")
    if prof:
        p = os.path.join(ROOT, "results/keepsets", prof, "coverage.json")
        if os.path.exists(p):
            return p
    cands = sorted(glob.glob(os.path.join(ROOT, "results/keepsets/*/coverage.json")))
    cands += sorted(glob.glob(os.path.join(ROOT, "results/trace-*/stats/coverage.json")))
    best, best_n = None, -1
    for c in cands:                       # the one that carries the most topics
        n = len(B.topic_names(c))
        if n > best_n:
            best, best_n = c, n
    return best


# --- state ------------------------------------------------------------------

class State:
    def __init__(self, host, index, stats_path, keep, max_seq, fmt, selection,
                 transient_slots=B.TRANSIENT_SLOTS_DEFAULT, keep_free_gb=B.KEEP_FREE_GB_DEFAULT):
        self.host, self.index, self.stats_path = host, index, stats_path
        self.keep, self.max_seq, self.fmt = keep, max_seq, fmt
        # The arena has to hold the kept set PLUS the transient ring, and the
        # engine's own default ring is 400 slots, not 8. Sizing against one
        # value and running with the other is how a keep-set that reads nothing
        # from NVMe quietly starts reading 392 experts a step.
        self.transient_slots, self.keep_free_gb = transient_slots, keep_free_gb
        self.sel = set(selection)
        self.cursor, self.scroll, self.filter, self.pane = 0, 0, "", 0
        self.typing = False
        self.msg = ""

    @property
    def visible(self):
        ts = self.index.topics if self.index else []
        f = self.filter.lower()
        return [t for t in ts if f in t.lower()] if f else list(ts)

    def plan(self):
        return B.plan(self.host, self.index, tuple(sorted(self.sel)), self.keep,
                      self.max_seq, fmt=self.fmt, transient_slots=self.transient_slots,
                      keep_free_gb=self.keep_free_gb)

    def curves(self):
        """Coverage under the CURRENT selection. With nothing selected the
        engine ranks on every topic, so that is what is shown."""
        if not self.index:
            return {}
        sel = tuple(sorted(self.sel)) if self.sel else tuple(self.index.topics)
        got = self.index.curves(sel)
        return got[0] if got else {}


# --- drawing ----------------------------------------------------------------

C = {}


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    n = curses.COLORS
    def mk(i, fg):
        curses.init_pair(i, fg, -1)
        return curses.color_pair(i)
    if n >= 256:
        C["accent"] = mk(1, 67)     # steel blue: structure
        C["good"] = mk(2, 71)
        C["warn"] = mk(3, 179)
        C["bad"] = mk(4, 167)
        C["muted"] = mk(5, 243)
        C["bright"] = mk(6, 254)
        C["on"] = mk(7, 80)         # selected topic
    else:
        C["accent"] = mk(1, curses.COLOR_BLUE)
        C["good"] = mk(2, curses.COLOR_GREEN)
        C["warn"] = mk(3, curses.COLOR_YELLOW)
        C["bad"] = mk(4, curses.COLOR_RED)
        C["muted"] = mk(5, curses.COLOR_WHITE)
        C["bright"] = mk(6, curses.COLOR_WHITE)
        C["on"] = mk(7, curses.COLOR_CYAN)


def sp(s: str) -> str:
    """Letter-spaced section label."""
    return " ".join(s)


def put(w, y, x, s, attr=0, maxw=None):
    h, W = w.getmaxyx()
    if y < 0 or y >= h or x >= W:
        return
    if maxw is not None:
        s = s[:maxw]
    s = s[:max(0, W - x - 1)]
    try:
        w.addstr(y, x, s, attr)
    except curses.error:
        pass


MIN_H, MIN_W = 20, 70


def draw(w, st: State):
    w.erase()
    h, W = w.getmaxyx()
    if h < MIN_H or W < MIN_W:
        put(w, 0, 0, f"window is {W}x{h}; this needs at least {MIN_W}x{MIN_H}", C.get("warn", 0))
        put(w, 1, 0, "resize, or use ./tune.sh --list / --print", C.get("muted", 0))
        w.noutrefresh()
        curses.doupdate()
        return
    p = st.plan()
    split = max(46, int(W * 0.54))          # left pane width
    rx = split + 3                          # right pane x

    # --- header
    title = " DeepSeek-V4.1-Flash "
    put(w, 0, 0, title, C["bright"] | curses.A_REVERSE | curses.A_BOLD)
    hostline = f"{st.host.name[:28]} · {st.host.total_gb:.1f} GB · {st.host.available_gb:.1f} free · {st.fmt} experts"
    put(w, 0, max(len(title) + 2, W - len(hostline) - 1), hostline, C["muted"])
    if st.host.busy:
        put(w, 1, 0, f" already running here: {st.host.busy} — this box holds one at a time",
            C["bad"] | curses.A_BOLD)
    elif st.host.note:
        put(w, 1, 0, " " + st.host.note, C["warn"])

    y = 2
    # --- left: topics
    focus = st.pane == 0
    put(w, y, 1, sp("TOPICS"), (C["accent"] if focus else C["muted"]) | curses.A_BOLD)
    if st.index and st.index.topics:
        sub = f"{len(st.index.topics)} available · {len(st.sel)} selected"
    else:
        sub = "this keep-set carries no per-topic histogram"
    hidden = 0
    if st.typing or st.filter:
        sub += f"   filter: {st.filter}" + ("_" if st.typing else "")
    put(w, y + 2, 1, "─" * split, C["muted"])

    curves = st.curves()
    vis = st.visible
    list_top = y + 3
    list_h = max(3, h - list_top - 9)   # 8 rows of controls + the key line
    if st.cursor < st.scroll:
        st.scroll = st.cursor
    if st.cursor >= st.scroll + list_h:
        st.scroll = st.cursor - list_h + 1
    bw = max(8, split - 37)

    put(w, y + 2, 22 + bw + 5, " traced ", C["muted"])

    if not vis:
        if st.index and st.index.topics:
            put(w, list_top, 3, "nothing matches that filter", C["muted"])
        else:
            for i, line in enumerate([
                "Build one with a traced corpus:",
                "  corpus/make_corpus.py --topic NAME:KIND:PATH",
                "  tools/expert_trace.py  then  tools/expert_stats.py",
                "",
                "Without topics the whole keep-set is used, which is",
                "what the shipped profiles do. The budget panel is live.",
            ]):
                put(w, list_top + i, 3, line, C["muted"], maxw=split)
    for i, t in enumerate(vis[st.scroll:st.scroll + list_h]):
        row = list_top + i
        idx = st.scroll + i
        on = t in st.sel
        here = idx == st.cursor and focus
        put(w, row, 0, "▌" if here else " ", C["accent"] | curses.A_BOLD)
        put(w, row, 2, "●" if on else "○", (C["on"] if on else C["muted"]) | (curses.A_BOLD if on else 0))
        nm = t[:16]
        put(w, row, 4, nm.ljust(17), (C["bright"] | curses.A_BOLD) if on else C["muted"])
        cv = curves.get(t)
        if cv is None:
            put(w, row, 22, "·" * bw + "    —", C["muted"])
        else:
            v = cv[B.keep_n(st.keep)]
            col = C["good"] if v >= COVERAGE_TARGET else (C["warn"] if v >= 0.7 else C["bad"])
            put(w, row, 22, bar(v, bw), col if on else C["muted"])
            put(w, row, 22 + bw + 1, f"{v:.2f}", (col | curses.A_BOLD) if on else C["muted"])
        n = (st.index.tokens.get(t, 0) if st.index else 0)
        thin = n < B.TopicIndex.THIN
        label = f"{n / 1000:.0f}k" if n >= 10000 else (f"{n / 1000:.1f}k" if n >= 1000 else str(n))
        put(w, row, 22 + bw + 6, f"{label:>5}", C["bad"] if thin else C["muted"])
    hidden = max(0, len(vis) - list_h - st.scroll)

    if hidden:
        sub += f" · {hidden} below"
    if st.scroll:
        sub += f" · {st.scroll} above"
    put(w, y + 1, 1, sub.ljust(split), C["muted"], maxw=split)

    # --- right: budget
    put(w, y, rx, sp("BUDGET"), C["accent"] | curses.A_BOLD)
    rw = max(24, W - rx - 2)
    put(w, y + 2, rx, "─" * rw, C["muted"])

    def row(yy, label, value, attr=0, unit=""):
        s = f"{value}{unit}"
        put(w, yy, rx, label[:max(0, rw - len(s) - 1)], C["muted"])
        put(w, yy, rx + rw - len(s), s, attr or C["bright"])

    ry = y + 3
    row(ry, "experts resident", f"{p.kept:,} / {B.N_ROUTED:,}", C["bright"] | curses.A_BOLD)
    row(ry + 1, "", f"{p.resident_frac * 100:.1f} %", C["bright"])
    row(ry + 3, "expert arena", f"{p.arena:.1f} GB")
    row(ry + 4, "dense weights", f"{p.dense:.1f} GB")
    row(ry + 5, "drafter experts", f"{p.dspark:.1f} GB")
    row(ry + 6, f"KV cache · {st.max_seq // 1024}k", f"{p.kv:.1f} GB" if p.kv >= 1 else f"{p.kv * 1000:.0f} MB")
    put(w, ry + 7, rx, "─" * rw, C["muted"])
    row(ry + 8, "resident", f"{p.resident:.1f} GB", C["bright"] | curses.A_BOLD)
    row(ry + 9, "free after load", f"{p.free_after_load:.1f} GB",
        C["good"] if p.free_after_load >= p.floor else C["bad"])
    row(ry + 11, "room to launch", f"{p.launch_slack:+.1f} GB",
        C["good"] if p.launch_slack >= 3 else (C["warn"] if p.launch_slack >= 0 else C["bad"]))
    vmark = {"ok": (" FITS ", C["good"]), "tight": (" TIGHT ", C["warn"]), "over": (" WILL NOT LOAD ", C["bad"])}
    txt, col = vmark[p.verdict]
    put(w, ry + 12, rx + rw - len(txt), txt, col | curses.A_REVERSE | curses.A_BOLD)

    if h >= 28:
        note = ["A step reads only the experts a token",
                "activates. Topics change the keep",
                "fraction you need — and that is the",
                "arena, not the step."]
        for i, ln in enumerate(note):
            put(w, ry + 15 + i, rx, ln, C["muted"], maxw=rw)

    # --- sliders, anchored to the bottom edge
    sy = h - 8
    put(w, sy, 1, "─" * split, C["muted"])
    kf = st.pane == 1
    put(w, sy + 1, 1, sp("RESIDENT EXPERTS"), (C["accent"] if kf else C["muted"]) | curses.A_BOLD)
    mk = p.max_keep()
    kb = max(10, split - 22)
    put(w, sy + 2, 1, f"◂ {st.keep * 100:4.0f} % ▸", (C["bright"] | curses.A_BOLD) if kf else C["muted"])
    filled = bar(st.keep / 0.6, kb)
    limit = int(min(1.0, mk / 0.6) * kb)
    put(w, sy + 2, 12, filled[:limit], C["good"] if p.verdict == "ok" else C["warn"])
    put(w, sy + 2, 12 + limit, filled[limit:], C["bad"])
    put(w, sy + 2, 12 + kb + 2, f"max {mk * 100:.0f} % here", C["muted"])

    cf = st.pane == 2
    put(w, sy + 4, 1, sp("CONTEXT"), (C["accent"] if cf else C["muted"]) | curses.A_BOLD)
    ctx = f"{st.max_seq // 1024}k" if st.max_seq >= 1024 else str(st.max_seq)
    put(w, sy + 5, 1, f"◂ {ctx:>5} ▸", (C["bright"] | curses.A_BOLD) if cf else C["muted"])
    room = max(0.0, p.free_after_load - p.floor) * B.GB / B.KV_BYTES_PER_TOKEN
    fits = f"{room / 1e6:.1f}M" if room >= 1e6 else f"{room / 1000:.0f}k"
    if st.max_seq <= B.VALIDATED_MAX_SEQ:
        msg = f"run to {B.VALIDATED_MAX_SEQ // 1024}k here; the cache alone has room for {fits}"
        if len(msg) > split - 12:
            msg = f"cache has room for {fits}"
        put(w, sy + 5, 12, msg, C["muted"], maxw=split - 12)
    else:
        msg = f"past the {B.VALIDATED_MAX_SEQ // 1024}k run here — prefill is the limit, not the cache"
        if len(msg) > split - 12:
            msg = f"past the {B.VALIDATED_MAX_SEQ // 1024}k run here"
        put(w, sy + 5, 12, msg, C["warn"], maxw=split - 12)

    # --- the one line that matters
    fy = h - 2
    t, v = p.weakest
    if t:
        col = C["good"] if v >= COVERAGE_TARGET else (C["warn"] if v >= 0.7 else C["bad"])
        put(w, fy, 1, "weakest selected topic  ", C["muted"])
        put(w, fy, 25, f"{t} {v:.2f}", col | curses.A_BOLD)
        need = st.index.keep_for(tuple(sorted(st.sel)), COVERAGE_TARGET) if st.index else None
        if need and abs(need - st.keep) > 0.005:
            verb = "raise to" if need > st.keep else "enough at"
            q = B.plan(st.host, st.index, tuple(sorted(st.sel)), need, st.max_seq, fmt=st.fmt)
            tail = "" if q.verdict != "over" else "  — more than this box holds"
            rec = (f"{verb} {need * 100:.0f} % for {COVERAGE_TARGET:.2f} on every one{tail}"
                   if W >= 96 else f"{verb} {need * 100:.0f} %{tail}")
            put(w, fy, min(45, W - len(rec) - 2), rec,
                C["warn"] if q.verdict == "over" else C["muted"])
    elif st.index and st.index.topics:
        put(w, fy, 1, "no topic selected — the keep-set would use all of them", C["muted"])
    if st.msg:
        # transient, and worth the key line for one keypress
        put(w, h - 1, 1, st.msg.ljust(W - 2)[:W - 2], C["warn"] | curses.A_BOLD)
    else:
        keys = ("↑↓ topic   space select   ←→ adjust   tab pane   a all   n none   / filter   "
                "m fit   r RUN   q quit")
        if len(keys) > W - 2:
            keys = "↑↓ space ←→ tab · a all · n none · / filter · m fit · r RUN · q quit"
        put(w, h - 1, 1, keys[:W - 2], C["muted"])
    w.noutrefresh()
    curses.doupdate()


# --- interaction ------------------------------------------------------------

def step(vals, cur, d):
    if cur in vals:
        i = vals.index(cur)
    else:
        i = min(range(len(vals)), key=lambda j: abs(vals[j] - cur))
    return vals[max(0, min(len(vals) - 1, i + d))]


def loop(w, st: State) -> str | None:
    curses.curs_set(0)
    try:
        curses.set_escdelay(25)
    except Exception:  # noqa: BLE001
        pass
    init_colors()
    w.keypad(True)
    while True:
        draw(w, st)
        try:
            k = w.getch()
        except KeyboardInterrupt:
            return None
        st.msg = ""   # a message lasts until the next key
        vis = st.visible
        st.cursor = max(0, min(st.cursor, len(vis) - 1)) if vis else 0

        if st.typing:
            if k in (27,):                      # esc
                st.typing, st.filter = False, ""
            elif k in (10, 13, curses.KEY_ENTER):
                st.typing = False
            elif k in (curses.KEY_BACKSPACE, 127, 8):
                st.filter = st.filter[:-1]
            elif 32 <= k < 127:
                st.filter += chr(k)
                st.cursor = 0
            continue

        if k == ord("q"):
            return None
        if k == ord("/"):
            st.typing = True
        elif k == 9:                            # tab
            st.pane = (st.pane + 1) % 3
        elif k == curses.KEY_BTAB:
            st.pane = (st.pane - 1) % 3
        elif k in (curses.KEY_DOWN, ord("j")):
            st.cursor = min(len(vis) - 1, st.cursor + 1) if vis else 0
        elif k in (curses.KEY_UP, ord("k")):
            st.cursor = max(0, st.cursor - 1)
        elif k == curses.KEY_NPAGE:
            st.cursor = min(len(vis) - 1, st.cursor + 10) if vis else 0
        elif k == curses.KEY_PPAGE:
            st.cursor = max(0, st.cursor - 10)
        elif k == ord(" ") and vis:
            t = vis[st.cursor]
            st.sel.symmetric_difference_update({t})
        elif k == ord("a"):
            st.sel |= set(vis)
        elif k == ord("n"):
            st.sel -= set(vis)
        elif k in (curses.KEY_RIGHT, curses.KEY_LEFT):
            d = 1 if k == curses.KEY_RIGHT else -1
            if st.pane == 2:
                st.max_seq = step(CTX_STEPS, st.max_seq, d)
            else:
                st.keep = step(KEEP_STEPS, st.keep, d)
        elif k == ord("f"):
            st.fmt = "fp4" if st.fmt == "cb3" else "cb3"
        elif k == ord("m"):                     # snap to the coverage target
            need = st.index.keep_for(tuple(sorted(st.sel)), COVERAGE_TARGET) if st.index and st.sel else None
            if need:
                st.keep = step(KEEP_STEPS, need, 0)
                if st.keep < need:
                    st.keep = step(KEEP_STEPS, st.keep, 1)
            else:
                st.msg = "no keep fraction reaches that on every selected topic"
        elif k in (ord("r"), ord("R")):
            p = st.plan()
            if st.host.busy:
                st.msg = "something is already running — ./stop.sh first, then w to write and run"
                continue
            if p.verdict == "over":
                st.msg = "this will not load — lower the keep fraction first"
                continue
            return "run"
        elif k in (ord("w"), ord("W")):
            return "write"


# --- output -----------------------------------------------------------------

MANAGED = ("EXPERT_TOPICS", "PRUNE_KEEP", "MAX_SEQ", "ARENA_GB", "TRACE_STATS", "EXPERT_FORMAT",
           "TRANSIENT_SLOTS", "KEEP_FREE_GB")


def env_for(st: State) -> dict:
    p = st.plan()
    e = {
        "PRUNE_KEEP": f"{st.keep:.2f}",
        "MAX_SEQ": str(st.max_seq),
        "ARENA_GB": f"{math.ceil(p.arena)}",
        "EXPERT_FORMAT": st.fmt,
        # written because the arena above was sized against them
        "TRANSIENT_SLOTS": str(st.transient_slots),
        "KEEP_FREE_GB": f"{st.keep_free_gb:g}",
        "TRACE_STATS": os.path.relpath(st.stats_path, ROOT) if st.stats_path else "",
    }
    if st.sel:
        e["EXPERT_TOPICS"] = ",".join(sorted(st.sel))
    return e


def write_env(env: dict, path: str) -> str:
    if not os.path.exists(path):
        ex = os.path.join(ROOT, "env.example")
        if os.path.exists(ex):
            shutil.copyfile(ex, path)
    lines = open(path).read().splitlines() if os.path.exists(path) else []
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    seen = set()
    out = []
    for ln in lines:
        k = ln.split("=", 1)[0].strip() if "=" in ln and not ln.lstrip().startswith("#") else None
        if k in env:
            if env[k]:
                out.append(f"{k}={env[k]}")
            seen.add(k)
        elif k in MANAGED and k not in env:
            seen.add(k)              # drop a managed key this selection does not set
        else:
            out.append(ln)
    for k, v in env.items():
        if k not in seen and v:
            out.append(f"{k}={v}")
    open(path, "w").write("\n".join(out) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="choose what this box should be good at")
    ap.add_argument("--stats", help="coverage.json to read topics from")
    ap.add_argument("--topics", default=os.environ.get("EXPERT_TOPICS", ""))
    ap.add_argument("--keep", type=float, default=float(os.environ.get("PRUNE_KEEP", "0.39")))
    ap.add_argument("--max-seq", type=int, default=int(os.environ.get("MAX_SEQ", "32768")))
    ap.add_argument("--format", default=os.environ.get("EXPERT_FORMAT", "cb3"), choices=("cb3", "fp4"))
    ap.add_argument("--transient-slots", type=int,
                    default=int(os.environ.get("TRANSIENT_SLOTS") or B.TRANSIENT_SLOTS_DEFAULT),
                    help="prefill slots outside the LRU; the arena is sized to hold these too")
    ap.add_argument("--keep-free-gb", type=float,
                    default=float(os.environ.get("KEEP_FREE_GB") or B.KEEP_FREE_GB_DEFAULT),
                    help="host memory the launcher leaves free")
    ap.add_argument("--list", action="store_true", help="print the topics and exit")
    ap.add_argument("--print", dest="show", action="store_true", help="print the environment and exit")
    ap.add_argument("--write", action="store_true", help="write the selection into .env and exit")
    a = ap.parse_args()

    host = B.read_host()
    sp_ = find_stats(a.stats)
    index = B.TopicIndex(sp_) if sp_ else None
    sel = [t.strip() for t in a.topics.split(",") if t.strip()]
    unknown = [t for t in sel if not index or t not in index.topics] if sel else []
    interactive = sys.stdout.isatty() and not (a.list or a.show or a.write)
    if unknown and not interactive:
        # a script asked for something this keep-set cannot serve: say so and stop
        where = os.path.relpath(sp_, ROOT) if sp_ else "any coverage.json in the checkout"
        print(f"not in {where}: {', '.join(unknown)}", file=sys.stderr)
        print(f"have: {', '.join(index.topics) if index else '(none)'}", file=sys.stderr)
        return 2
    if unknown:
        sel = [t for t in sel if t not in unknown]   # the screen is where this gets fixed

    st = State(host, index, sp_, a.keep, a.max_seq, a.format, sel,
               transient_slots=a.transient_slots, keep_free_gb=a.keep_free_gb)
    if unknown:
        st.msg = f"dropped, not in this keep-set: {', '.join(unknown)}"

    if a.list:
        if not index or not index.topics:
            print(f"no per-topic histograms in {sp_ or 'any coverage.json'}")
            return 1
        cur = index.curves(tuple(index.topics))[0]
        n = B.keep_n(a.keep)
        print(f"{os.path.relpath(sp_, ROOT)} — {len(index.topics)} topics, coverage at keep {a.keep:.0%}")
        for t in index.topics:
            nt = index.tokens.get(t, 0)
            flag = "  thin" if nt < B.TopicIndex.THIN else ""
            print(f"  {t:<14} {bar(cur[t][n], 24)} {cur[t][n]:.2f}  {nt:>8,} tokens{flag}")
        return 0

    if a.show or a.write or not sys.stdout.isatty():
        p = st.plan()
        env = env_for(st)
        if a.write:
            write_env(env, os.path.join(ROOT, ".env"))
            print(f"wrote {len(env)} settings to .env (previous kept as .env.bak)")
        if host.busy:
            print(f"# already running here: {host.busy} — this box holds one at a time", file=sys.stderr)
        for k, v in env.items():
            print(f"{k}={v}")
        print(f"# {p.kept:,} experts resident ({p.resident_frac:.1%}), {p.resident:.1f} GB resident, "
              f"{p.free_after_load:.1f} GB free after load — {p.verdict}", file=sys.stderr)
        return 0 if p.verdict != "over" else 1

    action = curses.wrapper(loop, st)
    if action is None:
        return 0
    env = env_for(st)
    write_env(env, os.path.join(ROOT, ".env"))
    p = st.plan()
    print(f"{p.kept:,} experts resident ({p.resident_frac:.1%}) · arena {p.arena:.0f} GB · "
          f"context {st.max_seq // 1024}k · {p.free_after_load:.1f} GB free after load")
    if st.sel:
        print(f"topics: {', '.join(sorted(st.sel))}")
    print(".env written (previous kept as .env.bak)")
    if action == "write":
        return 0
    print("starting the server — ./stop.sh to stop it\n")
    return subprocess.call([os.path.join(ROOT, "start.sh")], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
