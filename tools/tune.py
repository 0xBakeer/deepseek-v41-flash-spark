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
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget as B  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCKS = " ▏▎▍▌▋▊▉█"

# --- the easy view ----------------------------------------------------------
# Named bundles of topics, for people who know what they want the box to do and
# not which experts that implies. A profile fixes only WHICH topics; the keep
# fraction it needs is computed from whatever coverage file is loaded, so these
# stay correct as the trace behind them improves.
#
# Every one of them carries a natural-language topic, and that is not padding.
# Selecting markup and stylesheets alone drops English coverage to 0.31, deep
# into the range where long output falls apart, and the prose inside an HTML
# page is English. Adding it back costs the markup topics about five points of
# coverage and buys English forty-five.
# The fourth field is whether the profile has been through the generation gate.
# None of them has yet, and the screen says so, because coverage cannot see a
# gap in the topic CATALOGUE -- only a gap in the selection. The Chat profile
# scored 0.85 or better on all five of its topics and still reasoned in circles
# on a two-train arithmetic question, because no topic in the catalogue carries
# the register thinking mode writes in. Coverage warned about nothing, because
# there was nothing to warn about. Only a generation gate finds that.
#
# The fifth is the ranking rule the profile is budgeted and served with, and
# every one of them says `maxmin` (DSV41_PRUNE_RANK). A profile is a bundle of
# topics that ONE request spans at once -- a coding request with thinking on
# writes prose, deliberation, HTML, CSS and JavaScript in a single generation --
# and such a request degenerates at whichever topic in the bundle the keep-set
# serves least. `sum` optimises the total routing mass kept, which is free to
# let an already well-served topic go on taking slots while another starves; on
# the box, over {english, html, python, reasoning, css, javascript, typescript}
# at keep 0.36, that is english at 0.522 against css at 0.820. `maxmin` spends
# the same budget on the worst-served topic instead and lands every one of the
# seven between 0.676 and 0.699, which is the quantity a bundle is chosen for.
PROFILES = [
    # Every profile carries `reasoning` and `reasoning_code`: a user runs any of these with
    # thinking on, and the experts that write deliberation -- and the ones that END it and begin
    # the answer -- live in those two topics and nowhere else. A profile without them can score
    # well on every topic it names and still never close a think block (docs/tune-tasks.md).
    # Under maxmin the two cost the other topics about 0.01 of coverage each.
    ("Frontend", "HTML, CSS, JavaScript, TypeScript, config, and the English around them",
     ["html", "css", "javascript", "typescript", "english", "technical", "config",
      "reasoning", "reasoning_code", "reasoning_design"], False, "maxmin"),
    ("Backend", "Python, Go, Java, SQL, configuration files, technical prose",
     ["python", "go", "java", "sql", "config", "technical", "english",
      "reasoning", "reasoning_code"], False, "maxmin"),
    # "Programming, broadly" carried fifteen topics and passed 4 of 16 on its gate (2026-09-13):
    # at this box's budget a profile that wide serves nothing. Its registers are now two
    # profiles of nine -- the web half is Frontend above, the systems half is this.
    ("Systems programming", "Rust, C++, Go, Java, SQL, config, and the prose around them",
     ["rust", "cpp", "go", "java", "sql", "config", "technical", "english",
      "reasoning", "reasoning_code"], False, "maxmin"),
    ("Chat and explanation", "Everyday questions, essays, summaries, technical explanation",
     ["english", "technical", "academic", "journalism", "translation",
      "reasoning", "reasoning_code"], False, "maxmin"),
    ("Medicine", "Clinical and pharmacological register, with academic prose",
     ["medical", "academic", "technical", "english", "reasoning", "reasoning_code"], False, "maxmin"),
    ("Law and finance", "Contracts, statutes, filings, financial reporting",
     ["legal", "finance", "academic", "english", "reasoning", "reasoning_code"], False, "maxmin"),
    ("Data and research", "Python, R, SQL, LaTeX, notebooks and config, academic writing",
     ["python", "rlang", "sql", "latex", "academic", "technical", "english", "config",
      "reasoning", "reasoning_code"], False, "maxmin"),
    # "Many languages" carried fourteen topics and passed 2 of 15 on its gate (2026-09-13) --
    # every natural-language prompt failed. Split by script family into two profiles of nine.
    # With thinking off every one of these languages came out clean on the same keep-set; with it
    # on, French, German, Chinese and Japanese corrupted a word and looped. The catalogue's
    # deliberation was English-only, so `reasoning_lang` -- thinking IN the language -- was added
    # and gated (2026-09-14): European went from 6 of 10 to 3 of 10. A small, spread topic under
    # `maxmin` swaps about thirteen experts a layer out of the tail (513 of 5,560 at keep 0.36),
    # and the tail is where the deliberation in these languages routes. It is not in the bundle.
    # Thinking on in a non-English language is the open item; thinking off is served.
    ("European languages", "English, German, French, Spanish, Italian, Portuguese, translation",
     ["english", "german", "french", "spanish", "italian", "portuguese", "translation",
      "reasoning", "reasoning_code"], False, "maxmin"),
    ("World languages", "English, Arabic, Chinese, Japanese, Russian, Turkish, translation",
     ["english", "arabic", "chinese", "japanese", "russian", "turkish", "translation",
      "reasoning", "reasoning_code"], False, "maxmin"),
    ("Writing", "Journalism, marketing copy, essays, translation",
     ["english", "journalism", "marketing", "academic", "translation",
      "reasoning", "reasoning_code"], False, "maxmin"),
    # "Everything" -- all 37 topics -- passed 4 of 39 on its gate (2026-09-13). It is not a
    # profile this box can serve and is no longer offered as one; the topic screen still lets a
    # user select every topic by hand, and the screen will show what that costs.
]
KEEP_STEPS = [round(0.02 * i, 2) for i in range(3, 31)]          # 6 % .. 60 %
CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
# The coverage a selected topic should reach. There is no universal right
# value: 0.85 is where the shipped keep-sets sit for the domains they were
# built for, and generation starts to degrade well below 0.7.
#
# It was calibrated on `sum` numbers, and so were the status thresholds on the
# profile screen. `maxmin` spends the same budget on the worst-served topic, so
# it reads LOWER at the top of the range and higher at the bottom: a coding
# bundle that showed 0.59 on its weakest topic and 0.82 on its strongest shows
# 0.68-0.74 on all of them. Both numbers describe the same keep-set; only a
# generation gate can say which side of the trade holds together, so the
# thresholds are left where they are until one has been run on maxmin.
COVERAGE_TARGET = float(os.environ.get("DSV41_COVERAGE_TARGET", "0.85"))


def bar(frac: float, width: int, solid: bool = True) -> str:
    """A meter with eighth-block resolution, so small differences are visible.

    A hollow bar means the topic behind it was traced on too few tokens to
    rank 384 experts. Its coverage is not merely uncertain, it is biased
    upward: the number is measured on the very sample that chose the experts,
    so a topic seen for 300 tokens scores as if it were well served.
    """
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    rem = int((frac * width - full) * 8)
    if not solid:
        return ("▒" * full).ljust(width, "·")
    s = "█" * full + (BLOCKS[rem] if rem and full < width else "")
    return s.ljust(width, "·")


def short_path(path: str | None) -> str:
    """Relative when it is inside the checkout, absolute when it is not --
    `../../../elsewhere/coverage.json` helps nobody."""
    if not path:
        return "no keep-set"
    rel = os.path.relpath(path, ROOT)
    return path if rel.startswith("..") else rel


class MissingStats(Exception):
    """--stats named a file that is not there. Silence here is dangerous: the
    tool would carry on with no keep-set, write no TRACE_STATS, and the
    launcher would fall back to whichever results/trace-* directory sorts last
    -- which may carry no topics at all and fail three minutes into a load."""


def find_stats(explicit: str | None, source: str = B.SOURCE_DEFAULT) -> str | None:
    if explicit:
        if not os.path.exists(explicit):
            raise MissingStats(explicit)
        return explicit
    prof = os.environ.get("EXPERT_PROFILE")
    if prof:
        p = os.path.join(ROOT, "results/keepsets", prof, "coverage.json")
        if os.path.exists(p):
            return p
    cands = sorted(glob.glob(os.path.join(ROOT, "results/keepsets/*/coverage.json")))
    cands += sorted(glob.glob(os.path.join(ROOT, "results/trace-*/stats/coverage.json")))
    best, best_n = None, -1
    # the one that carries the most topics IN THE FAMILY that will rank them: a
    # file traced before the expert-output norms has every counts_* and no
    # saliency_*, and picking it under --source saliency leaves no topics at all
    for c in cands:
        n = len(B.topic_names(c, source))
        if n > best_n:
            best, best_n = c, n
    return best


# --- profiles a user writes -------------------------------------------------
# PROFILES above is what ships. Two JSON files extend it, so a selection that
# turned out well can be kept without editing Python:
#
#   results/keepsets/profiles.json                            with the keep-sets
#   $XDG_CONFIG_HOME/deepseek-v41-flash-spark/profiles.json   this user's own
#
# Both are optional, both are read, the user's file last, so a profile in it
# replaces one of the same name from the checkout file or from the list above.
# `s` on the topic screen and --save-profile write to the user's file: a
# profile kept there survives a re-clone and leaves the working tree clean.
#
# A user profile is a name, one line of description and a list of topic names.
# It is never gated. The gate is a generation run on a keep-set, not a property
# of a name and a list, so a profile from a file reads as untested exactly as
# the shipped ones do until somebody runs one. It names no ranking rule either,
# and applying one therefore leaves the rule alone rather than resetting it:
# whatever --rank or DSV41_PRUNE_RANK asked for is what it is budgeted with.

PROFILES_BASENAME = "profiles.json"
CONFIG_DIRNAME = "deepseek-v41-flash-spark"
BUILT_IN = "built-in"
BRIEF_FILENAME = "tune-brief.md"


class ProfileError(Exception):
    """A profiles file that cannot be read, or one a save refuses to write
    over. Reading is never fatal: a broken file costs the profiles in it, not
    the tool, which still starts on the built-in ones."""


def user_profiles_path() -> str:
    """The user's own file. Outside the checkout on purpose: it is a
    preference, not a measurement, and `results/` is tracked."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, CONFIG_DIRNAME, PROFILES_BASENAME)


def profiles_files(explicit: str | None = None) -> list:
    """Read order, lowest precedence first. An explicit path replaces both
    defaults, including as the file a save is written to."""
    if explicit:
        return [os.path.expanduser(explicit)]
    return [os.path.join(ROOT, "results", "keepsets", PROFILES_BASENAME), user_profiles_path()]


def _entries(raw, where: str) -> list:
    """The list of profiles out of a parsed file, in either accepted shape."""
    if isinstance(raw, dict):
        raw = raw.get("profiles")
    if not isinstance(raw, list):
        raise ProfileError(f'{where}: expected {{"profiles": [...]}}, or a list of profiles')
    return raw


def read_profiles(path: str) -> tuple:
    """(profiles, problems) from one file. Every problem names the file and the
    profile it is in, and the rest of the file is still used: one bad entry
    should not cost the others."""
    if not os.path.exists(path):
        return [], []
    where = short_path(path)
    try:
        raw = json.load(open(path))
    except ValueError as e:
        return [], [f"{where}: not valid JSON ({e})"]
    except OSError as e:
        return [], [f"{where}: {e.strerror or e}"]
    try:
        entries = _entries(raw, where)
    except ProfileError as e:
        return [], [str(e)]
    out, problems = [], []
    for i, e in enumerate(entries, start=1):
        if not isinstance(e, dict):
            problems.append(f"{where}: profile {i} is not an object")
            continue
        name, desc, topics = e.get("name"), e.get("description", ""), e.get("topics")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"{where}: profile {i} has no name")
            continue
        label = f"{where}: {name.strip()!r}"
        if not isinstance(desc, str):
            problems.append(f"{label}: description is not a line of text")
            continue
        if not isinstance(topics, list) or not topics or not all(
                isinstance(t, str) and t.strip() for t in topics):
            problems.append(f"{label}: topics must be a non-empty list of topic names")
            continue
        # A repeated topic would get two votes in the per-layer sum, which is
        # not what writing it twice means. First mention wins the order.
        seen, want = set(), []
        for t in (t.strip() for t in topics):
            if t not in seen:
                seen.add(t)
                want.append(t)
        out.append((" ".join(name.split()), " ".join(desc.split()), want, False, where, None))
    return out, problems


def load_profiles(paths) -> tuple:
    """Every user profile the given files carry, and everything wrong with them."""
    profiles, problems = [], []
    for p in paths:
        got, bad = read_profiles(p)
        profiles += got
        problems += bad
    return profiles, problems


def merge_profiles(built_in, user) -> list:
    """The shipped profiles in their own order, with a user profile of the same
    name replacing one in place rather than appearing twice below it.

    (name, description, topics, gated, source, rank) either way -- the rank last
    so that a shipped profile and one from a file are read the same way, with
    None for "this one does not ask for a ranking rule"."""
    out = [(n, b, t, g, BUILT_IN, r) for n, b, t, g, r in built_in]
    at = {n.lower(): i for i, n in enumerate(x[0] for x in out)}
    for pr in user:
        i = at.get(pr[0].lower())
        if i is None:
            at[pr[0].lower()] = len(out)
            out.append(pr)
        else:
            out[i] = pr
    return out


def unknown_topics(user, index) -> list:
    """Topic names in a user profile that this keep-set does not carry. They
    cannot be selected either way; saying which ones is the whole point, because
    a profile silently short two topics still looks like it applied."""
    have = set(index.topics) if index else set()
    out = []
    for name, _blurb, topics, _gated, source, _rank in user:
        miss = [t for t in topics if t not in have]
        if miss:
            out.append(f"{source}: {name!r} names {len(miss)} topic"
                       f"{'s' if len(miss) > 1 else ''} this keep-set does not carry: "
                       f"{', '.join(miss)}")
    return out


def save_profile(path: str, name: str, description: str, topics) -> str:
    """Add or replace one profile in a user profiles file, keeping the rest of
    it. Refuses to write over a file it could not read: overwriting takes the
    profiles already in it with no way back, and a save is not worth that."""
    entries = []
    if os.path.exists(path):
        where = short_path(path)
        try:
            entries = _entries(json.load(open(path)), where)
        except ValueError as e:
            raise ProfileError(f"{where} is not valid JSON ({e}); fix or move it, then save again")
        except OSError as e:
            raise ProfileError(f"{where}: {e.strerror or e}")
    name = " ".join(name.split())
    rec = {"name": name, "description": " ".join(description.split()), "topics": sorted(topics)}
    for i, e in enumerate(entries):
        # compared the way the loader will read it, so two entries cannot end up
        # displaying the same name with only one of them reachable
        if isinstance(e, dict) and " ".join(str(e.get("name", "")).split()).lower() == name.lower():
            entries[i] = rec
            break
    else:
        entries.append(rec)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"profiles": entries}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def describe_selection(topics) -> str:
    """The line a saved profile gets when none was given. The topic names say
    more than a sentence about them would."""
    s = ", ".join(sorted(topics))
    return s if len(s) <= 76 else s[:73].rstrip(", ") + "..."


# --- state ------------------------------------------------------------------

class State:
    def __init__(self, host, index, stats_path, keep, max_seq, fmt, selection,
                 transient_slots=B.TRANSIENT_SLOTS_DEFAULT, keep_free_gb=B.KEEP_FREE_GB_DEFAULT,
                 user_profiles=(), profiles_path=None, rank=B.RANK_DEFAULT, source=None):
        self.host, self.index, self.stats_path = host, index, stats_path
        self.keep, self.max_seq, self.fmt = keep, max_seq, fmt
        # How the selected topics are combined into one ranking (DSV41_PRUNE_RANK).
        # Every coverage number on this screen is read off a keep-set built with
        # it, and the engine will build the keep-set with it too, or the bars
        # describe a server nobody is going to run.
        self.rank = rank
        # WHICH histograms are being ranked (DSV41_PRUNE_SOURCE): routing
        # frequency, or REAP saliency. It is a property of the loaded index --
        # one index reads one family -- and mirrored here so that the screen can
        # say it and .env can be written with it. Taken from the index when it
        # has one, so the two can never disagree.
        self.source = source or getattr(index, "source", None) or B.SOURCE_DEFAULT
        # The arena has to hold the kept set PLUS the transient ring, and the
        # engine's own default ring is 400 slots, not 8. Sizing against one
        # value and running with the other is how a keep-set that reads nothing
        # from NVMe quietly starts reading 392 experts a step.
        self.transient_slots, self.keep_free_gb = transient_slots, keep_free_gb
        self.sel = set(selection)
        self.cursor, self.scroll, self.filter, self.pane = 0, 0, "", 0
        # "easy" names what the box should be good at; "advanced" is the topic
        # list with every number on it. Easy is the default because the
        # advanced screen asks you to know which experts a job implies.
        self.view = "easy"
        self.pcursor, self.pscroll = 0, 0
        self._profiles = None
        # Profiles read from a file, and where a saved one goes. Passed in
        # rather than read here, so that drawing a screen in a test never
        # depends on what is in the config directory of the box it runs on.
        self.user_profiles = list(user_profiles)
        self.profiles_path = profiles_path or user_profiles_path()
        self.typing = False
        self.asking = None          # a one-line prompt: {"label", "buf"}
        self.msg = ""
        self.problem = ""           # a profiles file that could not be read

    @property
    def visible(self):
        ts = self.index.topics if self.index else []
        f = self.filter.lower()
        return [t for t in ts if f in t.lower()] if f else list(ts)

    def profile_defs(self) -> list:
        """The shipped profiles with the user's own merged over them by name."""
        return merge_profiles(PROFILES, self.user_profiles)

    def add_profile(self, name: str, blurb: str, topics) -> None:
        """Take a just-saved profile without re-reading the file, so the screen
        shows it, with its budget, on the next frame."""
        self.user_profiles = [p for p in self.user_profiles if p[0].lower() != name.lower()]
        self.user_profiles.append((name, blurb, sorted(topics), False,
                                   short_path(self.profiles_path), None))
        self._profiles = None

    def profiles(self):
        """Each profile with the budget it needs, computed once. A profile fixes
        the topics; the keep fraction comes from the coverage file in use."""
        if self._profiles is not None:
            return self._profiles
        out = []
        have = set(self.index.topics) if self.index else set()

        def fits(k):
            return B.plan(self.host, None, (), k, self.max_seq, fmt=self.fmt,
                          transient_slots=self.transient_slots,
                          keep_free_gb=self.keep_free_gb).verdict != "over"


        # The largest STEP that fits, not the continuous ceiling: keep_n rounds
        # the per-layer count up, so a plan at the continuous maximum is already
        # over it. Walk down until one actually fits.
        ceiling = next((k for k in reversed(KEEP_STEPS) if fits(k)), KEEP_STEPS[0])

        for name, blurb, topics, gated, source, rank in self.profile_defs():
            want = list(self.index.topics) if (topics is None and self.index) else (topics or [])
            avail = [t for t in want if t in have]
            missing = [t for t in want if t not in have]
            rank = rank or self.rank
            need = (self.index.keep_for(tuple(sorted(avail)), COVERAGE_TARGET, rank=rank)
                    if avail else None)
            keep = ceiling
            if need is not None:
                # smallest step that reaches the target, then clamp to what fits
                want_step = next((k for k in KEEP_STEPS if k >= need), KEEP_STEPS[-1])
                keep = min(want_step, ceiling)
            capped = need is None or need > keep + 1e-9
            p = B.plan(self.host, self.index, tuple(sorted(avail)), keep, self.max_seq, fmt=self.fmt,
                       transient_slots=self.transient_slots, keep_free_gb=self.keep_free_gb,
                       rank=rank)
            # What differs between profiles is not whether they load -- most of
            # them land on the same ceiling -- but how well the budget covers
            # the weakest topic in the bundle. Say that, in words.
            worst = min((p.coverage.get(t, 0.0) for t in avail), default=0.0)
            # The bands are the sum rule's (see COVERAGE_TARGET): under maxmin a
            # bundle reads flatter and lower at the top, so a profile that said
            # "good" on sum's strongest topic can say "uneven" on the same
            # keep-set. Left as they are until a generation gate is run on it.
            if not avail:
                status, tone = "not in this keep-set", "bad"
            elif p.verdict == "over":
                status, tone = "needs a bigger box", "bad"
            elif worst >= COVERAGE_TARGET:
                status, tone = "serves all of it", "good"
            elif worst >= 0.75:
                status, tone = "good", "good"
            elif worst >= 0.65:
                status, tone = "uneven", "warn"
            else:
                status, tone = "spread thin", "bad"
            out_worst = worst
            if not gated and avail:
                status, tone = status + " · untested", "warn"
            out.append({"name": name, "blurb": blurb, "topics": avail, "missing": missing,
                        "keep": keep, "plan": p, "status": status, "tone": tone,
                        "capped": capped, "worst": out_worst, "gated": gated, "rank": rank,
                        "source": source, "mine": source != BUILT_IN})
        self._profiles = out
        return out

    def set_context(self, seq):
        if seq != self.max_seq:
            self.max_seq, self._profiles = seq, None   # every profile's budget moves with it

    def set_rank(self, rank):
        if rank != self.rank:
            self.rank, self._profiles = rank, None     # and so does every profile's coverage

    def apply_profile(self, pr):
        self.sel = set(pr["topics"])
        self.keep = pr["keep"]
        self.set_rank(pr["rank"])

    def plan(self):
        return B.plan(self.host, self.index, tuple(sorted(self.sel)), self.keep,
                      self.max_seq, fmt=self.fmt, transient_slots=self.transient_slots,
                      keep_free_gb=self.keep_free_gb, rank=self.rank)

    def curves(self):
        """Coverage under the CURRENT selection. With nothing selected the
        engine ranks on every topic, so that is what is shown."""
        if not self.index:
            return {}
        sel = tuple(sorted(self.sel)) if self.sel else tuple(self.index.topics)
        got = self.index.curves(sel, rank=self.rank)
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


def draw_easy(w, st: State):
    """Name the job, not the experts. Each row is a bundle of topics with the
    budget it needs on this box, computed from the coverage file in use."""
    h, W = w.getmaxyx()
    title = " DeepSeek-V4.1-Flash "
    put(w, 0, 0, title, C["bright"] | curses.A_REVERSE | curses.A_BOLD)
    hostline = f"{st.host.name[:28]} · {st.host.total_gb:.1f} GB · {st.host.available_gb:.1f} free"
    put(w, 0, max(len(title) + 2, W - len(hostline) - 1), hostline, C["muted"])
    if st.host.busy:
        put(w, 1, 0, f" already running here: {st.host.busy} — this box holds one at a time",
            C["bad"] | curses.A_BOLD)
    elif st.problem:
        put(w, 1, 0, " profiles: " + st.problem, C["bad"], maxw=W - 2)
    elif st.host.note:
        put(w, 1, 0, " " + st.host.note, C["warn"])

    put(w, 2, 1, sp("WHAT SHOULD THIS BOX BE GOOD AT?"), C["accent"] | curses.A_BOLD)
    n_top = len(st.index.topics) if st.index else 0
    ctx = f"{st.max_seq // 1024}k" if st.max_seq >= 1024 else str(st.max_seq)
    put(w, 3, 1, f"{n_top} topics · context ◂ {ctx} ▸ · v switches to the topic-by-topic view",
        C["muted"])
    put(w, 4, 1, "─" * (W - 2), C["muted"])

    profs = st.profiles()
    top, per = 5, 3
    rows = max(1, (h - top - 4) // per)
    if st.pcursor < st.pscroll:
        st.pscroll = st.pcursor
    if st.pcursor >= st.pscroll + rows:
        st.pscroll = st.pcursor - rows + 1
    tone = {"good": C["good"], "warn": C["warn"], "bad": C["bad"]}

    for i, pr in enumerate(profs[st.pscroll:st.pscroll + rows]):
        y = top + i * per
        here = st.pscroll + i == st.pcursor
        put(w, y, 0, "▌" if here else " ", C["accent"] | curses.A_BOLD)
        st_txt = pr["status"]
        put(w, y, 3, pr["name"][:max(10, W - len(st_txt) - 6)],
            (C["bright"] | curses.A_BOLD) if here else C["bright"])
        # where a profile came from, since a file can replace a shipped one and
        # the two would otherwise be indistinguishable
        if pr["mine"] and 3 + len(pr["name"]) + 10 < W - len(st_txt) - 2:
            put(w, y, 4 + len(pr["name"]), "· yours", C["muted"])
        put(w, y, max(3, W - len(st_txt) - 2), st_txt, tone[pr["tone"]] | (curses.A_BOLD if here else 0))
        # The profile names its own ranking rule; the histogram family it ranks is the screen's,
        # so both are here -- the pair is what decides which experts the keep fraction holds.
        cost = (f"{pr['keep'] * 100:.0f} % of experts · {pr['rank']} · {st.source} · "
                f"{st.max_seq // 1024}k context" if pr["topics"] else "")
        put(w, y + 1, 3, pr["blurb"][:max(10, W - len(cost) - 6)], C["muted"])
        if cost:
            put(w, y + 1, max(3, W - len(cost) - 2), cost, C["muted"])
        # A profile that names topics this keep-set does not carry still applies
        # -- with the rest. Which ones went missing has to be on the screen, or
        # a selection two topics short looks exactly like one that applied.
        if pr["topics"] and pr["missing"]:
            miss = f"not in this keep-set: {', '.join(pr['missing'])}"
            put(w, y + 2, 3, miss, C["warn"], maxw=max(10, W - 5))
    if len(profs) > rows:
        below = len(profs) - rows - st.pscroll
        if below > 0:
            put(w, 3, W - 14, f"{below} more below", C["muted"])

    cur = profs[st.pcursor] if profs else None
    if cur and cur["topics"]:
        p = cur["plan"]
        weakest = min(cur["topics"], key=lambda t: p.coverage.get(t, 0.0)) if cur["topics"] else None
        line = (f"{cur['name']} · {len(cur['topics'])} topics · {p.kept:,} of {B.N_ROUTED:,} experts "
                f"in memory · {p.arena:.0f} GB")
        if p.verdict == "tight":
            line += " · tight, little spare memory"
        put(w, h - 3, 1, line[:W - 2], C["muted"])
        if weakest and W >= 88:
            note = f"weakest of them: {weakest} at {p.coverage.get(weakest, 0):.2f} coverage"
            put(w, h - 2, 1, note[:W - 2], C["muted"])
    if st.msg:
        # the answer to the last keypress, and the only place it appears
        put(w, h - 1, 1, st.msg.ljust(W - 2)[:W - 2], C["warn"] | curses.A_BOLD)
    else:
        for keys in (
            "↑↓ choose · ←→ context · enter apply and inspect · v topic view · b brief · w write · "
            "r RUN · q quit",
            "↑↓ choose · ←→ context · enter inspect · v topics · b brief · w write · r RUN · q quit",
            "↑↓ choose · ←→ context · enter inspect · v topics · b brief · r RUN · q quit",
            "↑↓ ←→ · enter inspect · v topics · b brief · r RUN · q quit",
            "↑↓ ←→ · enter inspect · v topics · r RUN · q quit",
        ):
            if len(keys) <= W - 2:
                break
        put(w, h - 1, 1, keys[:W - 2], C["muted"])
    w.noutrefresh()
    curses.doupdate()


def draw(w, st: State):
    w.erase()
    h, W = w.getmaxyx()
    if h >= MIN_H and W >= MIN_W and st.view == "easy":
        return draw_easy(w, st)
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
    elif st.problem:
        put(w, 1, 0, " profiles: " + st.problem, C["bad"], maxw=W - 2)
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
            # kept short enough to fit the pane at the narrowest supported width
            for i, line in enumerate([
                "Nothing to select: this keep-set has",
                "no per-topic histograms in it.",
                "",
                "To build one:",
                "  corpus/fetch_topics.py",
                "  corpus/make_corpus.py",
                "  tools/expert_trace.py",
                "  tools/expert_stats.py",
                "",
                "docs/keep-sets.md walks the whole path.",
                "",
                "Without topics the engine ranks on all",
                "of them, which is what the shipped",
                "profiles do. The budget is live anyway.",
            ]):
                put(w, list_top + i, 3, line, C["muted"], maxw=split - 3)
    for i, t in enumerate(vis[st.scroll:st.scroll + list_h]):
        row = list_top + i
        idx = st.scroll + i
        on = t in st.sel
        here = idx == st.cursor and focus
        put(w, row, 0, "▌" if here else " ", C["accent"] | curses.A_BOLD)
        put(w, row, 2, "●" if on else "○", (C["on"] if on else C["muted"]) | (curses.A_BOLD if on else 0))
        nm = t[:16]
        put(w, row, 4, nm.ljust(17), (C["bright"] | curses.A_BOLD) if on else C["muted"])
        n = (st.index.tokens.get(t, 0) if st.index else 0)
        thin = n < B.TopicIndex.THIN
        cv = curves.get(t)
        if cv is None:
            put(w, row, 22, "·" * bw + "    —", C["muted"])
        else:
            v = cv[B.keep_n(st.keep)]
            col = C["good"] if v >= COVERAGE_TARGET else (C["warn"] if v >= 0.7 else C["bad"])
            if thin:
                col = C["muted"]
            put(w, row, 22, bar(v, bw, solid=not thin), col if on else C["muted"])
            put(w, row, 22 + bw + 1, f"{v:.2f}", (col | curses.A_BOLD) if on else C["muted"])
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
    # the same threshold the verdict uses, or this row reads green under a red badge
    row(ry + 9, "free after load", f"{p.free_after_load:.1f} GB",
        C["good"] if p.free_after_load >= p.prefill else C["bad"])
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
    # Which experts those are is the ranking rule's doing as much as the
    # fraction's, and the two rules put the same budget in different places, so
    # the rule belongs where the number it qualifies is. The histogram family it
    # ranks belongs there for the same reason: frequency and saliency order the
    # same layer differently, and the coverage bars above are read off whichever
    # one is named here.
    # A truncated `rank maxmin · sal` says less than `maxmin · saliency` does, so the longest
    # label that fits whole wins rather than the longest one clipped.
    room = split - 34
    rule = next((s for s in (f"rank {st.rank} · {st.source}", f"{st.rank} · {st.source}",
                             f"rank {st.rank}", st.rank) if len(s) <= room), st.rank)
    put(w, sy + 1, max(34, split - len(rule) - 1), rule, C["muted"], maxw=room)
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
    thin_sel = [t for t in st.sel if st.index and st.index.tokens.get(t, 0) < B.TopicIndex.THIN]
    t, v = p.weakest
    if thin_sel and not st.msg:
        n = len(thin_sel)
        long_ = (f"{n} selected topic{'s' if n > 1 else ''} traced on too little text — those bars "
                 f"read high because the sample chose the experts")
        short = f"{n} selected topic{'s' if n > 1 else ''} traced on too little text — bars read high"
        put(w, fy, 1, long_ if len(long_) <= W - 3 else short, C["bad"])
    elif t:
        col = C["good"] if v >= COVERAGE_TARGET else (C["warn"] if v >= 0.7 else C["bad"])
        put(w, fy, 1, "weakest selected topic  ", C["muted"])
        put(w, fy, 25, f"{t} {v:.2f}", col | curses.A_BOLD)
        need = (st.index.keep_for(tuple(sorted(st.sel)), COVERAGE_TARGET, rank=st.rank)
                if st.index else None)
        mk = p.max_keep()
        if need and need > mk:
            # the target is out of this box's reach: say how much of the
            # selection it CAN serve at the largest keep that fits
            curves = st.curves()
            names = sorted(st.sel) if st.sel else list(curves)
            n = B.keep_n(min(mk, 1.0))
            ok = sum(1 for t in names if curves.get(t, [0] * 385)[n] >= COVERAGE_TARGET)
            rec = f"at the {mk:.0%} this box holds, {ok} of {len(names)} reach {COVERAGE_TARGET:.2f}"
            put(w, fy, min(45, W - len(rec) - 2), rec, C["warn"])
        elif need and abs(need - st.keep) > 0.005:
            verb = "raise to" if need > st.keep else "enough at"
            rec = (f"{verb} {need * 100:.0f} % for {COVERAGE_TARGET:.2f} on every one"
                   if W >= 96 else f"{verb} {need * 100:.0f} %")
            put(w, fy, min(45, W - len(rec) - 2), rec, C["muted"])
    elif st.index and st.index.topics:
        put(w, fy, 1, "no topic selected — the keep-set would use all of them", C["muted"])
    if st.asking:
        put(w, h - 1, 1, f"{st.asking['label']} {st.asking['buf']}_".ljust(W - 2)[:W - 2],
            C["on"] | curses.A_BOLD)
    elif st.msg:
        # transient, and worth the key line for one keypress
        put(w, h - 1, 1, st.msg.ljust(W - 2)[:W - 2], C["warn"] | curses.A_BOLD)
    else:
        for keys in (
            "↑↓ topic  space select  ←→ adjust  tab pane  a all  n none  / filter  "
            "m fit  f format  s save  v profiles  w write  r RUN  q quit",
            "↑↓ space ←→ tab · a all · n none · / filter · m fit · f format · s save · v profiles · r RUN · q quit",
            "↑↓ space ←→ tab · / filter · m fit · s save · v profiles · r RUN · q quit",
            "↑↓ space ←→ tab · / filter · m fit · v profiles · r RUN · q quit",
            "space ←→ tab · v profiles · r RUN · q quit",
        ):
            if len(keys) <= W - 2:
                break
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


def save_current(st: State, name: str) -> str:
    """`s` on the topic screen: keep this selection under a name. Returns the
    line for the footer, because that is the only place the screen can answer."""
    name = name.strip()
    if not name:
        return "a profile needs a name"
    if not st.sel:
        return "nothing selected, so there is nothing to save"
    blurb = describe_selection(st.sel)
    try:
        where = save_profile(st.profiles_path, name, blurb, st.sel)
    except ProfileError as e:
        return str(e)
    st.add_profile(name, blurb, st.sel)
    return f"saved {name!r} to {short_path(where)}; v shows it on the profile screen"


def write_brief(st: State) -> str:
    """`b` on the profile screen. The brief is long and the screen is not the
    place to read it, so it goes to a file and the footer says which."""
    path = os.path.join(ROOT, BRIEF_FILENAME)
    try:
        with open(path, "w") as f:
            f.write(brief(st))
    except OSError as e:
        return f"could not write {short_path(path)}: {e.strerror or e}"
    return f"wrote the topic brief to {short_path(path)}"


def loop(w, st: State) -> str | None:
    curses.curs_set(0)
    try:
        curses.set_escdelay(25)
    except Exception:  # noqa: BLE001
        pass
    init_colors()
    w.keypad(True)
    # MemAvailable moves while the screen is open -- another process starts, the
    # kernel reclaims the last engine's arena -- so the budget has to be checked
    # against what is free now, not at startup. Wake once a second to re-read it.
    w.timeout(1000)
    while True:
        B.refresh(st.host)
        draw(w, st)
        try:
            k = w.getch()
        except KeyboardInterrupt:
            return None
        if k == -1:          # the once-a-second wake-up: just redraw
            continue
        st.msg = ""   # a message lasts until the next key
        vis = st.visible
        st.cursor = max(0, min(st.cursor, len(vis) - 1)) if vis else 0

        if st.asking:
            if k in (27,):                      # esc
                st.asking, st.msg = None, "not saved"
            elif k in (10, 13, curses.KEY_ENTER):
                ask, st.asking = st.asking, None
                st.msg = save_current(st, ask["buf"])
            elif k in (curses.KEY_BACKSPACE, 127, 8):
                st.asking["buf"] = st.asking["buf"][:-1]
            elif 32 <= k < 127:
                st.asking["buf"] += chr(k)
            continue

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
        if k in (ord("v"), ord("V")):
            st.view = "advanced" if st.view == "easy" else "easy"
            continue

        if st.view == "easy":
            profs = st.profiles()
            if k in (curses.KEY_DOWN, ord("j")):
                st.pcursor = min(len(profs) - 1, st.pcursor + 1)
            elif k in (curses.KEY_UP, ord("k")):
                st.pcursor = max(0, st.pcursor - 1)
            elif k in (curses.KEY_RIGHT, curses.KEY_LEFT):
                st.set_context(step(CTX_STEPS, st.max_seq, 1 if k == curses.KEY_RIGHT else -1))
            elif k in (10, 13, curses.KEY_ENTER, ord(" ")):
                if profs[st.pcursor]["topics"]:
                    st.apply_profile(profs[st.pcursor])
                    st.view = "advanced"      # show what it did, so it can be adjusted
                else:
                    st.msg = "this keep-set does not carry those topics"
            elif k in (ord("r"), ord("R")):
                pr = profs[st.pcursor]
                if not pr["topics"]:
                    st.msg = "this keep-set does not carry those topics"
                    continue
                st.apply_profile(pr)
                if st.host.busy:
                    st.msg = "something is already running — ./stop.sh first"
                    continue
                if st.plan().verdict == "over":
                    st.msg = "that will not load on this box"
                    continue
                return "run"
            elif k in (ord("w"), ord("W")):
                if profs[st.pcursor]["topics"]:
                    st.apply_profile(profs[st.pcursor])
                    return "write"
            elif k in (ord("b"), ord("B")):
                st.msg = write_brief(st)
            continue
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
        elif k in (ord("s"), ord("S")):         # keep this selection as a profile
            if not st.sel:
                st.msg = "select the topics first, then s keeps them as a profile"
            else:
                st.asking = {"label": f"save these {len(st.sel)} topics as:", "buf": ""}
        elif k == ord("m"):                     # snap to the coverage target
            need = (st.index.keep_for(tuple(sorted(st.sel)), COVERAGE_TARGET, rank=st.rank)
                    if st.index else None)
            if need:
                st.keep = step(KEEP_STEPS, need, 0)
                if st.keep < need:
                    st.keep = step(KEEP_STEPS, st.keep, 1)
                what = "every selected topic" if st.sel else "every topic in this keep-set"
                if st.plan().verdict == "over":
                    st.msg = f"{COVERAGE_TARGET:.2f} on {what} needs {need:.0%}, which this box cannot hold"
            elif st.index and st.index.topics:
                st.msg = f"nothing reaches {COVERAGE_TARGET:.2f} on all of them, even at 100 %"
            else:
                st.msg = "this keep-set carries no per-topic histogram to fit to"
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


class _Grid:
    """A window-shaped object that collects characters instead of drawing them,
    so the screen can be dumped without a terminal -- for `--render`, and for
    tools/test_tune_draw.py."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.g = [[" "] * w for _ in range(h)]

    def getmaxyx(self):
        return self.h, self.w

    def erase(self):
        self.g = [[" "] * self.w for _ in range(self.h)]

    def noutrefresh(self):
        pass

    def addstr(self, y, x, t, attr=0):
        if not (0 <= y < self.h) or x < 0 or x + len(t) > self.w:
            raise curses.error("out of bounds")
        for i, c in enumerate(t):
            self.g[y][x + i] = c

    def row(self, y):
        return "".join(self.g[y]).rstrip()


def render(st: State, h: int, w: int) -> str:
    """The screen as text. Used to keep docs/tune.md honest."""
    for k in ("accent", "good", "warn", "bad", "muted", "bright", "on"):
        C.setdefault(k, 0)
    save, curses.doupdate = curses.doupdate, lambda: None
    try:
        grid = _Grid(h, w)
        draw(grid, st)
    finally:
        curses.doupdate = save
    rows = [grid.row(y) for y in range(h)]
    while rows and not rows[-1].strip():
        rows.pop()
    return "\n".join(rows)


# --- the task brief ---------------------------------------------------------
# Coverage can only see a gap in the SELECTION. A gap in the CATALOGUE has no
# histogram, so it has no bar and no warning, and the tool cannot flag it. What
# the tool can do is hand somebody the whole task of closing one: what this
# keep-set carries, where the catalogue is thin, the commands with this
# checkout's own paths, and the two things that are easy to get wrong, which
# are how much text a topic needs and what a new topic costs the ones already
# there. All of it is read off the file that is loaded, so it stays true as the
# keep-set behind it changes.

BRIEF_TOPIC = "reasoning"        # the worked example, and the gap that prompted this


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _spread(names, k):
    """k names taken evenly across a list, so a sample is not just its head."""
    names = list(names)
    if k >= len(names) or k < 2:
        return names[:max(1, k)]
    return [names[round(i * (len(names) - 1) / (k - 1))] for i in range(k)]


def catalogue() -> dict:
    """The catalogue corpus/fetch_topics.py knows, read out of that script so
    this cannot drift from what the script will actually fetch."""
    path = os.path.join(ROOT, "corpus", "fetch_topics.py")
    try:
        spec = importlib.util.spec_from_file_location("_fetch_topics", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {"code": sorted(mod.CODE), "lang": sorted(mod.LANGS), "domain": sorted(mod.DOMAINS)}
    except Exception:  # noqa: BLE001
        return {}


def _pct(x) -> str:
    return "more than 100 %" if x is None else f"{x:.0%}"


def brief(st: State) -> str:
    """The task of adding a topic, written out of the loaded keep-set."""
    idx, keep = st.index, st.keep
    n = B.keep_n(keep)
    have = list(idx.topics) if idx else []
    sel = sorted(st.sel) if st.sel else list(have)
    cur = idx.curves(tuple(have), only=tuple(have), rank=st.rank)[0] if have else {}
    L = []

    def line(s=""):
        L.append(s)

    def para(s, indent=""):
        # break_on_hyphens off, or a path like results/trace-mine/ is split
        L.extend(textwrap.wrap(" ".join(s.split()), 96, break_long_words=False,
                               break_on_hyphens=False, subsequent_indent=indent))
        L.append("")

    line("# Task: add a topic to this keep-set")
    line()
    para(f"""
        Written by `./tune.sh --brief` from `{short_path(st.stats_path)}`, which carries
        {len(have)} topic{'' if len(have) == 1 else 's'}, at keep {keep:.0%} in the
        `{st.fmt}` layout, ranked by `{st.rank}` on the `{st.source}` histograms. Every figure below is
        computed from that file at the moment the brief was written, so re-run the command after the
        keep-set changes.""")

    line("## Why this is a task at all")
    line()
    para("""
        Coverage is the fraction of a topic's measured routing that lands on an expert the
        budget keeps resident. It can therefore see a gap in the SELECTION and never a gap in
        the CATALOGUE. A register nothing was ever traced on has no histogram, so it has no
        bar, no number and no warning, and a profile whose every topic scores well can still
        fail on it.""")
    para("""
        That is not hypothetical. A five-topic profile here scored 0.85 or better on every
        topic it had and still reasoned in circles on a two-train arithmetic question, because
        nothing in the catalogue carries the register a reasoning trace is written in. The
        screen warned about nothing, because there was nothing to warn about. Adding a topic is
        how a gap like that gets closed; there is no setting that fixes it.""")

    # --- what is already here ------------------------------------------------
    line("## What this keep-set already carries")
    line()
    if not have:
        para(f"""
            Nothing. `{short_path(st.stats_path)}` carries no per-topic histograms, only the
            mixed one, so `EXPERT_TOPICS` cannot be used with it and there is no topic to
            extend. The commands below build a keep-set that does carry them, which is the same
            work as adding a topic to one that does: tag every source with a topic name, and
            one trace yields one histogram per name.""")
    else:
        toks = [idx.tokens.get(t, 0) for t in have]
        thin = [t for t in have if idx.tokens.get(t, 0) < B.TopicIndex.THIN]
        para(f"""
            {len(have)} topics, traced on {min(toks):,} to {max(toks):,} tokens each, median
            {int(sorted(toks)[len(toks) // 2]):,}. Coverage is at keep {keep:.0%} with every
            topic selected, which is what the engine ranks on when `EXPERT_TOPICS` is unset.
            {len(thin) if thin else 'None'} of them {'is' if len(thin) == 1 else 'are'} below the
            {B.TopicIndex.THIN:,}-token line this tool calls thin{
            ': ' + ', '.join(thin) if thin else ''}.""")
        line(f"| topic | traced tokens | coverage at keep {keep:.0%} | note |")
        line("|---|---:|---:|---|")
        for t in have:
            nt = idx.tokens.get(t, 0)
            flag = "thin" if nt < B.TopicIndex.THIN else ""
            line(f"| `{t}` | {nt:,} | {cur[t][n]:.2f} | {flag} |")
        line()

    # --- where the catalogue is thin ----------------------------------------
    line("## Where the catalogue is thin")
    line()
    cat = catalogue()
    if not cat:
        para("""
            `corpus/fetch_topics.py` could not be read, so the catalogue could not be compared
            against this keep-set. Run `python3 corpus/fetch_topics.py --list` and compare by
            hand.""")
    else:
        here = set(have)
        para(f"""
            `corpus/fetch_topics.py` knows {len(cat['code'])} programming languages, taken from
            a source tree you point it at, {len(cat['lang'])} natural languages and
            {len(cat['domain'])} domain registers, both from Wikipedia. What this keep-set has of
            each group, and how well it was sampled:""")
        line("| catalogue group | in the catalogue | carried here | absent here | thinly traced |")
        line("|---|---:|---:|---:|---:|")
        for kind, names in cat.items():
            present = [t for t in names if t in here]
            absent = [t for t in names if t not in here]
            thin = [t for t in present if idx.tokens.get(t, 0) < B.TopicIndex.THIN]
            line(f"| {kind} | {len(names)} | {len(present)} | {len(absent)} | {len(thin)} |")
        line()
        for kind, names in cat.items():
            absent = [t for t in names if t not in here]
            if absent:
                para(f"* absent from this keep-set, {kind}: "
                     f"{', '.join('`%s`' % t for t in absent)}", indent="  ")
        extra = [t for t in have if not any(t in v for v in cat.values())]
        if extra:
            para(f"* carried here but not in the catalogue: "
                 f"{', '.join('`%s`' % t for t in extra)}", indent="  ")
        para(f"""
            What that table cannot show is a register the catalogue has no entry for at all.
            Its {len(cat.get('domain', []))} domain registers are
            {', '.join(cat.get('domain', []))}. None of them is step-by-step reasoning, none is
            dialogue or transcript, none is mathematics written out as prose, none is poetry or
            song, none is a patch or a diff. Those are candidates because nothing in the
            catalogue resembles them, which is a judgement about the list above rather than a
            number this tool can compute.""")

    # --- the commands --------------------------------------------------------
    line("## The commands")
    line()
    para(f"""
        Run from the root of the checkout. `$MODEL_DIR` is the checkpoint directory `.env`
        sets. The trace in step 5 is the only expensive step, at roughly a minute per layer;
        everything after it is arithmetic. The example topic is `{BRIEF_TOPIC}`, as prose;
        substitute your own name and `code` for source files.""")
    line("```bash")
    line("# 1. what the catalogue already has")
    line("python3 corpus/fetch_topics.py --list")
    line()
    line("# 2. gather the text. For a topic the catalogue has, this fetches it:")
    line("python3 corpus/fetch_topics.py --out topics --only german,rust --code-root ./sources")
    line("#    For one it does not have, write the file yourself: one plain-text file, one")
    line("#    topic, about 40,000 characters of the real register, at")
    line(f"#    topics/domain/{BRIEF_TOPIC}.txt")
    line()
    line("# 3. build the trace corpus. --tokenizer and --out are required; --target is tokens")
    line("#    per topic, and each --topic is NAME:KIND:PATH[,PATH...] with KIND code or prose.")
    line('python3 corpus/make_corpus.py --tokenizer "$MODEL_DIR" --target 3000 \\')
    line(f"    --topic {BRIEF_TOPIC}:prose:topics/domain/{BRIEF_TOPIC}.txt \\")
    line(f"    --out corpus/trace_{BRIEF_TOPIC}.jsonl")
    line()
    line("# 4. fetch the Engram rows this corpus touches. --model-dir, --corpus and --out are")
    line("#    required. The two n-gram tables are never downloaded whole.")
    line('python3 tools/engram_rows.py --model-dir "$MODEL_DIR" \\')
    line(f"    --corpus corpus/trace_{BRIEF_TOPIC}.jsonl --out engram_rows_{BRIEF_TOPIC}")
    line("#    add --local-shard \"$MODEL_DIR/model-00047-of-00048.safetensors\" and the same for")
    line("#    model-00048-of-00048 when those shards are already on disk")
    line()
    line("# 5. trace the routing, one layer shard at a time. --model-dir, --corpus,")
    line("#    --engram-dir and --out are required. --resume checkpoints after every layer.")
    line('python3 tools/expert_trace.py --model-dir "$MODEL_DIR" \\')
    line(f"    --corpus corpus/trace_{BRIEF_TOPIC}.jsonl --engram-dir engram_rows_{BRIEF_TOPIC} \\")
    line(f"    --out results/trace-{BRIEF_TOPIC} --layers 0-39 --resume")
    line()
    line("# 6. turn the trace into per-topic histograms. --trace and --out are required.")
    line(f"python3 tools/expert_stats.py --trace results/trace-{BRIEF_TOPIC} \\")
    line(f"    --out results/keepsets/{BRIEF_TOPIC}")
    line("```")
    line()
    para(f"""
        Step 6 writes `coverage.json` with one `counts_<topic>` histogram per category per
        layer, which is everything a keep-set needs. The raw per-layer trace arrays under
        `results/trace-{BRIEF_TOPIC}/trace/` are not needed afterwards and are not tracked.""")

    # --- how much text -------------------------------------------------------
    line("## How much text a new topic needs")
    line()
    picks = B.TOPK * B.N_LAYERS
    per_expert = lambda t: t * B.TOPK / B.N_EXPERTS
    para(f"""
        Aim for about 3,000 tokens, and treat 2,000 as a floor. The arithmetic is that every
        token contributes {B.TOPK} picks in each of the {B.N_LAYERS} layers, so a topic's
        histogram gets {picks} counts per token spread over {B.N_EXPERTS} experts per layer. At
        300 tokens that is a mean of {per_expert(300):.1f} counts per expert and a ranking that
        is mostly the difference between one and two; at {B.TopicIndex.THIN:,} it is
        {per_expert(B.TopicIndex.THIN):.0f}, which is where this tool stops drawing the bar
        hollow; at 3,000 it is {per_expert(3000):.0f}.""")
    para("""
        The reason to care is not noise. Coverage is measured on the very trace that chose the
        experts, so a topic seen for 300 tokens routes to whatever fired during those 300
        tokens, which are exactly the experts its own histogram ranked highest. The bias is
        upward and it is systematic: a thin topic scores as though it were well served.""")
    para("""
        That contamination has been measured here, on two traces of the same 35 topics
        (`results/keepsets/topics/GATE.md`). In the thin trace, where topics ran from 221 to
        2,778 tokens, the correlation between how much text a topic was traced on and its
        coverage is **-0.36**: the better-sampled topics scored LOWER, which is the artefact and
        not a property of those topics. Levelling the evidence at about 3,000 tokens each takes
        that correlation to **-0.00**, and coverage then measures the topic instead of the
        sample under it.""")
    if have and len(have) >= 3:
        r = _pearson([idx.tokens.get(t, 0) for t in have], [cur[t][n] for t in have])
        toks = [idx.tokens.get(t, 0) for t in have]
        if r is not None:
            para(f"""
                The same correlation on the file loaded here, over {len(have)} topics of
                {min(toks):,} to {max(toks):,} tokens, is **{r:+.2f}**. A new topic has to come
                in at the same weight as the ones already in the file, or it will not be
                comparable with them whichever way the number goes.""")

    # --- what it costs -------------------------------------------------------
    line("## What a new topic costs the topics already here")
    line()
    p = st.plan()
    # The sentence below names the ranking rule this brief was written under,
    # because the cost it goes on to measure is that rule's cost: `sum` pays for
    # a new topic out of every other topic's coverage, `maxmin` out of the
    # best-served one's.
    rank_says = {
        "sum": "every selected topic's per-layer histogram is normalised and summed, so each "
               "one gets an equal vote",
        "max": "every selected topic's per-layer histogram is normalised and the largest value "
               "wins the expert, so one topic wanting it is enough",
        "maxmin": "every selected topic's per-layer histogram is normalised and the layer's "
                  "slots go one at a time to whichever selected topic is least covered so far",
    }[st.rank]
    step_to = min(0.60, keep + 0.02)
    step_slots = (B.keep_n(step_to) - B.keep_n(keep)) * B.N_LAYERS
    step_gb = step_slots * B.EXPERT_BYTES[st.fmt] / B.GB
    para(f"""
        The budget is fixed and a new topic does not add to it. At keep {keep:.0%} this box
        holds {p.kept:,} of {B.N_ROUTED:,} routed experts, {B.keep_n(keep)} per layer, an arena
        of {p.arena:.1f} GB. Adding a topic does not add slots, it changes which experts fill
        them: {rank_says}, and one more claimant moves the ranking away from all the others.""")
    if len(sel) >= 2:
        got = idx.curves(tuple(sel), only=tuple(sel), rank=st.rank)[0]
        drops = _spread(sel, min(len(sel), 12))
        deltas, needs = [], []
        base_need = idx.keep_for(tuple(sel), COVERAGE_TARGET, rank=st.rank)
        for d in drops:
            rest = tuple(t for t in sel if t != d)
            cc = idx.curves(rest, only=rest, rank=st.rank)[0]
            deltas.append((d, _mean(cc[t][n] for t in rest) - _mean(got[t][n] for t in rest)))
            needs.append((d, idx.keep_for(rest, COVERAGE_TARGET, rank=st.rank)))
        worst = max(deltas, key=lambda x: x[1])
        avg = _mean(d for _t, d in deltas)
        best_need = min((x for x in needs if x[1] is not None), key=lambda x: x[1], default=None)
        para(f"""
            Measured on this file, with {len(sel)} topic{'' if len(sel) == 1 else 's'} selected
            at keep {keep:.0%}: dropping one of them and re-ranking raises the coverage of the
            other {len(sel) - 1} by **{avg:+.3f}** on average over the {len(drops)} drops
            tried, and by {worst[1]:+.3f} for the most expensive one, `{worst[0]}`. Read it in
            reverse, because that is the direction this task runs in: that is what one more
            topic costs each of the others once {len(sel)} are selected.""")
        if base_need is not None or best_need:
            no_topic = f"{_pct(best_need[1])} without `{best_need[0]}`" if best_need else "less without one of them"
            para(f"""
                The same cost in the currency that decides whether it loads: reaching
                {COVERAGE_TARGET:.2f} on every selected topic needs {_pct(base_need)} of the
                experts with all {len(sel)}, and {no_topic}. One 2-point step of the keep slider
                is {step_slots:,} more resident experts, {step_gb:.1f} GB of arena, and the
                arena is what the context window and the prefill chunk are competing with.""")
    if len(sel) >= 3:
        sample = _spread(sel, min(5, len(sel)))
        para(f"""How the cost accumulates, on {'' if len(sample) == len(sel) else 'a sample of '}
             this selection, each row adding one topic to the row above. The keep column is the
             smallest fraction at which every topic in that row reaches {COVERAGE_TARGET:.2f}:""")
        line(f"| topics selected | keep needed for {COVERAGE_TARGET:.2f} | arena | verdict |")
        line("|---|---:|---:|---|")
        for i in range(1, len(sample) + 1):
            sub = tuple(sorted(sample[:i]))
            need = idx.keep_for(sub, COVERAGE_TARGET, rank=st.rank)
            if need is None:
                line(f"| {', '.join('`%s`' % t for t in sub)} | more than 100 % | | |")
                continue
            q = B.plan(st.host, idx, sub, need, st.max_seq, fmt=st.fmt,
                       transient_slots=st.transient_slots, keep_free_gb=st.keep_free_gb,
                       rank=st.rank)
            verdict = {"ok": "fits", "tight": "tight", "over": "will not load"}[q.verdict]
            line(f"| {', '.join('`%s`' % t for t in sub)} | {need:.0%} | {q.arena:.0f} GB | {verdict} |")
        line()
        para("""
            Which is the whole trade. A topic that earns its place is one the traffic actually
            contains; a topic added in case it is needed is paid for by every other topic on
            every request.""")

    # --- no re-trace ---------------------------------------------------------
    line("## Nothing already traced has to be traced again")
    line()
    para("""
        A topic is one per-layer histogram and nothing in the ranking depends on the topics
        having been traced together: the engine normalises each topic's counts within its own
        layer before it combines them at all, whichever rule combines them
        (`engine/v41_engine.py`, and the same arithmetic in `tools/budget.py`). So a new topic is a separate, small trace of its own corpus, and
        the result is concatenated into the existing file by copying its `counts_<topic>` keys
        across.""")
    line("```python")
    line("import json, os")
    line()
    line(f'base = json.load(open("{short_path(st.stats_path) if st.stats_path else "results/keepsets/topics/coverage.json"}"))')
    line(f'add = json.load(open("results/keepsets/{BRIEF_TOPIC}/coverage.json"))')
    line('for layer, rows in add["per_layer"].items():')
    line('    for key, counts in rows.items():')
    line('        if key.startswith("counts_"):')
    line('            base["per_layer"][layer][key] = counts')
    line(f'os.makedirs("results/keepsets/{BRIEF_TOPIC}-merged", exist_ok=True)')
    line(f'json.dump(base, open("results/keepsets/{BRIEF_TOPIC}-merged/coverage.json", "w"))')
    line("```")
    line()
    para(f"""
        Then `./tune.sh --stats results/keepsets/{BRIEF_TOPIC}-merged/coverage.json --list`
        shows the new topic beside the old ones. Two details are worth knowing. The layer keys
        are strings in JSON, which is why the loop above does not convert them. And every other
        field in `per_layer` (`used`, `cov`, `top10`, `entropy_bits`, `block6_unique_mean`)
        still describes the base trace only, because it was not recomputed; nothing in the
        keep-set arithmetic reads them, but do not quote them for the merged file.""")

    # --- the gate ------------------------------------------------------------
    line("## Gate it before trusting it")
    line()
    para("""
        Coverage is a measurement. Sound generation is not, and the two come apart. The
        keep-set that could not write an HTML file measured BETTER on teacher-forced loss than
        the one that replaced it, and its markup coverage was 0.03 while its Python coverage
        was 0.51. A keep-set that never saw a domain does not get gradually worse in it, it
        produces structurally broken output.""")
    para(f"""
        So after the new topic is in, run free generation on it and on domains the corpus does
        not contain, at the keep fraction you intend to serve at, and write both results down
        next to the profile the way the shipped `GATE.md` files do. The measure used here is
        the distinct-token ratio of the generated text, where 0.15 or below is degenerate
        (`RESULTS.md`). Until that is done, a profile built on the new topic is untested, which
        is exactly what `./tune.sh` will call it: user profiles are never marked gated, because
        the gate is a generation run, not a name and a list of topics.""")
    para("""
        Keep the selection that passes as a profile, with `s` on the topic screen or with
        `--save-profile`, so that the next person gets the set rather than the search:""")
    line("```bash")
    line(f'./tune.sh --topics {",".join(sel[:3]) if sel else "a,b,c"} --save-profile "Name"')
    line("```")
    line()
    return "\n".join(L).rstrip() + "\n"


# --- output -----------------------------------------------------------------

MANAGED = ("EXPERT_TOPICS", "PRUNE_KEEP", "MAX_SEQ", "ARENA_GB", "TRACE_STATS", "EXPERT_FORMAT",
           "TRANSIENT_SLOTS", "KEEP_FREE_GB", "DSV41_PRUNE_RANK", "DSV41_PRUNE_SOURCE")


def env_for(st: State) -> dict:
    p = st.plan()
    e = {
        "PRUNE_KEEP": f"{st.keep:.2f}",
        # The coverage on the screen was read off a keep-set built with this
        # rule; written out so that the run reproduces the screen. The engine
        # reads it under its own name, straight out of the environment .env is
        # sourced into, which is why this one key is not the bare form.
        "DSV41_PRUNE_RANK": st.rank,
        # ... and off a keep-set built from THESE histograms. Same reasoning,
        # same bare engine-side name: a run that ranks frequency where the screen
        # ranked saliency keeps a different set of experts at the same budget.
        "DSV41_PRUNE_SOURCE": st.source,
        "MAX_SEQ": str(st.max_seq),
        "ARENA_GB": f"{math.ceil(p.arena)}",
        "EXPERT_FORMAT": st.fmt,
        # written because the arena above was sized against them
        "TRANSIENT_SLOTS": str(st.transient_slots),
        "KEEP_FREE_GB": f"{st.keep_free_gb:g}",
        "TRACE_STATS": short_path(st.stats_path) if st.stats_path else "",
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
    global COVERAGE_TARGET
    ap = argparse.ArgumentParser(description="choose what this box should be good at")
    ap.add_argument("--stats", help="coverage.json to read topics from")
    ap.add_argument("--topics", default=os.environ.get("EXPERT_TOPICS", ""))
    ap.add_argument("--keep", type=float, default=float(os.environ.get("PRUNE_KEEP", "0.39")))
    ap.add_argument("--max-seq", type=int, default=int(os.environ.get("MAX_SEQ", "32768")))
    ap.add_argument("--format", default=os.environ.get("EXPERT_FORMAT", "cb3"), choices=("cb3", "fp4"))
    ap.add_argument("--rank", default=B.rank_from_env(), metavar="RULE",
                    help="how several topics are combined into one ranking: "
                         f"{' | '.join(B.RANKS)} (default %(default)s, from DSV41_PRUNE_RANK). "
                         "A profile that names a rule overrides it.")
    ap.add_argument("--source", default=B.source_from_env(), metavar="WHICH",
                    help="which measurement ranks the experts: "
                         f"{' | '.join(B.SOURCES)} (default %(default)s, from DSV41_PRUNE_SOURCE). "
                         "counts is routing frequency; saliency is gate weight x expert-output "
                         "norm, and needs a coverage.json with saliency_<topic> histograms.")
    ap.add_argument("--transient-slots", type=int,
                    default=int(os.environ.get("TRANSIENT_SLOTS") or B.TRANSIENT_SLOTS_DEFAULT),
                    help="prefill slots outside the LRU; the arena is sized to hold these too")
    ap.add_argument("--keep-free-gb", type=float,
                    default=float(os.environ.get("KEEP_FREE_GB") or B.KEEP_FREE_GB_DEFAULT),
                    help="host memory the launcher leaves free")
    ap.add_argument("--coverage-target", type=float, default=COVERAGE_TARGET,
                    help="coverage every selected topic should reach (default %(default).2f)")
    ap.add_argument("--render", metavar="HxW", default=None,
                    help="print the screen as text at this size and exit (no terminal needed)")
    ap.add_argument("--profiles", action="store_true",
                    help="print the ready-made profiles with what each needs, and exit")
    ap.add_argument("--profile", default=None, metavar="NAME",
                    help="select a profile's topics and the keep fraction it needs")
    ap.add_argument("--profiles-file", default=os.environ.get("DSV41_TUNE_PROFILES") or None,
                    metavar="PATH",
                    help="read (and save) user profiles here instead of results/keepsets/"
                         "profiles.json and the file under $XDG_CONFIG_HOME")
    ap.add_argument("--save-profile", default=None, metavar="NAME",
                    help="keep the current --topics selection under this name and exit")
    ap.add_argument("--describe", default=None, metavar="TEXT",
                    help="the one-line description --save-profile gives the profile")
    ap.add_argument("--brief", action="store_true",
                    help="print the task of adding a topic to this keep-set, as Markdown, and exit")
    ap.add_argument("--list", action="store_true", help="print the topics and exit")
    ap.add_argument("--print", dest="show", action="store_true", help="print the environment and exit")
    ap.add_argument("--write", action="store_true", help="write the selection into .env and exit")
    a = ap.parse_args()

    COVERAGE_TARGET = a.coverage_target
    if a.rank not in B.RANKS:
        # Not argparse `choices`: this defaults from the environment, and a typo
        # in .env has to be refused here rather than served as `sum` -- which is
        # the very disagreement between screen and engine this flag exists for.
        print(f"unknown rank {a.rank!r} (DSV41_PRUNE_RANK): {' | '.join(B.RANKS)}", file=sys.stderr)
        return 2
    if a.source not in B.SOURCES:
        # same reasoning as --rank: it defaults from the environment, and a file
        # that says `salience` must stop the tool rather than be served frequency
        print(f"unknown source {a.source!r} (DSV41_PRUNE_SOURCE): {' | '.join(B.SOURCES)}",
              file=sys.stderr)
        return 2
    host = B.read_host()
    try:
        sp_ = find_stats(a.stats, a.source)
    except MissingStats as e:
        print(f"no such keep-set: {e}", file=sys.stderr)
        here = sorted(glob.glob(os.path.join(ROOT, "results/keepsets/*/coverage.json")))
        if here:
            print("available: " + ", ".join(short_path(x) for x in here), file=sys.stderr)
        return 2
    index = B.TopicIndex(sp_, a.source) if sp_ else None
    # Profiles from a file. A broken file costs its own profiles and nothing
    # else, so every problem is reported and the tool carries on with the
    # built-in ones -- including into the interactive screen, where stderr is
    # not visible, which is why the screen repeats the first problem itself.
    files = profiles_files(a.profiles_file)
    user, problems = load_profiles(files)
    problems += unknown_topics(user, index)
    for msg in problems:
        print(f"profiles: {msg}", file=sys.stderr)
    sel = [t.strip() for t in a.topics.split(",") if t.strip()]
    unknown = [t for t in sel if not index or t not in index.topics] if sel else []
    interactive = sys.stdout.isatty() and not (a.list or a.show or a.write or a.render or a.profiles
                                               or a.brief or a.save_profile)
    if unknown and not interactive:
        # a script asked for something this keep-set cannot serve: say so and stop
        where = short_path(sp_) if sp_ else "any coverage.json in the checkout"
        print(f"not in {where}: {', '.join(unknown)}", file=sys.stderr)
        print(f"have: {', '.join(index.topics) if index else '(none)'}", file=sys.stderr)
        if a.source != "counts" and (not index or not index.topics):
            # the likeliest cause by far: the file predates the tracer that records
            # expert-output norms, so it has every counts_* and no saliency_*
            print(f"this keep-set carries no {a.source}_<topic> histograms at all — re-run "
                  f"tools/expert_trace.py and tools/expert_stats.py, or use --source counts",
                  file=sys.stderr)
        return 2
    if unknown:
        sel = [t for t in sel if t not in unknown]   # the screen is where this gets fixed

    st = State(host, index, sp_, a.keep, a.max_seq, a.format, sel,
               transient_slots=a.transient_slots, keep_free_gb=a.keep_free_gb,
               user_profiles=user, profiles_path=files[-1], rank=a.rank, source=a.source)
    st.problem = problems[0] if problems else ""
    if unknown:
        st.msg = f"dropped, not in this keep-set: {', '.join(unknown)}"

    if a.profile:
        match = [p for p in st.profiles() if p["name"].lower().startswith(a.profile.lower())]
        if len(match) != 1:
            names = ", ".join(p["name"] for p in st.profiles())
            print(f"{'no' if not match else 'more than one'} profile matches {a.profile!r}; "
                  f"have: {names}", file=sys.stderr)
            return 2
        if not match[0]["topics"]:
            print(f"{match[0]['name']}: this keep-set carries none of its topics", file=sys.stderr)
            return 2
        st.apply_profile(match[0])

    if a.profiles:
        print(f"{short_path(sp_)} — {len(index.topics) if index else 0} topics, "
              f"{st.max_seq // 1024}k context")
        for f_ in sorted({p[4] for p in user}):
            print(f"{sum(1 for p in user if p[4] == f_)} of these are yours, from {f_}")
        print()
        for pr in st.profiles():
            head = f"  {pr['name']:<22} {pr['status']}"
            if pr["mine"]:
                head += "  (yours)"
            print(head)
            print(f"  {'':<22} {pr['blurb']}")
            p = pr["plan"]
            if pr["topics"]:
                n_t = len(pr["topics"])
                print(f"  {'':<22} {pr['keep']:.0%} of experts · rank {pr['rank']} · {st.source} · "
                      f"{p.arena:.0f} GB · {p.free_after_load:.0f} GB free · "
                      f"{n_t} topic{'' if n_t == 1 else 's'}")
            if pr["topics"] and pr["missing"]:
                print(f"  {'':<22} not in this keep-set: {', '.join(pr['missing'])}")
            print()
        return 0

    if a.brief:
        sys.stdout.write(brief(st))
        return 0

    if a.save_profile:
        if not st.sel:
            print("nothing to save: name the topics with --topics a,b or --profile NAME",
                  file=sys.stderr)
            return 2
        try:
            where = save_profile(files[-1], a.save_profile.strip(),
                                 a.describe or describe_selection(st.sel), st.sel)
        except ProfileError as e:
            print(str(e), file=sys.stderr)
            return 2
        print(f"saved {a.save_profile.strip()!r} to {short_path(where)}: "
              f"{', '.join(sorted(st.sel))}")
        return 0

    if a.render:
        try:
            rh, _, rw = a.render.partition("x")
            rh, rw = int(rh), int(rw)
        except ValueError:
            print("--render wants HxW, e.g. 30x96", file=sys.stderr)
            return 2
        print(render(st, rh, rw))
        return 0

    if a.list:
        if not index or not index.topics:
            print(f"no per-topic histograms in {sp_ or 'any coverage.json'}")
            return 1
        cur = index.curves(tuple(index.topics), rank=st.rank)[0]
        n = B.keep_n(a.keep)
        print(f"{short_path(sp_)} — {len(index.topics)} topics, coverage at keep {a.keep:.0%}, "
              f"rank {st.rank}, source {st.source}")
        for t in index.topics:
            nt = index.tokens.get(t, 0)
            flag = "  thin" if nt < B.TopicIndex.THIN else ""
            print(f"  {t:<14} {bar(cur[t][n], 24)} {cur[t][n]:.2f}  {nt:>8,} tokens{flag}")
        return 0

    if a.write and not sp_:
        print("refusing to write .env: no keep-set found, so TRACE_STATS would be dropped and "
              "the launcher would pick an arbitrary trace", file=sys.stderr)
        return 2
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
