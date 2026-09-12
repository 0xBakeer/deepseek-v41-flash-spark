# `./tune.sh` reference

Every flag, every key, every field on the screen and what it is computed from. The reasoning
behind the tool is in [`docs/tune.md`](tune.md); the ideas it works with — keep-set, topic,
coverage — are in [`docs/keep-sets.md`](keep-sets.md); the memory arithmetic behind the right-hand
panel is in [`docs/memory-budget.md`](memory-budget.md).

`tune.sh` sources `./.env` before running `tools/tune.py`, so every default below that names an
environment variable is read from `.env` as well as from the shell.

## Invocation

```bash
./tune.sh                          # interactive, needs a terminal of at least 70x20
./tune.sh --list                   # the topics this keep-set carries, with coverage
./tune.sh --topics coding --print  # the environment that selection implies
./tune.sh --render 30x96           # the screen as text, no terminal needed
```

The interactive screen is used only when stdout is a terminal **and** none of `--list`, `--print`,
`--write` or `--render` was given. In a pipe or a CI job, a bare `./tune.sh` behaves as `--print`.

## Flags

| flag | default | what it does |
|---|---|---|
| `--stats PATH` | see below | the `coverage.json` to read topics and histograms from |
| `--topics a,b` | `EXPERT_TOPICS` | the selection to start from, comma separated |
| `--keep F` | `PRUNE_KEEP`, else `0.39` | the fraction of each layer's 384 routed experts that stays resident |
| `--max-seq N` | `MAX_SEQ`, else `32768` | context length the KV cache is sized for |
| `--format cb3\|fp4` | `EXPERT_FORMAT`, else `cb3` | the arena's expert layout, which sets the slot size |
| `--transient-slots N` | `TRANSIENT_SLOTS`, else `8` | prefill slots outside the LRU; the arena is sized to hold these too |
| `--keep-free-gb F` | `KEEP_FREE_GB`, else `6.0` | host memory the launcher is told to leave free |
| `--coverage-target F` | `DSV41_COVERAGE_TARGET`, else `0.85` | the coverage every selected topic should reach; sets the bar colours and what `m` fits to |
| `--render HxW` | — | print the screen as text at that size and exit. **Height first**: `30x96` is 30 rows of 96 columns |
| `--list` | — | print the topics with their coverage at `--keep` and their traced token counts, then exit |
| `--print` | — | print the environment the current settings imply, then exit |
| `--write` | — | write those settings into `./.env` and exit |

Without `--stats`, the file is chosen in this order:

1. `results/keepsets/$EXPERT_PROFILE/coverage.json`, if `EXPERT_PROFILE` is set and that file exists;
2. otherwise every `results/keepsets/*/coverage.json` and `results/trace-*/stats/coverage.json` in
   the checkout is scanned and the one carrying the most topics wins, ties going to the first in
   that order.

A `coverage.json` written before per-category histograms were added carries no topics at all. The
tool still runs — the budget panel is live and the keep and context sliders work — but the topic
pane says `this keep-set carries no per-topic histogram` and `--list` exits 1.

## Exit codes

| code | meaning |
|---|---|
| 0 | the configuration will load (`FITS` or `TIGHT`); also a successful `--list` or `--render`, and quitting the interactive screen with `q` |
| 1 | the configuration will not load (`WILL NOT LOAD`); also `--list` on a keep-set with no per-topic histograms |
| 2 | a topic was named that this keep-set does not carry; also a `--render` argument that is not `HxW` |

`--print` is therefore usable as a check in a script: it exits non-zero exactly when the selection
would not serve. Note that `TIGHT` exits 0 — it means the margin is under 3 GB, not that it fails.

After `r` in the interactive screen the exit code is `./start.sh`'s.

## Keys

| key | effect |
|---|---|
| `↑` `↓`, `k` `j` | move the topic cursor |
| `PgUp` `PgDn` | move it ten rows |
| `space` | select or deselect the topic under the cursor |
| `a` | select every topic currently visible (the filter applies) |
| `n` | deselect every visible topic |
| `/` | start typing a filter; `Enter` keeps it, `Esc` clears it |
| `Tab`, `Shift-Tab` | cycle the focused pane: topics, resident experts, context |
| `←` `→` | adjust the focused slider. Context steps 4k, 8k, 16k, 32k, 64k, 128k, 256k; every other pane steps the keep fraction by 2 points between 6 % and 60 % |
| `m` | snap the keep fraction to the smallest one at which every selected topic reaches the coverage target |
| `f` | switch the arena format between `cb3` and `fp4` |
| `r` | write `.env` and run `./start.sh` |
| `w` | write `.env` and stop |
| `q` | quit without writing |

`f` and `w` are not in the on-screen key line. `r` refuses while something else is holding an arena,
and refuses a configuration whose verdict is `WILL NOT LOAD`; `w` does neither, so it can write an
over-budget `.env` on purpose.

## The screen

Minimum size is 70 columns by 20 rows; below that the tool prints the size it needs and the
non-interactive alternatives. The left pane is `max(46, 0.54 x width)` columns wide, the budget
panel starts three columns after it, and the coverage bar inside a topic row is
`max(8, left_pane - 37)` columns. The whole screen is redrawn once a second so that
`MemAvailable` is current; the probe for another process holding an arena runs every four
seconds because it walks `/proc`.

### Header

| field | computed from |
|---|---|
| box name | `/proc/device-tree/model` or the DMI product name, replaced by `nvidia-smi --query-gpu=name` when that is not already part of it, falling back to the host name |
| total GB | `MemTotal` in `/proc/meminfo` |
| free GB | `MemAvailable` in `/proc/meminfo`, re-read every second |
| format | the current `--format`, `cb3` or `fp4` |

On a GB10 there is no second pool to ask about. The memory is unified and `nvidia-smi` answers
`[N/A]` for `memory.total`, `memory.used` and `memory.free`, so `/proc/meminfo` is the only honest
source — and it is the one the engine's own pre-flight compares against.

Off Linux there is no `/proc/meminfo`, so the header falls back to a GB10's 130.6 GB total and
117.0 GB available and says so, which is what makes `--render` reproducible anywhere.

The row under the header carries one of two warnings, or nothing:

* `already running here: <script> (pid N) — this box holds one at a time`, when a Python process
  whose script argument is `v41_engine.py`, `app.py`, `expert_trace.py` or `engram_rows.py` is
  alive. Only the script argument counts, so a shell watching for those names does not match.
* `not a GB10 -- numbers are the model's, not this machine's`, when the box is not a GB10.

### Topics pane

The sub-line reads `N available · M selected`, plus `filter: X` while a filter is active, plus
`· K below` and `· K above` when the list is scrolled.

| column | content | computed from |
|---|---|---|
| 0 | `▌` | the cursor, when this pane has focus |
| 2 | `●` / `○` | selected or not |
| 4 | topic name | truncated to 16 characters |
| 22 | coverage bar | `curve[ceil(keep x 384)]` for that topic under the current selection, at eighth-block resolution |
| after the bar | the same number as `0.NN` | as above |
| right of that | traced tokens | `sum(histogram) / (40 x 6)`, printed as `NNk` at or above 10,000, `N.Nk` at or above 1,000, otherwise the count itself |

The bar is **solid** when the topic was traced on 2,000 tokens or more and **hollow** below that.
A hollow bar is not merely uncertain, it is biased upward; see
[`docs/keep-sets.md`](keep-sets.md#a-thinly-traced-topic-reports-coverage-that-is-too-high).

Colour follows the coverage target: at or above it the bar is green, at or above 0.70 amber, below
that red, and a thin topic is drawn muted whatever its number says. Unselected rows are muted.
A topic with no curve at all shows a row of dots and a dash.

The curves are computed for the **current selection**: with nothing selected the engine would rank
on every topic in the file, so that is what the bars show. Every topic in the file gets a curve,
selected or not, which is how the cost of a narrow selection to the rest of the file is visible on
the same screen.

### Budget panel

Each row is a term of the memory arithmetic. The full derivation, with where every constant comes
from, is in [`docs/memory-budget.md`](memory-budget.md).

| row | value |
|---|---|
| experts resident | `ceil(keep x 384) x 40` out of `15,360`, and the same as a percentage |
| expert arena | `(kept + transient_slots) x slot_bytes`, slot 14,454,784 B for `cb3` and 18,800,640 B for `fp4` |
| dense weights | 7.61 GB, fixed |
| drafter experts | `3 x 128 x 18,800,640` = 7.22 GB, fixed |
| KV cache · Nk | `max_seq x 3,200 + 180,355,072`, shown in MB below 1 GB |
| resident | the four rows above, added |
| free after load | `MemAvailable − resident` |
| room to launch | `MemAvailable − (arena + pack scratch + dense + keep-free floor)`, pack scratch 3 GB for `cb3` and 1 GB for `fp4` |
| verdict | `FITS`, `TIGHT` or `WILL NOT LOAD` |

The verdict is the two gates together:

```
will not load   room to launch < 0  or  free after load < one prefill chunk (10.24 GB at chunk 2,048)
tight           room to launch < 3 GB  or  free after load < that chunk + 3 GB
fits            otherwise
```

Two colouring details are worth knowing, because they are not the verdict rule:

* `free after load` turns red only below the keep-free floor (6 GB by default), so between that
  floor and the 10.24 GB a prefill chunk needs it is drawn green while the verdict already says
  `WILL NOT LOAD`. The badge is the number to act on.
* `room to launch` is green at 3 GB or more, amber down to 0, red below.

The four-line note about step time only appears when the window is at least 28 rows tall.

### Resident experts

`◂ NN % ▸` and a bar scaled so full width is 60 % keep. The part of the bar past `max NN % here` is
drawn red: that is the largest keep fraction this box can both start and survive a prefill chunk at,

```
max_arena = min( MemAvailable − pack scratch − dense − keep-free floor,
                 MemAvailable − dense − drafter − KV − one prefill chunk )
max_keep  = (max_arena / slot_bytes − 8) / 15,360
```

On a 121 GiB box that lands near 42 % in `cb3` and near 32 % in `fp4`. It moves with `MemAvailable`,
so it moves while the screen is open.

### Context

`◂ NNk ▸` and one of two messages:

* at or below 32,768, `run to 32k here; the cache alone has room for X` — where `X` is
  `(free after load − keep-free floor) / 3,200 bytes`, the tokens the cache could hold if nothing
  else wanted the memory;
* above 32,768, `past the 32k run here — prefill is the limit, not the cache`, in amber.

32,768 is the longest context this engine has actually loaded and generated from. Above it the KV
arithmetic still holds but the prefill path has not been run, so the tool marks the length rather
than predicting it.

Both messages have a short form for narrow windows.

### The line above the keys

One line, in priority order:

1. `N selected topics traced on too little text — those bars read high because the sample chose the
   experts`, when any selected topic is under 2,000 traced tokens. Narrow windows get the short
   form, `… — bars read high`.
2. `weakest selected topic  <name> <coverage>`, coloured by the same thresholds as the bars, plus
   an advice fragment on the right:
   * `at the NN % this box holds, K of N reach 0.85` when the keep fraction the target needs is
     larger than `max_keep`;
   * `raise to NN % for 0.85 on every one` or `enough at NN % …` when the needed fraction differs
     from the current one by more than half a point. Below 96 columns only the percentage is shown.
3. `no topic selected — the keep-set would use all of them`.

The bottom line carries the key hints, or a message from the last keypress. A message lasts until
the next key.

## Non-interactive output

### `--list`

```
results/keepsets/general/coverage.json — 2 topics, coverage at keep 39%
  coding         ███████████████████▊···· 0.83    20,694 tokens
  general        ██████████████████······ 0.75    15,556 tokens
```

Coverage is computed with every topic in the file selected, which is what the engine does when
`EXPERT_TOPICS` is unset. Unlike the interactive bars these are always drawn solid; a thinly traced
topic is marked by a trailing `thin` instead.

### `--print` and `--write`

Both print the managed keys on stdout and a one-line summary on stderr, so the settings can be
captured and the summary still read. This run is off Linux, so the memory figures behind it are the
117.0 GB stand-in:

```
$ ./tune.sh --topics coding --print
PRUNE_KEEP=0.39
MAX_SEQ=32768
ARENA_GB=87
EXPERT_FORMAT=cb3
TRANSIENT_SLOTS=8
KEEP_FREE_GB=6
TRACE_STATS=results/keepsets/general/coverage.json
EXPERT_TOPICS=coding
# 6,000 experts resident (39.1%), 102.0 GB resident, 15.0 GB free after load — ok
```

`ARENA_GB` is `ceil` of the arena the panel shows, and the engine floors it back into slots.
`tools/test_budget.py` checks at every keep step and every ring size that this rounding never
leaves a kept expert outside the arena.

`--write` adds `wrote N settings to .env (previous kept as .env.bak)`. It touches only the keys
listed above: existing lines are rewritten in place, a managed key the selection does not set is
dropped, anything else in the file is left alone, and the previous file is kept as `.env.bak`. If
there is no `.env` at all, `env.example` is copied first.

`EXPERT_TOPICS` is written only when at least one topic is selected. `TRACE_STATS` is written as a
path relative to the repository root and takes precedence over `EXPERT_PROFILE` in `start.sh`, so a
profile named in `.env` stops having any effect once the tool has written a selection.

When something else is already holding an arena, both print
`# already running here: <script> (pid N) — this box holds one at a time` on stderr and still
produce the settings.

### `--render`

Prints the screen as characters with no colour and no terminal, trailing blank rows removed. It is
what keeps the screen in [`docs/tune.md`](tune.md) honest, and it is what `tools/test_tune_draw.py`
draws into. Remember the argument is rows first: `--render 30x96` is 30 rows of 96 columns.

## Checks

```bash
python3 tools/test_budget.py      # the cost model against two loads this box actually ran
python3 tools/test_tune_draw.py   # the screen renders at seven sizes without colliding
```

Neither needs a GPU, the checkpoint or torch.
