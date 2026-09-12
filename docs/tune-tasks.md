# Working with `./tune.sh`

Five things people actually do with the tool. The screen itself is described in
[`docs/tune.md`](tune.md), every field and flag in [`docs/tune-reference.md`](tune-reference.md),
the ideas in [`docs/keep-sets.md`](keep-sets.md) and the arithmetic in
[`docs/memory-budget.md`](memory-budget.md).

## Choose the topics for a workload

Start from what the server will actually be sent, not from what would be nice to have. Every topic
selected spends part of a fixed budget, and the cost of the ones you do not need is paid by the
ones you do.

1. `./tune.sh`, or `./tune.sh --list` first to see what the keep-set carries.
2. Select the topics the traffic consists of with `space`. `/` filters the list, `a` selects
   everything visible, `n` clears it.
3. Press `m`. That snaps the keep fraction to the smallest one at which **every** selected topic
   reaches the coverage target, which is the cheapest configuration that serves the selection.
4. Read the line above the keys. It names the weakest selected topic and its coverage, which is the
   number that predicts whether long generations hold together.
5. Read the verdict in the budget panel, and if it is not `FITS`, see
   [fixing an over-budget selection](#fix-an-over-budget-selection).
6. `r` writes `.env` and starts the server. `w` writes and stops.

Selecting nothing is a legitimate answer: the engine then ranks on every topic in the file, which is
what the shipped profiles do. It is the right choice when the traffic is mixed or unpredictable.

Selecting one topic is the cheapest and the most brittle. On the shipped keep-set, `coding` alone
reaches 0.85 coverage at 32 % keep where both topics together need 51 % — but at that setting
`general` covers 0.45, and a keep-set that never saw a domain does not merely get worse at it, it
produces degenerate output in it. Check the other bars before committing to a narrow selection.

Without a terminal, the same decisions are available one at a time:

```bash
./tune.sh --topics coding --keep 0.32 --print   # exits non-zero if it will not load
./tune.sh --topics coding --keep 0.32 --write   # same, written into .env
```

## Read the verdict

| badge | what it means | what to do |
|---|---|---|
| `FITS` | both gates clear with at least 3 GB in hand | run it |
| `TIGHT` | it will load and serve, with under 3 GB of margin on one of the gates | fine on a box with nothing else on it; one more process and it becomes the other verdict |
| `WILL NOT LOAD` | either the engine's pre-flight refuses it, or the first prefill chunk would not fit | lower something before pressing `r`; the tool refuses to run it |

`WILL NOT LOAD` is mostly the second case, and that is the one worth understanding: the engine's own
pre-flight runs before the drafter experts, the KV cache and any prefill exist, so a configuration
can pass it, load for three minutes, log `ready`, and be killed by the memory watchdog on the first
request. On 2026-09-12 a 98 GB arena did exactly that with 5.5 GB free; an 87 GB arena left 16.5 GB
and served. The tool applies the stricter test, so it refuses configurations the engine would have
accepted.

One colouring detail: `free after load` turns red only below the keep-free floor, so it can be green
while the badge says `WILL NOT LOAD`. The badge is the number to act on.

## Fix an over-budget selection

In order of how much they buy, on a 121 GiB box with `cb3` experts:

1. **Press `m`.** If the keep fraction was set by hand it is probably higher than the selection
   needs. If `m` answers that the target needs a fraction this box cannot hold, the selection is too
   broad for this box, not the keep fraction too high.
2. **Deselect a topic.** Watch what the remaining bars do as it goes: a narrower selection reaches
   the same coverage at a smaller keep fraction, and the keep fraction is the arena. Each 2-point
   step of the keep slider is about 320 expert slots, which is 4.6 GB.
3. **Lower the keep fraction** with `←`, and read the weakest-topic line as you go. This is trading
   coverage for memory directly; below about 0.7 on a topic that matters, expect degeneration.
4. **Check the format.** `f` switches between `cb3` at 14.45 MB a slot and `fp4` at 18.80 MB. The
   same arena holds about 30 % more experts in `cb3`. At keep 39 % that is an 86.8 GB arena against
   113.0 GB, which is the difference between serving and not fitting at all.
5. **Shorten the context** only if the rest has been exhausted. It is the smallest term on the
   screen: the whole range from 32k to 256k is 0.73 GB, less than a sixth of one keep step.
6. **Check that nothing else holds the pool.** See below.

What not to do: raise `ARENA_GB` in `.env` by hand afterwards. The tool writes an arena sized to
hold exactly the kept experts plus the transient ring, and `tools/test_budget.py` checks that the
rounding never leaves a kept expert outside it. A larger arena does not make more experts routable —
`PRUNE_KEEP` decides that — it only takes memory the first prefill chunk then needs.

## Add a topic of your own

End to end, from source text to a keep-set the tool can compose. The trace is the expensive step, at
roughly a minute per layer; everything after it is arithmetic.

**1. Gather the text.** `corpus/fetch_topics.py` knows a 35-topic catalogue — sixteen programming
languages taken from a tree you point it at, eleven natural languages from Wikipedia, and eight
domain registers from a fixed set of English Wikipedia articles.

```bash
python3 corpus/fetch_topics.py --list                      # the catalogue
python3 corpus/fetch_topics.py --out topics --code-root ./sources --only rust,german
```

It writes `topics/<kind>/<topic>.txt`, skips a topic that is already there, and prints `THIN` for
any topic it could not fill — those are the ones that will draw a hollow bar later.

It asks Wikipedia for twenty article introductions per request, one request every two seconds,
single threaded, and backs off when told to. That is not politeness for its own sake: a burst of
parallel requests earns an IP-level 429 on every subsequent call, including single ones, for long
enough to stall the job.

For a topic the catalogue does not have, any text file works. One file per topic, not a directory.

**2. Build the trace corpus.** Each `--topic` becomes a sequence category, and one category becomes
one histogram in the finished `coverage.json`.

```bash
python3 corpus/make_corpus.py --tokenizer $MODEL_DIR --target 3000 \
    --out corpus/trace_mine.jsonl \
    $(python3 corpus/fetch_topics.py --out topics --print-topic-flags)
```

`--print-topic-flags` emits one `--topic NAME:KIND:PATH[:lang]` per file already in `topics/`, with
`KIND` being `code` or `prose`. The same flags can be written by hand:

```bash
python3 corpus/make_corpus.py --tokenizer $MODEL_DIR --target 3000 \
    --topic rust:code:topics/code/rust.txt:rust \
    --topic german:prose:topics/lang/german.txt \
    --out corpus/trace_mine.jsonl
```

`--target` is tokens per topic. Aim for a few thousand: below 2,000 the tool marks the topic thin,
and a thin topic's coverage is biased upward rather than merely noisy. The script prints what each
topic actually reached — that is the number to check, not the file size, because CJK text is far
denser per character than Latin text.

**3. Fetch the Engram rows the corpus needs.** The two n-gram tables are 101 GB each and are never
downloaded; only the rows this corpus touches are fetched, over multipart HTTP range requests.

```bash
python3 tools/engram_rows.py --model-dir $MODEL_DIR --corpus corpus/trace_mine.jsonl \
    --out engram_rows_mine
```

Add `--local-shard $MODEL_DIR/model-00047-of-00048.safetensors --local-shard
$MODEL_DIR/model-00048-of-00048.safetensors` when the shards are already on disk.

**4. Trace the routing.** One layer shard at a time, so at most one layer's weights are live.
`--resume` checkpoints after every layer, so this can run while the rest of the checkpoint is still
downloading.

```bash
python3 tools/expert_trace.py --model-dir $MODEL_DIR --corpus corpus/trace_mine.jsonl \
    --engram-dir engram_rows_mine --out results/trace-mine --layers 0-39 --resume
```

**5. Turn the trace into histograms.**

```bash
python3 tools/expert_stats.py --trace results/trace-mine --out results/keepsets/mine
```

That writes `coverage.json` with one `counts_<topic>` histogram per category per layer, which is
everything a keep-set needs. The raw per-layer trace arrays are not needed afterwards.

**6. Use it.**

```bash
./tune.sh --stats results/keepsets/mine/coverage.json --list
EXPERT_PROFILE=mine ./tune.sh
```

**7. Gate it.** Run free generation on your domains *and* on domains your corpus does not contain,
and write both results down next to the profile, the way the shipped `GATE.md` files do. A keep-set
that never saw a domain fails in it structurally, not gradually, and teacher-forced loss will not
show you that — the configuration that could not write an HTML file measured better on loss than the
one that replaced it.

## When the screen says something is already running

```
 already running here: v41_engine.py (pid 12345) — this box holds one at a time
```

`r` refuses while that line is up, and `--print` and `--write` repeat it on stderr but still emit
settings. Stop the other process and wait for the memory to come back:

```bash
./stop.sh          # SIGTERM -> SIGKILL, then blocks until MemAvailable recovers
```

The waiting is the part that matters. A pinned, page-cache-backed arena is reclaimed lazily, so
`MemAvailable` climbs back over seconds rather than instantly, and starting the next server before
it lands is the classic way to wedge the box: two arenas of this size push the host past the point
where `sshd` can fork, and it answers ping and accepts TCP on 22 while being unreachable until
someone power-cycles it. The header's free figure is re-read every second, so the screen can simply
be watched until it recovers.

The probe reads `/proc`, so it sees only processes on this host, and it matches the *script*
argument of a Python process against `v41_engine.py`, `app.py`, `expert_trace.py` and
`engram_rows.py`. A shell watching for one of those names does not trip it, and something holding
the pool under a different name — another inference server, a container — will not be named, though
`start.sh` refuses below `MIN_FREE_GIB` and does list what is holding memory.
