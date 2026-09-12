# Keep-sets, topics and coverage

The three ideas `./tune.sh` works with. The screen that uses them is described in
[`docs/tune.md`](tune.md), field by field in [`docs/tune-reference.md`](tune-reference.md); the
memory those ideas cost is in [`docs/memory-budget.md`](memory-budget.md); the shipped profiles and
what each was measured to be good and bad at are in
[`results/keepsets/README.md`](../results/keepsets/README.md).

## A keep-set is a cache policy

The model has 40 layers of 384 routed experts, 15,360 in all, and every token activates 6 of them in
every layer. At the FP4 layout the checkpoint ships, all 15,360 are 288.8 GB; a GB10 has about
121 GiB of unified memory for everything. So most of them cannot be resident, and the engine has to
choose.

A **keep-set** is that choice made once, before the server starts: per layer, the top *N* experts
stay in the arena and the router may only pick among them, where *N* is `ceil(PRUNE_KEEP x 384)`.
Nothing is requantised and nothing is approximated — the experts that stay are the checkpoint's own.
The experts that do not stay are simply not routable.

That is what makes it a cache policy rather than a compression scheme, and it is why the choice
matters so much: a policy is only as good as the workload sample it was learned from.

*N* is decided by a measured routing trace. `tools/expert_trace.py` pushes a corpus through the
model one layer at a time and records which experts each token picked; `tools/expert_stats.py`
turns that into the per-layer histograms in a `coverage.json`. The engine ranks each layer's 384
experts by those counts and keeps the top *N*.

## A topic is one per-layer histogram

`corpus/make_corpus.py` tags every sequence in the trace corpus with a category, and
`tools/expert_stats.py` writes a separate 384-wide histogram per category per layer
(`counts_<topic>` inside `coverage.json`) alongside the mixed one. A **topic** is one of those: the
routing a corpus of that subject alone produced, 40 layers deep.

Two consequences follow, and they are the reason the tool exists.

* One trace yields every topic in it. The trace is one pass over the corpus per layer and costs
  about the same whether the corpus carries five topics or thirty-five, so it is worth tagging
  generously.
* Composing a keep-set out of a subset of those topics is arithmetic on numbers already in the
  checkout. No GPU, no new trace, no model load — which is what makes a live screen possible at all.

`EXPERT_TOPICS` names the subset. Unset means every topic in the file, which is what the shipped
profiles do.

## How a selection becomes a ranking

For each selected topic, each layer's 384 counts are divided by their own sum. The normalised
histograms are added, and the top *N* of that sum is the layer's keep-set
(`engine/v41_engine.py`, and the same arithmetic in `tools/budget.py` so the screen agrees with the
engine).

That agreement holds for all three rules since 2026-09-12: `tools/budget.py` reimplements `max` and
the `maxmin` rule below as well, defaults to whatever `DSV41_PRUNE_RANK` is set to, takes
`./tune.sh --rank sum|max|maxmin`, shows the rule beside the keep fraction on both screens and
writes it into `.env` with them. `tools/test_budget_rank.py` lifts the engine's own function out
with `ast` and holds the two implementations to the same keep-set, layer by layer, as sets, at
seven keep fractions — because a screen that ranks by one rule while the engine ranks by another
reports coverage nothing will deliver: over the seven topics below at keep 0.36 that gap is 0.15 on
`english` alone.

Normalising before summing is the whole point. Without it a topic that contributed 20,000 tokens
would outvote one that contributed 3,000 by a factor of seven, and "serve this workload too" would
quietly mean "serve it if it happens to be the bigger half of the corpus". With it, every selected
topic gets one vote per layer.

The alternative combination — keeping an expert that matters to *any* selected topic, rather than
summing — exists as `DSV41_PRUNE_RANK=max` and was measured worse, so `sum` remained the default
until `maxmin`, below.

### One vote each is not the same as one outcome each

`sum` maximises the total routing mass the resident set keeps. That is the wrong quantity when one
request spans several topics at once, which is what a coding request with thinking on does: it
writes English prose, deliberation, HTML, CSS and JavaScript in a single generation, and it
degenerates at whichever of them the resident set serves least. A rule that maximises the total is
free to let an already well-served topic go on taking slots while another starves.

Normalising per layer has a second consequence that is easy to miss: it divides corpus size out, so
a *broad* topic — one that spreads its mass over many experts — scores low on every one of them and
loses slot after slot to a peaky specialist. Over
{english, html, python, reasoning, css, javascript} at `PRUNE_KEEP=0.36` on a GB10, `sum` leaves
english at 0.556 while css and javascript sit at 0.803 and 0.814.

`DSV41_PRUNE_RANK=maxmin` hands each layer's slots out one at a time to whichever selected topic is
currently least covered. An expert admitted for one topic counts for every topic that also routes
to it, so overlap is paid for once instead of per topic, and the selected topics converge on a
common coverage rather than a spread. The same budget then holds 0.676–0.688 across all six.

It also changes what adding a topic costs. Under `sum` the worst-served topic falls from 0.657 at
four topics to 0.410 at eighteen; under `maxmin` the same span costs 0.06. Breadth is affordable
under `maxmin` and ruinous under `sum` — but neither rule creates capacity, and a wide enough
selection runs every topic down. Measured on this box at `PRUNE_KEEP=0.36`, 139 experts a layer
(2026-09-12): under `maxmin` about **sixteen** topics is where the worst-served one lands at 0.671,
below the 0.7 line; 35 of the catalogue's 36 topics together run 0.580–0.636, with english the
hardest. The same full selection under `sum` puts chinese hardest at 0.464. Sixteen is this box at
this keep fraction, not a property of the rule — the boundary moves with both.

## Coverage

**Coverage** is the fraction of a topic's *measured* routing that lands on an expert the current
keep-set holds resident. It is read off a cumulative curve: index *n* of the curve is the routing
mass captured by keeping the top *n* experts of every layer, so the number on screen is
`curve[ceil(keep x 384)]`.

It is a measurement, not an estimate — the histogram it is computed from is the routing that
actually happened on that corpus. It is monotone in the keep fraction and reaches 1.0 at 100 %.

Coverage depends on the *selection*, not only on the keep fraction, because the selection decides
the ranking. On the shipped 2-topic keep-set at `PRUNE_KEEP=0.39`:

| selected | coverage of `coding` | coverage of `general` |
|---|---|---|
| `coding` | 0.90 | 0.51 |
| `general` | 0.44 | 0.85 |
| both | 0.83 | 0.75 |

Both topics are in the file either way. Selecting one does not remove the other from the world; it
spends the budget on the first, and the screen shows what that costs the second in the same glance.

## Why coverage predicts whether long generations hold together

An expert that is not resident is not routable. When the router's first choice for a token is
outside the keep-set, the token goes to its second choice, and then to the second choice again on
the next token, and the one after that. The errors do not cancel — they compound into repetition,
and then into structurally broken output.

This is not a theory about the format or the kernels. It is what this repository spent days chasing:

| keep-set, all at keep 44 % and otherwise identical | story | Python | HTML |
|---|---|---|---|
| a corpus whose only content marker was Python | 0.27 | 0.51 | **0.03** |
| the same, plus a 12-gram repeat ban | 0.32 | 0.51 | **0.07** |
| a corpus of web, code, configuration and technical prose | 0.43 | 0.50 | 0.40 |
| a corpus of narrative fiction and dialogue | 0.54 | 0.58 | **0.04** |
| the union of the last two | **0.56** | **0.47** | **0.59** |

(distinct-token ratio of the generated text; 0.15 or below is degenerate. `RESULTS.md` §4.2.)

The first row wrote `<!DOCTYPE><!DOCTYPE><!DOCTYPE>` for as long as it was allowed, and measured
*better* on teacher-forced loss than the configuration that replaced it. Markup coverage in that
keep-set was 0.03. Raising it to 0.40 fixed the generation. A corpus of fiction fixes prose and
loses markup; a corpus of code does the reverse; at 44 % of the experts the two rankings compete
for the same slots.

So the useful reading of the screen is not "how many topics" but **"what is the weakest selected
topic's coverage"**. Below about 0.7 is where generation starts to degrade, which is why the bars
turn amber there and red below. The default target of 0.85 is not a universal constant: it is where
the shipped keep-sets sit for the domains they were built for.

Two limits on that reading, both recorded in [`LIMITATIONS.md`](../LIMITATIONS.md):

* The generation gate that produced those rows runs to 2,000 tokens. Past that, generations still
  degenerate at the shipped keep fraction, and the cause is not yet attributed.
* Coverage says nothing about speed. A step reads the experts the token activates either way, so
  choosing fewer topics buys a smaller keep fraction rather than a faster step; see
  [`docs/tune.md`](tune.md#fewer-topics-are-not-faster-under-sum-they-are-cheaper-under-maxmin-breadth-is-cheap-but-not-free) for the mechanism, the one
  measurement that supports it, and the A/B that has not been run.

## How many experts a step actually reads

Six per layer is the per-token figure, and it is not the figure a decode step pays. DSpark verifies
a block of six tokens at once, and the routing of six consecutive tokens overlaps heavily, so the
36 `(token, expert)` pairs of a layer collapse to far fewer distinct reads — but to noticeably more
than six.

| measurement | value | how it was taken |
|---|---|---|
| distinct experts per layer per verify block | **20.96** | counted on the real decode path over ~20 verify blocks at keep 40 %, `RESULTS.md` §3.5 |
| `block6_unique_mean` in `coverage.json` | 22.6 mean, 18.2 to 29.5 by layer | counted on the trace, with **no** keep mask applied |

The two are close but they answer different questions, and the difference is the reason the speed
question in [`docs/tune.md`](tune.md#fewer-topics-are-not-faster-under-sum-they-are-cheaper-under-maxmin-breadth-is-cheap-but-not-free) is still open:
`block6_unique_mean` is measured without a keep-set, so it cannot say whether a keep-set matched to
its workload concentrates routing and lowers the count. Only an A/B at a fixed keep fraction, one
topic against many, can, and it has not been run.

What the first row does establish is the read volume a step pays: about 21 experts per layer across
40 layers at 14.45 MB a slot is 12.12 GB of expert reads per step, which over the 65.0 ms the two
expert kernels take is 186 GB/s — the rate the kernel reaches in isolation. The floor is real, and
the only way under it is fewer reads.

## A thinly traced topic reports coverage that is too high

Each token contributes 6 picks per layer spread over 384 experts. A topic traced on 300 tokens
therefore leaves each layer with 1,800 picks across 384 experts — a mean of under 5 each, so most of
the ranking is the difference between one count and two. At 2,000 tokens it is 12,000 picks, a mean
of about 31, which is where the tool stops calling a topic thin.

The problem is worse than noise, because **coverage is measured on the very trace that chose the
experts**. A topic seen for 300 tokens routes to whatever fired during those 300 tokens, and those
are exactly the experts its histogram ranked highest, so it scores as though it were well served.
The bias is upward and it is systematic.

The tool therefore draws such a topic's bar hollow, prints its traced token count next to it in red,
and puts a warning on the line above the keys whenever one is selected. Treat a hollow bar's number
as an upper bound. A few thousand tokens a topic is what makes a bar worth believing;
`corpus/fetch_topics.py` collects about that much and flags any topic it could not.

## Profiles and topics

A **profile** is a whole `coverage.json` directory under `results/keepsets/`, selected with
`EXPERT_PROFILE`. A **topic** is one histogram inside such a file, selected with `EXPERT_TOPICS`.
They compose: the profile decides which measurements are on the table, the topic selection decides
how the budget is spent across them.

Not every shipped profile carries topics. `results/keepsets/general/` has two (`coding` and
`general`); `code` and `prose` carry only the mixed histogram, so `./tune.sh` shows the budget panel
for them but no topic list, and `EXPERT_TOPICS` cannot be used with them. Each profile ships a
`GATE.md` recording which domains were measured, which passed and which failed. Read it before
choosing one.
