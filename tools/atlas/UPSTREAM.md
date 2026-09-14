# Weight Atlas, vendored

This directory is a **built copy of somebody else's web app**, not our code.

| | |
|---|---|
| upstream | <https://github.com/alesha-pro/atlas> |
| author | alesha-pro · <https://atlas.alesha.pro> |
| licence | MIT — `LICENSE` here is theirs, verbatim |
| pinned commit | `b57e75a583378fe073d106f122342718ec7f0887` |
| vendored | 2026-09-14 |

`index.html` and `assets/` are the output of `npm run build` at that commit with
`upstream/patch_ui.py` applied. Nothing else of the upstream tree is here: no sources, no
`node_modules`, no other model's data.

## Why it is here

Weight Atlas draws a MoE model's expert field as one grid of layers × experts, with a domain
slice, a ranked ordering and set outlines on top of it. That is exactly the shape of what
`results/keepsets/topics/coverage.json` measures, and a keep-set is much easier to argue
about when it is a visible band of columns than when it is a list of expert ids. `a` in
`./tune.sh` exports our routing into the files this page reads and serves it on loopback.

## What it is fed

`tools/atlas_export.py` writes, into the gitignored `models/` directory beside this file:

```
models/deepseek-v4.1-flash/insights.json   ~5 MB  the 40 × 384 matrices and the ten keep-sets
models/deepseek-v4.1-flash/atlas.jsonl            an architecture-derived weight inventory
models/manifest.json                              one entry, so the page opens on it
```

`models/` is generated, so it is not committed. Delete it and press `a` again.

## Rebuilding

The patch is idempotent and reverts with `--revert`, so it can be re-applied to a fresh clone
and the result compared against what is committed here:

```bash
cd tools/atlas
git clone https://github.com/alesha-pro/atlas upstream/atlas
git -C upstream/atlas checkout b57e75a583378fe073d106f122342718ec7f0887
python3 upstream/patch_ui.py                       # 22 replacements in 6 files, idempotent
( cd upstream/atlas && npm install && npm run build )
cp upstream/atlas/dist/index.html . && rm -rf assets && cp -R upstream/atlas/dist/assets .
```

The hashed asset names change when the sources do, so `index.html` and `assets/` are
replaced together; `models/` is generated and survives untouched.

`upstream/atlas/` is not committed either; `upstream/patch_ui.py` is, because it is the
record of what was changed and why.

## What the patch changes

Twenty-two exact string replacements in six files, `npx tsc --noEmit` clean, and the GLM
model that ships with the page still renders exactly as before. Three kinds:

* **defensive** — the twelve cards of the evidence region are built one at a time and a card
  whose data block is absent is skipped rather than taking the region down; the metric chips,
  the co-routing table, the prune-set outlines and the pair-cosine tiles become optional; the
  card title and the "uniform =" reference are derived from the data instead of hard-coded.
  Any of it would be a reasonable pull request upstream.
* **one real bug** — the expert grid uses a fixed 17 px column, which is what makes 288
  experts fit the 5800 px card. At more than about 330 experts the grid overflows the card
  and leaves the expert dossier beside it a *negative* width. Ours has 384, so the column now
  narrows to fit. This is not specific to our data: it is wrong for any model that wide.
* **ours, and nobody else's problem** — the page's prose ("288 → 8", "24 blocks", "the
  deployed NVFP4 checkpoint") is about the checkpoint it shipped with, so a manifest entry
  may now carry its own wording; and the two web fonts are dropped in favour of the fallback
  stacks, because this copy is served from a loopback port with no network behind it.
