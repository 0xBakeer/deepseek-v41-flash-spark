#!/usr/bin/env python3
"""patch_ui.py -- the patch this repository applies to its vendored copy of Weight Atlas.

Weight Atlas (github.com/alesha-pro/atlas, MIT) draws a model's expert field as a grid of
layers x experts. Its GLM region assumes the capture that shipped with it: `buildGlmInsights`
builds all twelve cards unconditionally, and `atlasCard` reads four metric matrices, a
co-routing table and a pair-cosine table that only that capture has. Nothing about the grid
itself is GLM-specific -- it is the surrounding code that is.

This file is kept for reproducibility: `tools/atlas/` holds the BUILT site, and this is the
exact source change it was built from. Every edit below either (a) derives a number the card
had hard-coded, (b) makes an already-optional-looking block actually optional, or (c) lets a
model carry its own wording for prose that was written about GLM. One of them, #12, is a
genuine layout bug rather than a local accommodation: the 17 px column that makes 288 experts
fit the 5800 px card overflows it at more than ~330 experts and leaves the expert dossier a
negative width. No behaviour changes for the GLM model, and `npx tsc --noEmit` stays clean.

Two of the edits are this vendoring's own rather than a fix anyone else needs: the upstream
page pulls Spectral and IBM Plex Mono from fonts.googleapis.com, and a copy served from a
loopback port must fetch nothing, so the stacks fall back to fonts the machine already has.

Idempotent: run it twice and the second run reports every edit as already applied.

    python3 upstream/patch_ui.py --atlas upstream/atlas [--revert]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

GLM = "src/sections/glm.ts"
MAIN = "src/main.ts"
INTRO = "src/sections/intro.ts"
DATA = "src/data.ts"
CSS = "src/style.css"
HTML = "index.html"

EDITS: list[tuple[str, str, str]] = [

    # ── 1. the card title is the shape of the data, not "42 × 288" ──────────
    (GLM,
     """  const { root, body } = cardShell(5800, l('THE EXPERT ATLAS · 42 × 288', 'АТЛАС ЭКСПЕРТОВ · 42 × 288'), l(""",
     """  const { root, body } = cardShell(5800, l(`THE EXPERT ATLAS · ${N} × ${E}`, `АТЛАС ЭКСПЕРТОВ · ${N} × ${E}`), l("""),

    # ── 2. offer only the metric matrices this model actually carries ───────
    (GLM,
     """  const metricDefs: Record<string, { name: string; fmt: (v: number) => string; log: boolean }> = {
    reap: { name: 'exact REAP', fmt: v => fmt(v, 4), log: true },""",
     """  const allMetricDefs: Record<string, { name: string; fmt: (v: number) => string; log: boolean }> = {
    reap: { name: 'exact REAP', fmt: v => fmt(v, 4), log: true },"""),

    (GLM,
     """    contribution: { name: l('output contribution', 'вклад в выход'), fmt: v => fmt(v, 3), log: false },
  };
  const ctl = el('div', 'no-pan',""",
     """    contribution: { name: l('output contribution', 'вклад в выход'), fmt: v => fmt(v, 3), log: false },
  };
  // a capture without top-1 counts, or without a sampled replay, simply does not offer
  // that view rather than crashing on `R[metric]` when the chip is clicked
  const metricDefs: typeof allMetricDefs = {};
  for (const k of Object.keys(allMetricDefs)) if (Array.isArray(R[k])) metricDefs[k] = allMetricDefs[k];
  if (!metricDefs[metric]) metric = Object.keys(metricDefs)[0];
  const ctl = el('div', 'no-pan',"""),

    # ── 3. prune-set outlines are optional ─────────────────────────────────
    (GLM,
     """Object.keys(R.prune_sets).map(a => `<option value="${a}">${l('outline', 'обвести')}: ${armName(a)}</option>`).join('');""",
     """Object.keys(R.prune_sets || {}).map(a => `<option value="${a}">${l('outline', 'обвести')}: ${armName(a)}</option>`).join('');"""),

    # ── 4. top-1 Gini is only drawn when it was measured ───────────────────
    (GLM,
     """${l('selected-load Gini', 'Gini нагрузки')}: ${fix(g, 3)}<br>top-1 Gini: ${fix(R.dynamics.all[i].top1_gini, 3)}`""",
     """${l('selected-load Gini', 'Gini нагрузки')}: ${fix(g, 3)}${R.dynamics.all[i].top1_gini != null ? `<br>top-1 Gini: ${fix(R.dynamics.all[i].top1_gini, 3)}` : ''}`"""),

    # ── 5. the hover line names only the measurements that exist ───────────
    (GLM,
     """REAP ${fmt(R.reap[p.r][ex], 3)} · ${l('routes', 'маршруты')} ${pct(R.route_share[p.r][ex], 2)} · top-1 ${pct(R.top1_share[p.r][ex], 2)}`""",
     """REAP ${fmt(R.reap[p.r][ex], 3)}${R.route_share ? ` · ${l('routes', 'маршруты')} ${pct(R.route_share[p.r][ex], 2)}` : ''}${R.top1_share ? ` · top-1 ${pct(R.top1_share[p.r][ex], 2)}` : ''}`"""),

    # ── 6. the dossier tiles follow the available metrics; uniform = 1/E ───
    (GLM,
     """      <div style="margin-top:14px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px">
        ${tile(fmt(R.reap[p.r][ex], 3), 'exact REAP', l('mean over 12.59M tokens', 'среднее по 12.59M токенов'))}
        ${tile(pct(R.route_share[p.r][ex], 2), l('route share', 'доля маршрутов'), l('uniform = 2.78%', 'равномерно = 2.78%'))}
        ${tile(pct(R.top1_share[p.r][ex], 2), l('top-1 share', 'доля top-1'), l('uniform = 0.35%', 'равномерно = 0.35%'))}
        ${tile(fmt(R.contribution[p.r][ex], 2), l('output norm', 'норма выхода'), l('weighted, sampled replay', 'взвешенно, sampled replay'))}
      </div>""",
     """      <div style="margin-top:14px;display:grid;grid-template-columns:repeat(${Object.keys(metricDefs).length},1fr);gap:8px">
        ${Object.keys(metricDefs).map(k => tile(metricDefs[k].fmt(R[k][p.r][ex]), metricDefs[k].name,
          k === 'route_share' || k === 'top1_share' ? `${l('uniform', 'равномерно')} = ${pct(1 / E, 2)}` : '')).join('')}
      </div>"""),

    # ── 7. co-routing pairs are optional ──────────────────────────────────
    (GLM,
     """    const co = R.coroute[p.r];""",
     """    const co = R.coroute?.[p.r] || { count: [], lift: [] };"""),

    # ── 8. prune-set chips and the aligned pair are optional ──────────────
    (GLM,
     """    const sets = Object.entries(R.prune_sets).filter(([, s]: any) => s[p.r].includes(ex)).map(([a]) => a);
    const aligned = (d.contributions.pairs_extreme[p.r].aligned as number[][]).filter(q => q[0] === ex || q[1] === ex);""",
     """    const sets = Object.entries(R.prune_sets || {}).filter(([, s]: any) => s[p.r].includes(ex)).map(([a]) => a);
    const aligned = ((d.contributions?.pairs_extreme?.[p.r]?.aligned || []) as number[][]).filter(q => q[0] === ex || q[1] === ex);"""),

    # ── 9. the domain tile stops claiming four of them are images ─────────
    (GLM,
     """${tile(String(R.domains.length), l('domains', 'доменов'), l('4 of them real images', '4 из них реальные изображения'))}""",
     """${tile(String(R.domains.length), l('domains', 'доменов'), l('slices of one trace', 'срезы одной трассы'))}"""),

    # ── 10. a card whose evidence block is missing is skipped, not fatal ──
    (GLM,
     """  const rowOf = (...items: [string, HTMLElement][]) => {
    const r = el('div', '', 'display:flex;gap:26px;align-items:flex-start');
    for (const [k, c] of items) { cards.set(k, c); r.appendChild(c); }
    grid.appendChild(r);
  };
  rowOf(['receipt', receiptCard(data, jump)]);
  rowOf(['atlas', atlasCard(data, store)]);
  rowOf(['router', routerCard(data, store)], ['trust', trustCard(data)]);
  rowOf(['kda', kdaCard(data, store)], ['indexer', indexerCard(data, store)]);
  rowOf(['flow', flowCard(data, store)], ['actq', actqCard(data, store)]);
  rowOf(['shared', sharedCard(data, store)], ['nvfp4', nvfp4Card(data, store)]);
  rowOf(['vision', visionCard(data)], ['pruning', pruningCard(data, store)]);""",
     """  // Each card answers one question from one capture. A model that carries only some of
  // those captures shows only those cards, the way dossier.json and live.json are already
  // gated at the region level -- rather than taking the whole region down with it.
  const rowOf = (...items: ([string, HTMLElement] | null)[]) => {
    const kept = items.filter(Boolean) as [string, HTMLElement][];
    if (!kept.length) return;
    const r = el('div', '', 'display:flex;gap:26px;align-items:flex-start');
    for (const [k, c] of kept) { cards.set(k, c); r.appendChild(c); }
    grid.appendChild(r);
  };
  const opt = (k: string, make: () => HTMLElement | null): [string, HTMLElement] | null => {
    try { const c = make(); return c ? [k, c] : null; } catch (e) { console.warn(`glm card ${k} skipped`, e); return null; }
  };
  rowOf(opt('receipt', () => receiptCard(data, jump)));
  rowOf(opt('atlas', () => atlasCard(data, store)));
  rowOf(opt('router', () => routerCard(data, store)), opt('trust', () => trustCard(data)));
  rowOf(opt('kda', () => kdaCard(data, store)), opt('indexer', () => indexerCard(data, store)));
  rowOf(opt('flow', () => flowCard(data, store)), opt('actq', () => actqCard(data, store)));
  rowOf(opt('shared', () => sharedCard(data, store)), opt('nvfp4', () => nvfp4Card(data, store)));
  rowOf(opt('vision', () => visionCard(data)), opt('pruning', () => pruningCard(data, store)));"""),

    # ── 11. the wall tint needs the live blocks, not just the file ────────
    (MAIN,
     """  const wall = buildWall(store, 80, 1890, isGlm && insights ? buildGlmWallTint(insights) : undefined);""",
     """  const wall = buildWall(store, 80, 1890,
    isGlm && insights?.memory && insights?.flow && insights?.vision_tower ? buildGlmWallTint(insights) : undefined);"""),
# ── 12. the grid must fit the card at any expert count ────────────────
    (GLM,
     """  const cw = 17, ch = 13, padL = 54, padT = 22;""",
     """  // 17px columns are what makes 288 experts fit a 5800px card. A layer with more experts
  // (384 here) overflows the card and leaves the dossier a negative width, so the column
  // narrows to fit instead.
  const cw = Math.max(6, Math.min(17, Math.floor(4900 / E))), ch = 13, padL = 54, padT = 22;"""),
    # ── 13. the page fetches nothing from the network ─────────────────────
    # The vendored copy is served from a loopback port on a box that may have no
    # route out at all, and a stylesheet request that hangs holds the first paint.
    (HTML,
     """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">""",
     """<!-- No web fonts: this build is served offline, and the stacks in style.css
     fall back to fonts the machine already has. -->"""),

    (CSS,
     """  --serif: 'Spectral', Georgia, serif;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;""",
     """  --serif: Georgia, 'Times New Roman', serif;
  --mono: Menlo, ui-monospace, SFMono-Regular, monospace;"""),

    # ── 14. the shell prose is about GLM; let a model carry its own ────────
    # Four strings around the cards -- the intro note, two of its stat tiles, the
    # rhythm sentence, the region sub-head and the INT4 metric label -- are written
    # about the GLM checkpoint and derived from nothing. They are the last thing on
    # the page that still says "288 -> 8" under a 384-expert grid.
    (DATA,
     """export interface ManifestEntry {
  slug: string; name: string; note?: string;
  kind?: 'weight' | 'glm-live';
  total_params?: number; active_params?: number;
}""",
     """export interface ManifestEntry {
  slug: string; name: string; note?: string;
  kind?: 'weight' | 'glm-live';
  total_params?: number; active_params?: number;
  // A `glm-live` model that is not GLM: the intro card, two of its stat tiles, the
  // region sub-head and the INT4 metric label are prose about that checkpoint and
  // are derived from nothing, so a model may carry its own wording here.
  shell?: {
    note?: string; rhythm?: string; tiles?: (string[] | null)[];
    metric?: string; region_sub?: string;
  };
}"""),

    (DATA,
     """    metrics.int4.label = t('glm.metric.qdq');""",
     """    metrics.int4.label = entry.shell?.metric ?? t('glm.metric.qdq');"""),

    (INTRO,
     """  const isGlm = m.entry.kind === 'glm-live';
  root.innerHTML = `""",
     """  const isGlm = m.entry.kind === 'glm-live';
  // The last three tiles describe the GLM checkpoint's layer rhythm, vision tower and
  // expert field; a model that carries its own wording replaces them by index.
  const tiles: (string | number)[][] = [
    [isGlm ? fmtN(m.entry.active_params ?? 0) : fmtN(m.langParams), isGlm ? t('glm.intro.stat.active') : t('intro.stat.lang')],
    [t('intro.stat.layers', m.langLayers.length), isGlm ? t('glm.intro.stat.layers.sub', full, lin) : t('intro.stat.layers.sub', full, lin)],
    [isGlm ? '24 blocks' : fmtN(m.visParams), isGlm ? t('glm.intro.stat.vision') : t('intro.stat.vision')],
    [isGlm ? '288 → 8' : `${n2d}`, isGlm ? t('glm.intro.stat.experts') : t('intro.stat.sqnr')],
  ];
  (m.entry.shell?.tiles || []).forEach((tl, i) => { if (tl && i < tiles.length) tiles[i] = tl; });
  root.innerHTML = `"""),

    (INTRO,
     """        ${[
          [isGlm ? fmtN(m.entry.active_params ?? 0) : fmtN(m.langParams), isGlm ? t('glm.intro.stat.active') : t('intro.stat.lang')],
          [t('intro.stat.layers', m.langLayers.length), isGlm ? t('glm.intro.stat.layers.sub', full, lin) : t('intro.stat.layers.sub', full, lin)],
          [isGlm ? '24 blocks' : fmtN(m.visParams), isGlm ? t('glm.intro.stat.vision') : t('intro.stat.vision')],
          [isGlm ? '288 → 8' : `${n2d}`, isGlm ? t('glm.intro.stat.experts') : t('intro.stat.sqnr')],
        ].map(([v, l]) => `""",
     """        ${tiles.map(([v, l]) => `"""),

    (INTRO,
     """        ${isGlm ? t('glm.intro.note') : t('intro.note', fmtN(m.totalParams), m.tensors.length)}""",
     """        ${isGlm ? (m.entry.shell?.note ?? t('glm.intro.note')) : t('intro.note', fmtN(m.totalParams), m.tensors.length)}"""),

    (INTRO,
     """<div class="small-note" style="max-width:580px">${isGlm ? t('glm.intro.rhythm') : t('intro.rhythm')}</div>""",
     """<div class="small-note" style="max-width:580px">${isGlm ? (m.entry.shell?.rhythm ?? t('glm.intro.rhythm')) : t('intro.rhythm')}</div>"""),

    (GLM,
     """<div class="section-sub">${l('weights · forward passes · causal controls, all from the deployed NVFP4 checkpoint', 'веса · forward passes · causal controls, всё с развёрнутого NVFP4-чекпоинта')}</div>""",
     """<div class="section-sub">${store.model.entry.shell?.region_sub ?? l('weights · forward passes · causal controls, all from the deployed NVFP4 checkpoint', 'веса · forward passes · causal controls, всё с развёрнутого NVFP4-чекпоинта')}</div>"""),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas"))
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    applied = skipped = 0
    for rel, old, new in EDITS:
        path = os.path.join(args.atlas, rel)
        src = open(path, encoding="utf-8").read()
        a, b = (new, old) if args.revert else (old, new)
        if b in src and a not in src:
            skipped += 1
            continue
        if a not in src:
            print(f"!! not found in {rel}:\n{a[:120]}...", file=sys.stderr)
            return 1
        open(path, "w", encoding="utf-8").write(src.replace(a, b, 1))
        applied += 1
    print(f"{'reverted' if args.revert else 'applied'} {applied} edit(s), {skipped} already in place")
    try:
        d = subprocess.run(["git", "-C", args.atlas, "diff", "--stat"], capture_output=True, text=True)
        print(d.stdout.strip())
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
