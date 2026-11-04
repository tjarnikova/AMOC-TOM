"""
MF_LIMNUT.py  --  Limiting-nutrient maps and shares, control and hose

Reads the REGRIDDED limiter files written by make_limiter_maps.py:
    <clims>/TOM12_TJ_HA01/HA01_limiter_maps{suffix}_rg.nc   (and the HH01 equivalent)
    suffix = ''      every sample is labelled by its limiting nutrient (Fe, P, Si, N)
    suffix = '_lv95' the same run with --unlimited-above 0.95: a fifth category 'None' (LV >= 0.95)

Three figures (/gpfs/home/mep22dku/scratch/AMOC-TOM/plts/):
  MF_LIMNUT_assemblage_<date>.png  2 rows x 5: share of biomass-months limited by Fe, P, Si, N (row 0 control,
        row 1 hose minus control) and a dominant-limiter map (row 0 control; row 1 where the dominant limiter
        CHANGED, coloured by the hose limiter)
  MF_LIMNUT_groups_<date>.png      4 rows x 6 (DIA..FIX): limitation value, its change, dominant limiter
        control, dominant limiter hose
  MF_LIMNUT_bars_<date>.png        regional stacked bars of the shares, control vs hose (from the stats CSV)

Dominant limiter = the category with the largest share, derived HERE from the regridded shares (a category code
cannot be regridded). 'mixed' = the largest share is below MIXED_BELOW of the total. Shares are monthly limiter
labels weighted by biomass x volume, top 14 levels (see make_limiter_maps.py). FIX 'N' means growth that depends on
N2 fixation (its N factor cannot fall below rn_munfix = 0.56).

Usage:  python MF_LIMNUT.py                      # plain files
        python MF_LIMNUT.py --suffix _lv95       # the 'None' version
        python MF_LIMNUT.py --mixed-below 0.5 --no-watermark
"""
import sys
import os
import glob
import csv
import re
import argparse

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

import map_maker as mm
import plot_style as ps

ap = argparse.ArgumentParser()
ap.add_argument('--suffix', default='', help="'' or '_lv95' (must match the files you regridded)")
ap.add_argument('--mixed-below', type=float, default=0.6, help='top share below this -> mixed')
ap.add_argument('--csv', default=None, help='stats csv for the bar figure (default: newest matching in /gpfs/home/mep22dku/scratch/AMOC-TOM/plts)')
ap.add_argument('--no-watermark', '--water-off', dest='no_watermark', action='store_true')
ap.add_argument('--table-only', action='store_true', help='print the assemblage table from the stats CSV and exit (no maps, no figures)')
args, _ = ap.parse_known_args()

# ---- tunable -----------------------------------------------------------
WATERMARK = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_LIMNUT.py"
OUTPUT_DIR = ps.output_dir()   # figure_settings.yml
CLIMS = os.environ.get('LIMNUT_CLIMS', '/gpfs/data/greenocean/users/mep22dku/clims')
RUNS = [('HA01', 'TOM12_TJ_HA01'), ('HH01', 'TOM12_TJ_HH01')]          # control, hose
SCHEME = 'top14'
SPECIES = ['DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
NUTS = ['Fe', 'N', 'P', 'Si']                    # column / legend order (None follows, then dominant)
COLORS = {'Fe': '#d95f02', 'N': '#5f8a4f', 'P': '#7570b3', 'Si': '#2f6690',      # burnt orange, moss, violet, Bauhaus blue
          'None': '#b8a48c', 'mixed': '#525252', 'no change': '#ececec'}          # taupe
ROW_LABEL_X = -0.18
LV_BASE_CMAP, LV_DIFF_CMAP = 'Greys_r', 'PuOr'      # as MF_PRIM: lighter = less limited
SUFFIX = args.suffix
WATER_ON = ps.watermark_from_settings() and not args.no_watermark   # yaml switch; flags only switch it off


def rg_path(label, folder):
    return '%s/%s/%s_limiter_maps%s_rg.nc' % (CLIMS, folder, label, SUFFIX)


def get(path, var, quiet=False):
    """(lon, lat, field) or None if the variable is not in the file."""
    try:
        lon, lat, f, _ = mm.load_field(path, var, 0)
    except Exception as exc:
        if not quiet:
            print('    [warn] %s not loaded from %s (%r)' % (var, os.path.basename(path), exc))
        return None
    return lon, lat, np.asarray(f, dtype=float)


def dominant(fracs, codes, ncat, mixed_below):
    """fracs: list of 2-D arrays; codes: their global category indices; ncat: index used for 'mixed'.
    Returns (dominant index array with NaN where no data, top share)."""
    st = np.stack([np.nan_to_num(f, nan=0.0) for f in fracs], axis=0)
    tot = st.sum(axis=0)
    valid = tot > 0
    share = st / np.where(valid, tot, 1.0)
    top = share.max(axis=0)
    idx = np.array(codes, dtype=float)[share.argmax(axis=0)]
    idx = np.where(top < mixed_below, float(ncat), idx)
    return np.where(valid, idx, np.nan), np.where(valid, top, np.nan)


LETTERS_ASSEMBLAGE = True      # panel letters a), b), ... on the assemblage figure (Fig 5)
LETTERS_GROUPS = True           # and on the per-group figure; set False to leave them off


def letter(i, on=True):
    '''Panel letter text: 'a)' for 0, 'b)' for 1, ... (row by row), or None when letters are off.'''
    return ("abcdefghijklmnopqrstuvwxyz"[i] + ")") if on else None


def panel(ax, lon, lat, data, cmap, vmin, vmax, ctx, tag, letter=None):
    try:
        mesh = mm._draw_panel(ax, lon, lat, data, cmap, vmin, vmax,
                              ctx['lon_m'], ctx['lat_m'], ctx['masks'], extent=ctx['extent'])
        if letter:
            ps.panel_letter_inside(ax, letter)
        return mesh
    except Exception as exc:
        print('    [warn] %s failed (%r), left blank' % (tag, exc))
        ax.clear()
        ax.axis('off')
        return None


CAT_LABELS = {'mixed': 'mix'}          # shorter label on the categorical colorbars
CAT_LABEL_PT = ps.font_small() - 1      # their label size (pt, 6 with the default 7); six labels must fit under one map width


def fit_cbar_to_map(ax, cax):
    '''Make a colorbar exactly as wide as its map.'''
    ax.apply_aspect()
    p, cp = ax.get_position(), cax.get_position()
    cax.set_position([p.x0, cp.y0, p.width, cp.height])


def cat_cbar(fig, cax, names, colors):
    cm = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(names) + 0.5, 1.0), cm.N)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), cax=cax, orientation='horizontal',
                      ticks=range(len(names)))
    labels = [CAT_LABELS.get(n, n) for n in names]
    # labels are drawn as text (tick labels reset their position at every draw); with six categories
    # n.c. would touch mix, so it is pushed a quarter of a segment to the right
    cb.ax.set_xticklabels([])
    shift = {len(names) - 1: 0.25} if len(names) > 5 else {}
    for i, lab in enumerate(labels):
        cb.ax.text(i + shift.get(i, 0.0), -0.6, lab, transform=cb.ax.get_xaxis_transform(),
                   ha='center', va='top', fontsize=CAT_LABEL_PT)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=CAT_LABEL_PT, length=0)
    return cb


def cont_cbar(fig, mesh, cax, label=None):
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=ps.font_small())
    try:
        cb.formatter = mm._value_formatter()
        cb.update_ticks()
    except Exception:
        pass
    if label:
        cb.set_label(label, fontsize=ps.font_small())
    return cb


def cell(fig, gs_cell, wide=False):
    inner = gs_cell.subgridspec(2, 1, height_ratios=[1, 0.07], hspace=0.06)
    ax = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
    wr = [0.01, 0.98, 0.01] if wide else [0.15, 0.70, 0.15]          # categorical bars need the width for their labels
    cb = inner[1, 0].subgridspec(1, 3, width_ratios=wr, wspace=0)
    return ax, fig.add_subplot(cb[0, 1])


REGION_NAMES = [('AB', 'ATLANTIC BASIN (AB = NHL + NA + SA + SO)'), ('NHL', 'NHL'), ('NA', 'NA'), ('SA', 'SA'),
                ('SO', 'SO (Atlantic sector)'), ('ocean', 'GLOBAL OCEAN (all ocean cells)')]


def find_stats_csv():
    pat = '%s/limiter_stats_%s%s_[0-9]*.csv' % (OUTPUT_DIR, SCHEME, SUFFIX)    # [0-9]: the date, so the plain pattern skips *_lv95_* files
    cands = [args.csv] if args.csv else sorted(glob.glob(pat), key=os.path.getmtime)
    return (cands[-1] if cands else None), pat


def read_stats(path):
    """(header, {(region, scope, metric, basis): (control, hose, delta)}); blank cells become NaN."""
    rows = list(csv.reader(open(path, newline='')))
    st = {}
    for r in rows[1:]:
        vals = []
        for x in r[4:7]:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(float('nan'))
        st[(r[0], r[1], r[2], r[3])] = tuple(vals)
    return rows[0], st


def print_table(path):
    """Assemblage means by region: share of biomass-months limited by each nutrient, control vs hose."""
    head, st = read_stats(path)
    ca, cb = head[4], head[5]
    m = re.search(r'_lv(\d+)', os.path.basename(path))
    thr = int(m.group(1)) / 100.0 if m else None
    cats = [c for c in NUTS + ['None'] if any(k[2] == 'frac_' + c for k in st)]
    print('\n' + '=' * 78)
    print('ASSEMBLAGE MEANS, %s, all six groups pooled, biomass x volume weighted' % SCHEME)
    print('stats file: %s' % path)
    if thr:
        print('None = nutrient factor LV >= %.2f (treated as not limited); the nutrient rows are the limited remainder' % thr)
    else:
        print('plain version: every sample is labelled by its smallest factor, including nearly unlimited ones;')
        print('see the "weight with LV >=" rows for how much of that is only weakly limited')
    print('shares are percent of biomass-months; change = hose minus control, in percentage points')
    print('=' * 78)
    for reg, name in REGION_NAMES:
        if not any(k[0] == reg for k in st):
            continue
        print('\n%s' % name)
        print('  %-34s %8s %8s %8s' % ('', ca, cb, 'change'))
        for c in cats:
            v = st.get((reg, 'ALL', 'frac_' + c, 'biomass'))
            if v is None:
                continue
            label = ('limited by ' + c) if c != 'None' else ('not limited (LV >= %.2f)' % thr if thr else 'None')
            print('  %-34s %8.1f %8.1f %+8.1f' % (label, 100 * v[0], 100 * v[1], 100 * v[2]))
        v = st.get((reg, 'ALL', 'LV', 'biomass'))
        if v:
            print('  %-34s %8.3f %8.3f %+8.3f' % ('mean limitation value LV', v[0], v[1], v[2]))
        for t in ('0.90', '0.95', '0.99'):
            v = st.get((reg, 'ALL', 'share_LV_ge_' + t, 'biomass'))
            if v:
                print('  %-34s %8.1f %8.1f %+8.1f' % ('weight with LV >= ' + t, 100 * v[0], 100 * v[1], 100 * v[2]))
        v = st.get((reg, 'ALL', 'dominant_changed', 'area'))
        if v and np.isfinite(v[2]):
            print('  %-34s %8s %8s %8.1f' % ('area where dominant limiter changed', '', '', 100 * v[2]))
    print('')


def main():
    if args.table_only:
        path, pat = find_stats_csv()
        if path is None:
            sys.exit('no stats csv matching %s (run make_limiter_maps.py in this folder, or pass --csv)' % pat)
        print_table(path)
        return
    ps.apply_figure_style()
    lon_min, lon_max, lat_min, lat_max = mm._load_extent()
    lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)
    ctx = dict(lon_m=lon_m, lat_m=lat_m, masks=masks, extent=[lon_min, lon_max, lat_min, lat_max])
    paths = {lab: rg_path(lab, fol) for lab, fol in RUNS}
    ca, cb_ = RUNS[0][0], RUNS[1][0]

    # does this file set have the None category?
    has_none = get(paths[ca], 'F_None_%s_annual' % SCHEME, quiet=True) is not None
    cats = NUTS + (['None'] if has_none else [])
    names = cats + ['mixed']
    colors = [COLORS[c] for c in names]
    MIXED = len(cats)
    print('categories: %s (mixed below %.2f)' % (names, args.mixed_below))

    def atl(lon, lat, f):
        v = mm._bbox_subset(lon, lat, f, lon_min, lon_max, lat_min, lat_max)
        return v[np.isfinite(v)]

    stamp = datetime.now().strftime('%Y%m%d')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------- figure 1: assemblage ----------------
    F = {lab: {} for lab in paths}
    for lab in paths:
        for c in cats:
            r = get(paths[lab], 'F_%s_%s_annual' % (c, SCHEME))
            if r is None:
                sys.exit('missing F_%s_%s_annual in %s: regrid the limiter file first' % (c, SCHEME, paths[lab]))
            F[lab][c] = r
    lon, lat = F[ca]['Fe'][0], F[ca]['Fe'][1]
    dom, top = {}, {}
    for lab in paths:
        dom[lab], top[lab] = dominant([F[lab][c][2] for c in cats], list(range(len(cats))), MIXED, args.mixed_below)
    diffs = {c: F[cb_][c][2] - F[ca][c][2] for c in cats}
    dmax = mm._diff_limit(np.concatenate([atl(lon, lat, d) for d in diffs.values()]))
    diff_cmap = mm._resolve_cmap(None, mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)

    ncols = len(cats) + 1
    fig = plt.figure(figsize=ps.figsize('MF_LIMNUT_assemblage', 2.5 * ncols + 0.6, 8.6))
    gs = fig.add_gridspec(2, ncols, hspace=ps.keep_gap(0.20, fig), wspace=0.03,
                          width_ratios=[1.0] * (ncols - 1) + [1.3])      # last column a bit wider for its colorbar labels
    fig.subplots_adjust(top=ps.keep_top(0.93, fig), bottom=ps.keep_bottom(0.07 if WATER_ON else 0.03, fig), left=0.03, right=0.99)
    for j, c in enumerate(cats):
        seq = LinearSegmentedColormap.from_list(c, ['#ffffff', COLORS[c]])
        for row, (data, cm, lo, hi, lab) in enumerate([
                (F[ca][c][2], seq, 0.0, 1.0, 'share of biomass-months'),
                (diffs[c], diff_cmap, -dmax, dmax, r'$\Delta$ share')]):
            ax, cax = cell(fig, gs[row, j])
            mesh = panel(ax, lon, lat, data, cm, lo, hi, ctx, '%s row %d' % (c, row),
                         letter=letter(row * ncols + j, LETTERS_ASSEMBLAGE))
            if row == 0:
                ax.set_title(c + (' (fixer: N2-fixation dependent)' if False else ''), fontweight='bold')
            if j == 0:
                ax.text(ROW_LABEL_X, 0.5, ['CONTROL', 'HOSE\u2212CONTROL'][row], transform=ax.transAxes,
                        rotation=90, ha='center', va='center', fontweight='bold')
            if mesh is None:
                cax.axis('off')
            else:
                cont_cbar(fig, mesh, cax, lab)
    cm_cat = ListedColormap(colors)
    ax, cax = cell(fig, gs[0, ncols - 1], wide=True)
    panel(ax, lon, lat, dom[ca], cm_cat, -0.5, len(names) - 0.5, ctx, 'dominant control',
          letter=letter(ncols - 1, LETTERS_ASSEMBLAGE))
    ax.set_title('Dominant limiter', fontweight='bold')
    cat_cbar(fig, cax, names, colors)
    fit_cbar_to_map(ax, cax)
    changed = np.where(np.isfinite(dom[ca]) & np.isfinite(dom[cb_]) & (dom[ca] != dom[cb_]), dom[cb_], np.nan)
    ok = np.isfinite(dom[ca]) & np.isfinite(dom[cb_])
    changed = np.where(ok & ~np.isfinite(changed), float(len(names)), changed)       # 'no change'
    names2 = names + ['n.c.']
    colors2 = colors + [COLORS['no change']]       # 'n.c.' = no change in the dominant limiter (explain in the caption)
    ax, cax = cell(fig, gs[1, ncols - 1], wide=True)
    panel(ax, lon, lat, changed, ListedColormap(colors2), -0.5, len(names2) - 0.5, ctx, 'dominant changed',
          letter=letter(2 * ncols - 1, LETTERS_ASSEMBLAGE))
    cat_cbar(fig, cax, names2, colors2)
    fit_cbar_to_map(ax, cax)
    if WATER_ON:
        fig.text(0.5, 0.008, WATERMARK, ha='center', va='bottom', fontsize=ps.font_watermark(), color='0.55')
    out = '%s/MF_LIMNUT_assemblage%s_%s.png' % (OUTPUT_DIR, SUFFIX, stamp)
    fig.savefig(out, bbox_inches='tight', dpi=ps.dpi())
    print('saved ' + out)
    plt.close(fig)

    # ---------------- figure 2: per group ----------------
    LV, FS, domg = {}, {}, {}
    for lab in paths:
        for sp in SPECIES:
            r = get(paths[lab], 'LV_%s_%s_annual' % (sp, SCHEME))
            LV[(lab, sp)] = r
            fr, codes = [], []
            for k, c in enumerate(cats):
                if c == 'Si' and sp != 'DIA':
                    continue
                g = get(paths[lab], 'F_%s_%s_%s_annual' % (c, sp, SCHEME))
                if g is not None:
                    fr.append(g[2])
                    codes.append(k)
            domg[(lab, sp)] = dominant(fr, codes, MIXED, args.mixed_below)[0] if fr else None
    if all(v is None for v in LV.values()):
        print('no per-group variables found, skipping the groups figure')
    else:
        lon, lat, _ = next(v for v in LV.values() if v is not None)
        base_all = np.concatenate([atl(lon, lat, LV[(ca, sp)][2]) for sp in SPECIES if LV[(ca, sp)] is not None])
        lv_lo, lv_hi = mm._percentile_limits(base_all, mm.BASE_PERCENTILES)
        dl = {sp: LV[(cb_, sp)][2] - LV[(ca, sp)][2] for sp in SPECIES
              if LV[(ca, sp)] is not None and LV[(cb_, sp)] is not None}
        dvl = mm._diff_limit(np.concatenate([atl(lon, lat, d) for d in dl.values()]))
        lv_base = mm._resolve_cmap(LV_BASE_CMAP, mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK)
        lv_diff = plt.get_cmap(LV_DIFF_CMAP)
        fig = plt.figure(figsize=ps.figsize('MF_LIMNUT_groups', 3.0 * len(SPECIES), 13.2))
        gs = fig.add_gridspec(4, 1, hspace=ps.keep_gap(0.16, fig))
        fig.subplots_adjust(top=ps.keep_top(0.95, fig), bottom=ps.keep_bottom(0.05 if WATER_ON else 0.02, fig), left=0.04, right=0.99)
        rows = [('LV control', 'lv', ca), (r'$\Delta$ LV (hose$-$control)', 'dlv', None),
                ('Dominant limiter, control', 'dom', ca), ('Dominant limiter, hose', 'dom', cb_)]
        for row, (rlabel, kind, lab) in enumerate(rows):
            block = gs[row].subgridspec(2, len(SPECIES), height_ratios=[1, 0.07], hspace=0.06, wspace=0.05)
            cax = fig.add_subplot(block[1, :].subgridspec(1, 3, width_ratios=[0.25, 0.5, 0.25], wspace=0)[0, 1])
            mesh = None
            for j, sp in enumerate(SPECIES):
                ax = fig.add_subplot(block[0, j], projection=mm.ATLANTIC_PROJ)
                if kind == 'lv' and LV[(ca, sp)] is not None:
                    mesh = panel(ax, lon, lat, LV[(ca, sp)][2], lv_base, lv_lo, lv_hi, ctx, 'LV ' + sp,
                                 letter=letter(row * len(SPECIES) + j, LETTERS_GROUPS)) or mesh
                elif kind == 'dlv' and sp in dl:
                    mesh = panel(ax, lon, lat, dl[sp], lv_diff, -dvl, dvl, ctx, 'dLV ' + sp,
                                 letter=letter(row * len(SPECIES) + j, LETTERS_GROUPS)) or mesh
                elif kind == 'dom' and domg[(lab, sp)] is not None:
                    panel(ax, lon, lat, domg[(lab, sp)], ListedColormap(colors), -0.5, len(names) - 0.5, ctx, 'dom ' + sp,
                          letter=letter(row * len(SPECIES) + j, LETTERS_GROUPS))
                if row == 0:
                    ax.set_title(sp + ('*' if sp == 'FIX' else ''), fontweight='bold')
                if j == 0:
                    ax.text(-0.22, 0.5, rlabel, transform=ax.transAxes, rotation=90, ha='center',
                            va='center', fontweight='bold', fontsize=ps.font_small())
            if kind == 'dom':
                cat_cbar(fig, cax, names, colors)
            elif mesh is not None:
                cont_cbar(fig, mesh, cax)
            else:
                cax.axis('off')
        fig.text(0.5, 0.025 if WATER_ON else 0.008,
                 '*FIX: an "N" label means growth that depends on N2 fixation (its N factor cannot fall below 0.56), not DIN starvation.',
                 ha='center', va='bottom', fontsize=ps.font_small(), color='0.35')
        if WATER_ON:
            fig.text(0.5, 0.005, WATERMARK, ha='center', va='bottom', fontsize=ps.font_watermark(), color='0.55')
        out = '%s/MF_LIMNUT_groups%s_%s.png' % (OUTPUT_DIR, SUFFIX, stamp)
        fig.savefig(out, bbox_inches='tight', dpi=ps.dpi())
        print('saved ' + out)
        plt.close(fig)

    # ---------------- figure 3: regional stacked bars ----------------
    path, pat = find_stats_csv()
    if path is None:
        print('no stats csv matching %s, skipping the bar figure and the table' % pat)
        return
    print_table(path)
    head, st_ = read_stats(path)
    vals = {}
    for (rg, scope, metric, basis), v in st_.items():
        if scope == 'ALL' and metric.startswith('frac_') and basis == 'biomass':
            vals[(rg, metric[5:])] = (v[0], v[1])
    regs = [r for r in ('NHL', 'NA', 'SA', 'SO', 'AB', 'ocean') if any(k[0] == r for k in vals)]
    fig, ax = plt.subplots(figsize=ps.figsize('MF_LIMNUT_bars', 1.9 * len(regs) + 1.5, 4.6))
    w = 0.36
    for i, reg in enumerate(regs):
        for k, lab in enumerate((head[4], head[5])):
            bottom = 0.0
            for c in cats:
                v = vals.get((reg, c), (0.0, 0.0))[k]
                if np.isfinite(v):
                    ax.bar(i + (k - 0.5) * (w + 0.04), v, w, bottom=bottom, color=COLORS[c], edgecolor='white', linewidth=0.5,
                           label=c if (i == 0 and k == 0) else None)
                    bottom += v
            ax.text(i + (k - 0.5) * (w + 0.04), -0.045, ['C', 'H'][k], ha='center', va='top', fontsize=ps.font_small())
    ax.set_xticks(range(len(regs)))
    ax.set_xticklabels(regs, fontweight='bold')
    ax.tick_params(axis='x', pad=22)
    ax.set_ylim(0, 1)
    ax.set_ylabel('share of biomass-months limited')
    ax.legend(ncol=len(cats), loc='upper center', bbox_to_anchor=(0.5, 1.12), frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('C = control, H = hose; assemblage, %s' % SCHEME, fontsize=ps.font_small(), color='0.4', pad=24)
    if WATER_ON:
        fig.text(0.5, 0.005, WATERMARK, ha='center', va='bottom', fontsize=ps.font_watermark(), color='0.55')
    out = '%s/MF_LIMNUT_bars%s_%s.png' % (OUTPUT_DIR, SUFFIX, stamp)
    fig.savefig(out, bbox_inches='tight', dpi=ps.dpi())
    print('saved %s (from %s)' % (out, path))


if __name__ == '__main__':
    main()
