"""
MF_ASS.py  —  Main Figure: Assemblage Composition

Default (2 rows x 3 cols):
    col 0 -- Bray-Curtis dissimilarity maps (phyto top, zoo bottom)
    col 1 -- Baseline % composition bars by region
    col 2 -- Composition shift bars by region

With --with-dom (2 rows x 5 cols):
    col 0 -- Dominant species in the control run
    col 1 -- Species that is dominant in the hose run, ONLY where the
             dominant species has flipped; everywhere else white with grey
             stippling
    col 2 -- Bray-Curtis dissimilarity maps (as before)
    col 3 -- Baseline % composition bars by region
    col 4 -- Composition shift bars by region (no legend or region labels; species colours
             are given by the legends under the dominance maps)

With --nobac, bacteria (BAC) are left out of the zooplankton row everywhere:
percent composition, Bray-Curtis, dominant species and the bars are all
computed over PRO, PTE, MES, GEL, CRU only. Species colours are unchanged,
so the two versions can be compared directly.

Usage:
    python MF_ASS.py                        # original figure
    python MF_ASS.py --with-dom             # adds the two dominance columns
    python MF_ASS.py --nobac                # zooplankton only, no bacteria
    python MF_ASS.py --with-dom --nobac

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_ASS_YYYYMMDD.png  (MF_ASS_dom_ with --with-dom, and a
        _nobac suffix with --nobac)
"""

import sys, os, argparse
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import map_maker as mm
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

parser = argparse.ArgumentParser()
parser.add_argument("--with-dom", action="store_true",
                    help="add dominant-species and dominance-flip map columns")
parser.add_argument("--nobac", action="store_true",
                    help="exclude bacteria (BAC) from the zooplankton row")
args, _ = parser.parse_known_args()
WITH_DOM = args.with_dom
NOBAC    = args.nobac

WATERMARK      = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_ASS.py"
WATERMARK_ON   = True    # False = no script-path watermark (or run with --no-watermark / --water-off)
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR     = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
FIGURE_CODE    = ("MF_ASS_dom" if WITH_DOM else "MF_ASS") + ("_nobac" if NOBAC else "")

BIOMASS_THRESH = 1e-10
SHIFT_XLIM     = 20.0 if NOBAC else 20.0   # shares are of a smaller total without BAC
BC_VMAX        = 0.7

FONT_BASE   = 16
TITLE_FONT  = 18
LABEL_FONT  = 16
TICK_FONT   = 14
LEGEND_FONT = 14

_viridis = plt.get_cmap("viridis")
_magma   = plt.get_cmap("magma")
PHYTO_SPECIES = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"]
ZOO_ALL       = ["BAC", "PRO", "PTE", "MES", "GEL", "CRU"]
ZOO_SPECIES   = [sp for sp in ZOO_ALL if not (NOBAC and sp == "BAC")]
ZOO_LABEL     = "Zooplankton" if NOBAC else "Zoo + Bacteria"
PHYTO_COLORS  = {sp: _viridis(p) for sp, p in zip(
    PHYTO_SPECIES, [0.00, 0.26, 0.47, 0.66, 0.83, 0.97])}
ZOO_COLORS    = {sp: _magma(p) for sp, p in zip(
    ZOO_ALL, [0.05, 0.32, 0.50, 0.63, 0.76, 0.90])}   # fixed per species
REGION_COLORS = mm.MASK_COLORS
REGIONS       = ["NHL", "NA", "SA", "SO"]
N_REG         = len(REGIONS)
REG_Y         = {reg: N_REG - 1 - ri for ri, reg in enumerate(REGIONS)}

STIPPLE_COLOR = "0.6"

mm.set_presentation_style()
plt.rcParams.update({"font.size": FONT_BASE})

lon_min, lon_max, lat_min, lat_max = mm._load_extent()
lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)
run_labels, runs = mm._load_models()
run_a, run_b = run_labels[0], run_labels[1]
ptrc = mm._paths_for_filetype(runs, "ptrc_T_int")

def load(paths, run, var):
    lon, lat, field, _ = mm.load_field(paths[run], var, 0)
    return lon, lat, field

eps = 1e-30
lon = lat = None
fields_a, fields_b = {}, {}
for sp in PHYTO_SPECIES + ZOO_SPECIES:
    lon, lat, fields_a[sp] = load(ptrc, run_a, sp)
    _,   _,   fields_b[sp] = load(ptrc, run_b, sp)

def pct(fields, species):
    total   = sum(fields[sp] for sp in species)
    mask    = total > BIOMASS_THRESH
    total_m = np.where(mask, total, np.nan)
    return {sp: np.where(mask, fields[sp] / total_m * 100, np.nan) for sp in species}

phy_pct_a = pct(fields_a, PHYTO_SPECIES); phy_pct_b = pct(fields_b, PHYTO_SPECIES)
zoo_pct_a = pct({sp: fields_a[sp] for sp in ZOO_SPECIES}, ZOO_SPECIES)
zoo_pct_b = pct({sp: fields_b[sp] for sp in ZOO_SPECIES}, ZOO_SPECIES)
phy_shift = {sp: phy_pct_b[sp] - phy_pct_a[sp] for sp in PHYTO_SPECIES}
zoo_shift = {sp: zoo_pct_b[sp] - zoo_pct_a[sp] for sp in ZOO_SPECIES}

def interp_mask(mask_field, lon_m, lat_m, lon_data, lat_data):
    interp = RegularGridInterpolator(
        (lat_m, lon_m), mask_field.astype(float),
        method="nearest", bounds_error=False, fill_value=0.0)
    ll  = np.meshgrid(lon_data, lat_data)
    pts = np.stack([ll[1].ravel(), ll[0].ravel()], axis=1)
    return interp(pts).reshape(lat_data.size, lon_data.size)

region_idx = np.full(fields_a[PHYTO_SPECIES[0]].shape, -1, dtype=int)
for ri, reg in enumerate(REGIONS):
    m = interp_mask(masks[reg], lon_m, lat_m, lon, lat)
    region_idx[m > 0.5] = ri

def region_means(pct_dict, species):
    return {reg: {sp: np.nanmean(pct_dict[sp][region_idx == ri])
                  for sp in species}
            for ri, reg in enumerate(REGIONS)}

phy_means_a     = region_means(phy_pct_a, PHYTO_SPECIES)
zoo_means_a     = region_means(zoo_pct_a, ZOO_SPECIES)
phy_means_shift = region_means(phy_shift,  PHYTO_SPECIES)
zoo_means_shift = region_means(zoo_shift,  ZOO_SPECIES)

def bc_map(fa, fb, species):
    num = sum(np.abs(fa[sp] - fb[sp]) for sp in species)
    den = sum(fa[sp] + fb[sp]          for sp in species)
    return num / np.where(den > eps, den, eps)

bc_phy = bc_map(fields_a, fields_b, PHYTO_SPECIES)
bc_zoo = bc_map({sp: fields_a[sp] for sp in ZOO_SPECIES},
                {sp: fields_b[sp] for sp in ZOO_SPECIES}, ZOO_SPECIES)

# ---------------------------------------------------------------------------
# Dominant species (only needed with --with-dom)
# ---------------------------------------------------------------------------
def dominant(fields, species):
    """Index (into `species`) of the largest-biomass member at each grid point,
    plus a validity mask (total biomass above threshold)."""
    arr   = np.stack([np.nan_to_num(fields[sp], nan=-1.0) for sp in species])
    total = sum(np.nan_to_num(fields[sp], nan=0.0) for sp in species)
    return np.argmax(arr, axis=0), total > BIOMASS_THRESH

def dominance_fields(species):
    idx_a, ok_a = dominant(fields_a, species)
    idx_b, ok_b = dominant(fields_b, species)
    valid     = ok_a & ok_b
    flipped   = valid & (idx_a != idx_b)
    unchanged = valid & (idx_a == idx_b)
    return {
        "control":   np.where(ok_a, idx_a, np.nan).astype(float),
        "new_dom":   np.where(flipped, idx_b, np.nan).astype(float),
        "unchanged": unchanged,
        "valid":     valid,
    }

heat_cmap = mm._resolve_cmap("heat", mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
if WITH_DOM:
    COL_DOM, COL_FLIP, COL_BC, COL_BASE, COL_SHIFT = 0, 1, 2, 3, 4
    fig = plt.figure(figsize=(11, 8))
    # maps: three tight columns sized to the (height-limited) map aspect
    gs_maps = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.05,
                               left=0.04, right=0.60, top=0.90, bottom=0.10)
    # bars: own grid so the gap left of the baseline bars can hold the
    # region labels without widening the gaps between the maps
    gs_bars = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.85],
                               hspace=0.22, wspace=0.30,
                               left=0.665, right=0.975, top=0.90, bottom=0.10)
    def cell(r, c):
        return gs_maps[r, c] if c < 3 else gs_bars[r, c - 3]
else:
    COL_BC, COL_BASE, COL_SHIFT = 0, 1, 2
    fig = plt.figure(figsize=(12, 8))
    main_gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.4, 1.4],
                               hspace=0.22, wspace=0.32,
                               left=0.06, right=0.97, top=0.90, bottom=0.10)
    def cell(r, c):
        return main_gs[r, c]

MAP_EXTENT = [lon_min, lon_max, lat_min, lat_max]

def make_map_axes(gs_cell):
    inner = gs_cell.subgridspec(2, 1, height_ratios=[1, 0.06], hspace=0.08)
    ax    = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
    return ax, inner[1, 0]

def add_row_label(ax, row_label):
    if row_label:
        ax.text(-0.14, 0.5, row_label, transform=ax.transAxes,
                rotation=90, ha="center", va="center",
                fontweight="bold", fontsize=LABEL_FONT)

def draw_bc_map(gs_cell, field, row_label=None, show_cbar=True):
    ax, slot = make_map_axes(gs_cell)
    cb_w     = [0.2, 0.6, 0.2] if WITH_DOM else [0.1, 0.8, 0.1]
    cbar_gs  = slot.subgridspec(1, 3, width_ratios=cb_w, wspace=0)
    cax      = fig.add_subplot(cbar_gs[0, 1]) if show_cbar else None
    mesh     = mm._draw_panel(ax, lon, lat, field, heat_cmap, 0, BC_VMAX,
                              lon_m, lat_m, masks, extent=MAP_EXTENT)
    add_row_label(ax, row_label)
    if not show_cbar:
        return
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.set_ticks([0, BC_VMAX / 2, BC_VMAX])
    cb_font = TICK_FONT - 3 if WITH_DOM else TICK_FONT
    cbar.ax.tick_params(labelsize=cb_font)
    cbar.set_label("BC dissimilarity", fontsize=cb_font, labelpad=2)

def species_cmap(species, colors):
    """Discrete colormap: index i -> colour of species[i]."""
    return ListedColormap([colors[sp] for sp in species])

def draw_dom_map(gs_cell, field, species, colors, row_label=None,
                 legend=True):
    """Map of species index. Legend of species colours sits under the map."""
    ax, slot = make_map_axes(gs_cell)
    cmap = species_cmap(species, colors)
    mm._draw_panel(ax, lon, lat, field, cmap, -0.5, len(species) - 0.5,
                   lon_m, lat_m, masks, extent=MAP_EXTENT)
    add_row_label(ax, row_label)
    lax = fig.add_subplot(slot)
    lax.axis("off")
    if legend:
        handles = [mpatches.Patch(color=colors[sp], label=sp) for sp in species]
        lax.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                   fontsize=LEGEND_FONT - 3, handlelength=1.0,
                   columnspacing=0.8, borderaxespad=0.0)
    return ax, lax

def draw_flip_map(gs_cell, dom, species, colors):
    """New dominant species where dominance flipped; white + grey stipple
    everywhere else (where the dominant species is unchanged)."""
    ax, slot = make_map_axes(gs_cell)
    cmap = species_cmap(species, colors)
    # white base under all valid ocean cells (drawn first, lowest zorder)
    ax.pcolormesh(lon, lat, np.where(dom["valid"], 1.0, np.nan),
                  cmap=ListedColormap(["white"]), vmin=0, vmax=1,
                  transform=ccrs.PlateCarree(), shading="auto", zorder=0)
    # flipped cells in the colour of the NEW dominant species
    mm._draw_panel(ax, lon, lat, dom["new_dom"], cmap, -0.5, len(species) - 0.5,
                   lon_m, lat_m, masks, extent=MAP_EXTENT)
    # grey stipple where nothing flipped
    unchanged = np.ma.masked_invalid(np.where(dom["unchanged"], 1.0, np.nan))
    with plt.rc_context({"hatch.color": STIPPLE_COLOR, "hatch.linewidth": 0.5}):
        ax.contourf(lon, lat, unchanged, levels=[0.5, 1.5], colors="none",
                    hatches=["...."], transform=ccrs.PlateCarree(), zorder=2)
    lax = fig.add_subplot(slot)
    lax.axis("off")
    stipple_patch = mpatches.Patch(facecolor="white", edgecolor=STIPPLE_COLOR,
                                   hatch="....", label="dominant unchanged")
    lax.legend(handles=[stipple_patch], loc="upper center", frameon=False,
               fontsize=LEGEND_FONT - 3, handlelength=1.6, borderaxespad=0.0)
    return ax

phy_dom = zoo_dom = None
if WITH_DOM:
    phy_dom = dominance_fields(PHYTO_SPECIES)
    zoo_dom = dominance_fields(ZOO_SPECIES)

    draw_dom_map(cell(0, COL_DOM),  phy_dom["control"], PHYTO_SPECIES,
                 PHYTO_COLORS, row_label="Phytoplankton")
    draw_dom_map(cell(1, COL_DOM),  zoo_dom["control"], ZOO_SPECIES,
                 ZOO_COLORS,   row_label=ZOO_LABEL)
    draw_flip_map(cell(0, COL_FLIP), phy_dom, PHYTO_SPECIES, PHYTO_COLORS)
    draw_flip_map(cell(1, COL_FLIP), zoo_dom, ZOO_SPECIES,       ZOO_COLORS)

    draw_bc_map(cell(0, COL_BC), bc_phy, show_cbar=False)  # one shared colorbar
    draw_bc_map(cell(1, COL_BC), bc_zoo)
else:
    draw_bc_map(cell(0, COL_BC), bc_phy, row_label="Phytoplankton")
    draw_bc_map(cell(1, COL_BC), bc_zoo, row_label=ZOO_LABEL)

def draw_bar_panel(gs_cell, species, means, colors, is_shift,
                   xlim=None, show_legend=False, show_ylabels=True):
    ax = fig.add_subplot(gs_cell)
    for ri, reg in enumerate(REGIONS):
        y = REG_Y[reg]
        if is_shift:
            pos_left = neg_left = 0.0
            for sp in species:
                delta = means[reg][sp]
                if np.isnan(delta): continue
                if delta >= 0:
                    ax.barh(y, delta, left=pos_left, color=colors[sp], height=0.65, linewidth=0)
                    pos_left += delta
                else:
                    ax.barh(y, delta, left=neg_left, color=colors[sp], height=0.65, linewidth=0)
                    neg_left += delta
        else:
            left = 0.0
            for sp in species:
                val = means[reg][sp]
                if np.isnan(val): continue
                ax.barh(y, val, left=left, color=colors[sp], height=0.65, linewidth=0)
                left += val

    if show_ylabels:
        y_positions = [REG_Y[reg] for reg in REGIONS]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(REGIONS, fontsize=TICK_FONT + 1)
        for tick, reg in zip(ax.get_yticklabels(), REGIONS):
            tick.set_color(REGION_COLORS[reg]); tick.set_fontweight("bold")
    else:
        ax.set_yticks([])

    if is_shift:
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlim(-xlim, xlim)
        ax.set_xticks([-xlim, 0, xlim])
        ax.set_xticklabels([f"\u2212{xlim:.0f}%", "0", f"+{xlim:.0f}%"], fontsize=TICK_FONT)
    else:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100%"], fontsize=TICK_FONT)

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, N_REG - 0.4)

    if show_legend:
        handles = [mpatches.Patch(color=colors[sp], label=sp) for sp in species]
        # store handles for figure-level legend placed after all panels are drawn
        ax._legend_handles = handles
    return ax

ax_phy_base  = draw_bar_panel(cell(0, COL_BASE),  PHYTO_SPECIES, phy_means_a,     PHYTO_COLORS, is_shift=False)
ax_phy_shift = draw_bar_panel(cell(0, COL_SHIFT), PHYTO_SPECIES, phy_means_shift, PHYTO_COLORS, is_shift=True,  xlim=SHIFT_XLIM, show_legend=not WITH_DOM, show_ylabels=not WITH_DOM)
ax_zoo_base  = draw_bar_panel(cell(1, COL_BASE),  ZOO_SPECIES,       zoo_means_a,     ZOO_COLORS,   is_shift=False)
ax_zoo_shift = draw_bar_panel(cell(1, COL_SHIFT), ZOO_SPECIES,       zoo_means_shift, ZOO_COLORS,   is_shift=True,  xlim=SHIFT_XLIM, show_legend=not WITH_DOM, show_ylabels=not WITH_DOM)

ax_phy_base.get_shared_x_axes().join(ax_phy_base,  ax_zoo_base)
ax_phy_shift.get_shared_x_axes().join(ax_phy_shift, ax_zoo_shift)
plt.setp(ax_phy_base.get_xticklabels(),  visible=False)
plt.setp(ax_phy_shift.get_xticklabels(), visible=False)
if WITH_DOM:
    for ax_top in (ax_phy_base, ax_phy_shift):
        ax_top.tick_params(axis="x", length=0)

# figure-level legends anchored to lower-right of each shift panel
for ax_shift in [ax_phy_shift, ax_zoo_shift]:
    if hasattr(ax_shift, "_legend_handles"):
        ax_shift.legend(handles=ax_shift._legend_handles,
                        loc="lower right",
                        bbox_to_anchor=(1.0, -0.02),
                        frameon=False, fontsize=LEGEND_FONT,
                        ncol=2, columnspacing=0.8, handlelength=1.2)

# column titles
if WITH_DOM:
    col_titles = {COL_DOM:   "Dominant Species\n(CONTROL)",
                  COL_FLIP:  "New Dominant\n(HOSE)",
                  COL_BC:    "Bray-Curtis Index",
                  COL_BASE:  "Baseline Composition",
                  COL_SHIFT: "Shift"}
    for col, title in col_titles.items():
        bbox = cell(0, col).get_position(fig)
        fig.text((bbox.x0 + bbox.x1) / 2, 0.93, title, ha="center", va="bottom",
                 fontsize=TITLE_FONT, fontweight="bold")
else:
    for x_pos, title in zip([0.16, 0.50, 0.80],
                             ["Bray-Curtis Index", "Baseline Composition", "Composition Shift"]):
        fig.text(x_pos, 0.93, title, ha="center", va="bottom",
                 fontsize=TITLE_FONT, fontweight="bold")

if WATERMARK_ON:
    fig.text(0.5, 0.02, WATERMARK, ha="center", va="bottom",
             fontsize=FONT_BASE - 4, color="0.55")

os.makedirs(OUTPUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d")
out = f"{OUTPUT_DIR}/{FIGURE_CODE}_{stamp}.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
print(f"saved {out}")
plt.show()
