"""
MF_PRIM.py  —  Main Figure: Primary Production and Limitation

2 rows x 5 cols:
    col 0    -- Phytoplankton biomass  (sum of the six phytoplankton groups)
    col 1    -- Integrated primary production
    cols 2-4 -- Nutrient / Light / Temperature limitation
    row 0    -- CONTROL
    row 1    -- HOSE-CONTROL

Self-contained: needs only map_maker.py (no figA_phyto_overview).

Data
    biomass : DIA+MIX+COC+PIC+PHA+FIX from ORCA2_7d_clim_<period>_ptrc_T_int_rg.nc
              (molC/m2 after fix_int_units.py; shown as mmol C m-2)
    PP      : PPT from ORCA2_7d_clim_<period>_diad_T_int_rg.nc (mol C m-2 s-1),
              the same field MF_FLUX and MF_ZOO use
    limitation : HA01_/HH01_limitation_maps_rg.nc (unchanged)

The biomass panel refuses to draw if the file has no units attribute or an
unexpected one: an unlabelled ptrc_T_int is molC/L*m, 1000x too small. Run
fix_int_units.py on the clim folders first.

Style reference for all other figures in this project.
Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_PRIM_YYYYMMDD.png
"""

import sys
import os
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import map_maker as mm

# ── tunable ────────────────────────────────────────────────────────────
WATERMARK   = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_PRIM.py"
WATERMARK_ON   = True    # False = no script-path watermark (or run with --no-watermark / --water-off)
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR  = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
FIGURE_CODE = "MF_PRIM"

ROW_LABELS  = ["CONTROL", "HOSE\u2212CONTROL"]
ROW_LABEL_X = -0.18

PHYTO_SPECIES = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"]
PP_VAR        = "PPT"

BIOMASS_CMAP = "algae"     # cmocean names; mm._resolve_cmap falls back to viridis
PP_CMAP      = "speed"

LIM_BASE_CMAP_NAME = "Greys_r"
LIM_DIFF_CMAP_NAME = "PuOr"
LIM_DIFF_VMAX      = 0.22

LIM_RUNS = ("HA01", "HH01")
LIM_RUN_PATHS = {
    "HA01": "/gpfs/data/greenocean/users/mep22dku/clims/TOM12_TJ_HA01/HA01_limitation_maps_rg.nc",
    "HH01": "/gpfs/data/greenocean/users/mep22dku/clims/TOM12_TJ_HH01/HH01_limitation_maps_rg.nc",
}
LIM_COLUMNS = [
    ("L_nut_top14_annual",   "Nutrient lim."),
    ("L_light_top14_annual", "Light lim."),
    ("L_temp_top14_annual",  "Temperature lim."),
]
SHARED_CBAR_WIDTH_RATIOS = [0.20, 0.60, 0.20]


# ── data loading ───────────────────────────────────────────────────────
def _mmol_from_units(units, label):
    """Factor that converts a depth-integrated carbon stock with the given
    units attribute to mmol C m-2. Raises if the units are missing or
    unexpected, so a wrong-by-1000 figure can never be drawn silently."""
    u = (units or "").replace(" ", "").lower()
    if u in ("molc/m2", "molc/m^2", "mol/m2"):
        return 1e3
    if u in ("mmolc/m2", "mmol/m2"):
        return 1.0
    raise ValueError(
        f"{label}: units attribute is {units!r}, expected 'molC/m2' (or 'mmolC/m2'). "
        "Files without units are molC/L*m (1000x too small): run fix_int_units.py "
        "on the clim folders first.")


def load_biomass(path):
    """Sum of the six phytoplankton groups, annual mean, in mmol C m-2."""
    total, lon, lat, units_seen = None, None, None, set()
    for sp in PHYTO_SPECIES:
        lon, lat, f, meta = mm.load_field(path, sp, 0)
        units_seen.add(meta["units"])
        total = f if total is None else total + f
    if len(units_seen) != 1:
        raise ValueError(f"phytoplankton groups have different units in {path}: {units_seen}")
    factor = _mmol_from_units(units_seen.pop(), os.path.basename(path))
    return lon, lat, total * factor


def load_pp(path):
    lon, lat, f, _ = mm.load_field(path, PP_VAR, 0)
    return lon, lat, f


# ── plotting helpers ───────────────────────────────────────────────────
def _draw_or_blank(ax, cax, lon, lat, data, cmap, vmin, vmax,
                   lon_m, lat_m, masks, extent, tag):
    try:
        return mm._draw_panel(ax, lon, lat, data, cmap, vmin, vmax,
                              lon_m, lat_m, masks, extent=extent)
    except Exception as exc:
        print(f"    [warn] {tag} failed ({exc!r}) -- leaving blank")
        ax.clear(); ax.axis("off")
        ax.text(0.5, 0.5, "render failed", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="0.5")
        if cax is not None:
            cax.axis("off")
        return None


def _style_cbar(cbar):
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=plt.rcParams["font.size"] - 2)


def make_figure(watermark=True):
    mm.set_presentation_style()

    lon_min, lon_max, lat_min, lat_max = mm._load_extent()
    extent = [lon_min, lon_max, lat_min, lat_max]
    lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)

    run_labels, runs = mm._load_models()
    run_a, run_b = run_labels[0], run_labels[1]
    ptrc = mm._paths_for_filetype(runs, "ptrc_T_int")
    diad = mm._paths_for_filetype(runs, "diad_T_int")

    lon, lat, bio_a = load_biomass(ptrc[run_a]); _, _, bio_b = load_biomass(ptrc[run_b])
    _,   _,   pp_a  = load_pp(diad[run_a]);      _, _, pp_b  = load_pp(diad[run_b])

    def atl(field):
        v = mm._bbox_subset(lon, lat, field, lon_min, lon_max, lat_min, lat_max)
        return v[np.isfinite(v)]

    diff_cmap = mm._resolve_cmap(None, mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)
    PHYTO_COLUMNS = [
        dict(title="Phytoplankton biomass", ctrl=bio_a, diff=bio_b - bio_a,
             cmap=mm._resolve_cmap(BIOMASS_CMAP, mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK),
             units=r"mmol C m$^{-2}$", diff_units=r"$\Delta$ mmol C m$^{-2}$"),
        dict(title="Integrated PP", ctrl=pp_a, diff=pp_b - pp_a,
             cmap=mm._resolve_cmap(PP_CMAP, mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK),
             units=r"mol C m$^{-2}$ s$^{-1}$", diff_units=r"$\Delta$ mol C m$^{-2}$ s$^{-1}$"),
    ]

    n_phyto = len(PHYTO_COLUMNS)
    n_lim   = len(LIM_COLUMNS)
    n_cols  = n_phyto + n_lim

    fig = plt.figure(figsize=(3.4 * n_cols, 9.0))
    outer_gs = fig.add_gridspec(2, n_cols, height_ratios=[3, 3],
                                 hspace=0.18, wspace=0.06)
    fig.subplots_adjust(top=0.94, bottom=0.07 if (watermark and WATERMARK_ON) else 0.03,
                        left=0.03, right=0.99)

    # phyto axes
    ax_phyto  = [[None] * n_phyto for _ in range(2)]
    cax_phyto = [[None] * n_phyto for _ in range(2)]
    for j in range(n_phyto):
        for row in range(2):
            inner = outer_gs[row, j].subgridspec(
                2, 1, height_ratios=[1, 0.06], hspace=0.06)
            ax_phyto[row][j]  = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
            cbar_gs = inner[1, 0].subgridspec(1, 3, width_ratios=[0.15, 0.70, 0.15], wspace=0)
            cax_phyto[row][j] = fig.add_subplot(cbar_gs[0, 1])

    # limitation axes
    ax_lim  = [[None] * n_lim for _ in range(2)]
    cax_lim = [None, None]
    for row in range(2):
        block = outer_gs[row, n_phyto:].subgridspec(
            2, n_lim, height_ratios=[1, 0.06], hspace=0.06, wspace=0.06)
        for k in range(n_lim):
            ax_lim[row][k] = fig.add_subplot(block[0, k], projection=mm.ATLANTIC_PROJ)
        cbar_gs = block[1, :].subgridspec(
            1, 3, width_ratios=SHARED_CBAR_WIDTH_RATIOS, wspace=0)
        cax_lim[row] = fig.add_subplot(cbar_gs[0, 1])

    # ── phyto columns ─────────────────────────────────────────────────
    for j, col in enumerate(PHYTO_COLUMNS):
        vmin, vmax = mm._percentile_limits(atl(col["ctrl"]), mm.BASE_PERCENTILES)
        dvmax = mm._diff_limit(atl(col["diff"]))
        for row, (data, cm, lo, hi, units) in enumerate([
            (col["ctrl"], col["cmap"], vmin,   vmax,  col["units"]),
            (col["diff"], diff_cmap,  -dvmax,  dvmax, col["diff_units"]),
        ]):
            ax, cax = ax_phyto[row][j], cax_phyto[row][j]
            mesh = _draw_or_blank(ax, cax, lon, lat, data, cm, lo, hi,
                                  lon_m, lat_m, masks, extent, f"{col['title']} row {row}")
            if row == 0:
                ax.set_title(col["title"], fontweight="bold")
            if mesh is None:
                continue
            cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
            _style_cbar(cbar)
            cbar.formatter = mm._value_formatter()
            cbar.update_ticks()
            cbar.set_label(units, fontsize=plt.rcParams["font.size"] - 2)

    # ── limitation columns ─────────────────────────────────────────────
    lim_a, lim_b = LIM_RUNS
    loaded = []
    for var, label in LIM_COLUMNS:
        lon_l, lat_l, fa, _ = mm.load_field(LIM_RUN_PATHS[lim_a], var, 0)
        _,     _,     fb, _ = mm.load_field(LIM_RUN_PATHS[lim_b], var, 0)
        loaded.append((var, label, lon_l, lat_l, fa, fb - fa))

    base_vals_all = np.concatenate([
        mm._bbox_subset(lo, la, fa, lon_min, lon_max, lat_min, lat_max).ravel()
        for _, _, lo, la, fa, _ in loaded])
    lim_vmin, lim_vmax = mm._percentile_limits(base_vals_all, mm.BASE_PERCENTILES)

    lim_base_cmap = mm._resolve_cmap(LIM_BASE_CMAP_NAME, mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK)
    lim_diff_cmap = plt.get_cmap(LIM_DIFF_CMAP_NAME)
    lim_specs = [
        (lim_base_cmap, lim_vmin,     lim_vmax),
        (lim_diff_cmap, -LIM_DIFF_VMAX, LIM_DIFF_VMAX),
    ]
    shared_mesh = [None, None]

    for k, (var, label, lon_l, lat_l, fa, diff) in enumerate(loaded):
        for row, data in enumerate((fa, diff)):
            cm, lo, hi = lim_specs[row]
            ax   = ax_lim[row][k]
            mesh = _draw_or_blank(ax, None, lon_l, lat_l, data, cm, lo, hi,
                                  lon_m, lat_m, masks, extent, f"{var} row {row}")
            if row == 0:
                ax.set_title(label, fontweight="bold")
            if mesh is not None:
                shared_mesh[row] = mesh

    for row in range(2):
        if shared_mesh[row] is None:
            cax_lim[row].axis("off")
            continue
        cbar = fig.colorbar(shared_mesh[row], cax=cax_lim[row], orientation="horizontal")
        _style_cbar(cbar)
        cbar.formatter = mm._value_formatter()
        cbar.update_ticks()

    # ── row labels + watermark ─────────────────────────────────────────
    for row in range(2):
        ax_phyto[row][0].text(ROW_LABEL_X, 0.5, ROW_LABELS[row],
                               transform=ax_phyto[row][0].transAxes,
                               rotation=90, ha="center", va="center",
                               fontweight="bold")
    if watermark and WATERMARK_ON:
        fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom",
                 fontsize=plt.rcParams["font.size"] - 6, color="0.55")

    stamp = datetime.now().strftime("%Y%m%d")
    return fig, f"{FIGURE_CODE}_{stamp}.png"


if __name__ == "__main__":
    fig, fname = make_figure(watermark=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")
