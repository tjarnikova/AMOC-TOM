"""
MF_PHYS.py  —  Main Figure: Physical Response

2 rows x 3 cols (or 4 cols with --with-ice), Atlantic-focused maps.
    col 0 -- tos      (SST, cmap thermal)
    col 1 -- sos      (SSS, cmap haline, BASE row fixed to 30-37 psu)
    col 2 -- mldr10_1 (MLD, cmap deep)
    col 3 -- icethic  (sea-ice thickness, cmap ice) -- only with --with-ice
    row 0 -- CONTROL
    row 1 -- HOSE-CONTROL

Run with `--with-ice` to append the sea-ice thickness column; omitted by
default since it's still undecided whether ice belongs in this figure.

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_PHYS_YYYYMMDD.png (or MF_PHYS-ice_YYYYMMDD.png with --with-ice)
"""

import os
import sys
import argparse
import datetime
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')

import matplotlib.pyplot as plt
import map_maker as mm

# ── tunable ────────────────────────────────────────────────────────────
WATERMARK      = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_PHYS.py"
WATERMARK_ON   = True    # False = no script-path watermark (or run with --no-watermark / --water-off)
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR     = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
FIGURE_CODE    = "MF_PHYS"

GRID_T_FILETYPE = "grid_T"
ICEMOD_FILETYPE = "icemod"

ROW_LABELS  = ["CONTROL", "HOSE\u2212CONTROL"]
ROW_LABEL_X = -0.18

# (variable, cmap, display title, units, (vmin,vmax) BASE override or None)
COLUMNS = [
    ("tos",      "thermal", "SST",  "\u00b0C", None),
    ("sos",      "haline",  "SSS",  "psu",     (30, 37)),
    ("mldr10_1", "deep",    "MLD",  "m",       None),
]

# optional extra column, added only when --with-ice is passed
ICE_COLUMN = ("icethic", "ice", "Sea ice thickness", "m", None)

# ── setup ──────────────────────────────────────────────────────────────
def make_figure(watermark=True, with_ice=False):
    mm.set_presentation_style()

    columns = COLUMNS + [ICE_COLUMN] if with_ice else COLUMNS

    lon_min, lon_max, lat_min, lat_max = mm._load_extent()
    lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)
    run_labels, runs = mm._load_models()
    run_a, run_b = run_labels[0], run_labels[1]
    paths = mm._paths_for_filetype(runs, GRID_T_FILETYPE)
    ice_paths = mm._paths_for_filetype(runs, ICEMOD_FILETYPE) if with_ice else None

    n_cols = len(columns)
    fig = plt.figure(figsize=(3.4 * n_cols, 9.0))
    outer_gs = fig.add_gridspec(2, n_cols, height_ratios=[3, 3],
                                 hspace=0.18, wspace=0.06)
    fig.subplots_adjust(top=0.94, bottom=0.07 if (watermark and WATERMARK_ON) else 0.03,
                        left=0.06, right=0.99)

    diff_cmap = mm._resolve_cmap(None, mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)

    def col_paths(var):
        return ice_paths if var == ICE_COLUMN[0] else paths

    for ci, (var, cmap_name, title, units, base_override) in enumerate(columns):
        cmap = mm._resolve_cmap(cmap_name, mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK)

        lon, lat, field_a, _ = mm.load_field(col_paths(var)[run_a], var, 0)
        _,   _,   field_b, _ = mm.load_field(col_paths(var)[run_b], var, 0)
        diff = field_b - field_a

        atl_a = mm._bbox_subset(lon, lat, field_a, lon_min, lon_max, lat_min, lat_max)
        atl_d = mm._bbox_subset(lon, lat, diff,    lon_min, lon_max, lat_min, lat_max)

        if base_override is not None:
            vmin, vmax = base_override
        else:
            vmin, vmax = mm._percentile_limits(atl_a, mm.BASE_PERCENTILES)
        dvmax = mm._diff_limit(atl_d)

        row_specs = [
            (field_a, cmap,      vmin,   vmax),
            (diff,    diff_cmap, -dvmax, dvmax),
        ]

        for row, (field, cm, lo, hi) in enumerate(row_specs):
            inner = outer_gs[row, ci].subgridspec(
                2, 1, height_ratios=[1, 0.06], hspace=0.06)
            ax  = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
            cbar_gs = inner[1, 0].subgridspec(
                1, 3, width_ratios=[0.15, 0.70, 0.15], wspace=0)
            cax = fig.add_subplot(cbar_gs[0, 1])

            try:
                mesh = mm._draw_panel(ax, lon, lat, field, cm, lo, hi,
                                       lon_m, lat_m, masks,
                                       extent=[lon_min, lon_max, lat_min, lat_max])
            except Exception as exc:
                print(f"[warn] {var} row {row} failed ({exc!r}) -- leaving blank")
                ax.clear(); ax.axis("off"); cax.axis("off")
                ax.text(0.5, 0.5, "render failed", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="0.5")
                continue

            if row == 0:
                ax.set_title(title, fontweight="bold")
            if ci == 0:
                ax.text(ROW_LABEL_X, 0.5, ROW_LABELS[row],
                        transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontweight="bold")

            cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
            cbar.outline.set_visible(False)
            cbar.formatter = mm._value_formatter()
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=plt.rcParams["font.size"] - 2)
            cbar.set_label(units, fontsize=plt.rcParams["font.size"] - 2)

    if watermark and WATERMARK_ON:
        fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom",
                 fontsize=plt.rcParams["font.size"] - 6, color="0.55")

    stamp = datetime.date.today().strftime("%Y%m%d")
    return fig, f"{FIGURE_CODE}_{stamp}.png"


def save_figure(watermark=True, dpi=300, with_ice=False):
    fig, fname = make_figure(watermark=watermark, with_ice=with_ice)
    if with_ice:
        fname = fname.replace(f"{FIGURE_CODE}_", f"{FIGURE_CODE}-ice_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    print(f"saved {out}")
    return out


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-ice", action="store_true",
                    help="add a 4th column for sea-ice thickness (icethic)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    save_figure(with_ice=args.with_ice)
