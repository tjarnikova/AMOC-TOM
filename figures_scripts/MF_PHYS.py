"""
MF_PHYS.py  —  Main Figure: Physical Response

2 rows x 4 cols (3 cols with --no-ice), Atlantic-focused maps.
    col 0 -- tos      (SST, cmap thermal)
    col 1 -- sos      (SSS, cmap haline, BASE row fixed to 30-37 psu)
    col 2 -- mldr10_1 (MLD, cmap deep)
    col 3 -- icethic  (sea-ice thickness, cmap ice) -- left out with --no-ice
    row 0 -- CONTROL
    row 1 -- HOSE-CONTROL

The sea-ice thickness column is included by default. Run with `--no-ice` to leave it
out. (`--with-ice` is still accepted and does nothing, so old commands keep working.)

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_PHYS_YYYYMMDD.png (or MF_PHYS-noice_YYYYMMDD.png with --no-ice)
"""

import os
import sys
import argparse
import datetime

import matplotlib.pyplot as plt
import map_maker as mm
import plot_style as ps

# ── tunable ────────────────────────────────────────────────────────────
WATERMARK      = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_PHYS.py"
WATERMARK_ON   = ps.watermark_from_settings()   # figure_settings.yml; --no-watermark / --water-off switch it off
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR     = ps.output_dir()   # figure_settings.yml
FIGURE_CODE    = "MF_PHYS"

GRID_T_FILETYPE = "grid_T"
ICEMOD_FILETYPE = "icemod"

ROW_LABELS  = ["CONTROL", "HOSE\u2212CONTROL"]
MAP_BORDER_LW = 1.5      # study-region outlines on the maps (pt); was 1.0 from figure_settings.yml
ROW_LABEL_X = -0.18

# (variable, cmap, display title, units, (vmin,vmax) of the CONTROL row or None for automatic,
#  (vmin,vmax) of the HOSE-CONTROL row or None for automatic)
COLUMNS = [
    ("tos",      "thermal", "SST",  "\u00b0C", None,      (-1.5, 1.5)),
    ("sos",      "haline",  "SSS",  "psu",     (32, 39),  None),
    ("mldr10_1", "deep",    "MLD",  "m",       (0, 200),  (-50, 50)),
]

# sea-ice column, included unless --no-ice is passed
ICE_COLUMN = ("icethic", "ice", "Sea ice thickness", "m", None, None)

# ── setup ──────────────────────────────────────────────────────────────
def make_figure(watermark=True, with_ice=True):
    ps.apply_figure_style()
    mm.MASK_LINEWIDTHS = {r: MAP_BORDER_LW for r in mm.MASK_REGIONS}   # thicker outlines, this figure only

    columns = COLUMNS + [ICE_COLUMN] if with_ice else COLUMNS

    lon_min, lon_max, lat_min, lat_max = mm._load_extent()
    lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)
    run_labels, runs = mm._load_models()
    run_a, run_b = run_labels[0], run_labels[1]
    paths = mm._paths_for_filetype(runs, GRID_T_FILETYPE)
    ice_paths = mm._paths_for_filetype(runs, ICEMOD_FILETYPE) if with_ice else None

    n_cols = len(columns)
    fig = plt.figure(figsize=ps.figsize(FIGURE_CODE, 3.4 * n_cols, 9.0))
    outer_gs = fig.add_gridspec(2, n_cols, height_ratios=[3, 3],
                                 hspace=ps.keep_gap(0.18, fig), wspace=0.06)
    fig.subplots_adjust(top=1 - 0.30 / fig.get_figheight(), bottom=ps.keep_bottom(0.07 if (watermark and WATERMARK_ON) else 0.03, fig),
                        left=0.06, right=0.99)

    diff_cmap = mm._resolve_cmap(None, mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)

    def col_paths(var):
        return ice_paths if var == ICE_COLUMN[0] else paths

    for ci, (var, cmap_name, title, units, base_override, diff_override) in enumerate(columns):
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
        if diff_override is not None:
            dlo, dhi = diff_override
        else:
            dvmax = mm._diff_limit(atl_d)
            dlo, dhi = -dvmax, dvmax

        row_specs = [
            (field_a, cmap,      vmin, vmax),
            (diff,    diff_cmap, dlo,  dhi),
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
                        ha="center", va="center", fontsize=ps.font_small(), color="0.5")
                continue

            ps.panel_letter_inside(ax, "%s)" % "abcdefgh"[row * n_cols + ci])
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
            cbar.ax.tick_params(labelsize=ps.font_small())
            cbar.set_label(units, fontsize=ps.font_small())

    if watermark and WATERMARK_ON:
        fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom",
                 fontsize=ps.font_watermark(), color="0.55")

    stamp = datetime.date.today().strftime("%Y%m%d")
    return fig, f"{FIGURE_CODE}_{stamp}.png"


def save_figure(watermark=True, dpi=None, with_ice=True):
    dpi = dpi or ps.dpi()
    fig, fname = make_figure(watermark=watermark, with_ice=with_ice)
    if not with_ice:
        fname = fname.replace(f"{FIGURE_CODE}_", f"{FIGURE_CODE}-noice_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    print(f"saved {out}")
    return out


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-ice", action="store_true",
                    help="leave out the 4th column (sea-ice thickness, icethic)")
    p.add_argument("--with-ice", action="store_true", help=argparse.SUPPRESS)   # old flag, now the default
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    save_figure(with_ice=not args.no_ice)
