"""
SF_GROWTH_TEMP.py

Supplementary figure: growth rate vs temperature.
One figure, two side-by-side panels (shared x and y axes):
    (a) phytoplankton: DIA, MIX, COC, PIC, PHA, FIX
    (b) zooplankton + bacteria: BAC, PRO, PTE, MES, GEL, CRU

Adapted from plot_growth_vs_temp.py (same CSV reading, same de-duplication).

DATA
    One CSV per PFT, {SPECIES}_growth.csv, in a folder (default PFT_growth_rates/
    next to this script; override with --data-dir).
    Columns used: Temperature, OBio (change Y_COLUMN below to plot GCB or SMor).
    OBio is a function of temperature only, so duplicate (T, OBio) rows are
    dropped and the rest sorted by T before plotting.

COLOURS
    Kept in ONE block (PHYTO_COLOR_POSITIONS / ZOO_COLOR_POSITIONS below).
    They are the viridis / magma positions from the old test script. If MF_ASS.py
    uses something different, edit that block so the two figures match.

Run from /gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts:
    python SF_GROWTH_TEMP.py
    python SF_GROWTH_TEMP.py --data-dir /path/to/PFT_growth_rates
    python SF_GROWTH_TEMP.py --no-watermark

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/SF_GROWTH_TEMP_<YYYYMMDD>.png (300 dpi)
"""

import os
import sys
import argparse
import datetime

sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/UTILS')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from plot_style import set_presentation_style
except ImportError:  # only so the script can be tried off the cluster
    print("WARNING: plot_style not found, using matplotlib defaults")

    def set_presentation_style():
        plt.rcParams.update({"font.size": 14})

# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PFT_growth_rates")
FILENAME_TEMPLATE = "{species}_growth.csv"
X_COLUMN = "Temperature"
Y_COLUMN = "OBio"

OUTPUT_DIR = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
CODE = "SF_GROWTH_TEMP"
WATERMARK_BASE_PATH = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts"

PHYTO_SPECIES = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"]
ZOO_SPECIES = ["BAC", "PRO", "PTE", "MES", "GEL", "CRU"]

# ---- colours (edit here to match MF_ASS) -----------------------------------
PHYTO_COLOR_POSITIONS = {
    "DIA": 0.00, "MIX": 0.26, "COC": 0.47, "PIC": 0.66, "PHA": 0.83, "FIX": 0.97,
}
ZOO_COLOR_POSITIONS = {
    "BAC": 0.05, "PRO": 0.32, "PTE": 0.50, "MES": 0.63, "GEL": 0.76, "CRU": 0.90,
}
_viridis = plt.get_cmap("viridis")
_magma = plt.get_cmap("magma")
COLORS = {s: _viridis(p) for s, p in PHYTO_COLOR_POSITIONS.items()}
COLORS.update({s: _magma(p) for s, p in ZOO_COLOR_POSITIONS.items()})
# ----------------------------------------------------------------------------


def load_species_curve(species, data_dir):
    """Return sorted, de-duplicated (temperature, growth) arrays for one PFT."""
    path = os.path.join(data_dir, FILENAME_TEMPLATE.format(species=species))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")  # strips a leading BOM
    missing = [c for c in (X_COLUMN, Y_COLUMN) if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    sub = df[[X_COLUMN, Y_COLUMN]].dropna().drop_duplicates().sort_values(X_COLUMN)
    return sub[X_COLUMN].to_numpy(), sub[Y_COLUMN].to_numpy()


def _draw_panel(ax, species_list, data_dir, title, letter):
    plotted, skipped = [], []
    for sp in species_list:
        try:
            t, g = load_species_curve(sp, data_dir)
        except (FileNotFoundError, KeyError) as err:
            print(f"WARNING: skipping {sp}: {err}")
            skipped.append(sp)
            continue
        ax.plot(t, g, color=COLORS[sp], linewidth=2.2, label=sp)
        plotted.append(sp)
    ax.set_title(title, loc="center", fontweight="bold")
    ax.text(-0.02, 1.08, f"({letter})", transform=ax.transAxes,
            fontweight="bold", ha="right", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted:
        ax.legend(frameon=False, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.20), columnspacing=1.2, handlelength=1.6)
    return plotted, skipped


def make_figure(data_dir=DATA_DIR, watermark=True):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"data folder {data_dir!r} not found (cwd = {os.getcwd()}); "
            "pass --data-dir /path/to/PFT_growth_rates")
    set_presentation_style()
    fig, (ax_p, ax_z) = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, sharey=True)

    p_ok, p_skip = _draw_panel(ax_p, PHYTO_SPECIES, data_dir, "Phytoplankton", "a")
    z_ok, z_skip = _draw_panel(ax_z, ZOO_SPECIES, data_dir, "Zooplankton and bacteria", "b")

    ylab = r"Growth rate ($\mathrm{d}^{-1}$)"
    ax_p.set_ylabel(ylab)                      # y is shared: label the left panel only
    for ax in (ax_p, ax_z):
        ax.set_xlabel("Temperature (\u00b0C)")

    if watermark:
        fig.text(0.5, 0.005, f"{WATERMARK_BASE_PATH}/{CODE}.py", ha="center", va="bottom",
                 fontsize=plt.rcParams.get("font.size", 10) - 6, color="0.55")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    print(f"plotted phyto: {p_ok}  skipped: {p_skip}")
    print(f"plotted zoo:   {z_ok}  skipped: {z_skip}")
    return fig


def save_figure(data_dir=DATA_DIR, watermark=True, dpi=300):
    fig = make_figure(data_dir=data_dir, watermark=watermark)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"{CODE}_{datetime.date.today():%Y%m%d}.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help=f"folder with {{species}}_growth.csv files (default {DATA_DIR!r})")
    ap.add_argument("--no-watermark", action="store_true")
    a = ap.parse_args()
    save_figure(data_dir=a.data_dir, watermark=not a.no_watermark)
