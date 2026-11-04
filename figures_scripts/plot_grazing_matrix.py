"""
plot_grazing_matrix.py

Grazing/predation matrix as a seaborn-style annotated heatmap (built
with plain matplotlib -- imshow + text annotations + white gridlines --
rather than an actual seaborn dependency, so nothing new needs to be
installed):

    rows    = zooplankton (the predators: PRO, PTE, MES, GEL, CRU)
    columns = food source (everything that can be eaten: the six
              phytoplankton groups plus all six zooplankton groups,
              DIA...CRU -- this is the transpose of the table as it was
              given, where food source was rows and predator was
              columns)

Cell shading uses a LOG-scale (not linear) grayscale colormap ("Greys")
-- the grazing coefficients span a wide relative range (0.1 to 3.0),
and a plain linear 0-3 scale compresses all the small values (0.1, 0.2,
0.5) into an almost-indistinguishable pale corner; log scaling spreads
them out instead. Cells with a value of exactly 0 are masked out
entirely (rendered white, no colour at all, and left unannotated --
see NUMBER FORMAT below) rather than forced onto the log scale, since
log(0) isn't defined anyway. Colourbar ticks are fixed at 1, 2, and 3
(the meaningful reference points -- 0 isn't representable on a log
axis, which is consistent with 0-valued cells having no colour/
annotation at all). Cell shading is NOT related to the species colour
scheme below.

NUMBER FORMAT -- a cell with value 0 gets NO annotation at all (fully
blank, not "0" or "0.0"). A whole-number value (e.g. 3.0) is annotated
as a plain integer ("3", not "3.0"). Anything else keeps one decimal
place (e.g. "0.1", "2.5") -- see _format_value().

LABEL COLOURS -- every row/column label that names a species gets a
small colour swatch behind the text (a coloured bounding box, not just
coloured text), with the text itself switched to black or white
automatically for contrast against that swatch -- plain coloured text
risks being unreadable for light colours (pale yellow, cream, etc.),
so a background box is used instead:

    - the six phytoplankton (DIA, MIX, COC, PIC, PHA, FIX) use the same
      custom viridis positions as plot_growth_vs_temp.py (chosen to
      maximise worst-case pairwise CIELAB distance -- see that script's
      PHYTO_COLOR_POSITIONS for the full rationale)
    - the six zooplankton (BAC, PRO, PTE, MES, GEL, CRU) use the
      REVISED magma positions from phyto_composition_bars.py /
      food_web_diagram.py (ZOO_COLOR_POSITIONS below) -- NOT the
      original set this script used to have, which put BAC and PRO too
      close together (both sit in magma's near-black range, where
      human vision loses hue sensitivity even when the raw CIELAB
      distance looks fine on paper -- see phyto_composition_bars.py's
      module docstring for the full story)

Since the same six zooplankton species appear as BOTH row labels
(as predators) and as some of the column labels (as prey/food source,
i.e. cannibalism/cross-feeding terms), their swatch colour is identical
in both places -- only the axis position differs.

DATA -- hardcoded from the table given in conversation (not read from
a file). If this needs to be swapped for a CSV later, replace
GRAZING_TABLE below with a pd.read_csv(...) call using the same
(food_source rows x predator columns) orientation, then the rest of the
script (transpose, plotting) is unchanged.

STYLE -- uses plot_style.py's set_presentation_style(), same house
style as every other script in this project, plus the small grey
"script path" watermark along the bottom (on by default).

Usage:
    python plot_grazing_matrix.py

or from a notebook:
    from plot_grazing_matrix import make_figure
    fig, fname = make_figure()
    fig.savefig(f"./plts/{fname}", bbox_inches="tight")
"""

import os
import sys
import datetime
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/UTILS')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter

from plot_style import set_presentation_style

OUTPUT_DIR = "plts"
WATERMARK_BASE_PATH = "/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts"
SCRIPT_NAME = "plot_grazing_matrix.py"

# ---------------------------------------------------------------------------
# Data -- as given: rows = food source (prey), columns = predator
# ---------------------------------------------------------------------------

FOOD_SOURCE = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX",
               "BAC", "PRO", "PTE", "MES", "GEL", "CRU"]
PREDATORS = ["PRO", "PTE", "MES", "GEL", "CRU"]

GRAZING_TABLE = pd.DataFrame(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0],   # DIA
        [0.7, 0.7, 0.7, 0.7, 0.7],   # MIX
        [2.5, 0.5, 0.5, 0.1, 0.1],   # COC
        [3.0, 0.1, 0.1, 0.1, 0.1],   # PIC
        [1.0, 1.0, 1.0, 1.0, 1.0],   # PHA
        [3.0, 0.1, 0.1, 0.1, 0.1],   # FIX
        [3.0, 0.1, 0.1, 0.1, 0.1],   # BAC
        [0.0, 1.0, 1.0, 1.0, 1.0],   # PRO
        [0.0, 0.0, 0.0, 3.0, 1.0],   # PTE
        [0.0, 0.2, 0.0, 3.0, 1.0],   # MES
        [0.0, 0.0, 0.0, 0.0, 0.5],   # GEL
        [0.0, 0.0, 0.0, 0.5, 0.0],   # CRU
    ],
    index=FOOD_SOURCE, columns=PREDATORS,
)

# ---------------------------------------------------------------------------
# Label colours -- swatches (background boxes), not just text colour
# ---------------------------------------------------------------------------

# same positions/rationale as plot_growth_vs_temp.py's PHYTO_COLOR_POSITIONS
PHYTO_COLOR_POSITIONS = {
    "DIA": 0.00, "MIX": 0.26, "COC": 0.47, "PIC": 0.66, "PHA": 0.83, "FIX": 0.97,
}
# REVISED positions (see module docstring's LABEL COLOURS section) --
# NOT this script's original set
ZOO_COLOR_POSITIONS = {
    "BAC": 0.05, "PRO": 0.32, "PTE": 0.50, "MES": 0.63, "GEL": 0.76, "CRU": 0.90,
}

_viridis = plt.get_cmap("viridis")
_magma = plt.get_cmap("magma")
PHYTO_COLORS = {sp: _viridis(pos) for sp, pos in PHYTO_COLOR_POSITIONS.items()}
ZOO_COLORS = {sp: _magma(pos) for sp, pos in ZOO_COLOR_POSITIONS.items()}
SPECIES_COLORS = {**PHYTO_COLORS, **ZOO_COLORS}


def _contrast_text_color(rgba):
    """Black or white, whichever reads better on top of this background
    colour (simple perceptual-luminance threshold)."""
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.6 else "white"


def _format_value(v):
    """Blank for exactly 0 (see module docstring's NUMBER FORMAT
    section), a plain integer for whole numbers (3.0 -> "3"), one
    decimal place otherwise (2.5 -> "2.5")."""
    if v == 0:
        return ""
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.1f}"


def _apply_label_swatches(ax, labels, tick_labels, axis):
    """Give each tick label (that names a species) a coloured background
    box, with automatically contrasting text colour. `axis` is "x" or "y"."""
    for label, name in zip(tick_labels, labels):
        color = SPECIES_COLORS.get(name)
        if color is None:
            continue
        label.set_color(_contrast_text_color(color))
        label.set_bbox(dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.3"))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(watermark=True):
    set_presentation_style()

    # transpose: rows = predator (zooplankton), columns = food source
    matrix = GRAZING_TABLE.T.loc[PREDATORS, FOOD_SOURCE]
    values = matrix.to_numpy()

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # log-scale shading, 0-valued cells masked out (rendered white, no
    # colour at all) rather than forced onto a log scale -- see module
    # docstring's "Cell shading" section
    nonzero = values[values > 0]
    vmin, vmax = float(nonzero.min()), float(values.max())
    masked_values = np.ma.masked_equal(values, 0)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad(color="white")
    norm = LogNorm(vmin=vmin, vmax=vmax)

    mesh = ax.imshow(masked_values, cmap=cmap, norm=norm, aspect="auto")

    # white gridlines between cells, seaborn-heatmap style
    ax.set_xticks(np.arange(-0.5, len(FOOD_SOURCE), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PREDATORS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # cell annotations, contrast-aware text colour, blank for 0
    for i in range(len(PREDATORS)):
        for j in range(len(FOOD_SOURCE)):
            value = matrix.iat[i, j]
            label = _format_value(value)
            if not label:
                continue
            text_color = "white" if norm(value) > 0.5 else "black"
            ax.text(j, i, label, ha="center", va="center",
                     color=text_color, fontsize=16)

    ax.set_xticks(range(len(FOOD_SOURCE)))
    ax.set_xticklabels(FOOD_SOURCE, fontsize=14)
    ax.set_yticks(range(len(PREDATORS)))
    ax.set_yticklabels(PREDATORS, fontsize=14)

    _apply_label_swatches(ax, FOOD_SOURCE, ax.get_xticklabels(), axis="x")
    _apply_label_swatches(ax, PREDATORS, ax.get_yticklabels(), axis="y")

    # extra labelpad -- more visual space between the matrix/tick labels
    # and the axis titles, per request
    ax.set_xlabel("Food source", fontweight="bold", fontsize=18, labelpad=22)
    ax.set_ylabel("Zooplankton", fontweight="bold", fontsize=18, labelpad=22)
    ax.spines[:].set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    # fixed ticks at 1, 2, 3 -- 0 isn't representable on a log axis,
    # consistent with 0-valued cells having no colour/annotation at all
    cbar.ax.yaxis.set_major_locator(FixedLocator([1, 2, 3]))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("grazing coefficient", fontsize=15)

    if watermark:
        script_path = f"{WATERMARK_BASE_PATH}/{SCRIPT_NAME}"
        fig.text(0.5, 0.01, script_path, ha="center", va="bottom",
                  fontsize=plt.rcParams.get("font.size", 10) - 6, color="0.55")

    fig.tight_layout()

    date_str = datetime.date.today().strftime("%Y%m%d")
    fname = f"grazing_matrix_{date_str}.png"
    return fig, fname


def save_figure(watermark=True, dpi=200):
    fig, fname = make_figure(watermark=watermark)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    save_figure()
