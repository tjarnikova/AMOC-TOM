"""
food_web_diagram.py

Circular food-web diagram of the same grazing preferences as
plot_grazing_matrix.py (imported directly from there -- GRAZING_TABLE,
FOOD_SOURCE, PREDATORS -- not duplicated), styled after a hand-drawn
reference diagram:

    - top arc: the 6 phytoplankton (DIA, MIX, COC, PIC, PHA, FIX), left
      to right, each outlined in its own colour
    - west point (side): BAC, also outlined in its own colour
    - bottom arc: the 5 predators (PRO, PTE, MES, GEL, CRU), left to
      right, each outlined in its OWN colour
    - one arrow per nonzero grazing-table cell, prey -> predator,
      coloured in the PREDATOR's colour (matching the reference image),
      thickness keyed to the trophic strength: thin if <1, medium if
      exactly 1, thick if >1
    - "zoo-family" edges (BAC or a predator preying on another predator)
      are routed as larger arcs bowing below the circle, same as the
      reference image's swooping bottom connections; phyto-to-predator
      edges use a small in-circle curve instead

COLOURS -- phytoplankton use the same custom viridis positions as
plot_growth_vs_temp.py / plot_grazing_matrix.py (PHYTO_COLOR_POSITIONS).
Zooplankton use the REVISED magma positions from phyto_composition_bars.py
(ZOO_COLOR_POSITIONS below), NOT plot_grazing_matrix.py's original set --
those made BAC and PRO too hard to tell apart (see
phyto_composition_bars.py's module docstring for the full story); since
BAC and PRO sit right next to each other here too (BAC feeds PRO
directly), the improved spacing matters just as much in this diagram.
Let me know if you'd rather match plot_grazing_matrix.py's original
palette exactly instead.

LAYOUT ASSUMPTION -- "bottom is PRO to GEL" is read as the canonical
zoo order minus BAC (PRO, PTE, MES, GEL, CRU), left to right along the
bottom arc -- CRU ends up rightmost (past GEL), not literally last
alphabetically-adjacent to GEL. Flag if you wanted a different order.

Usage:
    python food_web_diagram.py

or from a notebook:
    from food_web_diagram import make_figure
    fig, fname = make_figure()
    fig.savefig(f"/gpfs/home/mep22dku/scratch/AMOC-TOM/plts/{fname}", bbox_inches="tight")
"""

import os
import sys
import datetime
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/UTILS')

import numpy as np
import matplotlib
matplotlib.use("Agg")  # batch run -- no interactive display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_grazing_matrix import GRAZING_TABLE, FOOD_SOURCE, PREDATORS
from plot_style import set_presentation_style

PHYTO_ORDER = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"]  # top arc, left to right
ZOO_ORDER = ["PRO", "PTE", "MES", "GEL", "CRU"]           # bottom arc, left to right

# single continuous clockwise sequence around the whole circle, all 12
# species evenly spaced (360/12 = 30 degrees apart) -- this is what
# replaced the earlier "two separate arcs + BAC pinned at exactly 180"
# layout, which squeezed BAC into just a 15-degree gap on each side
# (DIA's arc and the zoo arc were each internally even, but the seam
# between them, where BAC sat alone, wasn't). Going clockwise from DIA
# (top-left) across the top to FIX (top-right), continuing clockwise
# down the right and back along the bottom lands on CRU, GEL, MES, PTE,
# PRO, BAC in that order -- which still reads left-to-right as PRO,
# PTE, MES, GEL, CRU along the bottom and BAC alone at the west side,
# same visual arrangement as before, just evenly spaced now.
CIRCLE_ORDER = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX",
                "CRU", "GEL", "MES", "PTE", "PRO", "BAC"]

# same positions as plot_growth_vs_temp.py / plot_grazing_matrix.py
PHYTO_COLOR_POSITIONS = {
    "DIA": 0.00, "MIX": 0.26, "COC": 0.47, "PIC": 0.66, "PHA": 0.83, "FIX": 0.97,
}
# REVISED positions (see module docstring's COLOURS section) -- NOT the
# same as plot_grazing_matrix.py's original set
ZOO_COLOR_POSITIONS = {
    "BAC": 0.05, "PRO": 0.32, "PTE": 0.50, "MES": 0.63, "GEL": 0.76, "CRU": 0.90,
}
_viridis = plt.get_cmap("viridis")
_magma = plt.get_cmap("magma")
PHYTO_COLORS = {sp: _viridis(pos) for sp, pos in PHYTO_COLOR_POSITIONS.items()}
ZOO_COLORS = {sp: _magma(pos) for sp, pos in ZOO_COLOR_POSITIONS.items()}
ALL_COLORS = {**PHYTO_COLORS, **ZOO_COLORS}

# arrow thickness by trophic strength
LW_THIN, LW_MEDIUM, LW_THICK = 1.1, 2.4, 4.0

OUTPUT_DIR = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
WATERMARK_BASE_PATH = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts"
SCRIPT_NAME = "food_web_diagram.py"


def _linewidth_for_value(v):
    if v > 1:
        return LW_THICK
    if v == 1:
        return LW_MEDIUM
    return LW_THIN  # 0 < v < 1 (zero-valued cells are skipped entirely)


def _node_positions(radius=1.0):
    """All 12 species placed evenly around the circle (30 degrees
    apart) -- see CIRCLE_ORDER's comment for why this replaced the
    earlier two-arc-plus-pinned-BAC layout."""
    n = len(CIRCLE_ORDER)
    start_angle = 165  # DIA's angle -- keeps the same overall orientation as before
    step = 360 / n
    positions = {}
    for i, sp in enumerate(CIRCLE_ORDER):
        ang = np.radians(start_angle - i * step)
        positions[sp] = (radius * np.cos(ang), radius * np.sin(ang))
    return positions


def make_figure(watermark=True):
    set_presentation_style()

    positions = _node_positions(radius=1.0)
    zoo_family = set(ZOO_ORDER) | {"BAC"}

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.9, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    for predator in PREDATORS:
        color = ALL_COLORS[predator]
        for prey in FOOD_SOURCE:
            if prey == predator:
                continue
            value = GRAZING_TABLE.loc[prey, predator]
            if value <= 0:
                continue

            lw = _linewidth_for_value(value)
            x0, y0 = positions[prey]
            x1, y1 = positions[predator]

            if prey in zoo_family:
                # same-family (or BAC) edge -- big arc bowing below the
                # circle, matching the reference image's swooping
                # bottom connections
                rad = 0.55
            else:
                # phyto -> predator edge -- small in-circle curve
                rad = 0.12

            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>", mutation_scale=16,
                color=color, linewidth=lw, alpha=0.85,
                shrinkA=20, shrinkB=20, zorder=2,
            )
            ax.add_patch(arrow)

    box_w, box_h = 0.34, 0.16
    for sp, (x, y) in positions.items():
        edge_color = ALL_COLORS[sp]
        rect = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0,rounding_size=0.025",
            facecolor="white", edgecolor=edge_color, linewidth=3.2,
            zorder=5,
        )
        ax.add_patch(rect)
        ax.text(x, y, sp, ha="center", va="center", fontsize=13,
                fontweight="bold", zorder=6)

    if watermark:
        script_path = f"{WATERMARK_BASE_PATH}/{SCRIPT_NAME}"
        fig.text(0.5, 0.01, script_path, ha="center", va="bottom",
                  fontsize=plt.rcParams.get("font.size", 10) - 6, color="0.55")

    date_str = datetime.date.today().strftime("%Y%m%d")
    fname = f"food_web_diagram_{date_str}.png"
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
