"""
Shared matplotlib style settings.

Two profiles:
    set_presentation_style()  -- default: larger fonts for slides/screens
    set_paper_style()         -- smaller, print-oriented fonts for journal figures

Both load Gill Sans if available (falls back to default font otherwise),
disable top/right spines, and set a default line width of 2.

Panels should default to a golden-ratio box (width:height = phi:1) unless
there's a good reason otherwise (e.g. square scatter/correlation plots,
wide time series). Use `golden(ax)` on any axes to apply this.
"""

import os
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

GOLDEN_RATIO = 1.6180339887

# ---- Colour palettes (qualitative, for up to N lines) ----

# Bauhaus poster palette (4 colours). Blue plotted first by request.
BAUHAUS = ['#0050FF', '#1A1A1A', '#FFB700', '#DE2E25']

# Paul Tol 'bright' qualitative scheme (7 colours) -- general default.
TOL_BRIGHT = ['#4477AA', '#EE6677', '#228833', '#CCBB44',
              '#66CCEE', '#AA3377', '#BBBBBB']

# Paul Tol 'muted' qualitative scheme (9 colours) -- when more lines are needed.
TOL_MUTED = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE',
             '#882255', '#44AA99', '#999933', '#AA4499']


def _load_gill_sans():
    try:
        gill_sans_path = "/gpfs/home/mep22dku/scratch/FONTS/gill-sans/GillSansMedium.otf"
        font_manager.fontManager.addfont(gill_sans_path)
        gill_font_name = font_manager.FontProperties(fname=gill_sans_path).get_name()
        rcParams['font.family'] = gill_font_name
        rcParams['mathtext.fontset'] = 'custom'
        rcParams['mathtext.rm'] = gill_font_name
        rcParams['mathtext.it'] = gill_font_name
        rcParams['mathtext.bf'] = gill_font_name
        print(f"  Using font: {gill_font_name}")
    except Exception as e:
        print(f"  Gill Sans not loaded ({e}), using default font.")
    rcParams['text.usetex'] = False


def _base_style(base_size, linewidth, axes_linewidth, dpi, savefig_dpi):
    _load_gill_sans()
    rcParams.update({
        'font.size': base_size,
        'axes.titlesize': base_size + 2,
        'axes.labelsize': base_size,
        'xtick.labelsize': base_size - 1,
        'ytick.labelsize': base_size - 1,
        'legend.fontsize': base_size - 2,
        'lines.linewidth': linewidth,
        'axes.linewidth': axes_linewidth,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.grid': False,
        'figure.dpi': dpi,
        'savefig.dpi': savefig_dpi,
        'axes.prop_cycle': plt.cycler(color=TOL_BRIGHT),
    })


def set_presentation_style(base_size=15):
    """Default style: larger fonts (14-16pt) for slides/screens."""
    _base_style(base_size=base_size, linewidth=2, axes_linewidth=0.8,
                dpi=100, savefig_dpi=200)


def set_paper_style(base_size=9):
    """Smaller, print-oriented style for journal figures."""
    _base_style(base_size=base_size, linewidth=1.5, axes_linewidth=0.6,
                dpi=150, savefig_dpi=300)


def golden(ax, ratio=GOLDEN_RATIO):
    """Force an axes' data box to a golden-ratio aspect (width:height = phi:1)."""
    ax.set_box_aspect(1 / ratio)


def add_provenance(fig, script_path=None, fontsize=7, color='#999999',
                    include_date=True, x=0.995, y=0.005, ha='right'):
    """
    Stamp the absolute path of the script that generated this figure
    (optionally prefixed with today's date), in small grey text at the
    bottom of the figure. Defaults to bottom-right; pass x=0.5, ha='center'
    for bottom-centre, and raise y if it's crowding the panels above it.

    Pass `script_path` explicitly (e.g. __file__ from the calling script).
    If omitted, tries to auto-detect via __main__.__file__, which only
    works when run as a script (not in a notebook/interactive session).
    """
    if script_path is None:
        import __main__
        script_path = getattr(__main__, '__file__', None)
        if script_path is None:
            raise ValueError(
                "script_path not provided and could not be auto-detected "
                "(e.g. running interactively). Pass it explicitly: "
                "add_provenance(fig, __file__)"
            )
    script_path = os.path.abspath(script_path)
    text = script_path
    if include_date:
        from datetime import date
        text = f"{date.today().isoformat()}  {script_path}"
    fig.text(x, y, text, ha=ha, va='bottom',
              fontsize=fontsize, color=color)

