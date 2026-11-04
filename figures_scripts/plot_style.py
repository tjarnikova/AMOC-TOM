"""
Shared matplotlib style settings.

Two profiles:
    set_presentation_style()  -- default: larger fonts for slides/screens
    set_paper_style()         -- smaller, print-oriented fonts for journal figures

A third profile, set_agu_style(), reads figure_settings.yml (page width, font
sizes, line widths, dpi, watermark, output folder). It is what the paper
figures use. figsize(code, design_w, design_h) turns a figure's design size
into the final size from that file.

Both load Gill Sans if available (falls back to default font otherwise),
disable top/right spines, and set a default line width of 2.

Panels should default to a golden-ratio box (width:height = phi:1) unless
there's a good reason otherwise (e.g. square scatter/correlation plots,
wide time series). Use `golden(ax)` on any axes to apply this.
"""

import os
import sys
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



# ===========================================================================
# Paper figures: everything below is driven by figure_settings.yml
# ===========================================================================

SETTINGS_ENV = "AMOC_FIG_SETTINGS"
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "figure_settings.yml")
_DEFAULTS = {
    "page_width_mm": 170.0,
    "max_height_mm": None,
    "font_base_pt": 8.0,
    "font_small_pt": 7.0,
    "font_watermark_pt": 6.0,
    "line_width_pt": 1.0,
    "mask_linewidth_pt": 1.0,
    "dpi": 300,
    "watermark": True,
    "output_dir": "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts",
}
_cache = {}
_LAST = {"design_h": None}      # design height of the figure made by the latest figsize() call


def load_settings(reload=False):
    """Defaults, overridden by figure_settings.yml (or the file named in the
    environment variable AMOC_FIG_SETTINGS). Read once per run."""
    path = os.environ.get(SETTINGS_ENV, SETTINGS_FILE)
    if reload or _cache.get("path") != path:
        cfg = dict(_DEFAULTS)
        cfg["figures"] = {}
        if os.path.isfile(path):
            import yaml
            with open(path) as f:
                user = yaml.safe_load(f) or {}
            figs = user.pop("figures", None) or {}
            unknown = [k for k in user if k not in _DEFAULTS]
            if unknown:
                print("[plot_style] WARNING: unknown keys in %s: %s" % (path, unknown))
            cfg.update(user)
            cfg["figures"] = figs
        else:
            print("[plot_style] WARNING: %s not found, using built-in defaults" % path)
        _cache["path"] = path
        _cache["cfg"] = cfg
    return _cache["cfg"]


def font_base():
    return float(load_settings()["font_base_pt"])


def font_small():
    return float(load_settings()["font_small_pt"])


def font_watermark():
    return float(load_settings()["font_watermark_pt"])


def line_width():
    return float(load_settings()["line_width_pt"])


def mask_linewidth():
    return float(load_settings()["mask_linewidth_pt"])


def dpi():
    return int(load_settings()["dpi"])


def output_dir():
    return os.path.expanduser(str(load_settings()["output_dir"]))


def watermark_from_settings():
    """The yaml switch only. The per-run flags --no-watermark / --water-off are
    handled in each script and can only switch it off."""
    return bool(load_settings()["watermark"])


def apply_figure_style():
    """Paper style from figure_settings.yml: Gill Sans, base and small font
    sizes, thin lines, no top/right spines."""
    cfg = load_settings()
    base, small = font_base(), font_small()
    _load_gill_sans()
    rcParams.update({
        'font.size': base,
        'axes.titlesize': base,
        'axes.labelsize': base,
        'figure.titlesize': base,
        'xtick.labelsize': small,
        'ytick.labelsize': small,
        'legend.fontsize': small,
        'lines.linewidth': line_width(),
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.grid': False,
        'figure.dpi': 150,
        'savefig.dpi': dpi(),
        'axes.prop_cycle': plt.cycler(color=TOL_BRIGHT),
    })


def figsize(code, design_w, design_h):
    """(width, height) in inches for the figure called `code`.

    design_w x design_h is the size the figure was originally laid out at.
    The width is page_width_mm (or the figure's own width_mm under `figures:`
    in figure_settings.yml). The height scales with the width so the panel
    arrangement is unchanged, unless the figure has its own height_mm.
    Prints the final size in mm."""
    cfg = load_settings()
    ov = cfg["figures"].get(code) or {}
    w_mm = float(ov.get("width_mm") or cfg["page_width_mm"])
    w = w_mm / 25.4
    if ov.get("height_mm"):
        h = float(ov["height_mm"]) / 25.4
    else:
        h = design_h * w / design_w
    note = ""
    mx = cfg.get("max_height_mm")
    if mx and h * 25.4 > float(mx):
        note = "  WARNING: taller than max_height_mm = %s" % mx
    print("[figsize] %s: %.0f x %.0f mm%s" % (code, w_mm, h * 25.4, note))
    _LAST["design_h"] = design_h
    return (w, h)


def width_in(code):
    """Final width in inches of the figure called `code` (no printing)."""
    cfg = load_settings()
    ov = cfg["figures"].get(code) or {}
    return float(ov.get("width_mm") or cfg["page_width_mm"]) / 25.4


def keep_bottom(frac, fig):
    """Bottom margin (a figure fraction chosen for the original design height) that keeps its
    size in inches when the figure is rescaled. Text does not shrink with the figure, so the
    margins that hold colorbar labels, titles and the watermark must not shrink either.
    Call right after the figure is made with figsize()."""
    d = _LAST.get("design_h")
    return frac if not d else frac * d / fig.get_figheight()


def keep_gap(frac, fig):
    """Gap between rows of panels (a gridspec hspace chosen for the original design height)
    that keeps its size in inches when the figure is rescaled, so colorbar tick labels and
    labels between rows still have room. Capped so the panels do not vanish."""
    d = _LAST.get("design_h")
    return frac if not d else min(frac * d / fig.get_figheight(), 0.7)


def keep_top(top, fig):
    """Same as keep_bottom, for the top edge (top = 1 - margin)."""
    d = _LAST.get("design_h")
    return top if not d else 1.0 - (1.0 - top) * d / fig.get_figheight()
