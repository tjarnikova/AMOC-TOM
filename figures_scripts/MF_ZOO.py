"""
MF_ZOO.py  --  Main Figure candidate: zooplankton overview with latitude profiles

Outputs (all in /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/, date-stamped):
    MF_ZOO_maps_YYYYMMDD.png          2 rows x 2 cols of maps
                                       col 0 zooplankton biomass, col 1 zooplankton grazing
                                       row 0 CONTROL, row 1 HOSE-CONTROL
    MF_ZOO_prof_lines_YYYYMMDD.png    latitude profiles, control (solid) and hose (dashed),
                                       difference shaded. Panels: PHY, ZOO, PP, grazing, ZOO/PHY, herbivory fraction
    MF_ZOO_prof_change_YYYYMMDD.png   latitude profiles of the change (hose minus control).
                                       Percent change for stocks and flows, absolute change for ratios.
    MF_ZOO_prof_overview_YYYYMMDD.png top: control biomass (PHY, ZOO; optional fill), PP and
                                       zooplankton grazing (dotted), herbivory fraction.
                                       bottom: ABSOLUTE change (hose minus control), one panel per unit:
                                       PHY and ZOO (mmol C m-2), PP and grazing (1e-6 mol C m-2 s-1),
                                       herbivory fraction (change in fraction). Same columns as the top row.
    MF_ZOO_full_YYYYMMDD.png          maps (left) + overview profiles (right), rows aligned in latitude

Also prints a regional table (control, hose, change) so the numbers can be checked against Table S1.

Conventions
    * Profiles and the regional table are computed on the NATIVE grid with csize weights and the
      native binary masks (same convention as TabK_summary).
    * Profile values are area-weighted bin means (stocks, flows) or ratios of bin sums (ratios).
      Never means of cell-wise ratios.
    * Latitude is the native nav_lat from meshmask_v5.nc.
    * Climatology is 12 monthly means averaged to one annual mean. No uncertainty is available.

Flags: --no-watermark / --water-off, --no-maps
Assumptions to check at the top of the CONFIG block (paths, variable names).
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import map_maker as mm
import plot_style as ps
from datetime import datetime

# ── CONFIG ─────────────────────────────────────────────────────────────
WATERMARK = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_ZOO.py"
WATERMARK_ON = ps.watermark_from_settings()   # figure_settings.yml; --no-watermark / --water-off switch it off
MAKE_MAPS = True
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
if "--no-maps" in sys.argv:
    sys.argv.remove("--no-maps")
    MAKE_MAPS = False

OUTPUT_DIR  = ps.output_dir()   # figure_settings.yml
FIGURE_CODE = "MF_ZOO"

# NATIVE (148 x 180) files. Check these paths and variable names.
MESHMASK    = "/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/masks/meshmask_v5.nc"
NATIVE_MASK = "/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL.nc"
LAT_NAMES   = ["nav_lat", "gphit", "lat"]
CSIZE_NAMES = ["csize"]
REGIONS     = ["NHL", "NA", "SA", "SO"]     # mask variable names; AB = union of these
MASK_ALIASES = {"NHL": ["NHL", "HA"]}          # fallback names if the file predates the rename

OVERVIEW_FILL = "between"   # biomass fill in the overview: "between" (PHY vs ZOO), "zero", or "none"
FILL_ALPHA    = 0.4

# Region boundary-line colours on the profile plots. Match the outline colours in map_maker (check by eye).
REGION_COLORS = {"NHL": "#e63227", "NA": "#ffb300", "SA": "#111111", "SO": "#1f5fff"}

BIN_DEG   = 2.0     # latitude bin width (degrees)
MIN_CELLS = 5       # bins with fewer ocean cells are left blank

C_CTRL = "0.15"
C_HOSE = "#c0392b"
C_PHY, C_PP, C_ZOO = "#00c853", "#e08a00", "#2979ff"   # PHY emerald, ZOO vivid blue (option C; was #2e7d32 / #1f5fa8)
LETTERS_FULL = True       # panel letters on the Fig 6 figure (MF_ZOO_full)
FULL_HEIGHT_MM = 140      # Fig 6 height (yaml: figures: MF_ZOO_full: {height_mm: ...} overrides it)
FULL_TOP_IN, FULL_BOTTOM_IN, FULL_GAP_IN = 0.50, 0.85, 0.60   # top margin, bottom margin (x labels + legend), gap between the rows
FULL_SPACER = 0.30         # width of the spacer between the maps and the profiles (room for the latitude label)
FULL_WSPACE = 0.20        # gap between columns, as a fraction of a column
FULL_LEGEND_Y_IN = 0.25    # Fig 6 legend: height of its bottom edge above the figure bottom, inches (larger = closer to the x labels)

ROW_LABELS  = ["CONTROL", "HOSE\u2212CONTROL"]
ROW_LABEL_X = -0.12

# Native-grid climatology files, loaded directly (no map_maker). Check root, run folders, pattern.
import glob
CLIM_ROOT    = "/gpfs/data/greenocean/users/mep22dku/clims"
RUN_DIRS     = {"BASE": "TOM12_TJ_HA01", "HOSE": "TOM12_TJ_HH01"}   # control, hose
CLIM_PATTERN = "ORCA2_7d_clim_2010_2019_%s.nc"                      # %s = ptrc_T_int, diad_T_int
run_a, run_b = "BASE", "HOSE"

def clim_path(run, ftype):
    p = os.path.join(CLIM_ROOT, RUN_DIRS[run], CLIM_PATTERN % ftype)
    if not os.path.exists(p):
        near = sorted(glob.glob(os.path.join(CLIM_ROOT, RUN_DIRS[run], "*%s*" % ftype)))
        raise IOError("not found: %s\nsimilar files:\n  %s" % (p, "\n  ".join(near) or "(none)"))
    return p

# ── native readers ─────────────────────────────────────────────────────
def read_native(path, var):
    """Read var, average a leading time axis, return (2D float array with NaN fill, units)."""
    with Dataset(path) as ds:
        v = ds.variables[var]
        units = getattr(v, "units", "")
        a = np.ma.filled(np.ma.asarray(v[:]).astype(float), np.nan)
    a = np.squeeze(a)
    if a.ndim == 3:
        a = np.nanmean(a, axis=0)
    if a.ndim != 2:
        raise ValueError("%s in %s has shape %s after squeeze, expected 2D or time x 2D"
                         % (var, path, a.shape))
    return a, units

def read_first(path, names):
    with Dataset(path) as ds:
        for n in names:
            if n in ds.variables:
                a = np.ma.filled(np.ma.asarray(ds.variables[n][:]).astype(float), np.nan)
                return np.squeeze(a)
    raise KeyError("none of %s found in %s" % (names, path))

def mmol_factor(units, label):
    u = (units or "").replace(" ", "").lower()
    if u in ("molc/m2", "molc/m^2", "mol/m2"):
        return 1e3
    if u in ("mmolc/m2", "mmol/m2"):
        return 1.0
    raise ValueError("%s: units attribute is %r, expected 'molC/m2'. "
                     "Run fix_int_units.py on the clim folders first." % (label, units))
def list_vars(path):
    with Dataset(path) as ds:
        return ["%s %s" % (n, tuple(v.shape)) for n, v in ds.variables.items()]

def read_mask(region):
    for name in MASK_ALIASES.get(region, [region]):
        try:
            a = read_first(NATIVE_MASK, [name])
        except KeyError:
            continue
        if name != region:
            print("WARNING: mask %s not found, using %r (old name?). Numbers use the OLD mask extent."
                  % (region, name))
        return np.nan_to_num(a) > 0.5
    raise KeyError("no variable for region %s (tried %s) in %s.\nVariables in file:\n  %s"
                   % (region, MASK_ALIASES.get(region, [region]), NATIVE_MASK,
                      "\n  ".join(list_vars(NATIVE_MASK))))
U = {}

MOLC_LABEL = r"mol C m$^{-2}$ s$^{-1}$"

def nice_units(u):
    """Mathtext label for the units attribute when it is the usual molC/m2/s; else the raw string."""
    if (u or "").replace(" ", "").lower() == "molc/m2/s":
        return MOLC_LABEL
    return u

# ── regional table (check against Table S1) ────────────────────────────
def regional(run, sel):
    cs = csize[sel]
    s = {}
    for v in ("ZOO", "PHY", "PPT", "FEED", "Herbiv"):
        s[v] = float(np.sum(F[run][v][sel] * cs))
    return s

def bsum(field):
    return np.bincount(idx, weights=field[sel_ab] * cs_ab, minlength=nb)

def blank(a):
    out = np.array(a, dtype=float)
    out[~okbin] = np.nan
    return out

def profiles(run):
    f = F[run]
    zoo, phy, ppt = bsum(f["ZOO"]), bsum(f["PHY"]), bsum(f["PPT"])
    feed, herb = bsum(f["FEED"]), bsum(f["Herbiv"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(
            PHY=blank(phy / area_b * TO_MMOL),
            ZOO=blank(zoo / area_b * TO_MMOL),
            PP=blank(ppt / area_b),
            FEED=blank(feed / area_b),
            ZP=blank(zoo / phy),
            HERB=blank(herb / feed),
        )

def pct(h, c):
    out = np.full_like(c, np.nan, dtype=float)
    good = np.isfinite(c) & (c > 0) & np.isfinite(h)
    out[good] = 100.0 * (h[good] - c[good]) / c[good]
    return out

# region latitude ranges for guide lines (percentiles of mask latitude, rounded; check by eye)
REG_LAT = {}

def add_regions(axes, label_axes):
    """Mark region boundaries with horizontal lines in the region (map-outline) colours.
    Where two regions meet, the line alternates between their two colours.
    Labels are written at the right edge of every axis in label_axes."""
    if not isinstance(label_axes, (list, tuple)):
        label_axes = [label_axes]
    pts = []
    for r, (lo, hi) in REG_LAT.items():
        col = REGION_COLORS.get(r, "0.5")
        for y in (lo, hi):
            if lat_min < y < lat_max:
                pts.append((float(y), col))
        lo_c, hi_c = max(lo, lat_min), min(hi, lat_max)
        if hi_c > lo_c:
            for lab in label_axes:
                lab.text(1.04, 0.5 * (lo_c + hi_c), r, transform=lab.get_yaxis_transform(),
                         ha="left", va="center", fontsize=ps.font_small(),
                         color=col, fontweight="bold")
    pts.sort(key=lambda t: t[0])
    groups = []
    for y, c in pts:
        if groups and y - groups[-1][0][0] <= 2.0:
            groups[-1].append((y, c))
        else:
            groups.append([(y, c)])
    for g in groups:
        yv = sum(y for y, _ in g) / len(g)
        cols = []
        for _, c in g:
            if c not in cols:
                cols.append(c)
        n = len(cols)
        for k, c in enumerate(cols):
            ls = "-" if n == 1 else (k * 5, (5, 5 * (n - 1)))
            for ax in axes:
                ax.axhline(yv, color=c, lw=ps.line_width(), ls=ls, zorder=1, alpha=0.9)

def finish(fig, name):
    if WATERMARK_ON:
        fig.text(0.5, 0.005, WATERMARK, ha="center", va="bottom",
                 fontsize=ps.font_watermark(), color="0.55")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = "%s/%s_%s_%s.png" % (OUTPUT_DIR, FIGURE_CODE, name, datetime.now().strftime("%Y%m%d"))
    fig.savefig(out, bbox_inches="tight", dpi=ps.dpi())
    print("saved " + out)

# ── figure: lines (control and hose) ───────────────────────────────────
def fig_lines():
    panels = [
        ("PHY",  "Phytoplankton biomass", r"mmol C m$^{-2}$"),
        ("ZOO",  "Zooplankton biomass",   r"mmol C m$^{-2}$"),
        ("PP",   "Particulate PP",        FLOW_LABEL),
        ("FEED", "Zooplankton grazing", FLOW_LABEL),
        ("ZP",   "ZOO / PHY",             "ratio"),
        ("HERB", "Herbivory fraction",    "ratio"),
    ]
    fig, axes = plt.subplots(1, len(panels), sharey=True,
                             figsize=ps.figsize("MF_ZOO_prof_lines", 2.7 * len(panels), 7.0))
    for ax, (key, title, units) in zip(axes, panels):
        sc = FLOW_SCALE if key in ("PP", "FEED") else 1.0
        c, h = P_a[key] * sc, P_b[key] * sc
        ax.plot(c, centers, color=C_CTRL, lw=ps.line_width(), label="Control")
        ax.plot(h, centers, color=C_HOSE, lw=ps.line_width(), ls="--", label="Hose")
        ax.fill_betweenx(centers, c, h, where=np.isfinite(c) & np.isfinite(h),
                         color=C_HOSE, alpha=0.18, lw=0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(units, fontsize=ps.font_small())
        ax.set_ylim(lat_min, lat_max)
        ax.grid(axis="x", color="0.9", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Latitude ($^\\circ$N)")
    axes[0].legend(frameon=False, loc="best", fontsize=ps.font_small())
    add_regions(axes, axes[-1])
    fig.subplots_adjust(top=ps.keep_top(0.93, fig), bottom=ps.keep_bottom(0.1, fig), left=0.06, right=0.95, wspace=0.12)
    finish(fig, "prof_lines")
    return fig

# ── figure: change (hose minus control) ────────────────────────────────
def fig_change():
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=ps.figsize("MF_ZOO_prof_change", 2.9 * 4, 7.0))
    ax = axes[0]
    ax.plot(pct(P_b["PHY"], P_a["PHY"]), centers, color=C_PHY, lw=ps.line_width(), label="PHY biomass")
    ax.plot(pct(P_b["PP"],  P_a["PP"]),  centers, color=C_PP,  lw=ps.line_width(), label="PP")
    ax.plot(pct(P_b["ZOO"], P_a["ZOO"]), centers, color=C_ZOO, lw=ps.line_width(), label="ZOO biomass")
    ax.set_title("Biomass and PP", fontweight="bold")
    ax.set_xlabel("% change")
    ax.legend(frameon=False, loc="best", fontsize=ps.font_small())

    ax = axes[1]
    ax.plot(pct(P_b["FEED"], P_a["FEED"]), centers, color=C_HOSE, lw=ps.line_width())
    ax.set_title("Zooplankton grazing", fontweight="bold")
    ax.set_xlabel("% change")

    ax = axes[2]
    ax.plot(P_b["ZP"] - P_a["ZP"], centers, color=C_HOSE, lw=ps.line_width())
    ax.set_title("ZOO / PHY", fontweight="bold")
    ax.set_xlabel("change in ratio")

    ax = axes[3]
    ax.plot(P_b["HERB"] - P_a["HERB"], centers, color=C_HOSE, lw=ps.line_width())
    ax.set_title("Herbivory fraction", fontweight="bold")
    ax.set_xlabel("change in fraction")

    for ax in axes:
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_ylim(lat_min, lat_max)
        ax.grid(axis="x", color="0.9", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Latitude ($^\\circ$N)")
    add_regions(axes, axes[-1])
    fig.subplots_adjust(top=ps.keep_top(0.93, fig), bottom=ps.keep_bottom(0.1, fig), left=0.07, right=0.94, wspace=0.14)
    finish(fig, "prof_change")
    return fig

# ── figure: maps (biomass and zooplankton grazing) ─────────────────────
def _map_setup():
    """Load the regridded fields for the map columns (map_maker loader, as in MF_ZOO.py)."""
    run_labels_m, runs_m = mm._load_models()
    ra, rb = run_labels_m[0], run_labels_m[1]
    mptrc = mm._paths_for_filetype(runs_m, "ptrc_T_int")
    mdiad = mm._paths_for_filetype(runs_m, "diad_T_int")
    lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)

    def load(paths, run, var):
        lon, lat, field, meta = mm.load_field(paths[run], var, 0)
        return lon, lat, field

    lon, lat, zoo_a = load(mptrc, ra, "ZOO")
    zoo_b = load(mptrc, rb, "ZOO")[2]
    herb_a, herb_b = load(mdiad, ra, "Herbiv")[2], load(mdiad, rb, "Herbiv")[2]
    carn_a, carn_b = load(mdiad, ra, "Carniv")[2], load(mdiad, rb, "Carniv")[2]
    detr_a, detr_b = load(mdiad, ra, "Detrit")[2], load(mdiad, rb, "Detrit")[2]
    graz_a = herb_a + carn_a + detr_a
    graz_b = herb_b + carn_b + detr_b

    columns = [
        dict(title="Zooplankton Biomass", cmap="matter",
             ctrl=zoo_a * TO_MMOL, diff=(zoo_b - zoo_a) * TO_MMOL,
             units=r"mmol C m$^{-2}$"),
        dict(title="Zooplankton Grazing", cmap="amp", vmin=0.0, vmax=1.0,   # 1.0 = 1e-6 mol C m-2 s-1
             ctrl=graz_a * FLOW_SCALE, diff=(graz_b - graz_a) * FLOW_SCALE, units=FLOW_LABEL),
    ]
    return dict(lon=lon, lat=lat, lon_m=lon_m, lat_m=lat_m, masks=masks, columns=columns,
                diff_cmap=mm._resolve_cmap(None, mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK))

def _draw_map_cell(fig, cell, row, ci, col, S):
    inner = cell.subgridspec(2, 1, height_ratios=[1, 0.06], hspace=0.06)
    ax = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
    ax.set_anchor("S")          # map sits on its colorbar strip
    cbar_gs = inner[1, 0].subgridspec(1, 3, width_ratios=[0.15, 0.70, 0.15], wspace=0)
    cax = fig.add_subplot(cbar_gs[0, 1])
    lon, lat = S["lon"], S["lat"]
    if row == 0:
        field = col["ctrl"]
        atl = mm._bbox_subset(lon, lat, field, lon_min, lon_max, lat_min, lat_max)
        vmin, vmax = mm._percentile_limits(atl, mm.BASE_PERCENTILES)
        if col.get("vmin") is not None:
            vmin = col["vmin"]
        if col.get("vmax") is not None:
            vmax = col["vmax"]
        cmap = mm._resolve_cmap(col["cmap"], mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK)
    else:
        field = col["diff"]
        atl = mm._bbox_subset(lon, lat, field, lon_min, lon_max, lat_min, lat_max)
        dvmax = mm._diff_limit(atl)
        vmin, vmax = -dvmax, dvmax
        cmap = S["diff_cmap"]
    mesh = mm._draw_panel(ax, lon, lat, field, cmap, vmin, vmax,
                          S["lon_m"], S["lat_m"], S["masks"],
                          extent=[lon_min, lon_max, lat_min, lat_max])
    if row == 0:
        ax.set_title(col["title"], fontweight="bold")
    if ci == 0:
        ax.text(ROW_LABEL_X, 0.5, ROW_LABELS[row], transform=ax.transAxes,
                rotation=90, ha="center", va="center", fontweight="bold")
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.formatter = mm._value_formatter()
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=ps.font_small())
    cbar.set_label(col["units"], fontsize=ps.font_small())
    ax.apply_aspect()           # colorbar exactly as wide as the map
    p, cp = ax.get_position(), cax.get_position()
    cax.set_position([p.x0, cp.y0, p.width, cp.height])
    return ax

def fig_maps():
    S = _map_setup()
    fig = plt.figure(figsize=ps.figsize("MF_ZOO_maps", 3.4 * 2, 9.0))
    outer_gs = fig.add_gridspec(2, 2, height_ratios=[3, 3], hspace=0.18, wspace=0.06)
    fig.subplots_adjust(top=ps.keep_top(0.94, fig), bottom=ps.keep_bottom(0.07, fig), left=0.06, right=0.99)
    for ci, col in enumerate(S["columns"]):
        for row in range(2):
            _draw_map_cell(fig, outer_gs[row, ci], row, ci, col, S)
    finish(fig, "maps")
    return fig

# ── figure: overview (top: control biomass, PP and grazing, herbivory fraction; bottom: % change) ──
def sym_xlim(ax, arrays, pad=1.08):
    """Centre the x axis on zero."""
    vals = [np.nanmax(np.abs(a)) for a in arrays if np.any(np.isfinite(a))]
    if vals:
        m = max(vals) * pad
        ax.set_xlim(-m, m)

def draw_overview_panels(ax_s, ax_f, ax_h, ax_bs, ax_bf, ax_hc):
    """Profile panels shared by fig_overview and fig_full."""
    from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
    # top: control biomass, with fill (see OVERVIEW_FILL)
    phy, zoo = P_a["PHY"], P_a["ZOO"]
    ok = np.isfinite(phy) & np.isfinite(zoo)
    if OVERVIEW_FILL == "between":
        ax_s.fill_betweenx(centers, phy, zoo, where=ok & (zoo >= phy), interpolate=True,
                           color=C_ZOO, alpha=FILL_ALPHA, lw=0)
        ax_s.fill_betweenx(centers, phy, zoo, where=ok & (phy > zoo), interpolate=True,
                           color=C_PHY, alpha=FILL_ALPHA, lw=0)
    elif OVERVIEW_FILL == "zero":
        ax_s.fill_betweenx(centers, 0, phy, where=np.isfinite(phy), color=C_PHY,
                           alpha=FILL_ALPHA, lw=0)
        ax_s.fill_betweenx(centers, 0, zoo, where=np.isfinite(zoo), color=C_ZOO,
                           alpha=FILL_ALPHA, lw=0)
    ax_s.plot(phy, centers, color=C_PHY, lw=ps.line_width())
    ax_s.plot(zoo, centers, color=C_ZOO, lw=ps.line_width())
    ax_s.set_title("Biomass", fontweight="bold")
    ax_s.set_xlabel(r"mmol C m$^{-2}$")

    pp, gr = P_a["PP"] * FLOW_SCALE, P_a["FEED"] * FLOW_SCALE
    okf = np.isfinite(pp) & np.isfinite(gr)
    if OVERVIEW_FILL == "between":
        ax_f.fill_betweenx(centers, pp, gr, where=okf & (pp >= gr), interpolate=True,
                           color=C_PHY, alpha=FILL_ALPHA, lw=0)
        ax_f.fill_betweenx(centers, pp, gr, where=okf & (gr > pp), interpolate=True,
                           color=C_ZOO, alpha=FILL_ALPHA, lw=0)
    ax_f.plot(pp, centers, color=C_PHY, lw=ps.line_width(), ls=":")
    ax_f.plot(gr, centers, color=C_ZOO, lw=ps.line_width(), ls=":")
    ax_f.set_title("PP and total grazing", fontweight="bold")
    ax_f.set_xlabel(FLOW_LABEL)

    ax_h.plot(P_a["HERB"], centers, color="0.25", lw=ps.line_width())
    ax_h.set_title("Herbivory\nfraction", fontweight="bold")
    ax_h.set_xlabel("fraction")

    # bottom: absolute change (hose minus control), each panel centred on zero with a heavy zero line
    d_phy, d_zoo = P_b["PHY"] - P_a["PHY"], P_b["ZOO"] - P_a["ZOO"]
    ax_bs.plot(d_phy, centers, color=C_PHY, lw=ps.line_width())
    ax_bs.plot(d_zoo, centers, color=C_ZOO, lw=ps.line_width())
    ax_bs.set_xlabel("change,\n" + r"mmol C m$^{-2}$")
    sym_xlim(ax_bs, [d_phy, d_zoo])

    d_pp   = (P_b["PP"]   - P_a["PP"])   * FLOW_SCALE
    d_feed = (P_b["FEED"] - P_a["FEED"]) * FLOW_SCALE
    ax_bf.plot(d_pp,   centers, color=C_PHY, lw=ps.line_width(), ls=":")
    ax_bf.plot(d_feed, centers, color=C_ZOO, lw=ps.line_width(), ls=":")
    ax_bf.set_xlabel("change,\n" + FLOW_LABEL)
    sym_xlim(ax_bf, [d_pp, d_feed])

    d_herb = P_b["HERB"] - P_a["HERB"]
    ax_hc.plot(d_herb, centers, color="0.25", lw=ps.line_width())
    ax_hc.set_xlabel("change in\nfraction")
    sym_xlim(ax_hc, [d_herb])

    for ax in (ax_bs, ax_bf, ax_hc):
        ax.axvline(0, color="0.1", lw=ps.line_width(), zorder=3)

    all_ax = (ax_s, ax_f, ax_h, ax_bs, ax_bf, ax_hc)
    for ax in all_ax:
        ax.set_ylim(lat_min, lat_max)
        ax.grid(axis="x", color="0.92", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax_h.xaxis.set_major_locator(MultipleLocator(0.1))      # control fraction: ticks at 0.7, 0.8
    ax_h.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    _lo, _hi = ax_hc.get_xlim()       # narrow panel: two ticks, at plus and minus a round value
    _t = float("%.1g" % (0.7 * _hi)) if _hi > 0 else 0.0
    if _t > 0:
        ax_hc.set_xticks([-_t, _t])
        ax_hc.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    for ax in (ax_f, ax_h, ax_bf, ax_hc):
        plt.setp(ax.get_yticklabels(), visible=False)
    ax_s.set_ylabel("Latitude ($^\\circ$N)")
    ax_bs.set_ylabel("Latitude ($^\\circ$N)")
    add_regions(list(all_ax), [ax_h, ax_hc])

def _overview_legend(fig, x, y=0.02):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=C_PHY, lw=ps.line_width()),
               Line2D([0], [0], color=C_ZOO, lw=ps.line_width()),
               Line2D([0], [0], color=C_PHY, lw=ps.line_width(), ls=":"),
               Line2D([0], [0], color=C_ZOO, lw=ps.line_width(), ls=":")]
    fig.legend(handles, ["PHY biomass", "ZOO biomass", "PP", "Zooplankton grazing"],
               loc="lower center", bbox_to_anchor=(x, y), ncol=4, frameon=False,
               fontsize=ps.font_small())

def fig_overview():
    fig = plt.figure(figsize=ps.figsize("MF_ZOO_prof_overview", 11.0, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[1, 1],
                          hspace=ps.keep_gap(0.26, fig), wspace=0.10)
    ax_s  = fig.add_subplot(gs[0, 0])
    ax_f  = fig.add_subplot(gs[0, 1], sharey=ax_s)
    ax_h  = fig.add_subplot(gs[0, 2], sharey=ax_s)
    ax_bs = fig.add_subplot(gs[1, 0], sharey=ax_s)
    ax_bf = fig.add_subplot(gs[1, 1], sharey=ax_s)
    ax_hc = fig.add_subplot(gs[1, 2], sharey=ax_s)
    draw_overview_panels(ax_s, ax_f, ax_h, ax_bs, ax_bf, ax_hc)
    _overview_legend(fig, 0.45)
    fig.subplots_adjust(top=ps.keep_top(0.92, fig), bottom=ps.keep_bottom(0.12, fig), left=0.07, right=0.94)
    finish(fig, "prof_overview")
    return fig

def fig_full():
    """Maps (left, two columns) and latitude profiles (right); rows = control / hose minus control.
    Biomass and PP/grazing panels are set to the width of a map axis, the herbivory panels to half
    of it. Profile axes take the vertical extent of the map axis of the same row, so latitudes line
    up (assumes latitude is the y axis of the map projection, e.g. plate carree)."""
    S = _map_setup()
    fig = plt.figure(figsize=ps.figsize("MF_ZOO_full", 15.0, 9.6, default_height_mm=FULL_HEIGHT_MM))
    H = fig.get_figheight()
    cell_h = (H - FULL_TOP_IN - FULL_BOTTOM_IN - FULL_GAP_IN) / 2.0      # margins and the row gap are set in inches
    outer = fig.add_gridspec(2, 6, width_ratios=[1, 1, FULL_SPACER, 1, 1, 0.5], height_ratios=[1, 1],   # column 2 is a spacer for the latitude label

                             hspace=FULL_GAP_IN / cell_h, wspace=FULL_WSPACE,
                             left=0.04, right=0.94, top=1 - FULL_TOP_IN / H, bottom=FULL_BOTTOM_IN / H)

    map_axes = [[None, None], [None, None]]
    for ci, col in enumerate(S["columns"]):
        for row in range(2):
            map_axes[row][ci] = _draw_map_cell(fig, outer[row, ci], row, ci, col, S)

    def prof_ax(cell, sharey=None):
        inner = cell.subgridspec(2, 1, height_ratios=[1, 0.06], hspace=0.06)
        return fig.add_subplot(inner[0, 0], sharey=sharey)

    ax_s  = prof_ax(outer[0, 3])
    ax_f  = prof_ax(outer[0, 4], ax_s)
    ax_h  = prof_ax(outer[0, 5], ax_s)
    ax_bs = prof_ax(outer[1, 3], ax_s)
    ax_bf = prof_ax(outer[1, 4], ax_s)
    ax_hc = prof_ax(outer[1, 5], ax_s)
    draw_overview_panels(ax_s, ax_f, ax_h, ax_bs, ax_bf, ax_hc)

    def place(ax, y0, height, width):
        p = ax.get_position()
        x0 = p.x0 + (p.width - width) / 2.0
        ax.set_position([x0, y0, width, height])

    m0 = map_axes[0][0]; m0.apply_aspect(); pos0 = m0.get_position()
    m1 = map_axes[1][0]; m1.apply_aspect(); pos1 = m1.get_position()
    place(ax_s,  pos0.y0, pos0.height, pos0.width)
    place(ax_f,  pos0.y0, pos0.height, pos0.width)
    place(ax_h,  pos0.y0, pos0.height, 0.5 * pos0.width)
    place(ax_bs, pos1.y0, pos1.height, pos1.width)
    place(ax_bf, pos1.y0, pos1.height, pos1.width)
    place(ax_hc, pos1.y0, pos1.height, 0.5 * pos1.width)

    if LETTERS_FULL:        # a) b) maps, c) d) e) profiles (top row); f) g) maps, h) i) j) profiles (bottom row)
        for ax, txt in ((map_axes[0][0], "a)"), (map_axes[0][1], "b)"), (ax_s, "c)"), (ax_f, "d)"), (ax_h, "e)"),
                        (map_axes[1][0], "f)"), (map_axes[1][1], "g)"), (ax_bs, "h)"), (ax_bf, "i)"), (ax_hc, "j)")):
            ps.panel_letter_inside(ax, txt)
    # legend centered under the profile block (panels c to j), close to the x labels
    x_mid = 0.5 * (ax_s.get_position().x0 + ax_h.get_position().x1)
    _overview_legend(fig, x_mid, FULL_LEGEND_Y_IN / H)
    finish(fig, "full")
    return fig


def main():
    """Run everything (this used to run at import time)."""
    global F, FLOW_LABEL, FLOW_SCALE, P_a, P_b, TO_MMOL, area_b, centers, cs_ab, csize, idx
    global lat_max, lat_min, lon_max, lon_min, nb, okbin, paths, run, sel_ab, var

    ps.apply_figure_style()
    lon_min, lon_max, lat_min, lat_max = mm._load_extent()

    ptrc = {r: clim_path(r, "ptrc_T_int") for r in (run_a, run_b)}
    diad = {r: clim_path(r, "diad_T_int") for r in (run_a, run_b)}
    for _d in (ptrc, diad):
        for _r, _p in _d.items():
            print("%s: %s" % (_r, _p))

    lat_n   = read_first(MESHMASK, LAT_NAMES)
    csize   = np.nan_to_num(read_first(MESHMASK, CSIZE_NAMES))

    masks_n = {r: read_mask(r) for r in REGIONS}
    masks_n["AB"] = np.zeros(csize.shape, dtype=bool)
    for r in REGIONS:
        masks_n["AB"] |= masks_n[r]
    for r, m in masks_n.items():
        if m.shape != csize.shape:
            raise ValueError("mask %s shape %s differs from csize %s" % (r, m.shape, csize.shape))
    if lat_n.shape != csize.shape:
        raise ValueError("lat shape %s differs from csize %s" % (lat_n.shape, csize.shape))

    F = {run_a: {}, run_b: {}}
    for run in (run_a, run_b):
        for var, paths in (("ZOO", ptrc), ("PHY", ptrc), ("PPT", diad),
                           ("Herbiv", diad), ("Carniv", diad), ("Detrit", diad)):
            arr, units = read_native(paths[run], var)
            if arr.shape != csize.shape:
                raise ValueError("%s (%s) has shape %s, native grid is %s. Check CLIM_PATTERN (not the _rg file)."
                                 % (var, run, arr.shape, csize.shape))
            F[run][var] = np.nan_to_num(arr)      # land and below-seafloor are 0 in _int files
            U[(run, var)] = units

    for var in ("ZOO", "PHY"):
        f_a = mmol_factor(U[(run_a, var)], var + " (control)")
        f_b = mmol_factor(U[(run_b, var)], var + " (hose)")
        assert f_a == f_b, "control and hose %s units differ" % var
    TO_MMOL = mmol_factor(U[(run_a, "ZOO")], "ZOO")
    assert TO_MMOL == mmol_factor(U[(run_a, "PHY")], "PHY"), "ZOO and PHY units differ"

    for run in (run_a, run_b):
        F[run]["FEED"] = F[run]["Herbiv"] + F[run]["Carniv"] + F[run]["Detrit"]

    PP_UNITS   = U[(run_a, "PPT")] or "units?"
    FEED_UNITS = U[(run_a, "Herbiv")] or "units?"

    PP_LABEL   = nice_units(PP_UNITS)
    FEED_LABEL = nice_units(FEED_UNITS)
    if PP_LABEL == MOLC_LABEL and FEED_LABEL == MOLC_LABEL:
        FLOW_SCALE = 1e6     # PP and grazing plotted x 1e6, with the 10^-6 written in the axis label
        FLOW_LABEL = r"$\times10^{-6}$ " + MOLC_LABEL
    else:
        FLOW_SCALE = 1.0
        FLOW_LABEL = PP_LABEL if PP_LABEL == FEED_LABEL else "%s (PP), %s (grazing)" % (PP_LABEL, FEED_LABEL)
    print("PPT units: %r; Herbiv units: %r (zooplankton grazing = Herbiv + Carniv + Detrit)"
          % (PP_UNITS, FEED_UNITS))

    print("\nRegional table (native grid, csize, binary masks). Stocks and flows are area sums in"
          " native units; ratios are ratios of sums.")
    print("%-4s %-12s %12s %12s %9s" % ("reg", "quantity", "control", "hose", "change"))
    for r in REGIONS + ["AB"]:
        sa, sb = regional(run_a, masks_n[r]), regional(run_b, masks_n[r])
        rows = [("PHY", sa["PHY"], sb["PHY"], True), ("ZOO", sa["ZOO"], sb["ZOO"], True),
                ("PP", sa["PPT"], sb["PPT"], True), ("GRAZING", sa["FEED"], sb["FEED"], True),
                ("ZOO/PHY", sa["ZOO"] / sa["PHY"], sb["ZOO"] / sb["PHY"], False),
                ("HERB/GRAZ", sa["Herbiv"] / sa["FEED"], sb["Herbiv"] / sb["FEED"], False)]
        for name, a, b, is_pct in rows:
            chg = ("%+8.2f%%" % (100.0 * (b - a) / a)) if is_pct else ("%+9.4f" % (b - a))
            print("%-4s %-12s %12.5g %12.5g %9s" % (r, name, a, b, chg))
    print("")

    # ── latitude profiles (native grid) ────────────────────────────────────
    edges = np.arange(np.floor(lat_min), np.ceil(lat_max) + BIN_DEG, BIN_DEG)
    nb = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])

    sel_ab = masks_n["AB"] & (lat_n >= edges[0]) & (lat_n < edges[-1])
    idx = np.clip(np.floor((lat_n[sel_ab] - edges[0]) / BIN_DEG).astype(int), 0, nb - 1)
    cs_ab = csize[sel_ab]
    area_b = np.bincount(idx, weights=cs_ab, minlength=nb)
    cnt_b  = np.bincount(idx, minlength=nb)
    okbin  = cnt_b >= MIN_CELLS

    P_a, P_b = profiles(run_a), profiles(run_b)

    # bin check: area-weighted sums over bins must equal the AB regional sum (within the lat range)
    chk_bins = float(np.sum(bsum(F[run_a]["ZOO"])))
    chk_ab   = float(np.sum(F[run_a]["ZOO"][sel_ab] * csize[sel_ab]))
    print("bin check, control ZOO summed over bins vs AB mask in lat range: %.6g vs %.6g" % (chk_bins, chk_ab))
    print("blank bins (< %d cells): %d of %d\n" % (MIN_CELLS, int(np.sum(~okbin)), nb))
    for r in REGIONS:
        lv = lat_n[masks_n[r]]
        if lv.size:
            lo, hi = np.percentile(lv, [1, 99])
            REG_LAT[r] = (round(lo), round(hi))

    if MAKE_MAPS:
        fig_maps()
        fig_full()
    fig_lines()
    fig_change()
    fig_overview()
    plt.show()


if __name__ == "__main__":
    main()
