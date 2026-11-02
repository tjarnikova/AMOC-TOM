"""
SF_CFLX_SIGN.py  --  Supplementary figure: air-sea C flux and where its sign changes

Two rows of Atlantic map panels (same machinery and conventions as MF_FLUX.py):

  Row 1
    1. Air-sea C flux, CONTROL
    2. Air-sea C flux, HOSE            (same symmetric color limits as panel 1)
    3. Cflx sign categories, HOSE vs CONTROL (4 colors, two reds and two blues)

  Row 2: do the SIGNS of the HOSE - CONTROL changes agree?
    4. Export and Cflx    (dots = Cflx increased)
    5. Export and PPINT   (dots = PPINT increased)
    6. PPINT and Cflx     (dots = PPINT increased)
       one color = agree (greenish), other = disagree (yellowish)

Cflx: positive = air to sea (into the ocean), negative = sea to air (out of the ocean).

Categories for row 1 (color family follows the HOSE state of the cell):
    stayed positive   CONTROL > 0, HOSE > 0   dark red     (into ocean in both)
    became positive   CONTROL < 0, HOSE > 0   light red    (flipped to into ocean)
    stayed negative   CONTROL < 0, HOSE < 0   dark blue    (out of ocean in both)
    became negative   CONTROL > 0, HOSE < 0   light blue   (flipped to out of ocean)

SIGN_EPS: cells with |Cflx| <= SIGN_EPS in either run are put in a grey
"near zero" class instead of being forced into a sign (default 0 = pure sign).
EXP_EPS / CFLX_EPS / PP_EPS do the same for the changes in row 2 (mol m-2 s-1).
EXP100 = mean of levels 9 and 10.  Cells where EXP100 is exactly zero in both runs are
treated as no data (assumed below the seafloor): EXCLUDE_EXACT_ZERO_EXP.
PPINT = PPT from the diad_T_int files.

Also prints, per region, the cos(latitude)-weighted share of each category on the
map_maker grid (1 degree _rg masks, approximate, for orientation only; use the
native-grid TabK_summary conventions for numbers that go in the paper).

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/SF_CFLX_SIGN_YYYYMMDD.png
"""

import sys, os
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')

WATERMARK_ON = True
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import map_maker as mm
from datetime import datetime

# -- tunable ------------------------------------------------------------
WATERMARK   = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/SF_CFLX_SIGN.py"
OUTPUT_DIR  = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
FIGURE_CODE = "SF_CFLX_SIGN"

SIGN_EPS          = 0.0      # mol m-2 s-1; 0 = pure sign
APPLY_PPINT_MASK  = False    # True = blank cells where PPINT < threshold, as in MF_FLUX Fig 6
PPINT_THRESHOLD   = 1e-9
CFLX_VLIM         = None     # None = shared symmetric limit from the Atlantic window
CMAP_CFLX         = "curl"

# row 2 (sign agreement of the HOSE - CONTROL changes)
EXP_DEPTH_LEVELS         = [9, 10]
EXP_EPS                  = 0.0      # mol m-2 s-1
CFLX_EPS                 = 0.0      # mol m-2 s-1
PP_EPS                   = 0.0      # mol m-2 s-1 (PPINT change)
EXCLUDE_EXACT_ZERO_EXP   = True
STIPPLE_STRIDE = 2          # one dot per STRIDE x STRIDE block of cells (1 = every cell)
STIPPLE_SIZE   = 3.0        # marker area, points^2
STIPPLE_COLOR  = "black"
STIPPLE_ALPHA  = 0.75
COLOR_AGREE    = "#7FC8B8"  # light teal   (light enough for black dots to read)
COLOR_DISAGREE = "#F2A65A"  # light orange
COLOR_NEAR0    = "#8C8C8C"

# category colors: reds = into ocean (positive), blues = out of ocean (negative);
# dark = stayed, light = became
CAT_COLORS = {
    "stay_pos": "#B2182B",
    "new_pos":  "#F4A582",
    "stay_neg": "#2166AC",
    "new_neg":  "#92C5DE",
    "near0":    "#BBBBBB",
}
CAT_ORDER  = ["stay_pos", "new_pos", "stay_neg", "new_neg", "near0"]   # integer codes 0..4
CAT_LABELS = {
    "stay_pos": "Stayed positive (into ocean)",
    "new_pos":  "Became positive (into ocean)",
    "stay_neg": "Stayed negative (out of ocean)",
    "new_neg":  "Became negative (out of ocean)",
    "near0":    "Near zero in either run",
}

# -- setup --------------------------------------------------------------
mm.set_presentation_style()
FS = plt.rcParams["font.size"]
lon_min, lon_max, lat_min, lat_max = mm._load_extent()
lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)
run_labels, runs = mm._load_models()
run_a, run_b = run_labels[0], run_labels[1]          # control, hose

diad      = mm._paths_for_filetype(runs, "diad_T")
diad_int  = mm._paths_for_filetype(runs, "diad_T_int")
cflx_cmap = mm._resolve_cmap(CMAP_CFLX, mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)

def load(paths, run, var, depth_index=0):
    lon, lat, field, _ = mm.load_field(paths[run], var, depth_index)
    return lon, lat, field

lon, lat, cflx_a = load(diad, run_a, "Cflx")
_,   _,   cflx_b = load(diad, run_b, "Cflx")
cflx_a = cflx_a.astype(float).copy()
cflx_b = cflx_b.astype(float).copy()

def exp100(run):
    stack = []
    for d in EXP_DEPTH_LEVELS:
        _, _, e = load(diad, run, "EXP", d)
        stack.append(e.astype(float))
    return np.nanmean(np.stack(stack, axis=0), axis=0)

exp_a, exp_b = exp100(run_a), exp100(run_b)
if EXCLUDE_EXACT_ZERO_EXP:
    below = (exp_a == 0) & (exp_b == 0)
    print("cells with EXP100 exactly zero in both runs (treated as no data): %d" % int(below.sum()))
    exp_a[below] = np.nan
    exp_b[below] = np.nan

# PPINT is always needed now (row 2); the mask is still optional
_, _, pp_a = load(diad_int, run_a, "PPT")
_, _, pp_b = load(diad_int, run_b, "PPT")
pp_a = pp_a.astype(float).copy()
pp_b = pp_b.astype(float).copy()

if APPLY_PPINT_MASK:
    bad = (pp_a < PPINT_THRESHOLD) | (pp_b < PPINT_THRESHOLD)
    for f in (cflx_a, cflx_b, exp_a, exp_b, pp_a, pp_b):
        f[bad] = np.nan

# -- sign categories (row 1, panel 3) ------------------------------------
valid = np.isfinite(cflx_a) & np.isfinite(cflx_b)
a_pos, a_neg = cflx_a > SIGN_EPS, cflx_a < -SIGN_EPS
b_pos, b_neg = cflx_b > SIGN_EPS, cflx_b < -SIGN_EPS

cat = np.full(cflx_a.shape, np.nan)
cat[valid] = 4                                        # near zero unless set below
cat[valid & a_pos & b_pos] = 0                        # stayed positive
cat[valid & a_neg & b_pos] = 1                        # became positive
cat[valid & a_neg & b_neg] = 2                        # stayed negative
cat[valid & a_pos & b_neg] = 3                        # became negative

cat_cmap = ListedColormap([CAT_COLORS[k] for k in CAT_ORDER])

# -- sign agreement of two changes (row 2) --------------------------------
dE = exp_b - exp_a
dC = cflx_b - cflx_a
dP = pp_b - pp_a

ag_cmap = ListedColormap([COLOR_AGREE, COLOR_DISAGREE, COLOR_NEAR0])

def agreement(dX, dY, epsX, epsY, dots_on):
    """Sign agreement of changes dX and dY.
    ag_cat codes: 0 both up, 1 both down, 2 X up / Y down, 3 X down / Y up, 4 near zero.
    draw codes:   0 agree, 1 disagree, 2 near zero.
    dots_on: "x" or "y" -> dots where that variable increased."""
    v = np.isfinite(dX) & np.isfinite(dY)
    x_up, x_dn = dX > epsX, dX < -epsX
    y_up, y_dn = dY > epsY, dY < -epsY
    c = np.full(dX.shape, np.nan)
    c[v] = 4
    c[v & x_up & y_up] = 0
    c[v & x_dn & y_dn] = 1
    c[v & x_up & y_dn] = 2
    c[v & x_dn & y_up] = 3
    draw = np.full(dX.shape, np.nan)
    draw[(c == 0) | (c == 1)] = 0
    draw[(c == 2) | (c == 3)] = 1
    draw[c == 4] = 2
    dots = ((c == 0) | (c == 3)) if dots_on == "y" else ((c == 0) | (c == 2))
    return c, draw, dots

def agree_note(nx, ny, dots_on):
    if dots_on == "y":
        return ("Agree: dots = both up; no dots = both down.\n"
                "Disagree: dots = %s down, %s up;\n"
                "no dots = %s up, %s down." % (nx, ny, nx, ny))
    return ("Agree: dots = both up; no dots = both down.\n"
            "Disagree: dots = %s up, %s down;\n"
            "no dots = %s down, %s up." % (nx, ny, nx, ny))

# (title, X name, Y name, dX, dY, epsX, epsY, which variable gets dots)
AGREE_SPECS = [
    ("Export and Cflx sign agreement",  "Export", "Cflx",  dE, dC, EXP_EPS, CFLX_EPS, "y"),
    ("Export and PPINT sign agreement", "Export", "PPINT", dE, dP, EXP_EPS, PP_EPS,   "y"),
    ("PPINT and Cflx sign agreement",   "PPINT",  "Cflx",  dP, dC, PP_EPS,  CFLX_EPS, "x"),
]

agree_panels = []
for title, nx, ny, dX, dY, eX, eY, don in AGREE_SPECS:
    c, draw, dots = agreement(dX, dY, eX, eY, don)
    dot_name = ny if don == "y" else nx
    agree_panels.append(dict(
        title=title, nx=nx, ny=ny, cat=c, draw=draw, dots=dots, dot_name=dot_name,
        note=agree_note(nx, ny, don), any_eps=(eX > 0 or eY > 0)))

# -- limits for the two flux maps ---------------------------------------
def atl_vals(field):
    return mm._bbox_subset(lon, lat, field, lon_min, lon_max, lat_min, lat_max)

def sym_limit(field, pct=95):
    v = atl_vals(field)
    return np.nanpercentile(np.abs(v[np.isfinite(v)]), pct) * mm.DIFF_LIMIT_SCALE

cflx_lim = CFLX_VLIM or max(sym_limit(cflx_a), sym_limit(cflx_b))

# -- summary tables (approximate, map_maker grid) --------------------------
def _regions_and_weights():
    lat2 = np.meshgrid(lon, lat)[1] if np.ndim(lat) == 1 else lat
    wgt = np.cos(np.deg2rad(lat2))
    regs = dict(masks)
    try:
        regs["Atlantic"] = np.any([np.asarray(masks[k]) > 0.5 for k in masks], axis=0)
    except Exception:
        pass
    return regs, wgt

def print_summary():
    regs, wgt = _regions_and_weights()
    print("\nShare of valid ocean area in each category (cos-lat weighted, map grid, approximate)")
    print("%-9s" % "" + "".join("%16s" % k for k in CAT_ORDER[:4]) + "   net-flip")
    for name, m in regs.items():
        m = np.asarray(m) > 0.5
        if m.shape != cat.shape:
            print("%-9s mask shape %s != field shape %s, skipped" % (name, m.shape, cat.shape))
            continue
        sel = m & np.isfinite(cat)
        tot = np.sum(wgt[sel])
        if tot == 0:
            continue
        shares = [np.sum(wgt[sel & (cat == i)]) / tot * 100 for i in range(4)]
        print("%-9s" % name + "".join("%15.1f%%" % s for s in shares)
              + "   %+.1f pp" % (shares[1] - shares[3]))
    print("net-flip = became positive minus became negative (percentage points of area).")

def print_agree_summary(ap):
    regs, wgt = _regions_and_weights()
    nx, ny, acat = ap["nx"], ap["ny"], ap["cat"]
    names = ["%s up, %s up" % (nx, ny), "%s dn, %s dn" % (nx, ny),
             "%s up, %s dn" % (nx, ny), "%s dn, %s up" % (nx, ny)]
    print("\n%s and %s sign agreement: share of valid ocean area "
          "(cos-lat weighted, map grid, approximate)" % (nx, ny))
    print("%-9s" % "" + "".join("%19s" % n for n in names) + "    agree")
    for name, m in regs.items():
        m = np.asarray(m) > 0.5
        if m.shape != acat.shape:
            print("%-9s mask shape %s != field shape %s, skipped" % (name, m.shape, acat.shape))
            continue
        sel = m & np.isfinite(acat)
        tot = np.sum(wgt[sel])
        if tot == 0:
            continue
        sh = [np.sum(wgt[sel & (acat == i)]) / tot * 100 for i in range(4)]
        print("%-9s" % name + "".join("%18.1f%%" % x for x in sh) + "   %5.1f%%" % (sh[0] + sh[1]))
    print("agree = both increased + both decreased (share of valid area).")

print_summary()
for ap in agree_panels:
    print_agree_summary(ap)

# -- figure ---------------------------------------------------------------
NCOL, NROW = 3, 2
fig = plt.figure(figsize=(3.4 * NCOL + 1.0, 6.6 * NROW))
LEG_H = [0.30, 0.30]      # legend-area height per row, relative to the map
gs = fig.add_gridspec(NROW, NCOL, wspace=0.14, hspace=0.14,
                      height_ratios=[1 + h for h in LEG_H],
                      left=0.03, right=0.99, top=0.95, bottom=0.04)

panels = [
    dict(title="CONTROL", field=cflx_a, cmap=cflx_cmap, vmin=-cflx_lim, vmax=cflx_lim, kind="flux"),
    dict(title="HOSE",    field=cflx_b, cmap=cflx_cmap, vmin=-cflx_lim, vmax=cflx_lim, kind="flux"),
    dict(title=r"Sign change, HOSE vs CONTROL", field=cat, cmap=cat_cmap,
         vmin=-0.5, vmax=4.5, kind="cat"),
] + [dict(title=ap["title"], field=ap["draw"], cmap=ag_cmap, vmin=-0.5, vmax=2.5,
          kind="agree", ap=ap) for ap in agree_panels]

for pi, p in enumerate(panels):
    row, col = divmod(pi, NCOL)
    inner = gs[row, col].subgridspec(2, 1, height_ratios=[1, LEG_H[row]], hspace=0.05)
    ax = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
    mesh = mm._draw_panel(ax, lon, lat, p["field"], p["cmap"], p["vmin"], p["vmax"],
                          lon_m, lat_m, masks,
                          extent=[lon_min, lon_max, lat_min, lat_max])
    ax.set_title(p["title"], fontweight="bold")

    if p["kind"] == "flux":
        cb_gs = inner[1, 0].subgridspec(2, 3, width_ratios=[0.15, 0.70, 0.15],
                                        height_ratios=[0.3, 0.7], wspace=0, hspace=0)
        cax = fig.add_subplot(cb_gs[0, 1])
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cbar.outline.set_visible(False)
        cbar.formatter = mm._value_formatter()
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=FS - 2)
        cbar.set_label(r"Air-sea C flux (mol C m$^{-2}$ s$^{-1}$)" + "\n(positive = into ocean)",
                       fontsize=FS - 2)
    elif p["kind"] == "agree":
        ap = p["ap"]
        lon2, lat2 = np.meshgrid(lon, lat) if np.ndim(lon) == 1 else (lon, lat)
        thin = np.zeros(ap["dots"].shape, dtype=bool)
        thin[::STIPPLE_STRIDE, ::STIPPLE_STRIDE] = True
        dsel = ap["dots"] & thin
        ax.scatter(lon2[dsel], lat2[dsel], s=STIPPLE_SIZE, marker="o", color=STIPPLE_COLOR,
                   alpha=STIPPLE_ALPHA, linewidths=0, transform=ccrs.PlateCarree(), zorder=4)
        lax = fig.add_subplot(inner[1, 0])
        lax.axis("off")
        handles = [
            mpatches.Patch(facecolor=COLOR_AGREE, edgecolor="black", linewidth=0.5,
                           label="Agree (same sign)"),
            mpatches.Patch(facecolor=COLOR_DISAGREE, edgecolor="black", linewidth=0.5,
                           label="Disagree (opposite signs)"),
            mlines.Line2D([], [], marker="o", linestyle="none", markersize=3.5,
                          markerfacecolor=STIPPLE_COLOR, markeredgecolor="none",
                          label="Dots: %s increased" % ap["dot_name"]),
        ]
        if ap["any_eps"]:
            handles.insert(2, mpatches.Patch(facecolor=COLOR_NEAR0, edgecolor="black", linewidth=0.5,
                                             label="Near zero in either change"))
        lax.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
                   fontsize=FS - 3, handlelength=1.4, labelspacing=0.5)
    else:
        lax = fig.add_subplot(inner[1, 0])
        lax.axis("off")
        keys = CAT_ORDER[:4] + (["near0"] if SIGN_EPS > 0 else [])
        handles = [mpatches.Patch(facecolor=CAT_COLORS[k], edgecolor="black",
                                  linewidth=0.5, label=CAT_LABELS[k]) for k in keys]
        lax.legend(handles=handles, loc="upper center", ncol=1, frameon=False,
                   fontsize=FS - 3, columnspacing=1.2, handlelength=1.2, labelspacing=0.5)

if WATERMARK_ON:
    fig.text(0.5, 0.004, WATERMARK, ha="center", va="bottom", fontsize=FS - 6, color="0.55")

os.makedirs(OUTPUT_DIR, exist_ok=True)
out = "{}/{}_{}.png".format(OUTPUT_DIR, FIGURE_CODE, datetime.now().strftime("%Y%m%d"))
fig.savefig(out, bbox_inches="tight", dpi=300)
print("saved " + out)
plt.show()
