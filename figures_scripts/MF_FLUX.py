"""
MF_FLUX.py  --  Main Figure 6: export, ef, air-sea carbon flux, and the regional
                 export budget (production vs efficiency).

Left block, 2 rows x 3 map cols (unchanged from the previous MF_FLUX):
    col 0 -- EXP100
    col 1 -- ef (EXP100 / PP, clipped to [0, 1] for the maps only)
    col 2 -- Air-sea C flux (Cflx)
    row 0 -- CONTROL
    row 1 -- HOSE-CONTROL

Right column (figure 1, MF_FLUX):
    regional budget bars (PgC/yr): production, efficiency, cross, total
    (numbers from MF_EXPD.decompose / regional_budget, so they are
    identical to the supplementary MF_EXPD figure), with the bar legend below.

Figure 2 (MF_FLUX_scatter), its own figure:
    Delta export vs Delta Cflx by region (PgC/yr). Has its own legend (region names),
    since it no longer sits next to the bar labels and map outlines.

    ΔEXP100 = ef_C ΔPP + PP_C Δef + ΔPP Δef        (default, control split)
    python MF_FLUX.py --midpoint                    (midpoint split, no cross term)

Region colors are shared by the map outlines, the scatter points and the
colored markers next to the bar labels (SCATTER_COLORS / AB_COLOR / GO_COLOR).

Run order after the NHL rename: updated map_maker, TabK_summary, MF_EXPD, then this.

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_FLUX_YYYYMMDD.png          maps + budget bars
        /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_FLUX_scatter_YYYYMMDD.png  export vs Cflx scatter
"""

import sys, os
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')

# command-line flags are handled BEFORE importing MF_EXPD (it also strips flags)
WATERMARK_ON = True      # False = no script-path watermark (or --no-watermark / --water-off)
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
SPLIT = "control"
if "--midpoint" in sys.argv:
    sys.argv.remove("--midpoint")
    SPLIT = "midpoint"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
import map_maker as mm
import TabK_summary as tk
import MF_EXPD as ex
from datetime import datetime

try:
    from plot_style import BAUHAUS
except ImportError:
    BAUHAUS = ["#CC6677", "#DDCC77", "#117733", "#332288",
               "#AA4499", "#44AA99", "#999933"]

# -- tunable ------------------------------------------------------------
WATERMARK        = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_FLUX.py"
OUTPUT_DIR       = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
FIGURE_CODE      = "MF_FLUX"

EXP_DEPTH_LEVELS = [9, 10]
PPINT_THRESHOLD  = 1e-9

EF_VMIN      = 0.0
EF_VMAX      = 0.35
EF_DIFF_VLIM = 0.1

CFLX_VLIM      = None
CFLX_DIFF_VLIM = None

CMAP_EXP  = "speed"
CMAP_EF   = "matter"
CMAP_CFLX = "curl"

ROW_LABELS  = ["CONTROL", "HOSE\u2212CONTROL"]
ROW_LABEL_X = -0.13

# scatter: set to a CSV path to skip recomputing, or None to recompute
SCATTER_CSV  = None
RUN_DISPLAY  = {"BASE": "CONTROL"}

BUDGET_TITLE = "Contributions of PP and ef\nto export change"

PROVINCE_FULL = {
    "NHL": "Northern High Latitudes", "NA": "North Atlantic",
    "SA": "South Atlantic",           "SO": "Southern Ocean",
    "AB": "Atlantic Basin",           "GO": "Global Ocean",
}
SCATTER_COLORS = {
    "NHL": BAUHAUS[3], "NA": BAUHAUS[2],
    "SA": BAUHAUS[1], "SO": BAUHAUS[0],
}
AB_COLOR = "forestgreen"
try:
    GO_COLOR = BAUHAUS[5]
except IndexError:
    GO_COLOR = "darkgoldenrod"

FS = None  # set after mm.set_presentation_style()

# -- map data setup -----------------------------------------------------
mm.set_presentation_style()
FS = plt.rcParams["font.size"]
lon_min, lon_max, lat_min, lat_max = mm._load_extent()
lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)
run_labels, runs = mm._load_models()
run_a, run_b = run_labels[0], run_labels[1]

diad     = mm._paths_for_filetype(runs, "diad_T")
diad_int = mm._paths_for_filetype(runs, "diad_T_int")
diff_cmap = mm._resolve_cmap(None,       mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)
cflx_cmap = mm._resolve_cmap(CMAP_CFLX,  mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)

def load(paths, run, var, depth_index=0):
    lon, lat, field, _ = mm.load_field(paths[run], var, depth_index)
    return lon, lat, field

lon, lat, ppint_a = load(diad_int, run_a, "PPT")
_,   _,   ppint_b = load(diad_int, run_b, "PPT")
ppint_mask = (ppint_a < PPINT_THRESHOLD) | (ppint_b < PPINT_THRESHOLD)

def apply_mask(field):
    out = field.astype(float).copy()
    out[ppint_mask] = np.nan
    return out

exp_a_stack, exp_b_stack = [], []
for d in EXP_DEPTH_LEVELS:
    _, _, ea = load(diad, run_a, "EXP", d)
    _, _, eb = load(diad, run_b, "EXP", d)
    exp_a_stack.append(ea); exp_b_stack.append(eb)
exp100_a = apply_mask(np.nanmean(np.stack(exp_a_stack, axis=0), axis=0))
exp100_b = apply_mask(np.nanmean(np.stack(exp_b_stack, axis=0), axis=0))

def ef_map(exp100, pp):
    """ef = EXP100 / PP, clipped to [0, 1] (maps only; the bars use unclipped ef)."""
    pp_m = apply_mask(pp)
    eps = 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = exp100 / np.where(np.abs(pp_m) > eps, pp_m, eps)
    return np.where(np.isfinite(ratio), np.clip(ratio, 0, 1), np.nan)

ef_a = ef_map(exp100_a, ppint_a)
ef_b = ef_map(exp100_b, ppint_b)

_, _, cflx_a_raw = load(diad, run_a, "Cflx")
_, _, cflx_b_raw = load(diad, run_b, "Cflx")
cflx_a = apply_mask(cflx_a_raw); cflx_b = apply_mask(cflx_b_raw)

def atl_vals(field):
    return mm._bbox_subset(lon, lat, field, lon_min, lon_max, lat_min, lat_max)

def sym_limit(field, pct=95):
    v = atl_vals(field)
    return np.nanpercentile(np.abs(v[np.isfinite(v)]), pct) * mm.DIFF_LIMIT_SCALE

cflx_ctrl_lim = CFLX_VLIM      or sym_limit(cflx_a)
cflx_diff_lim = CFLX_DIFF_VLIM or sym_limit(cflx_b - cflx_a)
exp_vmin, exp_vmax = mm._percentile_limits(
    atl_vals(exp100_a)[np.isfinite(atl_vals(exp100_a))], mm.BASE_PERCENTILES)
exp_diff_lim = mm._diff_limit(
    atl_vals(exp100_b - exp100_a)[np.isfinite(atl_vals(exp100_b - exp100_a))])

MAP_COLUMNS = [
    dict(title="EXP100",
         ctrl=exp100_a, diff=exp100_b - exp100_a,
         cmap_ctrl=mm._resolve_cmap(CMAP_EXP, mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK),
         cmap_diff=diff_cmap,
         vmin_ctrl=exp_vmin,      vmax_ctrl=exp_vmax,
         vmin_diff=-exp_diff_lim, vmax_diff=exp_diff_lim,
         units_ctrl=r"mol C m$^{-2}$ s$^{-1}$",
         units_diff=r"$\Delta$ mol C m$^{-2}$ s$^{-1}$"),
    dict(title="ef",
         ctrl=ef_a, diff=ef_b - ef_a,
         cmap_ctrl=mm._resolve_cmap(CMAP_EF, mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK),
         cmap_diff=diff_cmap,
         vmin_ctrl=EF_VMIN,        vmax_ctrl=EF_VMAX,
         vmin_diff=-EF_DIFF_VLIM,  vmax_diff=EF_DIFF_VLIM,
         units_ctrl="\u2014",
         units_diff=r"$\Delta$ ef"),
    dict(title="Air-sea C flux",
         ctrl=cflx_a, diff=cflx_b - cflx_a,
         cmap_ctrl=cflx_cmap, cmap_diff=cflx_cmap,
         vmin_ctrl=-cflx_ctrl_lim, vmax_ctrl=cflx_ctrl_lim,
         vmin_diff=-cflx_diff_lim, vmax_diff=cflx_diff_lim,
         units_ctrl=r"mol C m$^{-2}$ s$^{-1}$",
         units_diff=r"$\Delta$ mol C m$^{-2}$ s$^{-1}$"),
]

# -- regional data: scatter (TabK_summary) and budget (MF_EXPD, native grid) --
if SCATTER_CSV is not None:
    df_scatter = pd.read_csv(SCATTER_CSV, keep_default_na=False, na_values=[])
else:
    from TabK_summary import compute_province_means
    df_scatter, _ = compute_province_means()

run_a_disp = RUN_DISPLAY.get(run_a, run_a)
run_b_disp = RUN_DISPLAY.get(run_b, run_b)

def scatter_diff(var, prov):
    sub = df_scatter[(df_scatter["variable"] == var) & (df_scatter["province"] == prov)]
    piv = sub.set_index("run")["value"]
    return float(piv[run_b] - piv[run_a])

nat = {}
for run in (run_a, run_b):
    path = os.path.join(runs[run]["clim_dir"],
                        tk.DIAD_FILENAME_TEMPLATE.format(clim_period=str(runs[run]["clim_period"])))
    nat[run] = ex.load_native(path)
P_c, E_c = nat[run_a]
P_h, E_h = nat[run_b]
csize, masks_nat = ex.load_native_geometry()
dec_nat = ex.decompose(E_c, E_h, P_c, P_h, split=SPLIT)
budget = ex.regional_budget(E_c, E_h, P_c, P_h, dec_nat, csize, masks_nat)
print("\nRegional budget (PgC/yr), split =", SPLIT)
print(budget.round(4).to_string())
print("max |closure| = {:.2e} PgC/yr".format(budget["closure"].abs().max()))

def region_color(p):
    if p == "AB":
        return AB_COLOR
    if p == "GO":
        return GO_COLOR
    return SCATTER_COLORS[p]

def region_marker(p):
    return {"AB": "s", "GO": "*"}.get(p, "o")

# -- plotting helpers ------------------------------------------------------
def draw_budget_bars(ax, lax):
    """Bars in ax, legend in the separate axes lax."""
    provs = [p for p in ex.BAR_PROVINCES if p in budget.index]
    ypos, k = {}, 0
    for p in provs:
        if p == "AB":
            k += 0.6                      # gap before the aggregate
        ypos[p] = k
        k += 1
    terms = [("prod", "production"), ("eff", "efficiency")]
    if SPLIT == "control":
        terms.append(("cross", "cross"))
    show_resid = (budget.loc[provs, "pp_zero_cells"].abs().max()
                  > 0.02 * budget.loc[provs, "dEXP"].abs().max())
    if show_resid:
        terms.append(("resid", "pp_zero_cells"))
    h = 0.8 / len(terms)
    for ti, (key, col) in enumerate(terms):
        for p in provs:
            y = ypos[p] + (ti - (len(terms) - 1) / 2) * h
            ax.barh(y, budget.loc[p, col], height=h * 0.92,
                    color=ex.TERM_COLORS[key], edgecolor="black", linewidth=0.5,
                    hatch="///" if key == "resid" else None, zorder=3)
    for p in provs:
        ax.scatter(budget.loc[p, "dEXP"], ypos[p], marker="D", s=55, color="black",
                   edgecolor="white", linewidth=0.8, zorder=5)
    ax.axvline(0, color="0.4", linewidth=0.8, zorder=2)

    # region labels: text plus a marker in the region color (same as map outline / scatter)
    ax.set_yticks([ypos[p] for p in provs])
    ax.set_yticklabels(provs, fontweight="bold")
    ax.tick_params(axis="y", pad=16)
    for p in provs:
        ax.plot([-0.045], [ypos[p]], marker=region_marker(p), markersize=9,
                markerfacecolor=region_color(p), markeredgecolor="black",
                markeredgewidth=0.8, linestyle="none", clip_on=False,
                transform=ax.get_yaxis_transform(), zorder=6)
    ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ EXP100 ({}\u2212{})\nPgC yr$^{{-1}}$".format(run_b_disp, run_a_disp),
                  fontsize=FS - 1)
    ax.set_title(BUDGET_TITLE, fontweight="bold", fontsize=FS - 3)
    ax.tick_params(axis="x", labelsize=FS - 2)
    ax.tick_params(axis="y", labelsize=FS - 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="0.9", zorder=0)

    if SPLIT == "control":
        lab_prod  = r"Production ($\mathrm{ef}_C\,\Delta PP$)"
        lab_eff   = r"Efficiency ($PP_C\,\Delta\mathrm{ef}$)"
        lab_cross = r"Cross ($\Delta PP\,\Delta\mathrm{ef}$)"
    else:
        lab_prod  = r"Production ($\overline{\mathrm{ef}}\,\Delta PP$)"
        lab_eff   = r"Efficiency ($\overline{PP}\,\Delta\mathrm{ef}$)"
        lab_cross = None
    sq = dict(marker="s", linestyle="none", markersize=9, markeredgecolor="black")
    handles = [mlines.Line2D([], [], markerfacecolor=ex.TERM_COLORS["prod"], label=lab_prod, **sq),
               mlines.Line2D([], [], markerfacecolor=ex.TERM_COLORS["eff"],  label=lab_eff,  **sq)]
    if lab_cross:
        handles.append(mlines.Line2D([], [], markerfacecolor=ex.TERM_COLORS["cross"],
                                     label=lab_cross, **sq))
    if show_resid:
        handles.append(mlines.Line2D([], [], markerfacecolor="white",
                                     label=r"PP$\approx$0 cells", **sq))
    handles.append(mlines.Line2D([], [], marker="D", linestyle="none", markersize=7,
                                 markerfacecolor="black", markeredgecolor="white",
                                 label=r"Total $\Delta$EXP100"))
    lax.axis("off")
    lax.legend(handles=handles, loc="upper center", frameon=False,
               fontsize=FS - 3, ncol=1, handlelength=1.0, labelspacing=0.4)


FIT_PROVINCES = ["NHL", "NA", "SA", "SO"]      # regression uses these four only (not AB, GO)

def fit_export_cflx():
    """OLS of dCflx (y) on dEXP100 (x) across NHL, NA, SA, SO. Prints everything."""
    xs = np.array([scatter_diff("export", p) for p in FIT_PROVINCES])
    ys = np.array([scatter_diff("cflx",   p) for p in FIT_PROVINCES])
    res = stats.linregress(xs, ys)
    n = len(xs)
    tcrit = stats.t.ppf(0.975, n - 2)
    ci = (res.slope - tcrit * res.stderr, res.slope + tcrit * res.stderr)
    print("\nExport vs Cflx regression (n = {}: {})".format(n, ", ".join(FIT_PROVINCES)))
    print("  {:<5s} {:>14s} {:>14s}   (PgC/yr, {} minus {})".format(
        "", "dEXP100", "dCflx", run_b_disp, run_a_disp))
    for p, x, y in zip(FIT_PROVINCES, xs, ys):
        print("  {:<5s} {:>14.4f} {:>14.4f}".format(p, x, y))
    print("  slope  (dCflx / dEXP100) = {:.4f} PgC/yr per PgC/yr  (+/- {:.4f} s.e.)".format(
        res.slope, res.stderr))
    print("  95% CI on slope (t, df={}) = [{:.4f}, {:.4f}]".format(n - 2, ci[0], ci[1]))
    print("  intercept            = {:.4f} PgC/yr".format(res.intercept))
    print("  r                    = {:.4f}".format(res.rvalue))
    print("  R^2                  = {:.4f}".format(res.rvalue ** 2))
    print("  p (two-sided)        = {:.4g}".format(res.pvalue))
    print("  Change in Cflx per 0.1 PgC/yr change in export = {:.4f} PgC/yr".format(res.slope * 0.1))
    print("  Cflx > 0 is air to sea, so a positive slope means export falls with weaker uptake.")
    print("  Caveat: n = 4, so p and the CI rest on 2 degrees of freedom.")
    return res

def draw_scatter(ax):
    """Standalone scatter: legend with full region names (marker and color as in the
    map outlines and the bar labels)."""
    handles = []
    for prov in ["NHL", "NA", "SA", "SO", "AB", "GO"]:
        x = scatter_diff("export", prov); y = scatter_diff("cflx", prov)
        if prov == "AB":
            kw = dict(marker="s", s=180)
        elif prov == "GO":
            kw = dict(marker="*", s=400)
        else:
            kw = dict(marker="o", s=120)
        ax.scatter(x, y, color=region_color(prov), zorder=4,
                   edgecolor="black", linewidth=0.8, **kw)
        handles.append(mlines.Line2D(
            [], [], marker=region_marker(prov), linestyle="none",
            markersize=14 if prov == "GO" else 10,
            markerfacecolor=region_color(prov), markeredgecolor="black",
            markeredgewidth=0.8,
            label="{} ({})".format(PROVINCE_FULL[prov], prov)))
    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    ax.axvline(0, color="0.6", linewidth=0.8, zorder=1)

    # same scale on both axes, range set by every plotted point (incl. AB, GO)
    allv = np.array([scatter_diff(v, p) for v in ("export", "cflx")
                     for p in ["NHL", "NA", "SA", "SO", "AB", "GO"]])
    pad = 0.08 * (allv.max() - allv.min())
    lo, hi = allv.min() - pad, allv.max() + pad
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # regression line through NHL, NA, SA, SO
    res = fit_export_cflx()
    xx = np.array([lo, hi])
    ax.plot(xx, res.intercept + res.slope * xx, color="0.25", linewidth=1.5,
            linestyle="--", zorder=3)
    handles.append(mlines.Line2D([], [], color="0.25", linestyle="--", linewidth=1.5,
        label="Fit, NHL+NA+SA+SO\nslope = {:.2f}, $R^2$ = {:.2f}\np = {:.3g}".format(
            res.slope, res.rvalue ** 2, res.pvalue)))

    ax.set_xlabel("$\\Delta$ EXP100 ({}\u2212{})\nPgC yr$^{{-1}}$".format(run_b_disp, run_a_disp),
                  fontsize=FS - 1)
    ax.set_ylabel(r"$\Delta$ Cflx (PgC yr$^{-1}$)", fontsize=FS - 1)
    ax.set_title("Export vs Cflx", fontweight="bold")
    ax.tick_params(labelsize=FS - 2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5),
              frameon=False, fontsize=FS - 2, labelspacing=0.7, handletextpad=0.6)
    try:
        ax.set_box_aspect(1)          # square plotting area (matplotlib >= 3.3)
    except AttributeError:
        pass


def save_fig(fig, name=None):
    if WATERMARK_ON:
        fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom",
                 fontsize=FS - 6, color="0.55")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    code = FIGURE_CODE if name is None else "{}_{}".format(FIGURE_CODE, name)
    out = "{}/{}_{}.png".format(OUTPUT_DIR, code, stamp)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print("saved " + out)


# -- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(3.4 * 3 + 3.8, 10.0))
outer_gs = fig.add_gridspec(
    2, 4,
    width_ratios=[1, 1, 1, 0.95],
    hspace=0.18, wspace=0.14,
    left=0.05, right=0.99, top=0.94, bottom=0.07)

for ci, col in enumerate(MAP_COLUMNS):
    for row, (field, cmap, vmin, vmax, units) in enumerate([
        (col["ctrl"], col["cmap_ctrl"], col["vmin_ctrl"], col["vmax_ctrl"], col["units_ctrl"]),
        (col["diff"], col["cmap_diff"], col["vmin_diff"], col["vmax_diff"], col["units_diff"]),
    ]):
        inner = outer_gs[row, ci].subgridspec(2, 1, height_ratios=[1, 0.06], hspace=0.06)
        ax  = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
        cbar_gs = inner[1, 0].subgridspec(1, 3, width_ratios=[0.15, 0.70, 0.15], wspace=0)
        cax = fig.add_subplot(cbar_gs[0, 1])

        mesh = mm._draw_panel(ax, lon, lat, field, cmap, vmin, vmax,
                              lon_m, lat_m, masks,
                              extent=[lon_min, lon_max, lat_min, lat_max])
        if row == 0:
            ax.set_title(col["title"], fontweight="bold")
        if ci == 0:
            ax.text(ROW_LABEL_X, 0.5, ROW_LABELS[row],
                    transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontweight="bold")

        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cbar.outline.set_visible(False)
        cbar.formatter = mm._value_formatter()
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=FS - 2)
        cbar.set_label(units, fontsize=FS - 2)

# right column: bars, x-label row, then the bar legend in its own axes; equal padding rows
# above and below centre the whole block vertically (nudge the two 1.0s to taste)
right = outer_gs[:, 3].subgridspec(
    5, 1, height_ratios=[1.0, 4.6, 1.0, 1.4, 1.0], hspace=0.0)
ax_bars = fig.add_subplot(right[1, 0])
lax_bars = fig.add_subplot(right[3, 0])
draw_budget_bars(ax_bars, lax_bars)
save_fig(fig)

# -- figure 2: scatter on its own -------------------------------------------
fig_sc, ax_sc = plt.subplots(figsize=(7.2, 5.6))
draw_scatter(ax_sc)
fig_sc.subplots_adjust(left=0.12, right=0.62, top=0.92, bottom=0.16)
save_fig(fig_sc, "scatter")
plt.show()
