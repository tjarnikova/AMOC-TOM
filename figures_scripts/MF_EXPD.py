"""
MF_EXPD.py  —  Main Figure: Export decomposition (production vs efficiency)

    ΔExport100 = f_C·ΔPP  +  PP_C·Δf  +  ΔPP·Δf              (--split control, default)
               = f̄·ΔPP   +  PP̄·Δf                            (--split midpoint, no cross term)

with f = EXP100 / PPINT, evaluated on annual means, UNCLIPPED (clipping to
[0,1], as in MF_FLUX, would break the exact closure). C = CONTROL (run_a),
bar = mean of the two runs.

    2 rows x 3 map cols + regional bar panel (spans both rows):
        row 0 -- ΔPP            | Δf-ratio         | ΔExport100 (total)
        row 1 -- production term | efficiency term | cross term
        right -- regional bars (PgC/yr): NHL NA SA SO | AB, terms + total marker

Consistency with the existing pipeline
    * MAPS   : regridded (_rg) climatologies, same as MF_FLUX (EXP levels 9, 10
               averaged, PPINT, cells masked where PPINT < PP_THRESHOLD in
               either run).
    * TOTALS : NATIVE grid, same files / province masks / cell areas / unit
               conversion as TabK_summary.py (constants imported from it), so
               the regional numbers add up to TabK's export and productivity.
    * Cells with PPINT < PP_THRESHOLD in either run (f undefined) are not
      decomposed; their ΔExport goes into a separate "PP≈0 cells" term so the
      bars still add up to TabK's ΔExport exactly.
    * PPINT (model's own integral) is used; PPT in diad_T_int agrees with it
      to <0.1% (see test_int.py).

Outputs
    /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_EXPD_YYYYMMDD.png
    /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_EXPD_regional_YYYYMMDD.csv   (PgC/yr, feeds the budget figure)

Usage
    python MF_EXPD.py                       # control-referenced split
    python MF_EXPD.py --split midpoint
    python MF_EXPD.py --tabk-csv /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/TabK_summary_HA01.csv   # cross-check
    python MF_EXPD.py --no-plot             # numbers only
"""

import sys
import os
import argparse
import warnings
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')

from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import map_maker as mm
import TabK_summary as tk

try:
    from plot_style import BAUHAUS
except ImportError:
    BAUHAUS = ["#CC6677", "#DDCC77", "#117733", "#332288",
               "#AA4499", "#44AA99", "#999933"]

# ── tunable ────────────────────────────────────────────────────────────
WATERMARK   = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_EXPD.py"
WATERMARK_ON   = True    # False = no script-path watermark (or run with --no-watermark / --water-off)
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR  = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
FIGURE_CODE = "MF_EXPD"

PP_VAR       = "PPINT"
EXP_VAR      = "EXP"
EXP_LEVELS   = tuple(tk.EXP100_DEPTH_LEVELS)     # (9, 10) -> ~101 m
DEPTH_DIM    = tk.EXP100_DEPTH_DIM_NAME          # "grid_T_3D_inner"
PP_THRESHOLD = 1e-9                              # mol C m-2 s-1, as MF_FLUX

DF_VLIM      = 0.1                               # Δf colour limit, as MF_FLUX

CMAP_PP      = "speed"                           # unused for diffs, kept for reuse
ROW_LABELS   = ["HOSE\u2212CONTROL", "DECOMPOSITION"]
ROW_LABEL_X  = -0.13
RUN_DISPLAY  = {"BASE": "CONTROL", "HOSE": "HOSE"}

BAR_PROVINCES = ["NHL", "NA", "SA", "SO", "AB"]   # GO goes to the CSV only
TERM_COLORS = {
    "prod":  "#44AA99",
    "eff":   "#AA4499",
    "cross": "0.65",
    "resid": "white",
}

MOL_M2S_TO_PGC_YR_PER_M2 = tk.MOLAR_MASS_C * tk.PG_PER_G * tk.SECONDS_PER_YEAR


# ═══════════════════════════════════════════════════════════════════════
# Core numerics (no plotting) -- importable / testable on their own
# ═══════════════════════════════════════════════════════════════════════

def decompose(E_c, E_h, P_c, P_h, split="control", thr=PP_THRESHOLD):
    """Cell-wise decomposition of ΔE = E_h - E_c, with f = E/P (unclipped).

    Returns dict of arrays (NaN outside `valid`) plus the `valid` mask.
    Identity is exact on valid cells:
        control : ΔE = f_c ΔP + P_c Δf + ΔP Δf
        midpoint: ΔE = f̄ ΔP + P̄ Δf        (cross = 0)
    """
    E_c, E_h, P_c, P_h = (np.asarray(a, dtype="f8") for a in (E_c, E_h, P_c, P_h))
    valid = (np.isfinite(E_c) & np.isfinite(E_h) & np.isfinite(P_c) & np.isfinite(P_h)
             & (P_c >= thr) & (P_h >= thr))
    with np.errstate(divide="ignore", invalid="ignore"):
        f_c = np.where(valid, E_c / P_c, np.nan)
        f_h = np.where(valid, E_h / P_h, np.nan)
    dP = np.where(valid, P_h - P_c, np.nan)
    dE = np.where(valid, E_h - E_c, np.nan)
    df = f_h - f_c
    if split == "control":
        prod, eff, cross = f_c * dP, P_c * df, dP * df
    elif split == "midpoint":
        prod = 0.5 * (f_c + f_h) * dP
        eff = 0.5 * (P_c + P_h) * df
        cross = np.where(valid, 0.0, np.nan)
    else:
        raise ValueError("split must be 'control' or 'midpoint'")
    return dict(valid=valid, f_c=f_c, f_h=f_h, dP=dP, dE=dE, df=df,
                prod=prod, eff=eff, cross=cross)


def regional_budget(E_c, E_h, P_c, P_h, dec, csize, masks):
    """Area-integrated budget per province in PgC/yr (native grid).

    Same convention as TabK_summary.weighted_total(): sum(data * cell_area)
    over cells with weight > 0, NaN treated as no contribution.
    Columns: PP/E per run, dPP, dE (total, = TabK), the three terms, the
    'PP~0 cells' remainder, and `closure` (should be ~1e-15).
    """
    z = lambda a: np.nan_to_num(np.asarray(a, dtype="f8"))
    E_c0, E_h0, P_c0, P_h0 = z(E_c), z(E_h), z(P_c), z(P_h)
    valid = dec["valid"]
    rows = []
    for prov, m in masks.items():
        w = z(csize) * z(m)
        w = np.where(w > 0, w, 0.0)

        def tot(x):
            return float((z(x) * w).sum()) * MOL_M2S_TO_PGC_YR_PER_M2

        dE_tot = tot(E_h0 - E_c0)
        prod, eff, cross = tot(dec["prod"]), tot(dec["eff"]), tot(dec["cross"])
        resid = tot(np.where(valid, 0.0, E_h0 - E_c0))
        rows.append(dict(
            province=prov,
            PP_control=tot(P_c0), PP_hose=tot(P_h0), dPP=tot(P_h0 - P_c0),
            EXP_control=tot(E_c0), EXP_hose=tot(E_h0), dEXP=dE_tot,
            production=prod, efficiency=eff, cross=cross, pp_zero_cells=resid,
            closure=dE_tot - (prod + eff + cross + resid),
            frac_area_decomposed=float(((valid * w).sum()) / w.sum()) if w.sum() > 0 else np.nan,
        ))
    return pd.DataFrame(rows).set_index("province")


def clip_report(dec, csize, label=""):
    """How much would MF_FLUX-style clipping of f to [0, 1] have mattered?"""
    v = dec["valid"]
    w = np.nan_to_num(np.asarray(csize, dtype="f8"))
    out = []
    for run, f in (("control", dec["f_c"]), ("hose", dec["f_h"])):
        bad = v & ((f < 0) | (f > 1))
        out.append(f"{run}: {bad.sum()} of {v.sum()} cells "
                   f"({100 * (w * bad).sum() / max((w * v).sum(), 1e-30):.2f}% of area) outside [0,1]")
    print(f"[f-ratio clipping check{' ' + label if label else ''}] " + "; ".join(out))


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def _tdim(da):
    for c in ("time", "time_counter"):
        if c in da.dims:
            return c
    return None


def load_native(diad_path):
    """Annual-mean PPINT (y,x) and EXP100 (y,x) from the native diad_T clim."""
    with xr.open_dataset(diad_path) as ds:
        pp = ds[PP_VAR]
        if _tdim(pp):
            pp = pp.mean(_tdim(pp))
        ex = ds[EXP_VAR]
        if DEPTH_DIM not in ex.dims:
            raise ValueError(f"{EXP_VAR} has no dim {DEPTH_DIM!r}: {ex.dims}")
        if _tdim(ex):
            ex = ex.mean(_tdim(ex))
        ex = ex.isel({DEPTH_DIM: list(EXP_LEVELS)}).mean(DEPTH_DIM)
        return np.squeeze(pp.values).astype("f8"), np.squeeze(ex.values).astype("f8")


def load_native_geometry():
    """Cell areas and province masks on the native grid, as in TabK_summary."""
    with xr.open_dataset(tk.CELLSIZE_MASK_PATH) as mesh:
        csize = np.squeeze(mesh["csize"].values).astype("f8")
    masks = {}
    with xr.open_dataset(tk.PROVINCE_MASK_PATH) as ma:
        for name in ["NHL", "NA", "SA", "SO", "AB"]:
            masks[name] = np.squeeze(ma[name].values).astype("f8")
    masks["GO"] = np.ones_like(csize)
    return csize, masks


def load_rg(paths, run):
    """Annual-mean rg fields: lon, lat, PPINT, EXP100 (MF_FLUX convention)."""
    lon, lat, pp, _ = mm.load_field(paths[run], PP_VAR, 0)
    stack = []
    for d in EXP_LEVELS:
        _, _, e, _ = mm.load_field(paths[run], EXP_VAR, d)
        stack.append(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e100 = np.nanmean(np.stack(stack, 0), axis=0)
    return lon, lat, np.asarray(pp, "f8"), e100


# ═══════════════════════════════════════════════════════════════════════
# Cross-check against TabK_summary CSV
# ═══════════════════════════════════════════════════════════════════════

def check_against_tabk(budget, csv_path, run_a, run_b):
    if not csv_path or not os.path.exists(csv_path):
        print(f"[TabK check] {csv_path!r} not found -- skipped")
        return
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
    print(f"[TabK check] vs {csv_path}")
    for var, col in (("export", "dEXP"), ("productivity", "dPP")):
        sub = df[df["variable"] == var]
        piv = sub.pivot(index="province", columns="run", values="value")
        tab = (piv[run_b] - piv[run_a]).reindex(budget.index)
        err = (budget[col] - tab).abs()
        print(f"   {var:13s} max |mine - TabK| = {err.max():.3e} PgC/yr "
              f"(largest |TabK| = {tab.abs().max():.3e})")


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

def draw_bars(ax, budget, split, run_a_disp, run_b_disp):
    provs = [p for p in BAR_PROVINCES if p in budget.index]
    ypos, k = {}, 0
    for p in provs:
        if p == "AB":
            k += 0.6                      # gap before the aggregate
        ypos[p] = k
        k += 1
    terms = [("prod", "production"), ("eff", "efficiency")]
    if split == "control":
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
                    color=TERM_COLORS[key], edgecolor="black", linewidth=0.5,
                    hatch="///" if key == "resid" else None, zorder=3)
    for p in provs:
        ax.scatter(budget.loc[p, "dEXP"], ypos[p], marker="D", s=55, color="black",
                   edgecolor="white", linewidth=0.8, zorder=5)
    ax.axvline(0, color="0.4", linewidth=0.8, zorder=2)
    ax.set_yticks([ypos[p] for p in provs])
    ax.set_yticklabels(provs, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel(f"$\\Delta$ Export100 ({run_b_disp}\u2212{run_a_disp})\nPgC yr$^{{-1}}$",
                  fontsize=plt.rcParams["font.size"] - 1)
    ax.set_title("Regional budget", fontweight="bold")
    ax.tick_params(labelsize=plt.rcParams["font.size"] - 2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="0.9", zorder=0)

    if split == "control":
        lab_prod = r"Production ($f_C\,\Delta PP$)"
        lab_eff = r"Efficiency ($PP_C\,\Delta f$)"
        lab_cross = r"Cross ($\Delta PP\,\Delta f$)"
    else:
        lab_prod = r"Production ($\bar f\,\Delta PP$)"
        lab_eff = r"Efficiency ($\overline{PP}\,\Delta f$)"
        lab_cross = None
    handles = [mlines.Line2D([], [], marker="s", linestyle="none", markersize=9,
                             markerfacecolor=TERM_COLORS["prod"], markeredgecolor="black", label=lab_prod),
               mlines.Line2D([], [], marker="s", linestyle="none", markersize=9,
                             markerfacecolor=TERM_COLORS["eff"], markeredgecolor="black", label=lab_eff)]
    if lab_cross:
        handles.append(mlines.Line2D([], [], marker="s", linestyle="none", markersize=9,
                                     markerfacecolor=TERM_COLORS["cross"], markeredgecolor="black", label=lab_cross))
    if show_resid:
        handles.append(mlines.Line2D([], [], marker="s", linestyle="none", markersize=9,
                                     markerfacecolor="white", markeredgecolor="black", label=r"PP$\approx$0 cells"))
    handles.append(mlines.Line2D([], [], marker="D", linestyle="none", markersize=7,
                                 markerfacecolor="black", markeredgecolor="white", label=r"Total $\Delta$Export"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False, fontsize=plt.rcParams["font.size"] - 3, ncol=1,
              handlelength=1.0, labelspacing=0.4)


def make_figure(split="control", tabk_csv=None, watermark=True, plot=True):
    mm.set_presentation_style()
    lon_min, lon_max, lat_min, lat_max = mm._load_extent()
    extent = [lon_min, lon_max, lat_min, lat_max]
    lon_m, lat_m, masks_rg = mm.load_region_masks(mm.MASK_PATH)

    run_labels, runs = mm._load_models()
    run_a, run_b = run_labels[0], run_labels[1]
    run_a_disp, run_b_disp = RUN_DISPLAY.get(run_a, run_a), RUN_DISPLAY.get(run_b, run_b)

    # ---- native: regional budget -------------------------------------
    nat = {}
    for run in (run_a, run_b):
        path = os.path.join(runs[run]["clim_dir"],
                            tk.DIAD_FILENAME_TEMPLATE.format(clim_period=str(runs[run]["clim_period"])))
        nat[run] = load_native(path)
    P_c, E_c = nat[run_a]
    P_h, E_h = nat[run_b]
    csize, masks_nat = load_native_geometry()
    dec_nat = decompose(E_c, E_h, P_c, P_h, split=split)
    clip_report(dec_nat, csize, "native")
    budget = regional_budget(E_c, E_h, P_c, P_h, dec_nat, csize, masks_nat)
    print("\nRegional budget (PgC/yr), split =", split)
    print(budget.round(4).to_string())
    print(f"\nmax |closure| = {budget['closure'].abs().max():.2e} PgC/yr  (should be ~1e-14)")
    check_against_tabk(budget, tabk_csv, run_a, run_b)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    csv_out = f"{OUTPUT_DIR}/{FIGURE_CODE}_regional_{stamp}.csv"
    budget.to_csv(csv_out, float_format="%.6g")
    print(f"saved {csv_out}")
    if not plot:
        return None, None, budget

    # ---- rg: maps ------------------------------------------------------
    paths = mm._paths_for_filetype(runs, "diad_T")
    lon, lat, Pc_rg, Ec_rg = load_rg(paths, run_a)
    _, _, Ph_rg, Eh_rg = load_rg(paths, run_b)
    dec = decompose(Ec_rg, Eh_rg, Pc_rg, Ph_rg, split=split)

    def atl(f):
        v = mm._bbox_subset(lon, lat, f, lon_min, lon_max, lat_min, lat_max)
        return v[np.isfinite(v)]

    e_lim = max(mm._diff_limit(atl(dec[k])) for k in ("dE", "prod", "eff", "cross")
                if np.isfinite(dec[k]).any() and atl(dec[k]).size and np.nanmax(np.abs(atl(dec[k]))) > 0)
    p_lim = mm._diff_limit(atl(dec["dP"]))
    diff_cmap = mm._resolve_cmap(None, mm.DIFF_CMAP_NAME, mm.DIFF_CMAP_FALLBACK)

    u_flux = r"$\Delta$ mol C m$^{-2}$ s$^{-1}$"
    MAP_COLS = [
        [dict(title="Primary production", data=dec["dP"], lim=p_lim, units=u_flux),
         dict(title="f-ratio", data=dec["df"], lim=DF_VLIM, units=r"$\Delta$ f-ratio"),
         dict(title="Export100 (total)", data=dec["dE"], lim=e_lim, units=u_flux)],
        [dict(title="Production term", data=dec["prod"], lim=e_lim, units=u_flux),
         dict(title="Efficiency term", data=dec["eff"], lim=e_lim, units=u_flux),
         dict(title="Cross term",
              data=dec["cross"], lim=e_lim, units=u_flux)],
    ]

    fig = plt.figure(figsize=(3.4 * 3 + 3.2, 9.0))
    outer_gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.9],
                                hspace=0.32, wspace=0.10,
                                left=0.05, right=0.99, top=0.94, bottom=0.07)

    for row in range(2):
        for ci in range(3):
            spec = MAP_COLS[row][ci]
            no_panel = (row == 1 and ci == 2 and split == "midpoint")
            inner = outer_gs[row, ci].subgridspec(2, 1, height_ratios=[1, 0.06], hspace=0.06)
            ax = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
            cbar_gs = inner[1, 0].subgridspec(1, 3, width_ratios=[0.15, 0.70, 0.15], wspace=0)
            cax = fig.add_subplot(cbar_gs[0, 1])
            if no_panel:
                ax.axis("off"); cax.axis("off")
                continue
            try:
                mesh = mm._draw_panel(ax, lon, lat, spec["data"], diff_cmap,
                                      -spec["lim"], spec["lim"], lon_m, lat_m, masks_rg, extent=extent)
            except Exception as exc:
                print(f"    [warn] panel ({row},{ci}) failed ({exc!r}) -- leaving blank")
                ax.clear(); ax.axis("off"); cax.axis("off")
                continue
            ax.set_title(spec["title"], fontweight="bold")
            if ci == 0:
                ax.text(ROW_LABEL_X, 0.5, ROW_LABELS[row], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontweight="bold")
            cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
            cbar.outline.set_visible(False)
            cbar.formatter = mm._value_formatter()
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=plt.rcParams["font.size"] - 2)
            cbar.set_label(spec["units"], fontsize=plt.rcParams["font.size"] - 2)

    right = outer_gs[:, 3].subgridspec(3, 1, height_ratios=[0.25, 1.0, 0.45], hspace=0.0)
    ax_b = fig.add_subplot(right[1, 0])
    draw_bars(ax_b, budget, split, run_a_disp, run_b_disp)

    if watermark and WATERMARK_ON:
        fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom",
                 fontsize=plt.rcParams["font.size"] - 6, color="0.55")
    return fig, f"{FIGURE_CODE}_{stamp}.png", budget


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["control", "midpoint"], default="control")
    ap.add_argument("--tabk-csv", default=os.path.join(OUTPUT_DIR, "TabK_summary_HA01.csv"),
                    help="TabK_summary CSV to cross-check against (skipped if missing)")
    ap.add_argument("--no-plot", action="store_true", help="numbers only, no figure")
    args = ap.parse_args()

    fig, fname, budget = make_figure(split=args.split, tabk_csv=args.tabk_csv,
                                     plot=not args.no_plot)
    if not args.no_plot:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"saved {out}")
