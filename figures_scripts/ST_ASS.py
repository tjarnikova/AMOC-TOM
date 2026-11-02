"""
ST_ASS.py  --  Supplementary table: plankton assemblage by province, CONTROL vs HOSE (companion of ST_ECO.py)

Layout: four rows of tables (NHL, NA, SA, SO), two tables per row:
    left  = phytoplankton types (DIA, MIX, COC, PIC, PHA, FIX) + Total
    right = zooplankton types   (PRO, PTE, MES, GEL, CRU) + Total       (bacteria not included)
Columns: CONTROL, HOSE, % change, delta share.
    % change    = 100 * (HOSE - CONTROL) / CONTROL, of the type's own biomass (regional total).
    delta share = (type's share of its group total in HOSE) - (same in CONTROL), in percentage points.
                  Shares sum to 100 within a group, so the delta shares of a table sum to 0.
Only the two percentage columns are coloured, each on its own scale (PCT_VLIM, SHARE_VLIM).

Numbers are area-integrated TOTALS on the native grid. STANDALONE: loaders, masks, cell areas and unit
conversion are copied from ST_ECO.py / TabK_summary.py (no import), so the totals agree with ST_ECO's PHY and
ZOO rows by construction. The two files must be kept in step if a path or unit handling changes.
Decimals: one number of decimals per table, chosen so the smallest CONTROL/HOSE value has SIGFIGS
significant figures (capped at MAX_DECIMALS). % change is shown to 2 significant figures, delta share to
one decimal. The tidy CSV keeps full precision.

Outputs (/gpfs/home/mep22dku/scratch/AMOC-TOM/plts, date-stamped):
    ST_ASS_<date>.csv          block layout
    ST_ASS_<date>_tidy.csv     one row per region x group x type, full precision (also share_control, share_hose)
    ST_ASS_<date>.png          seaborn table figure
The technical caption is printed to the console.

Run from /gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts (needs models.yml there):
    python ST_ASS.py
    python ST_ASS.py --from-csv plts/ST_ASS_<date>_tidy.csv      # re-plot without recomputing
    python ST_ASS.py --no-watermark                              # (alias --water-off)
"""
import os
import sys
import csv
import argparse
import datetime
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/UTILS')

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ── tunable ────────────────────────────────────────────────────────────
WATERMARK   = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/ST_ASS.py"
WATERMARK_ON = True
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR  = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
CODE        = "ST_ASS"

REGIONS = ["NHL", "NA", "SA", "SO"]
GROUPS = {                                            # ptrc_T_int species
    "PHY": ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"],
    "ZOO": ["PRO", "PTE", "MES", "GEL", "CRU"],       # bacteria NOT included
}
GROUP_LABEL = {"PHY": "Phytoplankton", "ZOO": "Zooplankton"}

UNIT        = "TgC"            # "TgC" or "PgC" (1 TgC = 1e-3 PgC)
UNIT_FACTOR = {"PgC": 1.0, "TgC": 1e3}[UNIT]

# ── data settings (same as ST_ECO.py) ──────────────────────────────────
MODELS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yml")
PTRC_FILENAME_TEMPLATE = "ORCA2_7d_clim_{clim_period}_ptrc_T_int.nc"   # native, pre-integrated
PROVINCE_MASK_PATH = os.path.expanduser("~/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL.nc")
CELLSIZE_MASK_PATH = os.path.expanduser("~/scratch/AMOC-eco/NEMO5/masks/meshmask_v5.nc")
MOLAR_MASS_C = 12.011                    # g C / mol C
PG_PER_G = 1e-15

PCT_VLIM     = 30.0    # % change colour scale saturates at +/- this many percent (text is never clipped)
SHARE_VLIM   = 8.0     # delta share colour scale saturates at +/- this many percentage points
SIGFIGS      = 2       # significant figures of the smallest CONTROL/HOSE value in a table
MAX_DECIMALS = 4
FIG_WIDTH    = 9.4     # inches

CAPTION = [
    "Plankton functional type biomass by province, CONTROL vs HOSE; 2010-2019 climatology (annual mean of 12 monthly means).",
    "Area-integrated totals on the native ORCA2 grid in %s; one number of decimals per table. Total = sum of the group's types (bacteria not included)." % UNIT,
    "% change = 100*(HOSE-CONTROL)/CONTROL of the type's own biomass. Delta share = change in the type's share of its group total, in percentage points (HOSE share minus CONTROL share); shares sum to 100% within a group, so delta shares sum to 0.",
    "Colour: % change (saturating at +/-{:g}%) and delta share (saturating at +/-{:g} pp), each on its own scale.".format(PCT_VLIM, SHARE_VLIM),
    "Phytoplankton types: DIA, MIX, COC, PIC, PHA, FIX; zooplankton types: PRO, PTE, MES, GEL, CRU (full names: see the PlankTOM12 description).",
    "NHL >50N; NA 0-50N; SA 40S-0; SO <40S (Atlantic sector).",
    "HOSE: 0.3 Sv freshwater flux north of 50N (NAHosMIP).",
]


# ── loaders (copied from ST_ECO.py / TabK_summary.py) ──────────────────
def _load_models():
    with open(MODELS_CONFIG_PATH) as f:
        runs = yaml.safe_load(f)["runs"]
    run_labels = list(runs)
    ptrc = dict((k, os.path.join(r["clim_dir"], PTRC_FILENAME_TEMPLATE.format(clim_period=str(r["clim_period"]))))
                for k, r in runs.items())
    return run_labels, ptrc, runs[run_labels[0]]["run_id"]


def _time_dim(da):
    return "time" if "time" in da.dims else "time_counter"


def _standardize_spatial_dims(da):
    """Rename non-time dims positionally (_spatial0, _spatial1) so masks, areas and data align."""
    spatial = [d for d in da.dims if d not in ("time", "time_counter")]
    return da.rename(dict((d, "_spatial%d" % i) for i, d in enumerate(spatial)))


def load_annual_mean_da(path, var):
    da = xr.open_dataset(path)[var]
    return _standardize_spatial_dims(da.mean(_time_dim(da)))


def load_provinces():
    """{region: weight}: cell area * that region's 1/0 mask."""
    ma = xr.open_dataset(PROVINCE_MASK_PATH)
    mesh = xr.open_dataset(CELLSIZE_MASK_PATH)
    csize = _standardize_spatial_dims(mesh["csize"])
    return dict((name, csize * _standardize_spatial_dims(ma[name])) for name in REGIONS)


def weighted_total(var_data, prov_weight):
    """Area-integrated total: sum(data * cell area) where prov_weight > 0."""
    masked = var_data.where(prov_weight > 0, 0)
    dims = [d for d in masked.dims if d not in ("time", "time_counter")]
    return float((masked * prov_weight).sum(dim=dims).values)


def _mol_scale_from_units(units_str, debug_label=""):
    lowered = (units_str or "").lower()
    if "mmol" in lowered:
        return 1e-3
    if "mol" in lowered:
        return 1.0
    print("[%s] WARNING: no 'mol'/'mmol' in units %r, assuming mol" % (debug_label, units_str))
    return 1.0


def _to_pgc(total_native, units_str, debug_label=""):
    """Native mol C total -> PgC."""
    return total_native * _mol_scale_from_units(units_str, debug_label) * MOLAR_MASS_C * PG_PER_G


# ── compute ────────────────────────────────────────────────────────────
def compute():
    run_labels, ptrc_paths, base_id = _load_models()
    run_a, run_b = run_labels[0], run_labels[1]
    provs = load_provinces()
    val = {}                                          # (run, species, region) -> biomass in UNIT
    for run in (run_a, run_b):
        print("[%s] computing" % run, flush=True)
        for group, species in GROUPS.items():
            for sp in species:
                da = load_annual_mean_da(ptrc_paths[run], sp)
                units = da.attrs.get("units", "")
                for p in REGIONS:
                    lab = "%s/%s/%s" % (run, sp, p)
                    val[(run, sp, p)] = _to_pgc(weighted_total(da, provs[p]), units, lab) * UNIT_FACTOR

    rows = []
    for p in REGIONS:
        for group, species in GROUPS.items():
            tot_c = sum(val[(run_a, sp, p)] for sp in species)
            tot_h = sum(val[(run_b, sp, p)] for sp in species)
            for sp in species + ["TOTAL"]:
                if sp == "TOTAL":
                    c, h = tot_c, tot_h
                    sc = sh = 100.0
                    dsh = np.nan
                else:
                    c, h = val[(run_a, sp, p)], val[(run_b, sp, p)]
                    sc = 100.0 * c / tot_c if tot_c else np.nan
                    sh = 100.0 * h / tot_h if tot_h else np.nan
                    dsh = sh - sc
                rows.append(dict(region=p, group=group, species=sp, unit=UNIT, control=c, hose=h,
                                 delta=h - c, pct_change=(100.0 * (h - c) / abs(c)) if c else np.nan,
                                 share_control=sc, share_hose=sh, dshare=dsh, note=""))
    tidy = pd.DataFrame(rows)
    run_checks(tidy)
    return tidy


def run_checks(tidy):
    print("\n[checks]")
    for p in REGIONS:
        for group, species in GROUPS.items():
            sub = tidy[(tidy.region == p) & (tidy.group == group) & (tidy.species != "TOTAL")]
            tot = tidy[(tidy.region == p) & (tidy.group == group) & (tidy.species == "TOTAL")].iloc[0]
            print("   %-3s %-3s total %s: control %.5g, hose %.5g | share sums %.4f / %.4f | sum(delta share) = %.1e"
                  % (p, group, UNIT, tot.control, tot.hose, sub.share_control.sum(), sub.share_hose.sum(),
                     sub.dshare.sum()))
    print("   compare the PHY and ZOO totals with the ST_ECO rows (PgC; 1 TgC = 1e-3 PgC).")


# ── formatting ─────────────────────────────────────────────────────────
def _decimals(sub):
    """Decimals for one table (see module docstring)."""
    v = np.abs(np.concatenate([sub.control.values, sub.hose.values]).astype(float))
    v = v[np.isfinite(v) & (v > 0)]
    d = SIGFIGS - 1 - int(np.floor(np.log10(v.min()))) if v.size else 0
    return int(min(max(d, 0), MAX_DECIMALS))


def _fmt(x, d, signed=False):
    """Fixed number of decimals, trailing zeros kept; no '-0.00'."""
    if x is None or not np.isfinite(x):
        return ""
    txt = "%.*f" % (d, x)
    if float(txt) == 0:
        return "%.*f" % (d, 0.0)
    return ("+" + txt) if (signed and x > 0) else txt


def _pct_text(x):
    """% change to 2 significant figures, trailing zeros kept."""
    if x is None or not np.isfinite(x):
        return ""
    txt = np.format_float_positional(float(x), precision=2, unique=False, fractional=False, trim="k").rstrip(".")
    if float(txt) == 0:
        return "0%"
    return ("+" + txt if x > 0 else txt) + "%"


# ── block CSV ──────────────────────────────────────────────────────────
def write_block_csv(tidy, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["Region", "Group (%s)" % UNIT, "Type", "CONTROL", "HOSE", "% change", "\u0394 share (pp)"])
        for p in REGIONS:
            for group in GROUPS:
                sub = tidy[(tidy.region == p) & (tidy.group == group)]
                d = _decimals(sub)
                for _, r in sub.iterrows():
                    w.writerow([p, GROUP_LABEL[group], r["species"], _fmt(r["control"], d), _fmt(r["hose"], d),
                                _pct_text(r["pct_change"]).replace("%", ""),
                                _fmt(r["dshare"], 1, signed=True)])


# ── seaborn figure ─────────────────────────────────────────────────────
def _draw_panel(ax, tidy, region, group, cmap, fs, nmax, show_xlabels):
    from matplotlib.patches import Rectangle
    sub = tidy[(tidy.region == region) & (tidy.group == group)]
    types = GROUPS[group] + ["TOTAL"]
    n, ncol = 1 + len(types), 4
    d = _decimals(sub)
    pdata = np.full((n, ncol), np.nan)
    sdata = np.full((n, ncol), np.nan)
    texts = {}
    for i, sp in enumerate(types, start=1):
        r = sub[sub.species == sp].iloc[0]
        texts[i] = [_fmt(r["control"], d), _fmt(r["hose"], d), _pct_text(r["pct_change"]),
                    _fmt(r["dshare"], 1, signed=True)]
        if np.isfinite(r["pct_change"]):
            pdata[i, 2] = r["pct_change"]
        if np.isfinite(r["dshare"]):
            sdata[i, 3] = r["dshare"]
    sns.heatmap(pdata, mask=~np.isfinite(pdata), cmap=cmap, vmin=-PCT_VLIM, vmax=PCT_VLIM, ax=ax,
                linewidths=0, cbar=False)
    sns.heatmap(sdata, mask=~np.isfinite(sdata), cmap=cmap, vmin=-SHARE_VLIM, vmax=SHARE_VLIM, ax=ax,
                linewidths=0, cbar=False)
    ax.set_xlim(0, ncol)
    ax.set_ylim(nmax, 0)                       # same row height in every table
    ax.add_patch(Rectangle((0, 0), ncol, 1, facecolor="0.90", edgecolor="none", lw=0, zorder=0))
    ax.text(ncol / 2.0, 0.5, "%s (%s)" % (GROUP_LABEL[group], UNIT), ha="center", va="center",
            fontsize=fs + 0.5, fontweight="bold", zorder=5)
    ylabels = [""]
    for i, sp in enumerate(types, start=1):
        ylabels.append("   " + ("Total" if sp == "TOTAL" else sp))
        ax.hlines(i, 2, 4, colors="white", linewidth=0.8, zorder=4)         # separates the coloured cells only
        if sp == "TOTAL":
            ax.hlines(i, 0, ncol, colors="0.45", linewidth=0.8, zorder=3)
        for j in range(ncol):
            colour = "black"
            if j == 2 and np.isfinite(pdata[i, 2]) and abs(pdata[i, 2]) > 0.75 * PCT_VLIM:
                colour = "white"
            if j == 3 and np.isfinite(sdata[i, 3]) and abs(sdata[i, 3]) > 0.75 * SHARE_VLIM:
                colour = "white"
            ax.text(j + 0.5, i + 0.5, texts[i][j], ha="center", va="center", fontsize=fs, color=colour,
                    fontweight="bold" if j >= 2 else "normal", zorder=5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(ylabels, fontsize=fs, rotation=0)
    if show_xlabels:
        ax.set_xticks(np.arange(ncol) + 0.5)
        ax.set_xticklabels(["CONTROL", "HOSE", "% change", r"$\Delta$ share (pp)"], fontsize=fs, fontweight="bold")
        ax.xaxis.tick_top()
    else:
        ax.set_xticks([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_figure(tidy, watermark=True):
    try:
        from plot_style import set_presentation_style
        set_presentation_style()
    except ImportError:
        pass
    try:
        import cmocean
        cmap = cmocean.cm.balance
    except ImportError:
        cmap = "RdBu_r"
    nmax = 1 + max(len(v) for v in GROUPS.values()) + 1
    fig = plt.figure(figsize=(FIG_WIDTH, 0.27 * nmax * len(REGIONS) + 1.6))
    gs = fig.add_gridspec(len(REGIONS), 2, hspace=0.12, wspace=0.17, left=0.10, right=0.995, top=0.93,
                          bottom=0.03)
    for ri, region in enumerate(REGIONS):
        for ci, group in enumerate(GROUPS):
            ax = fig.add_subplot(gs[ri, ci])
            _draw_panel(ax, tidy, region, group, cmap, 8.5, nmax, show_xlabels=(ri == 0))
            if ci == 0:
                ax.set_ylabel(region, fontsize=12, fontweight="bold", rotation=90, labelpad=22)
    if watermark and WATERMARK_ON:
        fig.text(0.5, 0.0, WATERMARK, ha="center", va="bottom", fontsize=6, color="0.55")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-csv", default=None, help="tidy CSV from an earlier run (skips the computation)")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    if a.from_csv:
        tidy = pd.read_csv(a.from_csv, keep_default_na=False, na_values=[""])
        tidy["note"] = tidy["note"].fillna("")
    else:
        tidy = compute()
    stamp = datetime.date.today().strftime("%Y%m%d")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.join(OUTPUT_DIR, "%s_%s" % (CODE, stamp))
    tidy.to_csv(base + "_tidy.csv", index=False, float_format="%.6g")
    write_block_csv(tidy, base + ".csv")
    print("wrote %s.csv, %s_tidy.csv" % (base, base))
    print("\nCAPTION\n" + "\n".join(CAPTION) + "\n")
    if not a.no_plot:
        fig = draw_figure(tidy)
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        print("saved %s.png" % base)


if __name__ == "__main__":
    main()
