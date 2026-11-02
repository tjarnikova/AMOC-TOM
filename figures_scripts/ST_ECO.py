"""
ST_ECO.py  --  Supplementary table: ecosystem metrics by province, CONTROL vs HOSE

One block per metric, six regions per block (NHL, NA, SA, SO | AB | GO):
    PHY biomass (PgC) | ZOO biomass (PgC) | BAC biomass (PgC) | primary production (PgC/yr)
    | ZOO/PHY (ratio) | zooplankton grazing (PgC/yr) | herbivory fraction (ratio)
    | EXP100 (PgC/yr) | ef (ratio) | Cflx (PgC/yr, positive = air to sea)
Columns: CONTROL, HOSE, delta (HOSE - CONTROL), % change.

Numbers are area-integrated TOTALS on the native grid. STANDALONE: the loaders, masks, cell areas and
unit conversion are copied from TabK_summary.py (no import), so the numbers agree with TabK by
construction but the two files must be kept in step if a path or unit handling changes. NHL+NA+SA+SO = AB
for every additive metric.
ef is the ratio of regional totals (EXP100 / PP); it is not additive, AB != sum of the four.
% change = 100 * delta / |CONTROL|; it is not given for Cflx (near zero or changing sign in places).
Within each metric block CONTROL, HOSE and delta share one number of decimals (so the columns visibly
subtract; trailing zeros are kept, 3.0 not 3). The decimals are chosen so the smallest CONTROL/HOSE
value has SIGFIGS significant figures and the smallest delta has at least DELTA_SIGFIGS (override
per metric with DECIMALS). % change is always shown to 2 significant figures. The tidy CSV keeps
full precision.

Outputs (/gpfs/home/mep22dku/scratch/AMOC-TOM/plts, date-stamped):
    ST_ECO_<date>.csv          block layout, 2 significant figures
    ST_ECO_<date>_tidy.csv     one row per metric x region, full precision (for re-plotting / reuse)
    ST_ECO_<date>.png          seaborn table figure in two column sets; only % change is coloured
The technical caption is printed to the console (not written to a file).

Run from /gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts (needs models.yml there):
    python ST_ECO.py                          # compute from the climatologies, write all outputs
    python ST_ECO.py --from-csv plts/ST_ECO_<date>_tidy.csv    # re-plot / re-format without recomputing
    python ST_ECO.py --no-watermark           # (alias --water-off)
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
WATERMARK   = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/ST_ECO.py"
WATERMARK_ON = True    # False = no script-path watermark (or run with --no-watermark / --water-off)
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR  = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
CODE        = "ST_ECO"

REGIONS = ["NHL", "NA", "SA", "SO", "AB", "GO"]
ADDITIVE_PARTS = ["NHL", "NA", "SA", "SO"]            # these should sum to AB

GROUPS = {                                            # ptrc_T_int species
    "PHY": ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"],
    "ZOO": ["PRO", "PTE", "MES", "GEL", "CRU"],       # bacteria NOT included
    "BAC": ["BAC"],
}

# key, label, unit for the CSV, unit for the figure
METRICS = [
    ("PHY",    "PHY biomass",         "PgC",    r"PgC"),
    ("ZOO",    "ZOO biomass",         "PgC",    r"PgC"),
    ("ZP",     "ZOO / PHY",           "ratio",  "ratio"),
    ("BAC",    "BAC biomass",         "PgC",    r"PgC"),
    ("PP",     "Primary production",  "PgC/yr", r"PgC yr$^{-1}$"),
    ("GRAZ",   "Zooplankton grazing", "PgC/yr", r"PgC yr$^{-1}$"),
    ("HERB",   "Herbivory fraction",  "ratio",  "ratio"),
    ("EXP100", "EXP100",              "PgC/yr", r"PgC yr$^{-1}$"),
    ("ef",     "ef = EXP100/PP",      "ratio",  "ratio"),
    ("Cflx",   "Cflx (air to sea)",   "PgC/yr", r"PgC yr$^{-1}$"),
]
RATIO_KEYS = ("ef", "ZP", "HERB")      # ratios of regional totals: not additive, skipped in the closure check
LABEL = {k: (lab, ucsv, utex) for k, lab, ucsv, utex in METRICS}

# ── data settings (same as TabK_summary.py) ────────────────────────────
MODELS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yml")
PTRC_FILENAME_TEMPLATE = "ORCA2_7d_clim_{clim_period}_ptrc_T_int.nc"   # native, pre-integrated
DIAD_FILENAME_TEMPLATE = "ORCA2_7d_clim_{clim_period}_diad_T.nc"       # native raw
PROVINCE_MASK_PATH = os.path.expanduser("~/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL.nc")
CELLSIZE_MASK_PATH = os.path.expanduser("~/scratch/AMOC-eco/NEMO5/masks/meshmask_v5.nc")
EXP100_SOURCE_VAR = "EXP"
EXP100_DEPTH_LEVELS = (9, 10)
EXP100_DEPTH_DIM_NAME = "grid_T_3D_inner"
MOLAR_MASS_C = 12.011                    # g C / mol C
PG_PER_G = 1e-15
SECONDS_PER_YEAR = 365.25 * 24 * 3600    # Julian year, as TabK_summary.py
RATE_VARS = ("productivity", "export", "cflx", "grazing")
TABK_CSV_TEMPLATE = OUTPUT_DIR + "/TabK_summary_{base_id}.csv"   # optional cross-check only (read, not imported)

CFLX_SIGN    = 1.0    # +1: model Cflx is positive into the ocean (set -1 if it is the other way round)
PCT_VLIM     = 25.0   # colour scale saturates at +/- this many percent (text is never clipped)
LEFT_PANEL   = ["PHY", "ZOO", "ZP", "BAC", "PP"]  # metrics in the left column set; the rest go right
SIGFIGS      = 2     # significant figures of the smallest CONTROL/HOSE value in a metric block
DELTA_SIGFIGS = 1    # ... and of the smallest delta, whichever needs more decimals wins
MAX_DECIMALS = 4
FIG_WIDTH    = 8.8   # inches (the two column sets share this width)
DECIMALS     = {}    # manual override per metric, e.g. {"ef": 3, "PP": 1}

CAPTION = [
    "Ecosystem metrics by province, CONTROL vs HOSE; 2010-2019 climatology (annual mean of 12 monthly means).",
    "Area-integrated totals on the native ORCA2 grid: stocks in PgC, rates in PgC/yr; one number of decimals per metric block.",
    "PHY = DIA+MIX+COC+PIC+PHA+FIX; ZOO = PRO+PTE+MES+GEL+CRU; BAC separate.",
    "PP = PPINT (particulate PP only, excludes dissolved PP); EXP100 = mean of EXP levels 9-10 (~101 m), sinking particulate organic carbon only; ef = EXP100/PP, ratio of regional totals (not additive).",
    "Zooplankton grazing = Herbiv + Carniv + Detrit (total ingestion by the five zooplankton, native diad_T_int); herbivory fraction = Herbiv / grazing; ZOO/PHY = ZOO / PHY biomass; all three ratios are ratios of regional totals (not additive).",
    "Cflx positive = net air to sea; % change not given for Cflx.",
    "Delta = HOSE - CONTROL; % change = 100*Delta/|CONTROL|; colour = % change only, saturating at +/-{:g}%.".format(PCT_VLIM),
    "NHL >50N; NA 0-50N; SA 40S-0; SO <40S (Atlantic sector); AB = NHL+NA+SA+SO; GO = global ocean.",
    "HOSE: 0.3 Sv freshwater flux north of 50N (NAHosMIP).",
]


# ── loaders (copied from TabK_summary.py) ──────────────────────────────
def _load_models():
    with open(MODELS_CONFIG_PATH) as f:
        runs = yaml.safe_load(f)["runs"]
    run_labels = list(runs)
    ptrc = dict((k, os.path.join(r["clim_dir"], PTRC_FILENAME_TEMPLATE.format(clim_period=str(r["clim_period"]))))
                for k, r in runs.items())
    diad = dict((k, os.path.join(r["clim_dir"], DIAD_FILENAME_TEMPLATE.format(clim_period=str(r["clim_period"]))))
                for k, r in runs.items())
    return run_labels, ptrc, diad, runs[run_labels[0]]["run_id"]


def _time_dim(da):
    return "time" if "time" in da.dims else "time_counter"


def _standardize_spatial_dims(da):
    """Rename non-time dims positionally (_spatial0, _spatial1) so masks, areas and data align."""
    spatial = [d for d in da.dims if d not in ("time", "time_counter")]
    return da.rename(dict((d, "_spatial%d" % i) for i, d in enumerate(spatial)))


def load_annual_mean_da(path, var):
    da = xr.open_dataset(path)[var]
    return _standardize_spatial_dims(da.mean(_time_dim(da)))


def load_exp100_da(path):
    """EXP100 = mean of two depth levels of EXP (native diad_T)."""
    da = xr.open_dataset(path)[EXP100_SOURCE_VAR]
    da = da.mean(_time_dim(da))
    if EXP100_DEPTH_DIM_NAME not in da.dims:
        raise ValueError("EXP has no dim %r (dims %s)" % (EXP100_DEPTH_DIM_NAME, da.dims))
    da = da.isel({EXP100_DEPTH_DIM_NAME: list(EXP100_DEPTH_LEVELS)}).mean(EXP100_DEPTH_DIM_NAME)
    return _standardize_spatial_dims(da)


def load_provinces():
    """{region: weight}: GO = cell area everywhere; others = cell area * that region's 1/0 mask."""
    ma = xr.open_dataset(PROVINCE_MASK_PATH)
    mesh = xr.open_dataset(CELLSIZE_MASK_PATH)
    csize = _standardize_spatial_dims(mesh["csize"])
    provs = {"GO": csize}
    for name in ["AB", "NHL", "NA", "SA", "SO"]:
        provs[name] = csize * _standardize_spatial_dims(ma[name])
    return provs


def weighted_total(var_data, prov_weight, debug_label=""):
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


def _to_pgc(total_native, units_str, var_key, debug_label=""):
    """Native mol C (stock) or mol C/s (rate) total -> PgC or PgC/yr."""
    pgc = total_native * _mol_scale_from_units(units_str, debug_label) * MOLAR_MASS_C * PG_PER_G
    return pgc * SECONDS_PER_YEAR if var_key in RATE_VARS else pgc


# ── compute ────────────────────────────────────────────────────────────
def compute():
    run_labels, ptrc_paths, diad_paths, base_id = _load_models()
    run_a, run_b = run_labels[0], run_labels[1]
    provs = load_provinces()
    diadint_paths = dict((k, v.replace("_diad_T.nc", "_diad_T_int.nc")) for k, v in diad_paths.items())
    assert diadint_paths[run_a] != diad_paths[run_a], "could not derive the diad_T_int path from %s" % diad_paths[run_a]
    val = {}                                          # (run, metric, region) -> value

    for run in (run_a, run_b):
        print("[%s] computing" % run, flush=True)
        for key, species in GROUPS.items():
            tot = dict((p, 0.0) for p in REGIONS); units = None
            for sp in species:
                da = load_annual_mean_da(ptrc_paths[run], sp)
                units = units or da.attrs.get("units", "")
                for p in REGIONS:
                    tot[p] += weighted_total(da, provs[p], "%s/%s/%s" % (run, sp, p))
            for p in REGIONS:
                val[(run, key, p)] = _to_pgc(tot[p], units, "biomass", "%s/%s/%s" % (run, key, p))
        for key, da, var_key in (
                ("PP", load_annual_mean_da(diad_paths[run], "PPINT"), "productivity"),
                ("EXP100", load_exp100_da(diad_paths[run]), "export"),
                ("Cflx", load_annual_mean_da(diad_paths[run], "Cflx"), "cflx")):
            units = da.attrs.get("units", "")
            for p in REGIONS:
                v = _to_pgc(weighted_total(da, provs[p], "%s/%s/%s" % (run, key, p)),
                               units, var_key, "%s/%s/%s" % (run, key, p))
                val[(run, key, p)] = v * (CFLX_SIGN if key == "Cflx" else 1.0)
        herb = load_annual_mean_da(diadint_paths[run], "Herbiv")
        carn = load_annual_mean_da(diadint_paths[run], "Carniv")
        detr = load_annual_mean_da(diadint_paths[run], "Detrit")
        gunits = herb.attrs.get("units", "")
        for gkey, gda in (("GRAZ", herb + carn + detr), ("HERBnum", herb)):
            for p in REGIONS:
                val[(run, gkey, p)] = _to_pgc(weighted_total(gda, provs[p], "%s/%s/%s" % (run, gkey, p)),
                                              gunits, "grazing", "%s/%s/%s" % (run, gkey, p))
        for p in REGIONS:
            graz = val[(run, "GRAZ", p)]
            val[(run, "HERB", p)] = val[(run, "HERBnum", p)] / graz if graz else np.nan
            phy_t = val[(run, "PHY", p)]
            val[(run, "ZP", p)] = val[(run, "ZOO", p)] / phy_t if phy_t else np.nan
        for p in REGIONS:
            pp = val[(run, "PP", p)]
            val[(run, "ef", p)] = val[(run, "EXP100", p)] / pp if pp else np.nan

    rows = []
    for key, lab, ucsv, utex in METRICS:
        for p in REGIONS:
            c, h = val[(run_a, key, p)], val[(run_b, key, p)]
            d = h - c
            pct, note = (100.0 * d / abs(c) if c else np.nan), ""
            if key == "Cflx":
                pct, note = np.nan, "% change not given for Cflx"
            rows.append(dict(metric=key, unit=ucsv, region=p, control=c, hose=h,
                             delta=d, pct_change=pct, note=note))
    tidy = pd.DataFrame(rows)
    run_checks(tidy, base_id, ptrc_paths, run_a, provs)
    return tidy


def run_checks(tidy, base_id, ptrc_paths, run_a, provs):
    print("\n[checks]")
    for key, *_ in METRICS:
        if key in RATIO_KEYS:
            continue
        for col in ("control", "hose"):
            s = tidy[(tidy.metric == key)].set_index("region")[col]
            err = abs(s[ADDITIVE_PARTS].sum() - s["AB"]) / max(abs(s["AB"]), 1e-30)
            print("   closure %-7s %-7s |sum(NHL,NA,SA,SO)-AB|/|AB| = %.1e %s"
                  % (key, col, err, "" if err < 1e-6 else "  <-- CHECK"))
    for var, group in (("ZOO", "ZOO"), ("PHY", "PHY")):      # does the ptrc 'ZOO'/'PHY' variable match the species sums?
        try:
            da = load_annual_mean_da(ptrc_paths[run_a], var)
            tot = weighted_total(da, provs["AB"], "chk")
            mine = tidy[(tidy.metric == group) & (tidy.region == "AB")]["control"].iloc[0]
            conv = _to_pgc(tot, da.attrs.get("units", ""), "biomass", "chk")
            print("   ptrc variable %s (AB, control) = %.4g PgC vs species sum %.4g PgC  (ratio %.3f)"
                  % (var, conv, mine, conv / mine if mine else np.nan))
        except Exception as exc:
            print("   ptrc variable %s not compared (%s)" % (var, type(exc).__name__))
    tcsv = TABK_CSV_TEMPLATE.format(base_id=base_id[-4:])
    if os.path.isfile(tcsv):
        t = pd.read_csv(tcsv, keep_default_na=False, na_values=[""])
        pairs = (("biomass", "PHY"), ("productivity", "PP"), ("export", "EXP100"), ("cflx", "Cflx"))
        for tvar, key in pairs:
            sub = t[t["variable"] == tvar]
            worst = 0.0
            for (_, r) in sub.iterrows():
                m = tidy[(tidy.metric == key) & (tidy.region == r["province"])]
                if m.empty:
                    continue
                mine = m["control"].iloc[0] if r["run"] == t["run"].unique()[0] else m["hose"].iloc[0]
                mine = mine * (CFLX_SIGN if key == "Cflx" else 1.0)
                worst = max(worst, abs(mine - r["value"]))
            print("   vs %s %-12s max |mine - TabK| = %.2e" % (os.path.basename(tcsv), tvar, worst))
    else:
        print("   TabK CSV %s not found, cross-check skipped" % tcsv)


# ── formatting ─────────────────────────────────────────────────────────
def _decimals(tidy):
    """One number of decimals per metric block (see module docstring)."""
    out = {}
    for key, *_ in METRICS:
        if key in DECIMALS:
            out[key] = DECIMALS[key]
            continue
        sub = tidy[tidy.metric == key]
        v = np.abs(np.concatenate([sub.control.values, sub.hose.values]).astype(float))
        v = v[np.isfinite(v) & (v > 0)]
        dl = np.abs(sub.delta.values.astype(float))
        dl = dl[np.isfinite(dl) & (dl > 0)]
        d1 = SIGFIGS - 1 - int(np.floor(np.log10(v.min()))) if v.size else 0
        d2 = DELTA_SIGFIGS - 1 - int(np.floor(np.log10(dl.min()))) if dl.size else 0
        out[key] = int(min(max(d1, d2, 0), MAX_DECIMALS))
    return out


def _fmt(x, d, signed=False):
    """Fixed number of decimals, trailing zeros kept; no '-0.00'."""
    if x is None or not np.isfinite(x):
        return "n/a"
    txt = "%.*f" % (d, x)
    if float(txt) == 0:
        txt = "%.*f" % (d, 0.0)
        return txt
    return ("+" + txt) if (signed and x > 0) else txt


def _pct_text(r):
    """% change to 2 significant figures, trailing zeros kept (3.0%, not 3%)."""
    if r["note"] or not np.isfinite(r["pct_change"]):
        return ""
    x = float(r["pct_change"])
    txt = np.format_float_positional(x, precision=2, unique=False, fractional=False, trim="k").rstrip(".")
    if float(txt) == 0:
        return "0%"
    return ("+" + txt if x > 0 else txt) + "%"


# ── block CSV ──────────────────────────────────────────────────────────
def write_block_csv(tidy, path):
    dec = _decimals(tidy)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["Metric (units)", "Region", "CONTROL", "HOSE", "\u0394 (HOSE\u2212CONTROL)", "% change"])
        for key, lab, ucsv, _ in METRICS:
            w.writerow(["%s (%s)" % (lab, ucsv), "", "", "", "", ""])
            for p in REGIONS:
                r = tidy[(tidy.metric == key) & (tidy.region == p)].iloc[0]
                d = dec[key]
                w.writerow(["", p, _fmt(r["control"], d), _fmt(r["hose"], d), _fmt(r["delta"], d, signed=True),
                            _pct_text(r).replace("%", "")])


# ── seaborn figure ─────────────────────────────────────────────────────
def _draw_panel(ax, tidy, keys, cmap, fs, dec):
    from matplotlib.patches import Rectangle
    rows = []
    for key in keys:
        rows.append(("head", key, None))
        rows.extend(("reg", key, p) for p in REGIONS)
    n, ncol = len(rows), 4
    data = np.full((n, ncol), np.nan)
    mask = np.ones((n, ncol), dtype=bool)
    texts = {}
    for i, (kind, key, p) in enumerate(rows):
        if kind != "reg":
            continue
        r = tidy[(tidy.metric == key) & (tidy.region == p)].iloc[0]
        d = dec[key]
        texts[i] = [_fmt(r["control"], d), _fmt(r["hose"], d), _fmt(r["delta"], d, signed=True), _pct_text(r)]
        if np.isfinite(r["pct_change"]) and not r["note"]:
            data[i, 3] = r["pct_change"]
            mask[i, 3] = False
    sns.heatmap(data, mask=mask, cmap=cmap, vmin=-PCT_VLIM, vmax=PCT_VLIM, ax=ax,
                linewidths=0, cbar=False)
    ax.set_xlim(0, ncol)
    ax.set_ylim(n, 0)
    ylabels = []
    for i, (kind, key, p) in enumerate(rows):
        if kind == "head":
            ax.add_patch(Rectangle((0, i), ncol, 1, facecolor="0.90", edgecolor="none", lw=0, zorder=0))
            lab, ucsv, utex = LABEL[key]
            ax.text(ncol / 2.0, i + 0.5, "%s (%s)" % (lab, utex), ha="center", va="center",
                    fontsize=fs + 0.5, fontweight="bold", zorder=5)
            ylabels.append("")
        else:
            ylabels.append("   " + p)
            ax.hlines(i, 3, 4, colors="white", linewidth=0.8, zorder=4)     # separates the coloured cells only
            if p in ("AB", "GO"):
                ax.hlines(i, 0, ncol, colors="0.45", linewidth=0.8, zorder=3)
            for j in range(ncol):
                colour = "black"
                if j == 3 and np.isfinite(data[i, 3]) and abs(data[i, 3]) > 0.75 * PCT_VLIM:
                    colour = "white"
                ax.text(j + 0.5, i + 0.5, texts[i][j], ha="center", va="center", fontsize=fs,
                        color=colour, fontweight="bold" if j == 3 else "normal", zorder=5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(ylabels, fontsize=fs, rotation=0)
    ax.set_xticks(np.arange(ncol) + 0.5)
    ax.set_xticklabels(["CONTROL", "HOSE", r"$\Delta$", "% change"], fontsize=fs + 1, fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return n


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
    left = [k for k, *_ in METRICS if k in LEFT_PANEL]
    right = [k for k, *_ in METRICS if k not in LEFT_PANEL]
    n_left, n_right = 7 * len(left), 7 * len(right)
    fig = plt.figure(figsize=(FIG_WIDTH, 0.27 * n_left + 1.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[n_right, max(n_left - n_right, 1e-3)], hspace=0, wspace=0.17,
                          left=0.065, right=0.995, top=0.95, bottom=0.04)
    ax_l = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    dec = _decimals(tidy)
    print("decimals per metric:", dec)
    _draw_panel(ax_l, tidy, left, cmap, 9, dec)
    _draw_panel(ax_r, tidy, right, cmap, 9, dec)
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
