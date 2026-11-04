"""
TabK_summary.py

Summary of the HOSE-BASE change in Biomass, Productivity, Export at
100m, and Cflx, in physically meaningful carbon-mass units -- PgC for
Biomass (a stock), PgC/yr for the other three (rates) -- rather than
the % change or raw per-area units this script started out reporting.

Two outputs, both from the same underlying computation:

    make_summary_table() -- a plain, human-readable table: one row per
        province (NHL/NA/SA/SO/AB/GO), with BASE / HOSE / Diff columns
        for each of the four variables, in PgC or PgC/yr. Printed and
        saved as a CSV -- meant for pasting straight into a paper/slide
        table, not for colour-coded reading.

    make_figure() -- four heatmap panels side by side, one per
        variable, each showing just the HOSE-BASE difference, EACH WITH
        ITS OWN independent colour scale (unlike the old %-based
        version's one shared scale across the first three variables --
        these four are now wildly different magnitudes, since Biomass
        is a global stock and the other three are fluxes, so one shared
        scale would make some panels look uniformly blank). No colorbar
        on any panel -- every cell is already annotated with its value.

CARBON-MASS CONVERSION -- see weighted_total() / _to_pgc() /
CARBON_CONVERSION_FACTOR. All four variables here are natively
carbon-currency (confirmed), so that factor is 1.0 across the board --
kept as a per-variable dict rather than a bare constant purely so a
future model/variable swap that DOES need a currency conversion has an
obvious single place to adjust it. The mol vs. mmol scale is inferred
automatically from each variable's own "units" attribute
(_mol_scale_from_units()), with a printed warning if that attribute
doesn't contain a recognizable mol/mmol token.

DATA -- same sources/loaders as TabA_phyto.py:
    - Phyto biomass: sum of each species' own province TOTAL (DIA, MIX,
      COC, PIC, PHA, FIX), native ptrc_T_int.nc -- note this is now an
      area-INTEGRATED total (weighted_total()), not the area-weighted
      MEAN concentration the original version of this script computed;
      totals are what a PgC-family unit actually requires.
    - PPINT: native diad_T.nc, already 2-D+time at source.
    - EXP100: native diad_T.nc, mean of two depth levels (9, 10) of EXP
      -- EXP100_DEPTH_DIM_NAME confirmed via ncdump (see TabA_phyto.py's
      history), not guessed by position.
    - Cflx: native diad_T.nc, already 2-D+time at source -- same simple
      treatment as PPINT (no depth handling needed).

UNIT CONVERSION -- SECONDS_PER_YEAR assumes a standard 365.25-day
Julian year -- adjust that constant if the model actually runs on a
different calendar (360-day, no-leap, etc.). MOLAR_MASS_C is 12.011 g
C/mol C.

PROVINCES -- same 5-province mask (mask_atl_v5_5prov_NHL.nc, native) as
tab3_phyto.py / TabA_phyto.py / TabB_physics.py, plus GO (whole global
ocean, no mask).

NUMBER FORMAT -- heatmap cells: 3 significant figures (e.g. "+1.23e+02"
or "+0.456", whichever the value calls for). Summary table CSV: 6
significant figures.

Usage (from a notebook):
    from TabK_summary import compute_province_means, make_summary_table, make_figure

    df, base_run_id = compute_province_means()
    table = make_summary_table(df=df, base_run_id=base_run_id)
    fig, fname = make_figure(df=df, base_run_id=base_run_id)
    fig.savefig(f"./plts/{fname}", bbox_inches="tight")
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean

from plot_style import set_presentation_style
import plot_style as ps

# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------

MODELS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yml")
PTRC_FILENAME_TEMPLATE = "ORCA2_7d_clim_{clim_period}_ptrc_T_int.nc"  # native, pre-integrated at source
DIAD_FILENAME_TEMPLATE = "ORCA2_7d_clim_{clim_period}_diad_T.nc"      # native RAW -- NOT _int, NOT _rg

PROVINCE_MASK_PATH = os.path.expanduser("~/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL.nc")
CELLSIZE_MASK_PATH = os.path.expanduser("~/scratch/AMOC-eco/NEMO5/masks/meshmask_v5.nc")

PHYTO_VARS = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"]

# EXP100 isn't a variable in diad_T -- built from two depth levels of EXP
# (0-based indices), averaged. Depth dim name confirmed via ncdump
# against the real file (see TabA_phyto.py / depth_integrate_v9.py's
# history) -- not guessed by position.
EXP100_SOURCE_VAR = "EXP"
EXP100_DEPTH_LEVELS = (9, 10)
EXP100_DEPTH_DIM_NAME = "grid_T_3D_inner"

VARS_ORDER = ["biomass", "productivity", "export", "cflx"]
VAR_LABEL = {
    "biomass": "Biomass",
    "productivity": "Productivity",
    "export": "Export",
    "cflx": "Cflx",
}

# NHL/NA/SA/SO (the 4 latitude-banded subdivisions, north to south), then
# the two aggregates: AB (whole Atlantic basin), GO (whole global ocean)
PROVINCES_ORDER = ["NHL", "NA", "SA", "SO", "AB", "GO"]

DIFF_CMAP = cmocean.cm.balance
GRID_COLOR = "black"

# ASSUMED standard Julian year (365.25 days) -- adjust if the model runs
# on a different calendar (e.g. 360-day, no-leap)
SECONDS_PER_YEAR = 365.25 * 24 * 3600

# -- carbon-mass conversion -------------------------------------------------
# g C per mol C. All four variables here are natively carbon-currency
# (confirmed -- this isn't a PISCES-family model), so the per-variable
# factor below is 1.0 across the board, not a guess. Left as a dict
# (rather than a single hardcoded 1.0) purely so a future variable/model
# swap that DOES need a currency conversion (e.g. a different tracked
# element) has an obvious single place to adjust it, without touching
# the conversion logic itself.
MOLAR_MASS_C = 12.011  # g C / mol C
PG_PER_G = 1e-15       # Pg / g

# per-variable carbon-currency multiplier -- see note above (all 1.0,
# confirmed carbon-currency)
CARBON_CONVERSION_FACTOR = {
    "biomass": 1.0,
    "productivity": 1.0,
    "export": 1.0,
    "cflx": 1.0,
}

# which variables are RATES (need annualizing via SECONDS_PER_YEAR and
# are reported in PgC/yr) vs. a STOCK (reported in plain PgC)
RATE_VARS = {"productivity", "export", "cflx"}

VAR_UNIT_LABEL = {
    "biomass": "PgC",
    "productivity": "PgC/yr",
    "export": "PgC/yr",
    "cflx": "PgC/yr",
}

OUTPUT_DIR = ps.output_dir()   # figure_settings.yml (was a relative "plts")
CSV_FILENAME_TEMPLATE = "TabK_summary_{base_id}.csv"
SUMMARY_TABLE_FILENAME_TEMPLATE = "TabK_summary_table_{base_id}.csv"
PNG_FILENAME_TEMPLATE = "TabK_summary_{base_id}.png"
CHANGE_CSV_FILENAME_TEMPLATE = "TabK_change_{base_id}.csv"


def _load_models():
    """Re-read models.yml fresh each call (not cached at import)."""
    with open(MODELS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    runs = cfg["runs"]
    run_labels = list(runs)
    ptrc_paths = {
        label: os.path.join(r["clim_dir"], PTRC_FILENAME_TEMPLATE.format(clim_period=str(r["clim_period"])))
        for label, r in runs.items()
    }
    diad_paths = {
        label: os.path.join(r["clim_dir"], DIAD_FILENAME_TEMPLATE.format(clim_period=str(r["clim_period"])))
        for label, r in runs.items()
    }
    base_run_id = runs[run_labels[0]]["run_id"]
    return run_labels, ptrc_paths, diad_paths, base_run_id


# ---------------------------------------------------------------------------
# Data loading -- native grid, area-weighted. Same pattern as TabA_phyto.py.
# ---------------------------------------------------------------------------

def _time_dim(da):
    return "time" if "time" in da.dims else "time_counter"


def _standardize_spatial_dims(da):
    """Rename this DataArray's non-time dims positionally to generic names
    (_spatial0, _spatial1, ...)."""
    spatial = [d for d in da.dims if d not in ("time", "time_counter")]
    rename_map = {d: f"_spatial{i}" for i, d in enumerate(spatial)}
    return da.rename(rename_map)


def load_annual_mean_da(path, var):
    """Annual-mean DataArray for one already-2D(+time) variable (ptrc
    species, PPINT, or Cflx) -- standardized spatial dims."""
    ds = xr.open_dataset(path)
    da = ds[var]
    da = da.mean(_time_dim(da))
    return _standardize_spatial_dims(da)


def load_exp100_da(path):
    """EXP100 = mean of two specific depth levels of EXP, native
    diad_T.nc. Same construction as TabA_phyto.py / figA_phyto_overview.py."""
    ds = xr.open_dataset(path)
    da = ds[EXP100_SOURCE_VAR]
    da = da.mean(_time_dim(da))

    if EXP100_DEPTH_DIM_NAME not in da.dims:
        raise ValueError(
            f"Expected depth dim '{EXP100_DEPTH_DIM_NAME}' not found in EXP's "
            f"dims {da.dims}. Re-check with diagnose_dims.py if the file "
            "structure has changed."
        )
    i0, i1 = EXP100_DEPTH_LEVELS
    da = da.isel({EXP100_DEPTH_DIM_NAME: [i0, i1]}).mean(EXP100_DEPTH_DIM_NAME)

    return _standardize_spatial_dims(da)


def load_provinces():
    """{province_name: weight_mask}: GO (whole ocean, weight=csize) plus
    AB/NHL/NA/SA/SO (weight=csize * that region's 1/0 mask)."""
    ma = xr.open_dataset(PROVINCE_MASK_PATH)
    mesh = xr.open_dataset(CELLSIZE_MASK_PATH)
    csize = _standardize_spatial_dims(mesh["csize"])

    provinces = {"GO": csize}
    for name in ["AB", "NHL", "NA", "SA", "SO"]:
        provinces[name] = csize * _standardize_spatial_dims(ma[name])
    return provinces


def weighted_mean(var_data, prov_weight, debug_label=""):
    """Area-weighted (true grid-cell size) mean of var_data within
    prov_weight > 0."""
    masked_data = var_data.where(prov_weight > 0, 0)
    masked_weight = prov_weight.where(prov_weight > 0, 0)

    spatial_dims = [d for d in masked_data.dims if d not in ("time", "time_counter")]
    weighted_sum = (masked_data * masked_weight).sum(dim=spatial_dims)
    total_weight = masked_weight.sum(dim=spatial_dims)

    total = float(total_weight)
    if total <= 0:
        print(f"[{debug_label}] WARNING: zero total weight -> NaN.")
        return np.nan
    return float((weighted_sum / total_weight).values)


def weighted_total(var_data, prov_weight, debug_label=""):
    """Area-INTEGRATED total of var_data within prov_weight > 0 -- i.e.
    sum(data * cell_area), NOT divided by total area. This is what turns
    a per-m2 concentration or flux into an actual province-wide total
    (e.g. total mol C, or total mol C/s), the starting point for the
    PgC/PgC-yr conversion in _to_pgc()."""
    masked_data = var_data.where(prov_weight > 0, 0)
    spatial_dims = [d for d in masked_data.dims if d not in ("time", "time_counter")]
    total = (masked_data * prov_weight).sum(dim=spatial_dims)
    return float(total.values)


def _mol_scale_from_units(units_str, debug_label=""):
    """Multiplier to convert this variable's native mol-family unit to
    plain mol (e.g. 1e-3 for a "mmol/..." unit string), inferred from
    its "units" attribute. Prints a warning and assumes 1.0 (already
    mol) if the string doesn't contain a recognizable mol/mmol prefix --
    worth checking manually if that happens."""
    raw = units_str or ""
    lowered = raw.lower()

    if "mmol" in lowered:
        return 1e-3
    if "mol" in lowered:
        return 1.0

    print(f"[{debug_label}] WARNING: couldn't find 'mol'/'mmol' in units "
          f"attribute {raw!r} -- assuming already in mol. Verify this "
          f"manually against the source file.")
    return 1.0


def _to_pgc(total_native, units_str, var_key, debug_label=""):
    """Convert one area-integrated total (native mol-family units, e.g.
    mol C or mol C/s -- see weighted_total()) to PgC (stocks) or PgC/yr
    (rates, per RATE_VARS), applying CARBON_CONVERSION_FACTOR."""
    mol_scale = _mol_scale_from_units(units_str, debug_label=debug_label)
    total_mol = total_native * mol_scale
    total_gC = total_mol * MOLAR_MASS_C * CARBON_CONVERSION_FACTOR[var_key]
    total_pgc = total_gC * PG_PER_G
    if var_key in RATE_VARS:
        total_pgc *= SECONDS_PER_YEAR
    return total_pgc



# ---------------------------------------------------------------------------
# Step 1: province means -> CSV
# ---------------------------------------------------------------------------

def compute_province_means():
    """Despite the name (kept for continuity with the CSV/notebook usage
    this script already has), this now computes area-INTEGRATED,
    carbon-mass-converted totals -- PgC for biomass, PgC/yr for
    productivity/export/cflx -- not area-weighted mean concentrations.
    See weighted_total() / _to_pgc() for the actual conversion."""
    run_labels, ptrc_paths, diad_paths, base_run_id = _load_models()
    provinces = load_provinces()

    rows = []
    for run_label in run_labels:
        # biomass = sum of each species' own province total (totals are
        # additive across species, unlike means, so this is safe even
        # more directly than the old mean-based version was)
        biomass_totals = {prov: 0.0 for prov in PROVINCES_ORDER}
        biomass_units = None
        for var in PHYTO_VARS:
            da = load_annual_mean_da(ptrc_paths[run_label], var)
            biomass_units = biomass_units or da.attrs.get("units", "")
            for prov in PROVINCES_ORDER:
                val = weighted_total(da, provinces[prov], debug_label=f"{run_label}/{var}/{prov}")
                biomass_totals[prov] += val
        for prov in PROVINCES_ORDER:
            pgc = _to_pgc(biomass_totals[prov], biomass_units, "biomass",
                           debug_label=f"{run_label}/biomass/{prov}")
            rows.append({"variable": "biomass", "province": prov, "run": run_label, "value": pgc})

        # productivity (PPINT) -- already 2-D+time at source
        da_ppint = load_annual_mean_da(diad_paths[run_label], "PPINT")
        ppint_units = da_ppint.attrs.get("units", "")
        for prov in PROVINCES_ORDER:
            total = weighted_total(da_ppint, provinces[prov], debug_label=f"{run_label}/PPINT/{prov}")
            pgc = _to_pgc(total, ppint_units, "productivity", debug_label=f"{run_label}/productivity/{prov}")
            rows.append({"variable": "productivity", "province": prov, "run": run_label, "value": pgc})

        # export (EXP100) -- 2-level average of EXP
        da_exp100 = load_exp100_da(diad_paths[run_label])
        exp_units = da_exp100.attrs.get("units", "")
        for prov in PROVINCES_ORDER:
            total = weighted_total(da_exp100, provinces[prov], debug_label=f"{run_label}/EXP100/{prov}")
            pgc = _to_pgc(total, exp_units, "export", debug_label=f"{run_label}/export/{prov}")
            rows.append({"variable": "export", "province": prov, "run": run_label, "value": pgc})

        # cflx -- already 2-D+time at source, same treatment as PPINT
        da_cflx = load_annual_mean_da(diad_paths[run_label], "Cflx")
        cflx_units = da_cflx.attrs.get("units", "")
        for prov in PROVINCES_ORDER:
            total = weighted_total(da_cflx, provinces[prov], debug_label=f"{run_label}/Cflx/{prov}")
            pgc = _to_pgc(total, cflx_units, "cflx", debug_label=f"{run_label}/cflx/{prov}")
            rows.append({"variable": "cflx", "province": prov, "run": run_label, "value": pgc})

    df = pd.DataFrame(rows)
    df["units"] = df["variable"].map(VAR_UNIT_LABEL)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_name = CSV_FILENAME_TEMPLATE.format(base_id=base_run_id[-4:])
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    return df, base_run_id


# ---------------------------------------------------------------------------
# Step 2: human-readable summary table (BASE / HOSE / Diff per variable)
# ---------------------------------------------------------------------------

def make_summary_table(df=None, base_run_id=None):
    """Human-readable summary table: one row per province, with
    BASE / HOSE / Diff (HOSE-BASE) columns for each of biomass,
    productivity, export, and cflx, in PgC or PgC/yr. This is the same
    numbers as the heatmap figure, just in a plain, easy-to-read/copy
    tabular form (e.g. for pasting into a paper/slide table) rather than
    colour-coded cells.

    Pass df/base_run_id in directly if you've already called
    compute_province_means() (e.g. from make_figure()) to avoid
    recomputing everything from the source files a second time.
    """
    if df is None:
        df, base_run_id = compute_province_means()

    run_labels = list(df["run"].unique())
    run_a, run_b = run_labels[0], run_labels[1]

    table = pd.DataFrame(index=PROVINCES_ORDER)
    table.index.name = "province"
    for var in VARS_ORDER:
        sub = df[df["variable"] == var]
        piv = sub.pivot(index="province", columns="run", values="value").reindex(index=PROVINCES_ORDER)
        units = VAR_UNIT_LABEL[var]
        label = VAR_LABEL[var]
        table[f"{label} {run_a} ({units})"] = piv[run_a]
        table[f"{label} {run_b} ({units})"] = piv[run_b]
        table[f"{label} \u0394 ({units})"] = piv[run_b] - piv[run_a]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_name = SUMMARY_TABLE_FILENAME_TEMPLATE.format(base_id=base_run_id[-4:])
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    table.to_csv(csv_path, float_format="%.6g")
    print(f"wrote {csv_path}")
    print(table.round(4).to_string())

    return table


# ---------------------------------------------------------------------------
# Step 2b: long-form change table (BASE, HOSE, change, % change)
# ---------------------------------------------------------------------------

def make_change_csv(df=None, base_run_id=None):
    """One row per variable x province, in a plain long-form CSV:

        variable, province, units, base, hose, change, percent_change

    base/hose are the same PgC (biomass) or PgC/yr (productivity, export,
    cflx) totals as in the summary table; change = hose - base;
    percent_change = 100 * change / |base|. |base| is used so the sign of the
    percentage always follows the sign of the change, which matters for
    Cflx (a negative base is a net source). percent_change is NaN where
    base == 0.

    When reading this file back with pandas, pass keep_default_na=False,
    otherwise the province "NA" (North Atlantic) is read as missing.
    """
    if df is None:
        df, base_run_id = compute_province_means()

    run_labels = list(df["run"].unique())
    run_a, run_b = run_labels[0], run_labels[1]

    rows = []
    for var in VARS_ORDER:
        sub = df[df["variable"] == var]
        piv = sub.pivot(index="province", columns="run", values="value").reindex(index=PROVINCES_ORDER)
        for prov in PROVINCES_ORDER:
            base, hose = float(piv.loc[prov, run_a]), float(piv.loc[prov, run_b])
            change = hose - base
            pct = 100.0 * change / abs(base) if base != 0 else np.nan
            rows.append({"variable": var, "province": prov, "units": VAR_UNIT_LABEL[var],
                         "base": base, "hose": hose, "change": change, "percent_change": pct})
    out = pd.DataFrame(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_name = CHANGE_CSV_FILENAME_TEMPLATE.format(base_id=base_run_id[-4:])
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    out.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"wrote {csv_path}")

    # sanity check on the biomass stock (see the note in _mol_scale_from_units:
    # the ptrc_T_int files carry no units attribute and are assumed to be mol)
    go = out[(out["variable"] == "biomass") & (out["province"] == "GO")]
    if len(go):
        go_base = float(go["base"].iloc[0])
        if not (0.05 <= go_base <= 20):
            print(f"[warn] global phytoplankton biomass = {go_base:.3g} PgC, outside the "
                  "0.05-20 PgC range expected for the ocean. If it is about 1000x too "
                  "small, the ptrc_T_int units (molC/L integrated over m) are the likely cause.")
    print(out.round(4).to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Step 3: four-panel PgC/PgC-yr change heatmap
# ---------------------------------------------------------------------------

def make_figure(df=None, base_run_id=None):
    """Four panels side by side, one per variable (Biomass, Productivity,
    Export, Cflx), each showing the HOSE-BASE difference in PgC (biomass)
    or PgC/yr (the three rates). Each panel gets its OWN independent
    colour scale -- unlike the old %-based version, these four are wildly
    different magnitudes (a global biomass stock vs. a flux, for
    instance), so a single shared scale would make some panels look
    uniformly blank. No colorbar on any panel -- every cell is already
    annotated with its value.

    Pass df/base_run_id in directly if you've already called
    compute_province_means() (e.g. alongside make_summary_table()) to
    avoid recomputing everything from the source files a second time.
    """
    set_presentation_style()

    if df is None:
        df, base_run_id = compute_province_means()
    run_labels = list(df["run"].unique())
    run_a, run_b = run_labels[0], run_labels[1]

    diff_tables = {}
    for var in VARS_ORDER:
        sub = df[df["variable"] == var]
        piv = sub.pivot(index="province", columns="run", values="value").reindex(index=PROVINCES_ORDER)
        diff = piv[run_b] - piv[run_a]
        units = VAR_UNIT_LABEL[var]
        diff_tables[var] = diff.to_frame(name=f"{VAR_LABEL[var]}\n({units})")

    fig, axes = plt.subplots(
        1, len(VARS_ORDER), figsize=(2.4 * len(VARS_ORDER), 7),
        gridspec_kw={"wspace": 0.15},
    )

    for i, var in enumerate(VARS_ORDER):
        ax = axes[i]
        table = diff_tables[var]
        bound = float(np.nanmax(np.abs(table.values)))
        bound = bound if bound > 0 else 1.0
        _elementwise = getattr(table, "map", None) or table.applymap  # pandas < 2.1 has only applymap
        annot = _elementwise(lambda v: "" if not np.isfinite(v) else f"{v:+.3g}")

        sns.heatmap(
            table, ax=ax, cmap=DIFF_CMAP, vmin=-bound, vmax=bound,
            annot=annot, fmt="", linewidths=0.5, linecolor=GRID_COLOR, cbar=False,
            yticklabels=(i == 0),
        )
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labeltop=True, labelbottom=False, labelrotation=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("", pad=30)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.5)

    fig.suptitle(f"{run_b}\u2212{run_a}", fontweight="bold", y=1.04)

    fig.tight_layout()

    png_name = PNG_FILENAME_TEMPLATE.format(base_id=base_run_id[-4:])
    return fig, png_name


if __name__ == "__main__":
    df, base_run_id = compute_province_means()
    make_summary_table(df=df, base_run_id=base_run_id)
    make_change_csv(df=df, base_run_id=base_run_id)

    fig, png_name = make_figure(df=df, base_run_id=base_run_id)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, png_name)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"wrote {out_path}")
