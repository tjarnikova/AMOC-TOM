"""
SF_LIMCOMP.py  --  Supplementary figure + regional table: how do the light and temperature limitation
changes compare with the nutrient limitation change?   (hose minus control, annual, assemblage means)

FIGURE (maps, from the _rg limitation files, absolute change dL = L_hose - L_control)
    (a) sign map, using the cutoff TAU on |dL|. Cells where |dN| < TAU are left blank (same cells as the
        blanks in (b) and (c)); the other cells get one of three colors:
          opposing        : |dN| >= TAU and |dL| >= TAU, opposite signs
          same direction  : |dN| >= TAU and |dL| >= TAU, same sign
          light change small : |dN| >= TAU but |dL| < TAU
    (b) |dL_light| / |dL_nut|   linear 0 to 2, blank where |dN| < TAU
    (c) |dL_temp|  / |dL_nut|   same scale, same blank rule

TABLE (regional numbers, from the NATIVE limitation files, native binary masks, csize cell areas,
        area-weighted: the TabK_summary convention). Regions HA NA SA SO AB. Two versions side by side:
          abs : dX = X_hose - X_control
          rel : dX = ln(X_hose / X_control)
        Both versions use the SAME cells (the cell sets are fixed by the absolute cutoff TAU).
        Columns:
          mean_d*        signed area-weighted mean change of nutrient / light / temperature (all valid cells)
          meanabs_d*     area-weighted mean of |change| (all valid cells)
          share_light_lt_nut, share_temp_lt_nut
                         area share, among cells with |dN_abs| >= TAU, where |change| of light (temperature)
                         is smaller than |change| of nutrient
          share_opposing, share_same, share_nut_only, share_nut_small
                         area shares of all valid cells in the region (sum to 1):
                           opposing / same: both changes clear, opposite / same sign
                           nut_only:  nutrient change clear, light change small  ("light change small" in the map)
                           nut_small: nutrient change small, light anything      (the blank cells of the map)
                         These depend only on signs and cutoffs, so they are identical in abs and rel.
          frac_area_clearN   area share of the region with |dN_abs| >= TAU (denominator of the two shares above)
          n_cells        number of valid native cells

Valid cell = finite in all six fields (3 terms x 2 runs), total_weight > 0 in both runs, and all six
limitation values > 0 (needed for the log). Variables are dimensionless 0..1; the script refuses to run if a
limitation field falls outside 0..1.

Data (found from models.yml, override with NATIVE_PATHS / RG_PATHS):
    native : <clim_dir>/<run_id[-4:]>_limitation_maps.nc       (dims y, x; nav_lat, nav_lon)
    maps   : <clim_dir>/<run_id[-4:]>_limitation_maps_rg.nc    (bilinear, 1 degree)
Maps use the regridded files and the table uses the native files, so the area shares printed in the table
can differ slightly from what the maps show. Quote the table.

Geometry check: nav_lat/nav_lon of the limitation files are compared with the mask file, inside the regions only.

Run from figures_scripts:   python SF_LIMCOMP.py [--tau 0.01] [--suffix top14_annual] [--no-plot]
Output: ./plts/SF_LIMCOMP_YYYYMMDD.png and ./plts/SF_LIMCOMP_regional_YYYYMMDD.csv
"""

import sys
import os
import csv
import argparse

sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr

import map_maker as mm
import TabK_summary as tk

# -- tunable ---------------------------------------------------------------------------------------
WATERMARK   = "AMOC-eco/NEMO5/scripts/SF_LIMCOMP.py"
OUTPUT_DIR  = "./plts"
FIGURE_CODE = "SF_LIMCOMP"

DEFAULT_SUFFIX = "top14_annual"      # top 14 levels (about 150 m), the same window as Fig 3
DEFAULT_TAU    = 0.01                # |change| below this counts as "no clear change" (0..1 scale)
REGIONS        = ["HA", "NA", "SA", "SO", "AB"]

RATIO_MAX  = 2.0
RATIO_CMAP = "BrBG"                  # diverging, centered on a ratio of 1
CLASS_COLORS = ["#E69F00", "#0072B2", "#009E73"]   # opposing, same direction, light change small
CLASS_LABELS = ["opposing", "same\ndirection", "light change\nsmall"]

# Set to a dict {run label: path} to override the paths built from models.yml
NATIVE_PATHS = None
RG_PATHS     = None


# -- paths and loading -----------------------------------------------------------------------------
def lim_path(runs, label, native):
    name = "%s_limitation_maps%s.nc" % (str(runs[label]["run_id"])[-4:], "" if native else "_rg")
    return os.path.join(runs[label]["clim_dir"], name)


def var_names(suffix):
    return {"N": "L_nut_" + suffix, "L": "L_light_" + suffix, "T": "L_temp_" + suffix,
            "W": "total_weight_" + suffix}


def check_range(a, label):
    f = a[np.isfinite(a)]
    if f.size == 0:
        raise ValueError("%s: no finite values" % label)
    if f.min() < -1e-9 or f.max() > 1.0 + 1e-9:
        raise ValueError("%s: values %.4g .. %.4g are outside 0..1. A limitation factor is dimensionless "
                         "0..1, so this is not the expected variable or file." % (label, f.min(), f.max()))


def read_native(path, names):
    """Return ({key: 2D array}, nav_lat, nav_lon) from one native limitation file."""
    if not os.path.exists(path):
        raise IOError("native limitation file not found: %s" % path)
    out = {}
    with xr.open_dataset(path) as ds:
        for key, v in names.items():
            if v not in ds:
                raise KeyError("%s: variable %s not found" % (path, v))
            a = np.squeeze(ds[v].values).astype("f8")
            if a.ndim != 2:
                raise ValueError("%s: %s has shape %s after squeeze, expected 2D" % (path, v, a.shape))
            if key != "W":
                check_range(a, "%s:%s" % (os.path.basename(path), v))
            out[key] = a
        lat = np.squeeze(ds["nav_lat"].values).astype("f8") if "nav_lat" in ds else None
        lon = np.squeeze(ds["nav_lon"].values).astype("f8") if "nav_lon" in ds else None
    return out, lat, lon


def load_native_geometry():
    """Cell areas and binary province masks on the native grid, as in TabK_summary / MF_EXPD."""
    with xr.open_dataset(tk.CELLSIZE_MASK_PATH) as mesh:
        csize = np.squeeze(mesh["csize"].values).astype("f8")
    masks = {}
    mask_lat = mask_lon = None
    with xr.open_dataset(tk.PROVINCE_MASK_PATH) as ma:
        for name in ["HA", "NA", "SA", "SO", "AB"]:
            masks[name] = np.squeeze(ma[name].values).astype("f8")
        for cand in ("nav_lat", "lat", "latitude"):
            if cand in ma.variables and np.squeeze(ma[cand].values).ndim == 2:
                mask_lat = np.squeeze(ma[cand].values).astype("f8")
                break
        for cand in ("nav_lon", "lon", "longitude"):
            if cand in ma.variables and np.squeeze(ma[cand].values).ndim == 2:
                mask_lon = np.squeeze(ma[cand].values).astype("f8")
                break
    return csize, masks, mask_lat, mask_lon


def _coords_differ(lat1, lon1, lat2, lon2, tol=1e-3):
    """True where latitude differs by more than tol degrees, or longitude (modulo 360) does."""
    out = np.abs(lat1 - lat2) > tol
    if lon1 is not None and lon2 is not None:
        out = out | (np.abs(((lon1 - lon2) + 180.0) % 360.0 - 180.0) > tol)
    return out


# -- the calculation (shared by the maps and the table) ----------------------------------------------
def classify(f, tau):
    """f: dict of 2D arrays Nc Nh Lc Lh Tc Th Wc Wh (control/hose of nutrient, light, temperature, weight).
    Returns dict with valid mask, abs and rel changes, clear flags, class codes and ratios."""
    shape = f["Nc"].shape
    six = [f[k] for k in ("Nc", "Nh", "Lc", "Lh", "Tc", "Th")]
    with np.errstate(invalid="ignore"):
        basic = np.ones(shape, bool)
        for a in six:
            basic &= np.isfinite(a)
        basic &= (f["Wc"] > 0) & (f["Wh"] > 0)
        valid = basic.copy()
        for a in six:
            valid &= (a > 0)
    n_dropped_nonpos = int(basic.sum() - valid.sum())

    def safe(x):
        return np.where(valid, x, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        dN, dL, dT = safe(f["Nh"] - f["Nc"]), safe(f["Lh"] - f["Lc"]), safe(f["Th"] - f["Tc"])
        one = np.where(valid, 1.0, 1.0)
        rN = safe(np.log(np.where(valid, f["Nh"], one) / np.where(valid, f["Nc"], one)))
        rL = safe(np.log(np.where(valid, f["Lh"], one) / np.where(valid, f["Lc"], one)))
        rT = safe(np.log(np.where(valid, f["Th"], one) / np.where(valid, f["Tc"], one)))
        clearN = valid & (np.abs(np.where(valid, dN, 0.0)) >= tau)
        clearL = valid & (np.abs(np.where(valid, dL, 0.0)) >= tau)
        both = clearN & clearL
        prod = np.where(valid, dN * dL, 0.0)
    cls = np.full(shape, np.nan)
    cls[valid] = 0
    cls[both & (prod < 0)] = 1
    cls[both & (prod > 0)] = 2
    cls[clearN & ~clearL] = 3
    ratio_L = np.full(shape, np.nan)
    ratio_T = np.full(shape, np.nan)
    ratio_L[clearN] = np.abs(dL[clearN]) / np.abs(dN[clearN])
    ratio_T[clearN] = np.abs(dT[clearN]) / np.abs(dN[clearN])
    return dict(valid=valid, dN=dN, dL=dL, dT=dT, rN=rN, rL=rL, rT=rT, clearN=clearN, clearL=clearL,
                cls=cls, ratio_L=ratio_L, ratio_T=ratio_T, n_dropped_nonpos=n_dropped_nonpos)


def _wsum(x, w):
    return float(np.sum(x * w))


def region_stats(c, w, version):
    """One row of the table. w = csize * region mask (0 outside the region)."""
    v = c["valid"]
    wv = np.where(v, w, 0.0)
    A = wv.sum()
    row = {"version": version}
    keys = ("dN", "dL", "dT") if version == "abs" else ("rN", "rL", "rT")
    x = {k: np.where(v, c[kk], 0.0) for k, kk in zip(("N", "L", "T"), keys)}
    nan = float("nan")
    for k in ("N", "L", "T"):
        row["mean_d" + {"N": "nut", "L": "light", "T": "temp"}[k]] = _wsum(x[k], wv) / A if A > 0 else nan
    for k in ("N", "L", "T"):
        row["meanabs_d" + {"N": "nut", "L": "light", "T": "temp"}[k]] = (
            _wsum(np.abs(x[k]), wv) / A if A > 0 else nan)
    sel = v & c["clearN"]
    den = float(w[sel].sum())
    for k, name in (("L", "share_light_lt_nut"), ("T", "share_temp_lt_nut")):
        if den > 0:
            num = float(w[sel & (np.abs(x[k]) < np.abs(x["N"]))].sum())
            row[name] = num / den
        else:
            row[name] = nan
    for code, name in ((1, "share_opposing"), (2, "share_same"), (3, "share_nut_only"), (0, "share_nut_small")):
        row[name] = float(w[v & (c["cls"] == code)].sum()) / A if A > 0 else nan
    row["frac_area_clearN"] = den / A if A > 0 else nan
    row["n_cells"] = int(((w > 0) & v).sum())
    return row


COLUMNS = ["region", "version", "mean_dnut", "mean_dlight", "mean_dtemp",
           "meanabs_dnut", "meanabs_dlight", "meanabs_dtemp",
           "share_light_lt_nut", "share_temp_lt_nut",
           "share_opposing", "share_same", "share_nut_only", "share_nut_small",
           "frac_area_clearN", "n_cells"]


def build_table(runs, run_a, run_b, suffix, tau):
    names = var_names(suffix)
    paths = NATIVE_PATHS or {r: lim_path(runs, r, True) for r in (run_a, run_b)}
    A, lat_a, lon_a = read_native(paths[run_a], names)
    B, lat_b, lon_b = read_native(paths[run_b], names)
    csize, masks, mask_lat, mask_lon = load_native_geometry()

    shapes = {A["N"].shape, B["N"].shape, csize.shape}
    shapes.update(m.shape for m in masks.values())
    if len(shapes) != 1:
        raise ValueError("native grid shapes differ between limitation files, csize and masks: %s" % shapes)

    # Coordinate checks use only the cells inside the regions (AB). The limitation files hold placeholder
    # coordinates (0, 0) at some cells without data; those cells are never used, so they must not stop the run.
    inreg = masks["AB"] > 0
    if lat_a is not None and lat_b is not None:
        d = _coords_differ(lat_a, lon_a, lat_b, lon_b)
        if (d & inreg).any():
            raise ValueError("the two limitation files have different nav_lat/nav_lon in %d cells inside the "
                             "regions" % (d & inreg).sum())
    if lat_a is not None and mask_lat is not None:
        d = _coords_differ(lat_a, lon_a, mask_lat, mask_lon)
        n_in, n_out = int((d & inreg).sum()), int((d & ~inreg).sum())
        if n_in > 0:
            raise ValueError("nav_lat/nav_lon of the limitation file differ from the mask file in %d of %d cells "
                             "inside the regions: the grids may be flipped or shifted. Stop and check."
                             % (n_in, inreg.sum()))
        print("[geometry check] limitation-file and mask-file coordinates agree in all %d cells inside the "
              "regions (%s compared); %d cells outside the regions differ and are not used"
              % (inreg.sum(), "latitude and longitude" if (lon_a is not None and mask_lon is not None)
                 else "latitude only", n_out))
    else:
        print("[geometry check] only the array shape %s could be compared (the mask file carries no 2D "
              "latitude, or the limitation file has no nav_lat)" % (next(iter(shapes)),))

    f = dict(Nc=A["N"], Nh=B["N"], Lc=A["L"], Lh=B["L"], Tc=A["T"], Th=B["T"], Wc=A["W"], Wh=B["W"])
    fin_a = np.isfinite(A["N"])
    fin_b = np.isfinite(B["N"])
    print("[cells] finite nutrient values: %s %d, %s %d, identical pattern: %s"
          % (run_a, fin_a.sum(), run_b, fin_b.sum(), bool(np.array_equal(fin_a, fin_b))))
    c = classify(f, tau)
    print("[cells] valid cells: %d   dropped for a non-positive value: %d"
          % (c["valid"].sum(), c["n_dropped_nonpos"]))

    rows = []
    for reg in REGIONS:
        w = csize * masks[reg]
        in_reg = w > 0
        cover = float(w[in_reg & c["valid"]].sum() / w[in_reg].sum()) if in_reg.any() else float("nan")
        print("[coverage] %s: %.1f%% of the region area has valid limitation values" % (reg, 100 * cover))
        for ver in ("abs", "rel"):
            r = region_stats(c, w, ver)
            r["region"] = reg
            rows.append(r)
        tot = sum(rows[-2][k] for k in ("share_opposing", "share_same", "share_nut_only", "share_nut_small"))
        if abs(tot - 1.0) > 1e-9:
            raise AssertionError("%s: class shares sum to %.12f, not 1" % (reg, tot))
    return rows


def print_table(rows):
    hdr = ("%-3s %-4s %8s %8s %8s | %8s %8s %8s | %7s %7s | %6s %6s %6s %6s | %6s %6s"
           % ("reg", "ver", "m_dN", "m_dL", "m_dT", "|dN|", "|dL|", "|dT|", "L<N", "T<N",
              "opp", "same", "Nonly", "Nsmall", "clearN", "ncell"))
    print("\n" + hdr)
    for r in rows:
        print("%-3s %-4s %+8.4f %+8.4f %+8.4f | %8.4f %8.4f %8.4f | %7.3f %7.3f | %6.3f %6.3f %6.3f %6.3f "
              "| %6.3f %6d"
              % (r["region"], r["version"], r["mean_dnut"], r["mean_dlight"], r["mean_dtemp"],
                 r["meanabs_dnut"], r["meanabs_dlight"], r["meanabs_dtemp"],
                 r["share_light_lt_nut"], r["share_temp_lt_nut"],
                 r["share_opposing"], r["share_same"], r["share_nut_only"], r["share_nut_small"],
                 r["frac_area_clearN"], r["n_cells"]))


def write_csv(rows, path):
    with open(path, "w") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLUMNS)
        for r in rows:
            wr.writerow([r[k] for k in COLUMNS])


# -- maps -------------------------------------------------------------------------------------------
def _draw_or_blank(ax, cax, lon, lat, data, cmap, vmin, vmax, lon_m, lat_m, masks, extent, tag):
    try:
        return mm._draw_panel(ax, lon, lat, data, cmap, vmin, vmax, lon_m, lat_m, masks, extent=extent)
    except Exception as exc:
        print("    [warn] %s failed (%r) -- leaving blank" % (tag, exc))
        ax.clear()
        ax.axis("off")
        ax.text(0.5, 0.5, "render failed", transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="0.5")
        cax.axis("off")
        return None


def make_figure(runs, run_a, run_b, suffix, tau, watermark=True):
    mm.set_presentation_style()
    lon_min, lon_max, lat_min, lat_max = mm._load_extent()
    extent = [lon_min, lon_max, lat_min, lat_max]
    lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)

    names = var_names(suffix)
    paths = RG_PATHS or {r: lim_path(runs, r, False) for r in (run_a, run_b)}
    got = {}
    lon = lat = None
    for key, v in names.items():
        for tag, run in (("c", run_a), ("h", run_b)):
            lon, lat, a, _ = mm.load_field(paths[run], v, 0)
            if key != "W":
                check_range(a, "%s:%s" % (os.path.basename(paths[run]), v))
            got[key + tag] = a
    f = dict(Nc=got["Nc"], Nh=got["Nh"], Lc=got["Lc"], Lh=got["Lh"], Tc=got["Tc"], Th=got["Th"],
             Wc=got["Wc"], Wh=got["Wh"])
    c = classify(f, tau)

    fig = plt.figure(figsize=(3.8 * 3, 5.0))
    gs = fig.add_gridspec(1, 3, wspace=0.06)
    fig.subplots_adjust(top=0.90, bottom=0.17 if watermark else 0.10, left=0.03, right=0.99)

    cmap_cls = ListedColormap(CLASS_COLORS)
    cls_plot = np.where(c["cls"] == 0, np.nan, c["cls"])      # code 0 = nutrient change small: left blank
    cmap_rat = plt.get_cmap(RATIO_CMAP)
    panels = [
        dict(title=r"(a) Sign of $\Delta L_{light}$ vs $\Delta L_{nut}$", data=cls_plot, cmap=cmap_cls,
             vmin=0.5, vmax=3.5,
             label=r"clear: $|\Delta L|\geq %g$ (blank: $|\Delta L_{nut}|<%g$)" % (tau, tau)),
        dict(title=r"(b) $|\Delta L_{light}|\,/\,|\Delta L_{nut}|$", data=c["ratio_L"], cmap=cmap_rat,
             vmin=0.0, vmax=RATIO_MAX,
             label=r"ratio (blank: $|\Delta L_{nut}|<%g$)" % tau),
        dict(title=r"(c) $|\Delta L_{temp}|\,/\,|\Delta L_{nut}|$", data=c["ratio_T"], cmap=cmap_rat,
             vmin=0.0, vmax=RATIO_MAX,
             label=r"ratio (blank: $|\Delta L_{nut}|<%g$)" % tau),
    ]
    for j, p in enumerate(panels):
        inner = gs[0, j].subgridspec(2, 1, height_ratios=[1, 0.06], hspace=0.10)
        ax = fig.add_subplot(inner[0, 0], projection=mm.ATLANTIC_PROJ)
        cbar_gs = inner[1, 0].subgridspec(1, 3, width_ratios=[0.15, 0.70, 0.15], wspace=0)
        cax = fig.add_subplot(cbar_gs[0, 1])
        mesh = _draw_or_blank(ax, cax, lon, lat, p["data"], p["cmap"], p["vmin"], p["vmax"],
                              lon_m, lat_m, masks, extent, p["title"])
        ax.set_title(p["title"], fontweight="bold")
        if mesh is None:
            continue
        if j == 0:
            cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
            cbar.set_ticks([1, 2, 3])
            cbar.set_ticklabels(CLASS_LABELS)
            cbar.outline.set_visible(False)
        else:
            cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
            cbar.set_ticks([0, 0.5, 1.0, 1.5, 2.0])
            cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=plt.rcParams["font.size"] - 2)
        cbar.set_label(p["label"], fontsize=plt.rcParams["font.size"] - 2)
    if watermark:
        fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom",
                 fontsize=plt.rcParams["font.size"] - 6, color="0.55")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX)
    ap.add_argument("--no-plot", action="store_true", help="table and CSV only")
    a = ap.parse_args()

    run_labels, runs = mm._load_models()
    run_a, run_b = run_labels[0], run_labels[1]
    print("control = %s, hose = %s, suffix = %s, cutoff tau = %g" % (run_a, run_b, a.suffix, a.tau))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    rows = build_table(runs, run_a, run_b, a.suffix, a.tau)
    print_table(rows)
    csv_path = os.path.join(OUTPUT_DIR, "%s_regional_%s.csv" % (FIGURE_CODE, stamp))
    write_csv(rows, csv_path)
    print("\nsaved %s" % csv_path)

    if not a.no_plot:
        fig = make_figure(runs, run_a, run_b, a.suffix, a.tau)
        out = os.path.join(OUTPUT_DIR, "%s_%s.png" % (FIGURE_CODE, stamp))
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print("saved %s" % out)


if __name__ == "__main__":
    main()
