"""
MF_AMOC.py  —  Main Figure: AMOC Streamfunction + Study Regions
(MTbasins mask version)

Reads the MOC files recomputed with the test_withMTbasins.nc mask
(*_MOC_MTbasins.nc).

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/MF_AMOC_YYYYMMDD.png
"""

import os
import sys
import datetime
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')
sys.path.append('/gpfs/home/mep22dku/scratch/SOZONE/UTILS')

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plot_style import set_presentation_style, BAUHAUS

# ── tunable ────────────────────────────────────────────────────────────
WATERMARK   = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/MF_AMOC.py"
WATERMARK_ON   = True    # False = no script-path watermark (or run with --no-watermark / --water-off)
for _flag in ("--no-watermark", "--water-off"):
    if _flag in sys.argv:
        sys.argv.remove(_flag)
        WATERMARK_ON = False
OUTPUT_DIR  = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"
FIGURE_CODE = "MF_AMOC"

MODELS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yml")
# MOC files now live in one folder per run: <MOC_ROOT>/<run_id>/
MOC_ROOT           = "/gpfs/data/greenocean/software/runs/NEMO5"
MASK_TAG           = "MTbasins"
MOC_FILENAME_TEMPLATE = "{run_id}_7d_{year}0101_{year}1231_MOC_{mask}.nc"

YEARS      = range(2010, 2020)
VAR_NAME   = "zomsfatl"
LAT_VAR    = "nav_lat"
DEPTH_DIM  = "depthw"

AMOC_MIN_LAT   = 0       # AMOC strength: north of this latitude (deg N)
AMOC_MIN_DEPTH = 500     # ... and below this depth (m)

LEVEL_STEP      = 2.0
COLOR_MIN       = -11
COLOR_MAX       = 11
LAT_MIN         = -30
LAT_MAX         = 75
DEPTH_MAX       = -3000
N_FILLED_LEVELS = 61

RUN_DISPLAY_NAME = {"BASE": "CONTROL"}

GLOBE_PROJ = ccrs.Orthographic(central_longitude=-30, central_latitude=10)

MASK_PATH    = os.path.expanduser(
    "~/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL_rg.nc")
MASK_REGIONS = ["NHL", "NA", "SA", "SO"]
MASK_COLORS  = {"NHL": BAUHAUS[3], "NA": BAUHAUS[2],
                "SA": BAUHAUS[1], "SO": BAUHAUS[0]}
MASK_LINEWIDTHS  = {r: 1.8 for r in MASK_REGIONS}
MASK_LINESTYLES  = {r: "solid" for r in MASK_REGIONS}
MASK_ZORDERS     = {r: 4 for r in MASK_REGIONS}
MASK_LAT_OFFSET  = {"NHL": 1.0, "NA": 0.5, "SA": 0.0, "SO": -0.5}
MASK_HALO_COLOR  = "white"
MASK_HALO_EXTRA_WIDTH = 0.8

FIG_WIDTH    = 12.0
GOLDEN_RATIO = 1.618
COLUMN_WIDTH_RATIOS = [1.3, 0.06, 1.0]

# ── streamfunction helpers ─────────────────────────────────────────────
def _load_run_ids():
    with open(MODELS_CONFIG_PATH) as f:
        runs = yaml.safe_load(f)["runs"]
    labels = list(runs)
    ids = {label: run["run_id"] for label, run in runs.items()}
    return labels, ids


def _sv_scale(units, peak, run_id):
    """cdfmoc already writes Sv. Only rescale if values are clearly m3/s."""
    if peak > 1e3:          # a real AMOC is O(10) Sv; m3/s would be O(1e7)
        print(f"[{run_id}] WARNING: peak |psi|={peak:.3g} (units={units!r}), "
              "assuming m3/s -> Sv")
        return 1e-6
    return 1.0


def load_moc(run_id):
    yearly = []
    lat = depth = units = None
    for year in YEARS:
        path = os.path.join(MOC_ROOT, run_id, MOC_FILENAME_TEMPLATE.format(
            run_id=run_id, year=year, mask=MASK_TAG))
        if not os.path.exists(path):
            print(f"[{run_id}] missing {os.path.basename(path)}, skipping")
            continue
        ds = xr.open_dataset(path)
        da = ds[VAR_NAME].squeeze("x", drop=True)
        if units is None:
            units = da.attrs.get("units", "")
        tdim = "time_counter" if "time_counter" in da.dims else "time"
        if tdim in da.dims:
            da = da.mean(tdim)
        yearly.append(da.load())
        if lat is None:
            lat   = ds[LAT_VAR].squeeze("x", drop=True).values
            depth = ds[DEPTH_DIM].values
        ds.close()
    if not yearly:
        raise FileNotFoundError(
            f"No {MASK_TAG} MOC files found for {run_id} in {os.path.join(MOC_ROOT, run_id)}")
    stack = xr.concat(yearly, dim="year").values      # (year, depth, lat)
    sv    = stack.mean(axis=0)
    scale = _sv_scale(units, np.nanmax(np.abs(sv)), run_id)
    sv, stack = sv * scale, stack * scale
    print(f"[{run_id}] {len(yearly)} yrs, units={units!r}, "
          f"range {np.nanmin(sv):.1f} to {np.nanmax(sv):.1f} Sv")
    return lat, depth, sv, stack


def amoc_strength(lat, depth, sv):
    """Max of the streamfunction north of the equator and below 500 m.
    Returns (value in Sv, latitude, depth). Uses |depth| so it works
    whether depthw is stored positive or negative."""
    sel = (np.abs(depth)[:, None] > AMOC_MIN_DEPTH) & (lat[None, :] > AMOC_MIN_LAT)
    sub = np.where(sel, sv, np.nan)
    iz, iy = np.unravel_index(np.nanargmax(sub), sub.shape)
    return float(sub[iz, iy]), float(lat[iy]), float(abs(depth[iz]))


def report_amoc(label, lat, depth, sv, stack):
    clim, clat, cdep = amoc_strength(lat, depth, sv)
    yearly = np.array([amoc_strength(lat, depth, s)[0] for s in stack])
    print(f"[{label}] AMOC strength (max N of {AMOC_MIN_LAT}\u00b0N, "
          f"below {AMOC_MIN_DEPTH} m): {clim:.2f} Sv at "
          f"{clat:.1f}\u00b0N, {cdep:.0f} m (climatology of {len(yearly)} yrs); "
          f"annual max {yearly.mean():.2f} \u00b1 {yearly.std():.2f} Sv "
          f"(range {yearly.min():.2f}\u2013{yearly.max():.2f})")
    return clim


def _contour_levels(*fields):
    # levels only matter within the plotted colour range
    vmax = max(np.ceil(COLOR_MAX / LEVEL_STEP) * LEVEL_STEP, LEVEL_STEP)
    return np.arange(-vmax, vmax + LEVEL_STEP, LEVEL_STEP)


def _plot_panel(ax, lat, depth, sv, levels, title):
    cf = ax.contourf(lat, depth, sv,
                     levels=np.linspace(COLOR_MIN, COLOR_MAX, N_FILLED_LEVELS),
                     cmap=cmocean.cm.balance, extend="both")
    cs = ax.contour(lat, depth, sv, levels=levels, colors="black",
                    linewidths=1.1,
                    linestyles=["dashed" if l < 0 else "solid" for l in levels])
    ax.clabel(cs, inline=True, fontsize=9, fmt="%g")
    ax.set(xlim=(LAT_MIN, LAT_MAX), ylim=(DEPTH_MAX, 0),
           xlabel="Latitude (\u00b0N)", ylabel="Depth (m)", title=title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return cf


# ── globe helpers ──────────────────────────────────────────────────────
def _load_masks():
    ds  = xr.open_dataset(MASK_PATH)
    lon = ds["lon"].values
    lat = ds["lat"].values
    masks = {n: np.squeeze(ds[n].values) for n in MASK_REGIONS}
    if lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
        order = np.argsort(lon)
        lon = lon[order]
        masks = {n: m[:, order] for n, m in masks.items()}
    return lon, lat, masks


def _contour_halo(ax, lon, lat, mask, color, lw, ls, zorder):
    ax.contour(lon, lat, mask, levels=[0.5],
               colors=[MASK_HALO_COLOR], linewidths=lw + MASK_HALO_EXTRA_WIDTH,
               linestyles=ls, transform=ccrs.PlateCarree(), zorder=zorder - 0.1)
    ax.contour(lon, lat, mask, levels=[0.5],
               colors=[color], linewidths=lw, linestyles=ls,
               transform=ccrs.PlateCarree(), zorder=zorder)


def _plot_globe(ax, lon_m, lat_m, masks):
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="#dceaf2", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="0.85",   zorder=1)
    ax.coastlines(resolution="110m", linewidth=0.4, color="0.4", zorder=2)
    for name in MASK_REGIONS:
        _contour_halo(ax, lon_m, lat_m + MASK_LAT_OFFSET[name], masks[name],
                      MASK_COLORS[name], MASK_LINEWIDTHS[name],
                      MASK_LINESTYLES[name], MASK_ZORDERS[name])
        lon2d, lat2d = np.meshgrid(lon_m, lat_m)
        sel = masks[name] > 0.5
        cx  = float(np.mean(lon2d[sel]))
        cy  = float(np.mean(lat2d[sel])) + MASK_LAT_OFFSET[name]
        txt = ax.text(cx, cy, name, transform=ccrs.PlateCarree(),
                      color=MASK_COLORS[name], fontweight="bold", fontsize=12,
                      ha="center", va="center", zorder=20)
        txt.set_path_effects(
            [pe.withStroke(linewidth=2.5, foreground=MASK_HALO_COLOR)])
    ax.set_title("Study regions", fontweight="bold")


# ── main figure ────────────────────────────────────────────────────────
def make_figure(watermark=True):
    set_presentation_style()

    labels, run_ids = _load_run_ids()
    run_a, run_b    = labels[:2]

    lat_a, depth_a, sv_a, st_a = load_moc(run_ids[run_a])
    lat_b, depth_b, sv_b, st_b = load_moc(run_ids[run_b])
    levels = _contour_levels(sv_a, sv_b)

    amoc_a = report_amoc(RUN_DISPLAY_NAME.get(run_a, run_a),
                         lat_a, depth_a, sv_a, st_a)
    amoc_b = report_amoc(RUN_DISPLAY_NAME.get(run_b, run_b),
                         lat_b, depth_b, sv_b, st_b)
    print(f"AMOC strength change ({run_b} - {run_a}): {amoc_b - amoc_a:+.2f} Sv "
          f"({100 * (amoc_b - amoc_a) / amoc_a:+.1f}%)")

    lon_m, lat_m, masks = _load_masks()

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH / GOLDEN_RATIO))
    gs  = fig.add_gridspec(2, 3,
                            width_ratios=COLUMN_WIDTH_RATIOS,
                            height_ratios=[1, 1],
                            wspace=0.35, hspace=0.15)

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hose = fig.add_subplot(gs[1, 0], sharex=ax_hist, sharey=ax_hist)

    cf = _plot_panel(ax_hist, lat_a, depth_a, sv_a, levels,
                     RUN_DISPLAY_NAME.get(run_a, run_a))
    _plot_panel(ax_hose, lat_b, depth_b, sv_b, levels,
                RUN_DISPLAY_NAME.get(run_b, run_b))
    ax_hist.set_xlabel("")
    plt.setp(ax_hist.get_xticklabels(), visible=False)
    ax_hist.tick_params(axis="x", which="both", length=0)

    # shared colorbar
    cbar_ax = fig.add_subplot(gs[:, 1])
    cbar_ax.set_visible(False)
    real_cax = fig.add_axes(cbar_ax.get_position())
    cbar = fig.colorbar(cf, cax=real_cax)
    cbar.set_label("Streamfunction (Sv)")
    cbar.set_ticks(np.arange(-10, 11, 2))
    cbar.outline.set_visible(False)

    ax_globe = fig.add_subplot(gs[:, 2], projection=GLOBE_PROJ)
    _plot_globe(ax_globe, lon_m, lat_m, masks)

    if watermark and WATERMARK_ON:
        fig.text(0.5, 0.008, WATERMARK, ha="center", va="bottom",
                 fontsize=plt.rcParams.get("font.size", 10) - 6, color="0.55")

    stamp = datetime.date.today().strftime("%Y%m%d")
    return fig, f"{FIGURE_CODE}_{stamp}.png"


def save_figure(watermark=True, dpi=300):
    fig, fname = make_figure(watermark=watermark)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    save_figure()
