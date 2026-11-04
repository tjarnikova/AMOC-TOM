"""
map_maker.py

Generic "type A" map figure: 2 rows x 2 columns.

    Row1Col1 -- Atlantic-focused BASE field           (PlateCarree, as in
                figA_phyto_overview.py)
    Row1Col2 -- Atlantic-focused HOSE-BASE difference, centred on 0
    Row2Col1 -- Global BASE field                      (Mollweide)
    Row2Col2 -- Global HOSE-BASE difference, centred on 0

Unlike figA/figB/figC/figE (each hardcoded to specific variables/files),
this is a general-purpose plot maker: it takes a variable name, an
optional depth level, and a filetype, and works for any variable in any
of the project's "_rg" (regridded) climatology files -- diad_T, ptrc_T,
ptrc_T_int, grid_T, LNL_FULL_T, etc.

DEPTH HANDLING -- a variable may be 3D (time, lat, lon -- e.g. Cflx) or
4D (time, depth, lat, lon -- e.g. a ptrc_T tracer concentration). This is
detected by checking whether the variable's dims include the standard
depth-dim name used throughout this pipeline's regridded files,
`grid_T_3D_inner` (confirmed via ncdump against diad_T's EXP variable --
see TabA_phyto.py / figA_phyto_overview.py). If present, `d` (0-based
depth-level index, default 0 = surface) selects a level before the
annual mean is taken. If absent, `d` has no effect on the data at all --
it's still embedded in the output filename for interface consistency
with transect_maker.py / polar_maker.py (e.g. Cflx, which has no depth
dimension, is still saved as ".../MAP_Cflx_d0_YYYYMMDD.png").

DATA -- yearly-averaged (mean over the time dimension) values from
climatology files named
    ORCA2_7d_clim_{clim_period}_{filetype}_rg.nc
e.g. filetype="diad_T" -> ORCA2_7d_clim_2010_2019_diad_T_rg.nc,
filetype="ptrc_T_int" -> ORCA2_7d_clim_2010_2019_ptrc_T_int_rg.nc.
Same run/path bookkeeping (models.yml) as every fig*.py script.

MASKS -- NHL/NA/SA/SO province outlines overlaid on every panel (white
halo + BAUHAUS colours), same mask file/styling as
figA_phyto_overview.py / figC_map_amoc.py / figH_globe_regions.py.

KNOWN CARTOPY/SHAPELY BUG -- some cartopy/shapely version combinations
intermittently raise "'GeometryCollection' object is not subscriptable"
while reprojecting a filled path (a data mesh, or the land feature)
against certain projections -- a genuine, unresolved upstream bug (see
https://github.com/SciTools/cartopy/issues/2176), not something fixable
from this script. It's surfaced early per panel (via a forced
canvas.draw() right after drawing each panel, rather than waiting for
the deferred render inside fig.savefig()) and caught in make_figure():
if it happens, that ONE panel is left blank with a small "render failed"
note instead of losing the whole figure.

COLOUR SCALES -- BASE panels use the BASE_PERCENTILES-th percentile
range of that panel's own field (Atlantic row: percentile computed only
over the Atlantic bounding box; Global row: percentile computed over the
whole field), same percentile-fallback style as figA/figE. Diff panels
use the symmetric DIFF_PERCENTILE-th percentile of |diff|, widened by
DIFF_LIMIT_SCALE, computed the same way (row-appropriate subset).

SIGN CONVENTION -- diff = HOSE - BASE (run_b - run_a), matching
figA_phyto_overview.py / figE_limitation.py / TabB_physics.py /
TabA_phyto.py.

Uses the shared house style in plot_style.py (Gill Sans, Tol-bright
palette, no top/right spines), same as every other fig*.py script.

Usage (from a notebook):
    from map_maker import make_figure
    fig, fname = make_figure("PPINT", d=0, filetype="diad_T")
    fig.savefig(f"./plts/{fname}", bbox_inches="tight")

    # 4D tracer, 3rd depth level (0-based), from ptrc_T_int_rg.nc
    fig, fname = make_figure("DIA", d=2, filetype="ptrc_T_int")

    # small grey "script path" watermark along the bottom (on by default)
    fig, fname = make_figure("Cflx", filetype="diad_T", watermark=True)
"""

import os
import sys
import datetime

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
try:
    import cmocean
    _CMOCEAN_AVAILABLE = True
except ImportError:
    cmocean = None
    _CMOCEAN_AVAILABLE = False

from plot_style import set_presentation_style, BAUHAUS
import plot_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------

MODELS_CONFIG_PATH = os.path.join(HERE, "models.yml")
FILENAME_TEMPLATE = "ORCA2_7d_clim_{clim_period}_{filetype}_rg.nc"
DEFAULT_FILETYPE = "diad_T"

OUTPUT_DIR = ps.output_dir()   # figure_settings.yml
FILENAME_PREFIX = "MAP"

# used only when make_figure(watermark=True) -- see that flag's docstring
WATERMARK_BASE_PATH = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts"
SCRIPT_NAME = "map_maker.py"

# standard depth-dim name in this pipeline's regridded files -- see
# module docstring's DEPTH HANDLING section
DEPTH_DIM_NAME = "grid_T_3D_inner"

BASE_PERCENTILES = (5, 95)
DIFF_PERCENTILE = 95
DIFF_LIMIT_SCALE = 1.44

# BASE-row / diff-row default colormap NAMES (not objects) -- resolved
# at call time via _resolve_cmap() so a missing/misspelled name (or a
# cmocean-less environment) degrades gracefully instead of crashing at
# import. See _resolve_cmap()'s docstring.
DEFAULT_CMAP_NAME = "thermal"   # cmocean.cm.thermal
DIFF_CMAP_NAME = "balance"      # cmocean.cm.balance
CMAP_FALLBACK = "viridis"       # matplotlib builtin, always available
DIFF_CMAP_FALLBACK = "RdBu_r"   # matplotlib builtin diverging colormap


def _resolve_cmap(cmap, default_name, fallback):
    """Resolve `cmap` to an actual Colormap object.

    Accepts:
      - None -> looked up as `default_name` (see below)
      - an existing Colormap instance -> returned unchanged
      - a string -> looked up first as a cmocean colormap name (e.g.
        "thermal", "balance", "speed", "algae", "tempo" -- any name
        under cmocean.cm), then as a standard matplotlib colormap name
        (e.g. "viridis", "RdBu_r"). If neither resolves (typo, or
        cmocean isn't installed and the name is cmocean-only), falls
        back to `fallback` (e.g. "viridis") with a printed warning
        rather than raising.
    """
    if isinstance(cmap, mcolors.Colormap):
        return cmap

    name = cmap if cmap is not None else default_name

    if _CMOCEAN_AVAILABLE and hasattr(cmocean.cm, name):
        return getattr(cmocean.cm, name)

    try:
        return plt.get_cmap(name)
    except (ValueError, KeyError, TypeError):
        print(f"[map_maker] colormap {name!r} not found in cmocean or "
              f"matplotlib -- using {fallback!r} instead")
        return plt.get_cmap(fallback)


def _load_models():
    """Re-read models.yml fresh (not cached at import)."""
    with open(MODELS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    runs = cfg["runs"]
    run_labels = list(runs)  # e.g. ["BASE", "HOSE"], in file order
    return run_labels, runs


def _paths_for_filetype(runs, filetype):
    return {
        label: os.path.join(r["clim_dir"], FILENAME_TEMPLATE.format(
            clim_period=str(r["clim_period"]), filetype=filetype))
        for label, r in runs.items()
    }


# ---------------------------------------------------------------------------
# Region / projection
# ---------------------------------------------------------------------------

EXTENT_CONFIG_PATH = os.path.join(HERE, "map_extents.yml")
EXTENT_REGION = "atlantic"


def _load_extent():
    """Re-read map_extents.yml fresh (not cached at import)."""
    with open(EXTENT_CONFIG_PATH) as f:
        extent = yaml.safe_load(f)[EXTENT_REGION]
    return extent["lon_min"], extent["lon_max"], extent["lat_min"], extent["lat_max"]


ATLANTIC_PROJ = ccrs.PlateCarree(central_longitude=-30)
GLOBAL_PROJ = ccrs.Mollweide(central_longitude=-30)

# NHL / NA / SA / SO region outlines -- same 5-province mask/styling as
# figA_phyto_overview.py / figC_map_amoc.py / figH_globe_regions.py
MASK_PATH = os.path.expanduser("~/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL_rg.nc")
MASK_REGIONS = ["NHL", "NA", "SA", "SO"]
MASK_COLORS = {
    "NHL": BAUHAUS[3],
    "NA": BAUHAUS[2],
    "SA": BAUHAUS[1],
    "SO": BAUHAUS[0],
}
MASK_LINEWIDTHS = {r: ps.mask_linewidth() for r in MASK_REGIONS}   # figure_settings.yml
MASK_LINESTYLES = {"NHL": "solid", "NA": "solid", "SA": "solid", "SO": "solid"}
MASK_ZORDERS = {"NHL": 4, "NA": 4, "SA": 4, "SO": 4}
MASK_LAT_OFFSET = {"NHL": 1.0, "NA": 0.5, "SA": 0.0, "SO": -0.5}
MASK_HALO_COLOR = "white"
MASK_HALO_EXTRA_WIDTH = 0.8


def load_region_masks(path, regions=MASK_REGIONS):
    """Load binary region masks (1/0) sharing a common lon/lat grid."""
    ds = xr.open_dataset(path)
    lon = ds["lon"].values
    lat = ds["lat"].values
    masks = {name: np.squeeze(ds[name].values) for name in regions}

    if lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
        order = np.argsort(lon)
        lon = lon[order]
        masks = {name: mask[:, order] for name, mask in masks.items()}

    return lon, lat, masks


def contour_with_halo(ax, lon, lat, mask, color, linewidth, linestyle, zorder):
    """Draw a mask outline with a white halo behind it, so it stays legible
    over any colormap / any projection."""
    ax.contour(
        lon, lat, mask, levels=[0.5],
        colors=[MASK_HALO_COLOR], linewidths=linewidth + MASK_HALO_EXTRA_WIDTH,
        linestyles=linestyle, transform=ccrs.PlateCarree(), zorder=zorder - 0.1,
    )
    ax.contour(
        lon, lat, mask, levels=[0.5],
        colors=[color], linewidths=linewidth, linestyles=linestyle,
        transform=ccrs.PlateCarree(), zorder=zorder,
    )


# ---------------------------------------------------------------------------
# Data loading -- generic, dims-agnostic (3D or 4D variables)
# ---------------------------------------------------------------------------

def _time_dim(da):
    """Name of the time dimension in `da`, or None if there isn't one --
    some derived/already-averaged fields (e.g. LNL_FULL_T's mean_NUT_*/
    mean_LIGHT_* variables) have dims of exactly (lat, lon), no time
    axis at all."""
    for candidate in ("time", "time_counter"):
        if candidate in da.dims:
            return candidate
    return None


def _depth_dim(da):
    """Name of the depth dimension in `da`, or None if this variable has
    no vertical axis at all (3D: time, lat, lon).

    Checks the standard `grid_T_3D_inner` name first (see module
    docstring); falls back to "first dim that isn't time/lat/lon" for any
    file that happens to name it differently, same heuristic
    figA_phyto_overview.py uses for EXP100.
    """
    if DEPTH_DIM_NAME in da.dims:
        return DEPTH_DIM_NAME
    excluded = {"time", "time_counter", "lat", "lon"}
    for d in da.dims:
        if d not in excluded:
            return d
    return None


def _finish_field(da):
    """Collapse any stray extra dims, force (lat, lon) axis order, and
    convert 0-360 lon to -180-180. Identical logic to fig1/fig2/fig3/figA."""
    extra_dims = [d for d in da.dims if d not in ("lat", "lon")]
    if extra_dims:
        da = da.mean(extra_dims)

    da = da.transpose("lat", "lon")

    if float(da["lon"].max()) > 180:
        da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180).sortby("lon")

    return da["lon"].values, da["lat"].values, da.values


def load_field(path, var, depth_index=0):
    """Annual-mean 2D field for one variable, from any of this pipeline's
    "_rg" files. Handles both 3D (time, lat, lon) and 4D (time, depth,
    lat, lon) variables transparently -- see _depth_dim().

    Returns (lon, lat, field, meta) where meta is
    {"has_depth": bool, "units": str}.
    """
    ds = xr.open_dataset(path)
    da = ds[var]
    units = da.attrs.get("units", "")

    ddim = _depth_dim(da)
    has_depth = ddim is not None
    if has_depth:
        da = da.isel({ddim: depth_index})

    tdim = _time_dim(da)
    if tdim is not None:
        da = da.mean(tdim)
    lon, lat, field = _finish_field(da)
    return lon, lat, field, {"has_depth": has_depth, "units": units}


def _bbox_subset(lon, lat, field, lon_min, lon_max, lat_min, lat_max):
    lon_sel = (lon >= lon_min) & (lon <= lon_max)
    lat_sel = (lat >= lat_min) & (lat <= lat_max)
    return field[np.ix_(lat_sel, lon_sel)]


def _percentile_limits(values, percentiles):
    with np.errstate(invalid="ignore"):
        return np.nanpercentile(values, percentiles)


def _diff_limit(values):
    with np.errstate(invalid="ignore"):
        dvmax = np.nanpercentile(np.abs(values), DIFF_PERCENTILE)
    return dvmax * DIFF_LIMIT_SCALE


def _value_formatter():
    """ScalarFormatter that only switches to scientific notation outside
    a normal decimal range -- works reasonably for both tiny (mol/m2/s)
    and everyday-sized variables without per-variable tuning."""
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 4))
    return fmt


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _draw_panel(ax, lon, lat, field, cmap, vmin, vmax, lon_m, lat_m, masks,
                 extent=None):
    """Draw one panel's data + land + coastlines + province outlines,
    then force an early render (via canvas.draw()) so that a known,
    unresolved cartopy/shapely bug -- "'GeometryCollection' object is
    not subscriptable" while reprojecting a filled path (the data mesh
    OR the land feature) against certain projections; see
    https://github.com/SciTools/cartopy/issues/2176 -- surfaces here,
    per panel, rather than silently inside the final fig.savefig()
    call. Raises onward on failure; the caller (make_figure) is
    responsible for catching it and leaving that one panel blank rather
    than losing the whole figure."""
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    mesh = ax.pcolormesh(
        lon, lat, field, transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
    ax.coastlines(resolution="50m" if extent is not None else "110m",
                  linewidth=0.5, color="0.3", zorder=3)
    for name in MASK_REGIONS:
        contour_with_halo(
            ax, lon_m, lat_m + MASK_LAT_OFFSET[name], masks[name],
            color=MASK_COLORS[name], linewidth=MASK_LINEWIDTHS[name],
            linestyle=MASK_LINESTYLES[name], zorder=MASK_ZORDERS[name],
        )
    ax.figure.canvas.draw()  # force deferred cartopy rendering now, not at savefig()
    return mesh


def make_figure(variable, d=0, filetype=DEFAULT_FILETYPE, cmap=None,
                units=None, vmin=None, vmax=None, diff_vmax=None,
                watermark=True):
    """Build the 2x2 (Atlantic/Global x BASE/diff) figure for one
    variable.

    variable -- netCDF variable name (e.g. "PPINT", "Cflx", "DIA").
    d        -- 0-based depth-level index, used only if `variable` has a
                vertical dimension (see module docstring). Default 0
                (surface). Ignored (but still embedded in the output
                filename) for variables with no depth axis.
    filetype -- the "_rg" file family to read from, e.g. "diad_T",
                "ptrc_T", "ptrc_T_int", "grid_T", "LNL_FULL_T". Builds
                ORCA2_7d_clim_{clim_period}_{filetype}_rg.nc.
    cmap     -- BASE-row colormap. Accepts a cmocean colormap name (e.g.
                "thermal", "speed", "algae", "tempo"), a standard
                matplotlib colormap name (e.g. "viridis"), or an actual
                Colormap object. Defaults to cmocean's "thermal"; if a
                given name isn't found in either cmocean or matplotlib
                (or cmocean isn't installed), falls back to "viridis"
                with a printed warning -- see _resolve_cmap().
    units    -- axis/colorbar unit label. Defaults to the variable's own
                "units" netCDF attribute, if present.
    vmin, vmax -- optional fixed BASE-row colour limits. If either is
                given, it overrides the automatic BASE_PERCENTILES-based
                limit for BOTH the Atlantic and Global rows (they
                otherwise get independently-computed percentile limits
                -- see module docstring). Give both together for a
                normal fixed scale.
    diff_vmax -- optional fixed symmetric diff-row colour limit (applied
                as (-diff_vmax, +diff_vmax) to both the Atlantic and
                Global diff panels), overriding the automatic
                DIFF_PERCENTILE-based limit.
    watermark -- if True, prints the generating script's name and path
                in small grey text along the bottom.
    """
    set_presentation_style()
    cmap = _resolve_cmap(cmap, DEFAULT_CMAP_NAME, CMAP_FALLBACK)
    diff_cmap = _resolve_cmap(None, DIFF_CMAP_NAME, DIFF_CMAP_FALLBACK)

    lon_min, lon_max, lat_min, lat_max = _load_extent()
    run_labels, runs = _load_models()
    run_a, run_b = run_labels[0], run_labels[1]  # e.g. BASE, HOSE
    paths = _paths_for_filetype(runs, filetype)

    lon_m, lat_m, masks = load_region_masks(MASK_PATH)

    lon_a, lat_a, field_a, meta_a = load_field(paths[run_a], variable, d)
    _, _, field_b, meta_b = load_field(paths[run_b], variable, d)
    diff = field_b - field_a
    units = units if units is not None else (meta_a["units"] or meta_b["units"])

    # -- colour scales, computed separately per row (Atlantic bbox vs
    # whole/global field), same percentile-fallback style as figA/figE --
    # unless the caller passed explicit vmin/vmax/diff_vmax, in which
    # case those win outright for both rows.
    if vmin is not None or vmax is not None:
        atl_base_vals = _bbox_subset(lon_a, lat_a, field_a, lon_min, lon_max, lat_min, lat_max)
        auto_vmin_atl, auto_vmax_atl = _percentile_limits(atl_base_vals, BASE_PERCENTILES)
        auto_vmin_glb, auto_vmax_glb = _percentile_limits(field_a, BASE_PERCENTILES)
        vmin_atl = vmin if vmin is not None else auto_vmin_atl
        vmax_atl = vmax if vmax is not None else auto_vmax_atl
        vmin_glb = vmin if vmin is not None else auto_vmin_glb
        vmax_glb = vmax if vmax is not None else auto_vmax_glb
    else:
        atl_base_vals = _bbox_subset(lon_a, lat_a, field_a, lon_min, lon_max, lat_min, lat_max)
        vmin_atl, vmax_atl = _percentile_limits(atl_base_vals, BASE_PERCENTILES)
        vmin_glb, vmax_glb = _percentile_limits(field_a, BASE_PERCENTILES)

    if diff_vmax is not None:
        dvmax_atl = diff_vmax
        dvmax_glb = diff_vmax
    else:
        atl_diff_vals = _bbox_subset(lon_a, lat_a, diff, lon_min, lon_max, lat_min, lat_max)
        dvmax_atl = _diff_limit(atl_diff_vals)
        dvmax_glb = _diff_limit(diff)

    fig = plt.figure(figsize=(11, 10))
    outer_gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.32)

    axes = [[None, None], [None, None]]
    caxes = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            proj = ATLANTIC_PROJ if i == 0 else GLOBAL_PROJ
            inner = outer_gs[i, j].subgridspec(1, 2, width_ratios=[1, 0.05], wspace=0.06)
            axes[i][j] = fig.add_subplot(inner[0, 0], projection=proj)
            caxes[i][j] = fig.add_subplot(inner[0, 1])

    fig.subplots_adjust(top=0.90, bottom=0.07 if watermark else 0.04, left=0.05, right=0.96)

    panel_specs = [
        (0, 0, field_a, cmap, vmin_atl, vmax_atl, [lon_min, lon_max, lat_min, lat_max]),
        (0, 1, diff, diff_cmap, -dvmax_atl, dvmax_atl, [lon_min, lon_max, lat_min, lat_max]),
        (1, 0, field_a, cmap, vmin_glb, vmax_glb, None),
        (1, 1, diff, diff_cmap, -dvmax_glb, dvmax_glb, None),
    ]

    for i, j, data, cm, vmin, vmax, extent in panel_specs:
        try:
            mesh = _draw_panel(axes[i][j], lon_a, lat_a, data, cm, vmin, vmax,
                                lon_m, lat_m, masks, extent=extent)
        except Exception as exc:
            # known unresolved cartopy/shapely bug (see _draw_panel's
            # docstring) -- leave this one panel blank rather than
            # losing the whole figure
            print(f"    [warn] panel ({i},{j}) failed to render ({exc!r}) "
                  f"-- leaving it blank")
            axes[i][j].clear()
            axes[i][j].axis("off")
            caxes[i][j].axis("off")
            axes[i][j].text(0.5, 0.5, "render failed\n(see log)",
                             transform=axes[i][j].transAxes,
                             ha="center", va="center", fontsize=ps.font_small(), color="0.5")
            continue

        cbar = fig.colorbar(mesh, cax=caxes[i][j], orientation="vertical")
        cbar.outline.set_visible(False)
        cbar.formatter = _value_formatter()
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=ps.font_small())
        if units:
            cbar.set_label(units, fontsize=ps.font_small())

    # column headers (top row only): BASE / HOSE-BASE
    axes[0][0].set_title(run_a, fontweight="bold")
    axes[0][1].set_title(f"{run_b}\u2212{run_a}", fontweight="bold")

    # row labels (first column only): Atlantic / Global
    axes[0][0].text(-0.18, 0.5, "Atlantic", transform=axes[0][0].transAxes,
                     rotation=90, ha="center", va="center", fontweight="bold")
    axes[1][0].text(-0.18, 0.5, "Global", transform=axes[1][0].transAxes,
                     rotation=90, ha="center", va="center", fontweight="bold")

    depth_note = f"depth level {d}" if meta_a["has_depth"] else "no depth axis"
    fig.suptitle(f"{variable}  ({depth_note})", fontweight="bold", fontsize=plt.rcParams["font.size"] + 2)

    if watermark:
        script_path = f"{WATERMARK_BASE_PATH}/{SCRIPT_NAME}"
        fig.text(
            0.5, 0.008, script_path, ha="center", va="bottom",
            fontsize=ps.font_watermark(), color="0.55",
        )

    date_str = datetime.date.today().strftime("%Y%m%d")
    fname = f"{FILENAME_PREFIX}_{variable}_d{d}_{date_str}.png"
    return fig, fname


def save_figure(variable, d=0, filetype=DEFAULT_FILETYPE, cmap=None,
                 units=None, vmin=None, vmax=None, diff_vmax=None,
                 watermark=True, dpi=200):
    """Convenience wrapper: build the figure and write it straight to
    ./plts/, creating the directory if needed. Returns the full path
    written."""
    fig, fname = make_figure(variable, d=d, filetype=filetype, cmap=cmap,
                              units=units, vmin=vmin, vmax=vmax,
                              diff_vmax=diff_vmax, watermark=watermark)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    # example: python map_maker.py PPINT 0 diad_T
    var = sys.argv[1] if len(sys.argv) > 1 else "PPINT"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ftype = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_FILETYPE
    save_figure(var, d=depth, filetype=ftype)
