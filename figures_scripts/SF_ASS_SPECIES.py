"""
SF_ASS_SPECIES.py  —  Supporting Figure: Per-species Abundance Maps

Generates 4 figures:
    SF_ASS_PHYTO_ATL  -- Phytoplankton species, Atlantic
    SF_ASS_ZOO_ATL    -- Zooplankton + Bacteria species, Atlantic
    SF_ASS_PHYTO_GLB  -- Phytoplankton species, Global (Mollweide)
    SF_ASS_ZOO_GLB    -- Zooplankton + Bacteria species, Global (Mollweide)

Each figure: 2 rows x n_species cols (species along the top)
    top row:    CONTROL absolute abundance (one shared colorbar under the row)
    bottom row: HOSE-CONTROL anomaly (one shared colorbar under the row)

Output: /gpfs/home/mep22dku/scratch/AMOC-TOM/plts/SF_ASS_{PHYTO|ZOO}_{ATL|GLB}_YYYYMMDD.png
"""

import sys, os
sys.path.append('/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/scripts')
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import map_maker as mm
from datetime import datetime

WATERMARK  = "/gpfs/home/mep22dku/scratch/AMOC-TOM/figures_scripts/SF_ASS_SPECIES.py"
OUTPUT_DIR = "/gpfs/home/mep22dku/scratch/AMOC-TOM/plts"

BASE_PERCENTILES = (1, 99)
DIFF_PERCENTILE  = 99

FONT_BASE  = 13
TITLE_FONT = 14
TICK_FONT  = 10

FIG_WIDTH_ATL  = 13      # 6 species columns
FIG_WIDTH_GLB  = 16
ROW_HEIGHT_ATL = 3.2     # height of one map row; Atlantic maps are taller than wide
ROW_HEIGHT_GLB = 1.5     # Mollweide maps are about 2:1
CBAR_HEIGHT    = 1.0     # height of the strip under each row (bar + ticks + label)
GS_TOP, GS_BOTTOM = 0.91, 0.03

_viridis = plt.get_cmap("viridis")
_magma   = plt.get_cmap("magma")
PHYTO_SPECIES = ["DIA", "MIX", "COC", "PIC", "PHA", "FIX"]
ZOO_BAC       = ["BAC", "PRO", "PTE", "MES", "GEL", "CRU"]
PHYTO_COLORS  = {sp: _viridis(p) for sp, p in zip(
    PHYTO_SPECIES, [0.00, 0.26, 0.47, 0.66, 0.83, 0.97])}
ZOO_COLORS    = {sp: _magma(p) for sp, p in zip(
    ZOO_BAC, [0.05, 0.32, 0.50, 0.63, 0.76, 0.90])}

GLOBAL_PROJ = ccrs.Mollweide(central_longitude=-30)

mm.set_presentation_style()
plt.rcParams.update({"font.size": FONT_BASE})

lon_min, lon_max, lat_min, lat_max = mm._load_extent()
lon_m, lat_m, masks = mm.load_region_masks(mm.MASK_PATH)
run_labels, runs = mm._load_models()
run_a, run_b = run_labels[0], run_labels[1]
ptrc      = mm._paths_for_filetype(runs, "ptrc_T_int")
diff_cmap = mm._resolve_cmap(None,      mm.DIFF_CMAP_NAME,    mm.DIFF_CMAP_FALLBACK)
phy_cmap  = mm._resolve_cmap("speed",   mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK)
zoo_cmap  = mm._resolve_cmap("magma",   mm.DEFAULT_CMAP_NAME, mm.CMAP_FALLBACK)

def load(paths, run, var):
    lon, lat, field, _ = mm.load_field(paths[run], var, 0)
    return lon, lat, field

def mmol_factor(path, var):
    """Factor to mmol C m-2 from the units attribute (same check as MF_ZOO2).
    Stops if the units are not molC/m2 or mmolC/m2 (run fix_int_units.py first)."""
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        units = getattr(ds.variables[var], "units", "")
    u = (units or "").replace(" ", "").lower()
    if u in ("molc/m2", "molc/m^2", "mol/m2"):
        return 1e3
    if u in ("mmolc/m2", "mmol/m2"):
        return 1.0
    raise ValueError("%s in %s: units attribute is %r, expected 'molC/m2'. "
                     "Run fix_int_units.py on the clim folders first." % (var, path, units))

lon = lat = None
fields_a, fields_b = {}, {}
for sp in PHYTO_SPECIES + ZOO_BAC:
    f_a, f_b = mmol_factor(ptrc[run_a], sp), mmol_factor(ptrc[run_b], sp)
    assert f_a == f_b, "control and hose %s units differ" % sp
    lon, lat, fields_a[sp] = load(ptrc, run_a, sp)
    _,   _,   fields_b[sp] = load(ptrc, run_b, sp)
    fields_a[sp] = fields_a[sp] * f_a          # mmol C m-2
    fields_b[sp] = fields_b[sp] * f_b

os.makedirs(OUTPUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d")

def shared_limits(species, is_diff, is_atlantic):
    all_vals = []
    for sp in species:
        field = (fields_b[sp] - fields_a[sp]) if is_diff else fields_a[sp]
        if is_atlantic:
            vals = mm._bbox_subset(lon, lat, field, lon_min, lon_max, lat_min, lat_max)
        else:
            vals = field.ravel()
        all_vals.append(vals)
    combined = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in all_vals])
    if is_diff:
        dvmax = np.nanpercentile(np.abs(combined), DIFF_PERCENTILE) * mm.DIFF_LIMIT_SCALE
        return -dvmax, dvmax
    return np.nanpercentile(combined, BASE_PERCENTILES)

def draw_map_panel(ax, field, cmap, vmin, vmax, is_atlantic):
    if is_atlantic:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        res = "50m"
    else:
        ax.set_global(); res = "110m"
    mesh = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(),
                          cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
    ax.coastlines(resolution=res, linewidth=0.4, color="0.3", zorder=3)
    if is_atlantic:
        for name in mm.MASK_REGIONS:
            mm.contour_with_halo(ax, lon_m, lat_m + mm.MASK_LAT_OFFSET[name], masks[name],
                                  color=mm.MASK_COLORS[name], linewidth=mm.MASK_LINEWIDTHS[name],
                                  linestyle=mm.MASK_LINESTYLES[name], zorder=mm.MASK_ZORDERS[name])
    ax.figure.canvas.draw()
    return mesh

def make_figure(species, sp_colors, base_cmap, is_atlantic, title, code):
    n     = len(species)
    proj  = mm.ATLANTIC_PROJ if is_atlantic else GLOBAL_PROJ
    row_h = ROW_HEIGHT_ATL   if is_atlantic else ROW_HEIGHT_GLB
    fw    = FIG_WIDTH_ATL    if is_atlantic else FIG_WIDTH_GLB

    total_h = (2 * row_h + 2 * CBAR_HEIGHT) / (GS_TOP - GS_BOTTOM)

    fig = plt.figure(figsize=(fw, total_h))
    # rows: control maps, its colorbar strip, difference maps, its colorbar strip
    gs  = fig.add_gridspec(4, n,
                            height_ratios=[row_h, CBAR_HEIGHT, row_h, CBAR_HEIGHT],
                            hspace=0.0, wspace=0.04,
                            left=0.05, right=0.99, top=GS_TOP, bottom=GS_BOTTOM)

    vmin_base, vmax_base = shared_limits(species, False, is_atlantic)
    vmin_diff, vmax_diff = shared_limits(species, True,  is_atlantic)
    mesh_base = mesh_diff = None

    row_labels = ["CONTROL",
                  "HOSE \u2212 CONTROL" if is_atlantic else "HOSE \u2212\nCONTROL"]
    label_x = -0.12 if is_atlantic else -0.06

    for si, sp in enumerate(species):
        ax_b = fig.add_subplot(gs[0, si], projection=proj)
        mesh_base = draw_map_panel(ax_b, fields_a[sp], base_cmap,
                                    vmin_base, vmax_base, is_atlantic)
        ax_b.set_title(sp, fontweight="bold", fontsize=TITLE_FONT, color=sp_colors[sp])

        ax_d = fig.add_subplot(gs[2, si], projection=proj)
        mesh_diff = draw_map_panel(ax_d, fields_b[sp] - fields_a[sp],
                                    diff_cmap, vmin_diff, vmax_diff, is_atlantic)

        if si == 0:
            for ax, lab in ((ax_b, row_labels[0]), (ax_d, row_labels[1])):
                ax.text(label_x, 0.5, lab, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontweight="bold", fontsize=FONT_BASE)

    # one colorbar per row, centred under the row; thin bar at the top of its strip,
    # the rest of the strip holds the tick labels and the units label
    for gs_row, mesh, label in [
        (1, mesh_base, r"mmol C m$^{-2}$"),
        (3, mesh_diff, r"$\Delta$ mmol C m$^{-2}$"),
    ]:
        cax_gs = gs[gs_row, :].subgridspec(3, 3, width_ratios=[0.3, 0.4, 0.3],
                                           height_ratios=[0.25, 0.22, 0.53], hspace=0, wspace=0)
        cax    = fig.add_subplot(cax_gs[1, 1])
        cb     = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.formatter = mm._value_formatter()
        cb.update_ticks()
        cb.ax.tick_params(labelsize=TICK_FONT)
        cb.set_label(label, fontsize=TICK_FONT)

    fig.suptitle(title, fontsize=TITLE_FONT, fontweight="bold", y=1.0)
    fig.text(0.5, 0.001, WATERMARK, ha="center", va="bottom",
             fontsize=FONT_BASE - 5, color="0.55")

    out = f"{OUTPUT_DIR}/{code}_{stamp}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")
    plt.show()
    plt.close(fig)

make_figure(PHYTO_SPECIES, PHYTO_COLORS, phy_cmap, True,
            "Phytoplankton species \u2014 Atlantic", "SF_ASS_PHYTO_ATL")
make_figure(ZOO_BAC,       ZOO_COLORS,   zoo_cmap, True,
            "Zooplankton + Bacteria species \u2014 Atlantic", "SF_ASS_ZOO_ATL")
make_figure(PHYTO_SPECIES, PHYTO_COLORS, phy_cmap, False,
            "Phytoplankton species \u2014 Global", "SF_ASS_PHYTO_GLB")
make_figure(ZOO_BAC,       ZOO_COLORS,   zoo_cmap, False,
            "Zooplankton + Bacteria species \u2014 Global", "SF_ASS_ZOO_GLB")
