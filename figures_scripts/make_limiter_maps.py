"""
make_limiter_maps.py: biomass-weighted limiting-nutrient FRACTIONS and nutrient-limitation
values, per phytoplankton group and for the whole assemblage, control and hose, from the
2010-2019 climatologies. Native grid. Same weighting as the nutrients-temp-light notebook:

    weight(species, month, level, cell) = biomass (molC/L, ptrc_T) x cell volume (meshmask_v5 vol)

For nutrient k in {Fe, P, Si, N} and a set of (species, month, level) samples:
    F_k = sum(weight where the limiting nutrient (LN) is k) / sum(weight)     (sums to 1)
    LV  = sum(weight x limitation value) / sum(weight)                       (as the notebook's L_nut)
Annual = all 12 months pooled with their weights (the notebook's biomass-weighted annual mean).

Differences from the notebook weighting, on purpose:
  * a sample counts in numerator AND denominator only if LN, LV and weight are all valid. The notebook
    sums with skipna in the numerator but keeps those weights in the denominator, which pulls
    L_nut towards 0 wherever LV is NaN while biomass is present. This script reports how much
    weight that affects (Wmiss) and compares with the notebook's L_nut when --compare finds it.
Optional --unlimited-above X (e.g. 0.99): samples whose LV >= X are counted in a fifth category 'None'
(not nutrient limited) instead of being labelled by their argmin nutrient. Without it every sample gets
a nutrient label, including nearly unlimited ones (in the raw weekly data 15-30% of non-diatom samples
have LV >= 0.99). The share of weight above LV 0.90 / 0.95 / 0.99 is always reported.
Dominant limiter is NOT stored (an argmax cannot be regridded): it is derived from the fractions,
after regridding, by the plotting script; here it is used only for the area statistics.

Outputs
  <clims>/<folder>/<RUN>_limiter_maps.nc   (with --unlimited-above 0.95: <RUN>_limiter_maps_lv95.nc, so nothing is overwritten)   native grid, same layout as <RUN>_limitation_maps.nc
        F_<Nut>_top<N>_{annual,monthly}, LV_top<N>_{annual,monthly}, W_top<N>_annual, Wmiss_frac_top<N>_annual
        F_<Nut>_<SP>_top<N>_annual, LV_<SP>_top<N>_annual, W_<SP>_top<N>_annual      (SP = DIA ... FIX)
  <output_dir>/limiter_stats_top<N>_<date>.csv   (output_dir from figure_settings.yml)   long format: region, scope, metric, basis, ctrl, hose, delta
Then regrid the .nc exactly as you did for the limitation maps (-> <RUN>_limiter_maps_rg.nc).

Usage:   python make_limiter_maps.py            # top 14 levels
         python make_limiter_maps.py --nlev 20
Python 3.7; numpy, xarray, netCDF4. The mask reader is built in (it used to be imported from
SF_CARBCLOSE.py, which needs carb_port.py and is not part of this repository).
"""
import argparse
import csv
import datetime
import os
import sys
import warnings

import numpy as np
import xarray as xr

from netCDF4 import Dataset

import plot_style as ps

warnings.filterwarnings('ignore')

# ---- CONFIG (edit) -----------------------------------------------------------
CLIMS = os.environ.get('LIM_CLIMS', '/gpfs/data/greenocean/users/mep22dku/clims')
RUNS = [('HA01', 'TOM12_TJ_HA01'), ('HH01', 'TOM12_TJ_HH01')]     # control, hose
PERIOD = '2010_2019'
MESH = os.environ.get('LIM_MESH', '/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/masks/meshmask_v5.nc')
MASKFILE = '/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL.nc'
# region label -> candidate variable names in the mask file (first one found is used); 'HA' is the old name of NHL
REGION_VARS = [('NHL', ['NHL', 'HA']), ('NA', ['NA']), ('SA', ['SA']), ('SO', ['SO'])]
OUTDIR = ps.output_dir()      # figure_settings.yml
# ------------------------------------------------------------------------------
SPECIES = ['DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
NUT = [('Fe', 3), ('P', 4), ('Si', 5), ('N', 6)]
CATS = list(NUT)          # becomes NUT + [('None', 0)] when --unlimited-above is used
LV_SHARES = (0.90, 0.95, 0.99)


def read_masks(shape):
    """Boolean region masks on the native grid from MASKFILE (same reader as SF_CARBCLOSE.py).
    Returns {label: bool array} for NHL, NA, SA, SO and AB (their union, only if all four exist)."""
    masks = {}
    if not os.path.exists(MASKFILE):
        print('mask file not found, regional statistics skipped: %s' % MASKFILE)
        return masks
    nc = Dataset(MASKFILE)
    print('mask file: %s' % MASKFILE)
    print('  variables: %s' % ', '.join(sorted(nc.variables.keys())))
    for label, cands in REGION_VARS:
        name = next((c for c in cands if c in nc.variables), None)
        if name is None:
            print('  region %s: none of %s found, skipped' % (label, cands))
            continue
        m = np.ma.filled(nc.variables[name][:], 0).squeeze()
        if m.shape != shape:
            print('  region %s (%s): shape %s, expected %s, skipped' % (label, name, m.shape, shape))
            continue
        masks[label] = m > 0.5
        print('  region %s <- %s, %d cells' % (label, name, masks[label].sum()))
    nc.close()
    if all(r in masks for r, _ in REGION_VARS):
        masks['AB'] = np.any([masks[r] for r, _ in REGION_VARS], axis=0)   # NHL+NA+SA+SO
    else:
        print('  AB not built: needs all four regions')
    return masks


def top_levels(da, nlev):
    """(time, z, y, x) values for the first nlev levels, by position."""
    if da.ndim != 4 or not da.dims[0].startswith('time'):
        sys.exit('%s has dims %s, expected (time, depth, y, x)' % (da.name, da.dims))
    return da.isel({da.dims[1]: slice(0, nlev)}).values.astype('float64')


def load_run(folder, nlev):
    base = '%s/%s/ORCA2_7d_clim_%s_%%s.nc' % (CLIMS, folder, PERIOD)
    for ft in ('LoP_T', 'ptrc_T'):
        if not os.path.exists(base % ft):
            sys.exit('missing %s' % (base % ft))
    lop = xr.open_dataset(base % 'LoP_T')
    ptrc = xr.open_dataset(base % 'ptrc_T')
    d = {'ln': {}, 'lv': {}, 'bio': {}}
    for sp in SPECIES:
        d['ln'][sp] = top_levels(lop['LN_' + sp], nlev)
        d['lv'][sp] = top_levels(lop['LV_' + sp], nlev)
        d['bio'][sp] = top_levels(ptrc[sp], nlev)
        if not (d['ln'][sp].shape == d['lv'][sp].shape == d['bio'][sp].shape):
            sys.exit('%s shapes differ: LN %s LV %s biomass %s' % (
                sp, d['ln'][sp].shape, d['lv'][sp].shape, d['bio'][sp].shape))
    d['lat'] = lop['nav_lat'].values
    d['lon'] = lop['nav_lon'].values
    return d


def accumulate(d, vol, thr):
    """Per scope ('ALL' and each species): monthly sums over levels of weight, weight where LN==k,
    weight x LV, weight lost to missing LN/LV, and weight above LV thresholds."""
    acc = {}
    for sp in SPECIES:
        w = d['bio'][sp] * vol[None]
        pos = np.isfinite(w) & (w > 0)
        ok = pos & np.isfinite(d['ln'][sp]) & np.isfinite(d['lv'][sp])
        wv = np.where(ok, w, 0.0)
        lvf = np.where(ok, d['lv'][sp], 0.0)
        a = {'W': wv.sum(axis=1),
             'Wmiss': np.where(pos & ~ok, w, 0.0).sum(axis=1),
             'WLV': (wv * lvf).sum(axis=1)}
        limflag = (lvf < thr) if thr else np.ones(lvf.shape, bool)
        for name, code in NUT:
            a['W' + name] = (wv * ((np.round(d['ln'][sp]) == code) & limflag)).sum(axis=1)
        if thr:
            a['WNone'] = (wv * (lvf >= thr)).sum(axis=1)
        for t in LV_SHARES:
            a['Wge%.2f' % t] = (wv * (lvf >= t)).sum(axis=1)
        acc[sp] = a
    keys = list(acc['DIA'].keys())
    acc['ALL'] = {k: sum(acc[sp][k] for sp in SPECIES) for k in keys}
    return acc


def safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def scope_fields(a, nlev, tag):
    """Dataset-ready dict for one scope. tag '' (assemblage) or '_<SP>'."""
    out = {}
    suff = 'top%d' % nlev
    Wa = a['W'].sum(axis=0)
    for name, _ in CATS:
        if tag and name == 'Si' and tag != '_DIA':
            continue
        out['F_%s%s_%s_annual' % (name, tag, suff)] = (('y', 'x'), safe_div(a['W' + name].sum(axis=0), Wa))
        if not tag:
            out['F_%s_%s_monthly' % (name, suff)] = (('time', 'y', 'x'), safe_div(a['W' + name], a['W']))
    out['LV%s_%s_annual' % (tag, suff)] = (('y', 'x'), safe_div(a['WLV'].sum(axis=0), Wa))
    if not tag:
        out['LV_%s_monthly' % suff] = (('time', 'y', 'x'), safe_div(a['WLV'], a['W']))
        tot = Wa + a['Wmiss'].sum(axis=0)
        out['Wmiss_frac_%s_annual' % suff] = (('y', 'x'), safe_div(a['Wmiss'].sum(axis=0), tot))
    out['W%s_%s_annual' % (tag, suff)] = (('y', 'x'), np.where(Wa > 0, Wa, np.nan))
    return out


def wsum(a, m):
    return float(np.nansum(a[m]))


def region_stats(acc, area, masks, ocean):
    """Rows for one run: dict[(region, scope, metric, basis)] -> value."""
    rows = {}
    regs = [('ocean', ocean)] + list(masks.items())
    doms = {}
    for scope in ['ALL'] + SPECIES:
        a = acc[scope]
        Wa = a['W'].sum(axis=0)
        Fa = np.stack([safe_div(a['W' + n].sum(axis=0), Wa) for n, _ in CATS], axis=0)      # (ncat,y,x)
        valid = np.isfinite(Fa).all(axis=0) & (Wa > 0)
        dom = np.where(valid, np.argmax(np.nan_to_num(Fa), axis=0), -1)
        doms[scope] = dom
        for rn, rm in regs:
            m = rm & ocean
            Wr = float(Wa[m].sum())
            tot = Wr + float(a['Wmiss'].sum(axis=0)[m].sum())
            Wall = float(acc['ALL']['W'].sum(axis=0)[m].sum())
            if scope != 'ALL':
                rows[(rn, scope, 'weight_share', 'biomass')] = Wr / Wall if Wall > 0 else np.nan
            ar = float(area[m & valid].sum())
            for k, (name, _) in enumerate(CATS):
                if scope not in ('ALL', 'DIA') and name == 'Si':
                    continue
                rows[(rn, scope, 'frac_' + name, 'biomass')] = float(a['W' + name].sum(axis=0)[m].sum()) / Wr if Wr > 0 else np.nan
                rows[(rn, scope, 'frac_' + name, 'area')] = float(np.nansum((Fa[k] * area)[m & valid])) / ar if ar > 0 else np.nan
                rows[(rn, scope, 'domarea_' + name, 'area')] = float(area[m & (dom == k)].sum()) / ar if ar > 0 else np.nan
            for t in LV_SHARES:
                rows[(rn, scope, 'share_LV_ge_%.2f' % t, 'biomass')] = float(a['Wge%.2f' % t].sum(axis=0)[m].sum()) / Wr if Wr > 0 else np.nan
            rows[(rn, scope, 'LV', 'biomass')] = float(a['WLV'].sum(axis=0)[m].sum()) / Wr if Wr > 0 else np.nan
            rows[(rn, scope, 'Wmiss_share', 'biomass')] = float(a['Wmiss'].sum(axis=0)[m].sum()) / tot if tot > 0 else np.nan
    return rows, doms


def main():
    global CATS, MASKFILE
    ap = argparse.ArgumentParser()
    ap.add_argument('--nlev', type=int, default=14)
    ap.add_argument('--maskfile', default=MASKFILE)
    ap.add_argument('--no-compare', action='store_true')
    ap.add_argument('--unlimited-above', type=float, default=0.0,
                    help='LV threshold above which a sample counts as not nutrient limited (0 = off)')
    args = ap.parse_args()
    nlev = args.nlev
    thr = args.unlimited_above
    MASKFILE = args.maskfile
    CATS = list(NUT) + ([('None', 0)] if thr else [])
    tag = '' if not thr else '_lv%02d' % int(round(thr * 100))
    suff = 'top%d' % nlev

    mesh = xr.open_dataset(os.path.expanduser(MESH))
    vol = np.squeeze(mesh['vol'].values).astype('float64')
    if vol.ndim != 3:
        sys.exit('mesh vol has shape %s, expected (z, y, x)' % (vol.shape,))
    vol = vol[:nlev]
    ocean = vol[0] > 0
    if 'e1t' in mesh and 'e2t' in mesh:
        area = np.squeeze((mesh['e1t'] * mesh['e2t']).values).astype('float64')
    else:
        print('e1t/e2t not in the mesh: area statistics unweighted')
        area = np.ones(vol.shape[1:])

    results = {}
    for label, folder in RUNS:
        print('loading %s (%s)' % (label, folder))
        d = load_run(folder, nlev)
        if d['bio']['DIA'].shape[1:] != vol.shape:
            sys.exit('data grid %s vs mesh vol %s: trim or fix before running' % (
                d['bio']['DIA'].shape[1:], vol.shape))
        acc = accumulate(d, vol, thr)
        fields = scope_fields(acc['ALL'], nlev, '')
        for sp in SPECIES:
            fields.update(scope_fields(acc[sp], nlev, '_' + sp))
        ds = xr.Dataset(fields)
        ds = ds.assign_coords(nav_lat=(('y', 'x'), d['lat']), nav_lon=(('y', 'x'), d['lon']))
        ds.attrs['made_in'] = 'make_limiter_maps.py'
        ds.attrs['run'] = label
        ds.attrs['description'] = ('Biomass x volume weighted fractions of the assemblage (or one group) limited by '
                                   'Fe, P, Si, N (from LN in LoP_T) and weighted limitation value LV. '
                                   'Samples with missing LN, LV or weight are excluded from numerator and denominator.')
        out = '%s/%s/%s_limiter_maps%s.nc' % (CLIMS, folder, label, tag)
        ds.to_netcdf(out)
        print('  wrote %s' % out)
        results[label] = dict(acc=acc, ds=ds, folder=folder, d=d)

    shape2d = ocean.shape
    masks = read_masks(shape2d)
    labs = [r[0] for r in RUNS]
    stats = {}
    doms = {}
    for lab in labs:
        stats[lab], doms[lab] = region_stats(results[lab]['acc'], area, masks, ocean)

    os.makedirs(OUTDIR, exist_ok=True)
    stamp = datetime.date.today().strftime('%Y%m%d')
    csvp = '%s/limiter_stats_%s%s_%s.csv' % (OUTDIR, suff, tag, stamp)
    keys = sorted(stats[labs[0]].keys())
    with open(csvp, 'w') as fh:
        wr = csv.writer(fh)
        wr.writerow(['region', 'scope', 'metric', 'basis', labs[0], labs[1], 'delta'])
        for k in keys:
            a, b = stats[labs[0]][k], stats[labs[1]][k]
            wr.writerow(list(k) + ['%.5f' % a, '%.5f' % b, '%.5f' % (b - a)])
        # area share where the dominant limiter changed
        regs = [('ocean', ocean)] + list(masks.items())
        for scope in ['ALL'] + SPECIES:
            for rn, rm in regs:
                m = rm & ocean
                both = (doms[labs[0]][scope] >= 0) & (doms[labs[1]][scope] >= 0) & m
                ch = both & (doms[labs[0]][scope] != doms[labs[1]][scope])
                share = float(area[ch].sum() / area[both].sum()) if both.any() else np.nan
                wr.writerow([rn, scope, 'dominant_changed', 'area', '', '', '%.5f' % share])
    print('wrote %s' % csvp)

    print('\nAssemblage, biomass-weighted fraction limited (%s): control, hose, delta' % suff)
    print('%-7s' % 'region' + ''.join('%-24s' % n for n, _ in CATS) + 'LV (ctrl, hose)     Wmiss ctrl')
    for rn in ['ocean'] + list(masks.keys()):
        line = '%-7s' % rn
        for n, _ in CATS:
            a = stats[labs[0]][(rn, 'ALL', 'frac_' + n, 'biomass')]
            b = stats[labs[1]][(rn, 'ALL', 'frac_' + n, 'biomass')]
            line += '%5.3f %5.3f %+6.3f   ' % (a, b, b - a)
        line += '%5.3f %5.3f   %5.2f%%' % (stats[labs[0]][(rn, 'ALL', 'LV', 'biomass')],
                                           stats[labs[1]][(rn, 'ALL', 'LV', 'biomass')],
                                           100 * stats[labs[0]][(rn, 'ALL', 'Wmiss_share', 'biomass')])
        print(line)

    print('\nBiomass x volume weight share of each group in the assemblage (%s), control / hose, percent' % suff)
    print('%-7s' % 'region' + ''.join('%-14s' % sp for sp in SPECIES))
    for rn in ['ocean'] + list(masks.keys()):
        print('%-7s' % rn + ''.join('%5.1f / %-6.1f' % (100 * stats[labs[0]][(rn, sp, 'weight_share', 'biomass')],
                                                         100 * stats[labs[1]][(rn, sp, 'weight_share', 'biomass')]) for sp in SPECIES))

    print('\nShare of assemblage weight with LV above a threshold (nearly unlimited samples), control / hose')
    for rn in ['ocean'] + list(masks.keys()):
        line = '%-7s' % rn
        for t in LV_SHARES:
            line += 'LV>=%.2f: %5.1f%% / %5.1f%%   ' % (
                t, 100 * stats[labs[0]][(rn, 'ALL', 'share_LV_ge_%.2f' % t, 'biomass')],
                100 * stats[labs[1]][(rn, 'ALL', 'share_LV_ge_%.2f' % t, 'biomass')])
        print(line)

    if not args.no_compare:
        print('\nComparison with the notebook weighting (L_nut_%s_annual in <RUN>_limitation_maps.nc)' % suff)
        for lab, folder in RUNS:
            p = '%s/%s/%s_limitation_maps.nc' % (CLIMS, folder, lab)
            var = 'L_nut_%s_annual' % suff
            if not os.path.exists(p):
                print('  %s: %s not found, skipped' % (lab, p))
                continue
            nb = xr.open_dataset(p)
            if var not in nb:
                print('  %s: %s not in the file, skipped' % (lab, var))
                continue
            nbv = nb[var].values
            mine = results[lab]['ds']['LV_%s_annual' % suff].values
            if nbv.shape != mine.shape:
                print('  %s: shapes differ %s vs %s' % (lab, nbv.shape, mine.shape))
                continue
            m = np.isfinite(nbv) & np.isfinite(mine) & ocean
            d = nbv[m] - mine[m]
            print('  %s: cells %d, mean(notebook - this) %+.4f, rms %.4f, max |diff| %.4f, cells with |diff|>0.01: %.1f%%' % (
                lab, m.sum(), d.mean(), np.sqrt((d ** 2).mean()), np.abs(d).max(), 100 * (np.abs(d) > 0.01).mean()))
            for rn, rm in [('ocean', ocean)] + list(masks.items()):
                mm_ = m & rm
                if mm_.any():
                    print('      %-6s area-weighted mean notebook %.4f, this %.4f' % (
                        rn, np.average(nbv[mm_], weights=area[mm_]), np.average(mine[mm_], weights=area[mm_])))
    print('\ndone. Next: regrid the two *_limiter_maps.nc files like the limitation maps.')


if __name__ == '__main__':
    main()
