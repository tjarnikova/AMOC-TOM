"""
SF_CARBCLOSE.py: Gate 0 closure test for the offline carbonate port (carb_port.py).

Compares the port's surface pCO2 (from monthly T, S, DIC, alkalinity) with the
model's own monthly pCO2 output, for control and hose, and checks the closure of
the CHANGE (hose minus control), which is what the swap decomposition needs.
Reports bias, RMS and 99th percentile of |residual| globally and per region
(unweighted cell statistics, native grid). Writes a three-panel residual map.

Expect a small nonzero residual: monthly means of nonlinear inputs do not
reproduce the monthly mean of the model's pCO2 exactly.

Run from the figure scripts directory. Python 3.7; needs numpy, netCDF4, matplotlib.
CHECK the VARS block with inspect_nc.py before the first run: names are guesses.
"""
import argparse
import datetime
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import carb_port as cp

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---- CONFIG (edit) -----------------------------------------------------------
CLIM_ROOT = '/gpfs/data/greenocean/users/mep22dku/clims'
RUNS = {'ctrl': 'TOM12_TJ_HA01', 'hose': 'TOM12_TJ_HH01'}
FILE_PATTERN = 'ORCA2_7d_clim_2010_2019_{}.nc'      # {} = file type
# role: (file type, variable name). NAMES ARE GUESSES: check with inspect_nc.py
VARS = {
    'T':    ('grid_T', 'tos'),        # degC, surface
    'S':    ('grid_T', 'sos'),        # practical salinity, surface
    'dic':  ('ptrc_T', 'DIC'),        # molC/L (surface level taken if 4D)
    'alk':  ('ptrc_T', 'Alkalini'),   # eq/L   (surface level taken if 4D)
    'pco2': ('diad_T', 'pCO2'),       # model pCO2, uatm
}
MASKFILE = '/gpfs/home/mep22dku/scratch/AMOC-eco/NEMO5/masks/mask_atl_v5_5prov_NHL.nc'
# region label -> candidate variable names in the mask file (first one found is used).
# 'HA' is the old name of NHL. CHECK the names the script prints on the first run.
REGION_VARS = [('NHL', ['NHL', 'HA']), ('NA', ['NA']), ('SA', ['SA']), ('SO', ['SO'])]
OUTDIR = './plts'
# ------------------------------------------------------------------------------


def read_var(path, name):
    if not os.path.exists(path):
        sys.exit('missing file: %s' % path)
    nc = Dataset(path)
    if name not in nc.variables:
        print('variable %s not in %s. Available:' % (name, path))
        print(', '.join(sorted(nc.variables.keys())))
        nc.close()
        sys.exit(1)
    a = np.ma.filled(nc.variables[name][:], np.nan).astype('float64')
    nc.close()
    a[np.abs(a) > 1.0e19] = np.nan
    if a.ndim == 4:
        a = a[:, 0]          # surface level
    if a.ndim != 3:
        sys.exit('%s in %s has shape %s; expected (time, y, x)' % (name, path, a.shape))
    return a


def load_run(run, need_model_pco2=True):
    d = {}
    for role, (ftype, vname) in VARS.items():
        if role == 'pco2' and not need_model_pco2:
            continue
        path = os.path.join(CLIM_ROOT, run, FILE_PATTERN.format(ftype))
        d[role] = read_var(path, vname)
    return d


def check_units(d, label):
    """Fail loudly on the units traps listed in the project notes."""
    med = lambda x: np.nanmedian(x)
    ok = (1.0e-3 < med(d['dic']) < 4.0e-3 and 1.0e-3 < med(d['alk']) < 4.0e-3
          and -3. < np.nanmin(d['T']) and np.nanmax(d['T']) < 45.
          and 0. <= np.nanmin(d['S']) and np.nanmax(d['S']) < 45.)
    if not ok:
        sys.exit('%s: units look wrong. median DIC %.3e, alk %.3e (expect ~2e-3 mol/L), '
                 'T range %.1f..%.1f, S range %.1f..%.1f' %
                 (label, med(d['dic']), med(d['alk']), np.nanmin(d['T']),
                  np.nanmax(d['T']), np.nanmin(d['S']), np.nanmax(d['S'])))


def valid_inputs(d):
    return (np.isfinite(d['T']) & np.isfinite(d['S']) & np.isfinite(d['dic'])
            & np.isfinite(d['alk']) & (d['S'] > 0) & (d['dic'] > 0) & (d['alk'] > 0))


def port_pco2(d):
    ok = valid_inputs(d)
    # as the model does for dry points, use harmless fill values, then mask
    T = np.where(ok, d['T'], 20.)
    S = np.where(ok, d['S'], 35.)
    dic = np.where(ok, d['dic'], 2.1e-3)
    alk = np.where(ok, d['alk'], 2.3e-3)
    out = cp.pco2_surface(T, S, dic, alk)
    return np.where(ok, out, np.nan)


def read_masks(shape):
    masks = {}
    if not os.path.exists(MASKFILE):
        print('mask file not found, regional statistics skipped: %s' % MASKFILE)
        return masks
    nc = Dataset(MASKFILE)
    have = sorted(nc.variables.keys())
    print('mask file: %s' % MASKFILE)
    print('  variables: %s' % ', '.join(have))
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


def stats(res, mask2d):
    """res: (time, y, x). Unweighted cell statistics over valid cells and months."""
    m = np.broadcast_to(mask2d, res.shape) & np.isfinite(res)
    x = res[m]
    if x.size == 0:
        return (0, np.nan, np.nan, np.nan)
    return (x.size, x.mean(), np.sqrt((x ** 2).mean()), np.percentile(np.abs(x), 99))


def print_table(title, res, masks, allocean):
    print('\n%s  (uatm; bias, RMS, 99th pct of |res|)' % title)
    print('%-8s %9s %9s %9s %9s' % ('region', 'n', 'bias', 'rms', 'p99'))
    rows = [('ocean', allocean)] + [(k, v) for k, v in masks.items()]
    for name, m in rows:
        n, b, r, p = stats(res, m)
        print('%-8s %9d %9.3f %9.3f %9.3f' % (name, n, b, r, p))


def main():
    global CLIM_ROOT, MASKFILE
    ap = argparse.ArgumentParser()
    ap.add_argument('--clim-root', default=CLIM_ROOT)
    ap.add_argument('--maskfile', default=MASKFILE)
    args = ap.parse_args()
    CLIM_ROOT, MASKFILE = args.clim_root, args.maskfile

    data = {}
    for key, run in RUNS.items():
        print('loading %s (%s)' % (key, run))
        data[key] = load_run(run)
        check_units(data[key], key)
    shape2d = data['ctrl']['T'].shape[1:]
    masks = read_masks(shape2d)

    port = {k: port_pco2(data[k]) for k in data}
    model = {}
    for k in data:
        mp = data[k]['pco2'].copy()
        mp[~(mp > 0)] = np.nan          # land is 0 or NaN
        model[k] = mp

    res = {k: port[k] - model[k] for k in data}
    dres = (port['hose'] - port['ctrl']) - (model['hose'] - model['ctrl'])
    allocean = np.isfinite(model['ctrl'][0]) & np.isfinite(model['hose'][0])

    print_table('closure, control: port minus model', res['ctrl'], masks, allocean)
    print_table('closure, hose: port minus model', res['hose'], masks, allocean)
    print_table('closure of the CHANGE: (port hose-ctrl) minus (model hose-ctrl)',
                dres, masks, allocean)
    print('\nmodel mean dpCO2 per region, for scale:')
    dmodel = model['hose'] - model['ctrl']
    for name, m in [('ocean', allocean)] + list(masks.items()):
        n, b, r, p = stats(dmodel, m)
        print('%-8s mean %8.3f  rms %8.3f' % (name, b, r))

    # ---- figure ---------------------------------------------------------------
    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    panels = [('control: port minus model', np.nanmean(res['ctrl'], axis=0)),
              ('hose: port minus model', np.nanmean(res['hose'], axis=0)),
              ('change: port minus model', np.nanmean(dres, axis=0))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels = ['pCO$_2$ residual ($\\mu$atm)', 'pCO$_2$ residual ($\\mu$atm)',
              '$\\Delta$pCO$_2$ residual ($\\mu$atm)']
    for ax, (title, a), lab in zip(axes, panels, labels):
        lim = np.nanpercentile(np.abs(a), 99) if np.isfinite(a).any() else 1.
        lim = max(lim, 1e-6)
        im = ax.imshow(a, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(title + ' (annual mean)')
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.06)
        cb.set_label(lab)
    stamp = datetime.date.today().strftime('%Y%m%d')
    out = os.path.join(OUTDIR, 'SF_CARBCLOSE_%s.png' % stamp)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
