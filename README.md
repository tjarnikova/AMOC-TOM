# AMOC-TOM

Plotting code for the manuscript on the response of the Atlantic plankton ecosystem to an AMOC weakening, using PlankTOM12 in NEMO v5.0 (ORCA2). Two runs are compared: a historical control (`TOM12_TJ_HA01`) and a 0.3 Sv freshwater hosing run (`TOM12_TJ_HH01`). All figures show hose minus control for the 2010 to 2019 climatology.

Scripts by Tereza Jarnikova; Claude was used to adapt and expand existing scripts.

## Contents

All code is in `figures_scripts/`.

| Script | Output |
|---|---|
| `food_web_diagram.py` | Fig 1: PlankTOM12 food web |
| `SF_GROWTH_TEMP.py` | Fig 1: growth rate vs temperature |
| `MF_AMOC.py` | Fig 2: AMOC streamfunction and study regions |
| `MF_PHYS.py` | Fig 3: SST, SSS, MLD (sea ice with `--with-ice`) |
| `MF_PRIM.py` | Fig 4: phytoplankton biomass, primary production, limitation |
| `MF_LIMNUT.py` | Fig 5: limiting nutrients |
| `MF_ZOO.py` | Fig 6: zooplankton |
| `MF_ASS.py` | Fig 7: assemblage (`--with-dom --nobac`) |
| `MF_FLUX.py` | Fig 8, 9: export, export efficiency, air-sea carbon flux |
| `ST_ECO.py` | Table S1: ecosystem metrics by region |
| `SF_LIMCOMP.py` | Fig S2: light vs nutrient limitation |
| `ST_ASS.py` | Table S2: assemblage by region |
| `SF_ASS_SPECIES.py` | Fig S3, S4: per-species maps |
| `MF_EXPD.py` | Fig S5: export decomposition |
| `SF_CFLX_SIGN.py` | Fig S6: air-sea flux sign |
| `run_models.py` | Runs all of the above |

Figure numbers follow the manuscript draft and may change.

## Running

Python 3.7 in the `swamp2` conda environment on the cluster.

```bash
cd figures_scripts
python MF_PRIM.py                        # one figure
python run_models.py                     # everything
python run_models.py --list              # what runs, with flags
python run_models.py --only MF_PRIM MF_ZOO
```

Figures and tables are written to `plts/` with a date stamp, and each figure has its script path printed as a watermark. `run_models.py` writes one log per script to `logs/`. Neither folder is tracked.

## Requirements and paths

- Data paths are absolute and point to cluster locations: climatologies under `/gpfs/data/greenocean/users/mep22dku/clims/`, region masks under `~/scratch/AMOC-eco/NEMO5/masks/`, and MOC files under `/gpfs/data/greenocean/software/runs/NEMO5/`. Run IDs and the climatology period are set in `models.yml`.
- Scripts import `map_maker.py`, `TabK_summary.py` and (for the food web) `plot_grazing_matrix.py`, and read `models.yml`, `map_extents.yml` and `PFT_growth_rates/`, all expected in `figures_scripts/`. `plot_style` is imported from `/gpfs/home/mep22dku/scratch/SOZONE/UTILS` and is not part of this repository.
- Region names: NHL (northern high latitudes), NA, SA, SO (Atlantic Southern Ocean). Run IDs keep their original `HA` prefix.
