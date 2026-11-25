# Relict charcoal mounds and canopy height in the Swiss National Park

GEO 715 *Application of GIS in Protected Areas*, University of Zurich, fall semester 2026/27  
Moritz Birle, Paul Ghisletti, Fabian Ott, Lars Ruhstaller

Do relict charcoal mounds (RCMs, *Kohlemeiler*) still leave a measurable trace in today's canopy height?
We sample the 1 m normalised surface model (nDSM) along 8 rays around each mound, aggregate the samples in
six concentric 6 m rings (0–36 m) and compare every mound with up to three control points in similar
forest and terrain. A repetition with 10 m LiDAR metrics serves as a robustness check.

## Repository structure

```
data/
  raw/                  input data (mounds, park boundary, forest; rasters not in git, see below)
  processed/            derived data: reduced mound set (66), control points, mound checks
src/                    Python modules used by the notebooks
  paths.py              central paths (data/, results/, figures/)
  kiln_ring_analysis.py multi-ring ray sampling around points
  plot_kiln_rings.py    standard three-panel ring plot
  semivariogram.py      empirical semivariogram with bootstrap bands
  reduce_kilns.py       remove excluded mounds from the point layer
  helpers.py            save_fig(): saves PNG + PDF into figures/<notebook>/
notebooks/              the analysis, in the order of the report
  00_example_ring_analysis.ipynb   how to use the ring analysis functions
  01_control_points.ipynb          forest/park filter, mound checks, control point placement, maps
  02_semivariogram.ipynb           spatial autocorrelation of canopy height
  03_ring_analysis_36m.ipynb       36 m ring analysis, overlap of study areas, maps of every mound
  04_kilns_vs_controls.ipynb       ring statistics and paired tests, mounds vs. controls
  05_robustness_10m_lidar.ipynb    repetition with the 10 m LiDAR metrics
  archive/parameter_tests.ipynb    early tests of ray count and ring layout
results/                output of the ring analysis, one folder per run
  <raster>_rays<n>_step<m>_ring<width>x<count>[_reduced|_controls]/
  ring_analysis_36m/    delta table of the 36 m analysis
  archive/              early runs before per-run folders existed
figures/                all figures (PNG + PDF), one folder per notebook
  01_control_points/    r36_d150 = final settings; r36 and first_run = earlier settings
  parameter_tests/      exploratory plots that are not in the report
archive/legacy_scripts/ first command-line versions of the ring analysis
```

## Setup

```bash
pip install -r requirements.txt
```

The rasters are too large for GitHub. Copy them into `data/raw/` before running the notebooks:

| File | Content | Source |
|---|---|---|
| `merged_dem.tif` | 1 m normalised surface model (canopy height) | provided by the seminar / SNP |
| `metriche_z_raster.tiff` | 10 m LiDAR point cloud metrics (band 2 mean, 3 vertical SD, 26 P90) | Christian Rossi, SNP |
| `dtm.tif` | terrain model for slope, aspect and hillshade | swissALTI3D, swisstopo |

Then start Jupyter in the repository folder and run the notebooks in order. Paths are resolved
through `src/paths.py`, so the notebooks work from any folder.

## Final analysis settings

- 66 of 80 mounds (forest share in the 36 m ring area ≥ 90 % inside the park); `data/processed/kohlmeiler_2056_reduced.gpkg`
- control points: up to 3 per mound, 72–150 m away, elevation ±30 m, slope ±10°, aspect ±45°; `data/processed/controls_2056_36_150.gpkg`
- 8 rays, samples every 1 m, rings 0–6, 6–12, …, 30–36 m, unweighted

## History

This repository merges the branches `main`, `analysis_fabian` and `paul` of the earlier working
repository (`oneObjectMVP` was empty). Where files differed, the newest version was used:
the Python modules from `paul`, the 36 m visualisation from `analysis_fabian` (24.11.2026).
The former `experimentation.ipynb` was split into notebooks 04, 05 and `archive/parameter_tests.ipynb`.
No figures were dropped; only byte-identical duplicates, `__pycache__` and editor settings were removed.
