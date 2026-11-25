"""
Ring analysis of any GeoTIFF around charcoal kilns
==================================================

What it does
------------
For every kiln, rays are sent out in N_RAYS directions. Every STEP metres along
each ray, the raster value is read. Each sample is assigned to one ring by its
distance from the kiln, and a mean and median are calculated per ring.

Switches
--------
WEIGHTED = True   each sample counts in proportion to the area it represents
                  (= its distance from the kiln). Recommended for area means.
WEIGHTED = False  every sample counts equally (simple average of all samples).
INCLUDE_CENTRE    whether the value at the kiln itself is used (first ring only).

Outputs (named after the raster, so different rasters don't overwrite each other)
-------
<raster>_rings.csv       one row per kiln: mean, median, n, valid share per ring
<raster>_summary.csv     overview across all kilns per ring (count, median, mean, std, ...)
<raster>_samples.gpkg    every single sample as a point (open in QGIS to check)
<raster>_samples.csv     the same samples as a table

Usage
-----
    python kiln_ring_analysis.py                       # uses TIF below
    python kiln_ring_analysis.py my_raster.tif         # any other GeoTIFF
    python kiln_ring_analysis.py my_raster.tif 3       # ... and band 3

pip install geopandas rasterio numpy pandas
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

# =============================================================================
# SETTINGS
# =============================================================================
GPKG = "kohlmeiler_2056.gpkg"     # kiln points
ID_COLUMN = "ID_PAROLINI"           # column that identifies each kiln
TIF = "merged_dem.tif"              # raster to analyse (or give it on the command line)
BAND = 1                            # band number (or give it on the command line)

# Rings (inner, outer) in metres, from inside to outside.
# Rule: inner < distance <= outer. The centre (distance 0) only counts in the
# first ring, and only if INCLUDE_CENTRE = True.
RINGS = [(0, 6), (6, 12), (12, 18), (18, 24)]

N_RAYS = 16              # number of directions, evenly spread over 360°
STEP = 1               # spacing of samples along each ray, in metres
WEIGHTED = False         # True: weight by distance (area-correct); False: all samples equal
INCLUDE_CENTRE = False   # True: the value at the kiln itself is part of the first ring
MIN_VALID = 0.9          # ring is left empty if less than this share of samples has data
SUMMARY_OF = "mean"      # which per-kiln ring value the overview across kilns uses: "mean" or "median"

# Command line overrides: python kiln_ring_analysis.py <tif> [band]
if len(sys.argv) > 1:
    TIF = sys.argv[1]
if len(sys.argv) > 2:
    BAND = int(sys.argv[2])

OUT_PREFIX = Path(TIF).stem   # e.g. "metriche_z_raster_1"


# =============================================================================
# STEP 1: Check the ring definition
# =============================================================================
for (in_a, out_a), (in_b, out_b) in zip(RINGS, RINGS[1:]):
    if in_b < out_a:
        raise ValueError(f"Rings overlap: ({in_a}, {out_a}) and ({in_b}, {out_b})")
R_MAX = RINGS[-1][1]


# =============================================================================
# STEP 2: Sampling design (the same for every kiln)
# =============================================================================
# One entry per sample: which ray, which direction, how far from the kiln.
# The centre is ray -1 at distance 0.
bearings = np.arange(N_RAYS) * 360.0 / N_RAYS            # 0° = north, 90° = east
dists_on_ray = np.arange(STEP, R_MAX + STEP / 2, STEP)    # STEP, 2*STEP, ..., R_MAX

design_ray, design_bearing, design_dist = [], [], []
if INCLUDE_CENTRE:
    design_ray.append(-1)
    design_bearing.append(np.nan)
    design_dist.append(0.0)
for r, b in enumerate(bearings):
    for d in dists_on_ray:
        design_ray.append(r)
        design_bearing.append(b)
        design_dist.append(d)

design_ray = np.array(design_ray)
design_bearing = np.array(design_bearing)
design_dist = np.array(design_dist)

# Ring of every sample (-1 = outside all rings, e.g. in a gap between rings)
design_ring = np.full(design_dist.size, -1)
for k, (r_in, r_out) in enumerate(RINGS):
    if k == 0:
        in_ring = (design_dist >= r_in) & (design_dist <= r_out)
    else:
        in_ring = (design_dist > r_in) & (design_dist <= r_out)
    design_ring[in_ring] = k

# Weight of every sample
if WEIGHTED:
    # A sample at distance d stands for an area of STEP * 2*pi*d / N_RAYS,
    # i.e. proportional to d. The centre stands for a small disc around the
    # kiln, which in the same units is N_RAYS * STEP / 8.
    design_weight = design_dist.copy()
    design_weight[design_dist == 0] = N_RAYS * STEP / 8
else:
    design_weight = np.ones(design_dist.size)


# =============================================================================
# STEP 3: Load kilns and raster, bring them into the same coordinate system
# =============================================================================
kilns = gpd.read_file(GPKG)

with rasterio.open(TIF) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds
    raster_res = src.res
    raster_nodata = src.nodata
    band_count = src.count
    band_desc = src.descriptions[BAND - 1] if 1 <= BAND <= src.count else None

if not 1 <= BAND <= band_count:
    raise SystemExit(f"BAND = {BAND}, but the raster only has {band_count} band(s).")
if raster_crs is None:
    raise SystemExit("The raster has no coordinate system.")
if raster_crs.is_geographic:
    raise SystemExit(f"The raster is in degrees ({raster_crs}). Distances in metres need a "
                     "projected CRS such as EPSG:2056. Reproject the raster first.")

if kilns.crs != raster_crs:
    print(f"Reprojecting kilns from {kilns.crs} to the raster's {raster_crs}")
    kilns = kilns.to_crs(raster_crs)

print("=" * 70)
print(f"Raster:        {TIF}")
print(f"Band:          {BAND} of {band_count} ({band_desc or 'no name'})")
print(f"CRS:           {raster_crs}")
print(f"Pixel size:    {raster_res[0]} x {raster_res[1]} m")
print(f"Nodata value:  {raster_nodata}")
print(f"Kilns:         {len(kilns)} from {GPKG}")
print(f"Rays:          {N_RAYS} (every {360 / N_RAYS:g}°), sample every {STEP} m up to {R_MAX} m")
print(f"Weighted:      {WEIGHTED}")
print(f"Centre used:   {INCLUDE_CENTRE}")
print("Samples per ring (identical for every kiln):")
for k, (r_in, r_out) in enumerate(RINGS):
    n = int((design_ring == k).sum())
    print(f"   {r_in:>5g}-{r_out:<5g} m: {n} samples")
if STEP < raster_res[0]:
    print(f"Note: STEP ({STEP} m) is smaller than the pixel size ({raster_res[0]} m), "
          "so neighbouring samples often read the same pixel.")
print("=" * 70)


# =============================================================================
# STEP 4: Read the raster value at every sample position of every kiln
# =============================================================================
sample_tables = []

with rasterio.open(TIF) as src:
    for _, kiln in kilns.iterrows():
        kx, ky = kiln.geometry.x, kiln.geometry.y

        # Positions of the samples: bearing 0° = north, so
        # east offset = d * sin(bearing), north offset = d * cos(bearing)
        rad = np.radians(np.nan_to_num(design_bearing))
        sx = kx + design_dist * np.sin(rad)
        sy = ky + design_dist * np.cos(rad)

        # Read the values
        values = np.array([v[0] for v in src.sample(zip(sx, sy), indexes=BAND)],
                          dtype="float64")

        # Nodata -> NaN
        if raster_nodata is not None:
            values[values == raster_nodata] = np.nan
        # Outside the raster -> NaN
        outside = ((sx < raster_bounds.left) | (sx > raster_bounds.right) |
                   (sy < raster_bounds.bottom) | (sy > raster_bounds.top))
        values[outside] = np.nan

        sample_tables.append(pd.DataFrame({
            "kiln_id": kiln[ID_COLUMN],
            "kiln_x": kx,
            "kiln_y": ky,
            "ray": design_ray,
            "bearing": design_bearing,
            "distance": design_dist,
            "ring": [f"{RINGS[k][0]}-{RINGS[k][1]}" if k >= 0 else "none" for k in design_ring],
            "ring_index": design_ring,
            "x": sx,
            "y": sy,
            "value": values,
            "valid": ~np.isnan(values),
            "weight": design_weight,
        }))

samples = pd.concat(sample_tables, ignore_index=True)


# =============================================================================
# STEP 5: Statistics per kiln and ring
# =============================================================================
def weighted_median(values, weights):
    """Sort the values, add up their weights, and return the first value at
    which the running sum reaches half of the total weight.
    With all weights = 1 this is the ordinary (lower) median."""
    order = np.argsort(values)
    v_sorted = values[order]
    cum = np.cumsum(weights[order])
    return v_sorted[np.searchsorted(cum, cum[-1] / 2)]


rows = []
for kiln_id, s_kiln in samples.groupby("kiln_id", sort=False):
    row = {"kiln_id": kiln_id,
           "x": s_kiln["kiln_x"].iloc[0],
           "y": s_kiln["kiln_y"].iloc[0]}

    for k, (r_in, r_out) in enumerate(RINGS):
        name = f"{r_in}_{r_out}"
        s_ring = s_kiln[s_kiln["ring_index"] == k]
        s_valid = s_ring[s_ring["valid"]]

        n = len(s_ring)
        valid_share = len(s_valid) / n if n else 0.0

        if n > 0 and valid_share >= MIN_VALID:
            v = s_valid["value"].to_numpy()
            w = s_valid["weight"].to_numpy()
            mean = np.sum(v * w) / np.sum(w)        # weighted mean (= plain mean if all w = 1)
            median = weighted_median(v, w)
        else:
            mean = np.nan
            median = np.nan

        row[f"mean_{name}"] = mean
        row[f"median_{name}"] = median
        row[f"n_{name}"] = n
        row[f"valid_{name}"] = valid_share

    rows.append(row)

result = pd.DataFrame(rows).set_index("kiln_id")


# =============================================================================
# STEP 6: Save and summarise
# =============================================================================
result_csv = f"{OUT_PREFIX}_rings.csv"
summary_csv = f"{OUT_PREFIX}_summary.csv"
samples_csv = f"{OUT_PREFIX}_samples.csv"
samples_gpkg = f"{OUT_PREFIX}_samples.gpkg"

mean_cols = [f"mean_{a}_{b}" for a, b in RINGS]
value_cols = [f"{SUMMARY_OF}_{a}_{b}" for a, b in RINGS]

# Overview across all kilns: each kiln contributes ONE value per ring
# (its ring mean, or its ring median if SUMMARY_OF = "median").
# Note: "median" and "50%" are the same number; both are kept for readability.
values = result[value_cols]
summary = pd.DataFrame({
    "count":  values.count(),
    "median": values.median(),
    "mean":   values.mean(),
    "std":    values.std(),
    "min":    values.min(),
    "25%":    values.quantile(0.25),
    "50%":    values.quantile(0.50),
    "75%":    values.quantile(0.75),
    "max":    values.max(),
}).T
summary.columns = [f"{a}-{b} m" for a, b in RINGS]

result.to_csv(result_csv)
summary.to_csv(summary_csv)
samples.to_csv(samples_csv, index=False)
gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples["x"], samples["y"]),
                 crs=raster_crs).to_file(samples_gpkg, layer="samples")

print(f"\nOverview across all kilns (each kiln = one ring {SUMMARY_OF}, "
      f"{'weighted' if WEIGHTED else 'unweighted'}):")
print(summary.round(2))

empty = result[mean_cols].isna()
if empty.any().any():
    print("\nRings left empty (too little valid data):")
    for col in mean_cols:
        ids = list(result.index[empty[col]])
        if ids:
            print(f"   {col}: kilns {ids}")

print(f"\nSaved: {result_csv}  (one row per kiln)")
print(f"       {summary_csv}  (overview across all kilns)")
print(f"       {samples_csv}  ({len(samples)} samples)")
print(f"       {samples_gpkg}  (same samples as points for QGIS)")
