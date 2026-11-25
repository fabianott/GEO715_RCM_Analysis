"""
Ring analysis of any GeoTIFF around charcoal kilns
==================================================
 
What it does
------------
For every kiln, rays are sent out in n_rays directions. Every `step` metres along
each ray, the raster value is read. Each sample is assigned to one ring by its
distance from the kiln, and a mean and median are calculated per ring.
 
Switches
--------
weighted = True   each sample counts in proportion to the area it represents
                  (= its distance from the kiln). Recommended for area means.
weighted = False  every sample counts equally (simple average of all samples).
include_centre    whether the value at the kiln itself is used (first ring only).
 
Outputs (named after the raster, so different rasters don't overwrite each other)
-------
<raster>_rings.csv       one row per kiln: mean, median, n, valid share per ring
<raster>_summary.csv     overview across all kilns per ring (count, median, mean, std, ...)
<raster>_samples.gpkg    every single sample as a point (open in QGIS to check)
<raster>_samples.csv     the same samples as a table
 
Usage in a notebook
-------------------
    from kiln_ring_analysis import kiln_ring_analysis
 
    params = {
        "gpkg": "kohlmeiler_2056.gpkg",
        "tif": "merged_dem.tif",
        "rings": [(0, 6), (6, 12), (12, 18), (18, 24)],
        "weighted": True,
    }
    out = kiln_ring_analysis(params)
    out["rings"]      # one row per kiln
    out["summary"]    # overview across kilns
    out["samples"]    # every sample
 
Any key left out falls back to DEFAULTS below.
 
Usage from the command line
---------------------------
    python kiln_ring_analysis.py                       # uses DEFAULTS
    python kiln_ring_analysis.py my_raster.tif         # any other GeoTIFF
    python kiln_ring_analysis.py my_raster.tif 3       # ... and band 3
 
pip install geopandas rasterio numpy pandas
"""
import sys
import re
from pathlib import Path
 
import numpy as np

from paths import RESULTS, data
import pandas as pd
import geopandas as gpd
import rasterio
 
# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================
DEFAULTS = {
    "gpkg": data("kohlmeiler_2056.gpkg"),   # kiln points
    "id_column": "ID_PAROLINI",       # column that identifies each kiln
    "tif": data("merged_dem.tif"),          # raster to analyse
    "band": 1,                        # band number
 
    # Rings (inner, outer) in metres, from inside to outside.
    # Rule: inner < distance <= outer. The centre (distance 0) only counts in the
    # first ring, and only if include_centre = True.
    "rings": [(0, 6), (6, 12), (12, 18), (18, 24)],
 
    "n_rays": 16,             # number of directions, evenly spread over 360°
    "step": 1,                # spacing of samples along each ray, in metres
    "weighted": False,        # True: weight by distance (area-correct); False: all samples equal
    "include_centre": False,  # True: the value at the kiln itself is part of the first ring
    "min_valid": 0.9,         # ring is left empty if less than this share of samples has data
    "summary_of": "mean",     # per-kiln ring value used in the overview: "mean" or "median"
 
    "out_dir": str(RESULTS),  # parent folder; each run gets its own subfolder in here
    "out_prefix": None,       # None -> name of the raster without extension
    "comment": None,          # optional text appended to the subfolder name
    "save": True,             # write the output files
    "verbose": True,          # print the overview
}
 
 
# =============================================================================
# HELPERS
# =============================================================================
def _check_rings(rings):
    """Make sure the rings don't overlap."""
    for (in_a, out_a), (in_b, out_b) in zip(rings, rings[1:]):
        if in_b < out_a:
            raise ValueError(f"Rings overlap: ({in_a}, {out_a}) and ({in_b}, {out_b})")

def _run_folder_name(tif, n_rays, step, rings, comment):
    """Folder name from the parameters, e.g. merged_dem_rays16_step1_ring6x4_mycomment."""
    widths = {b - a for a, b in rings}
    contiguous = all(r1[0] == r0[1] for r0, r1 in zip(rings, rings[1:]))
    if rings[0][0] == 0 and contiguous and len(widths) == 1:
        ring_part = f"ring{widths.pop():g}x{len(rings)}"            # equal rings: width x count
    else:
        ring_part = "rings" + "_".join(f"{a:g}-{b:g}" for a, b in rings)  # otherwise list them
    name = f"{Path(tif).stem}_rays{n_rays}_step{step:g}_{ring_part}"
    if comment:
        name += "_" + re.sub(r"[^\w-]+", "-", str(comment)).strip("-")  # spaces etc. -> "-"
    return name

def _sampling_design(rings, n_rays, step, include_centre, weighted):
    """
    One entry per sample (the same for every kiln): which ray, which direction,
    how far from the kiln, which ring, and which weight.
    The centre is ray -1 at distance 0.
    """
    r_max = rings[-1][1]
    bearings = np.arange(n_rays) * 360.0 / n_rays            # 0° = north, 90° = east
    dists_on_ray = np.arange(step, r_max + step / 2, step)   # step, 2*step, ..., r_max
 
    ray, bearing, dist = [], [], []
    if include_centre:
        ray.append(-1)
        bearing.append(np.nan)
        dist.append(0.0)
    for r, b in enumerate(bearings):
        for d in dists_on_ray:
            ray.append(r)
            bearing.append(b)
            dist.append(d)
 
    ray = np.array(ray)
    bearing = np.array(bearing)
    dist = np.array(dist)
 
    # Ring of every sample (-1 = outside all rings, e.g. in a gap between rings)
    ring = np.full(dist.size, -1)
    for k, (r_in, r_out) in enumerate(rings):
        if k == 0:
            in_ring = (dist >= r_in) & (dist <= r_out)
        else:
            in_ring = (dist > r_in) & (dist <= r_out)
        ring[in_ring] = k
 
    # Weight of every sample
    if weighted:
        # A sample at distance d stands for an area of step * 2*pi*d / n_rays,
        # i.e. proportional to d. The centre stands for a small disc around the
        # kiln, which in the same units is n_rays * step / 8.
        weight = dist.copy()
        weight[dist == 0] = n_rays * step / 8
    else:
        weight = np.ones(dist.size)
 
    return {"ray": ray, "bearing": bearing, "dist": dist, "ring": ring, "weight": weight}
 
 
def weighted_median(values, weights):
    """Sort the values, add up their weights, and return the first value at
    which the running sum reaches half of the total weight.
    With all weights = 1 this is the ordinary (lower) median."""
    order = np.argsort(values)
    v_sorted = values[order]
    cum = np.cumsum(weights[order])
    return v_sorted[np.searchsorted(cum, cum[-1] / 2)]
 
 
# =============================================================================
# MAIN FUNCTION
# =============================================================================
def kiln_ring_analysis(params=None):
    """
    Run the ring analysis for one raster.
 
    Parameters
    ----------
    params : dict, optional
        Any subset of the keys in DEFAULTS. Unknown keys raise an error.
 
    Returns
    -------
    dict with
        "rings"   : DataFrame, one row per kiln (mean, median, n, valid share per ring)
        "summary" : DataFrame, overview across all kilns per ring
        "samples" : DataFrame, every single sample
        "files"   : dict of the written file paths (empty if save = False)
        "params"  : the full parameter set that was used
    """
    params = params or {}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(sorted(unknown))}. "
                       f"Allowed: {', '.join(DEFAULTS)}")
    p = {**DEFAULTS, **params}
 
    gpkg, tif, band = p["gpkg"], p["tif"], p["band"]
    rings = [tuple(r) for r in p["rings"]]
    n_rays, step = p["n_rays"], p["step"]
    weighted, include_centre = p["weighted"], p["include_centre"]
    min_valid, summary_of = p["min_valid"], p["summary_of"]
    verbose = p["verbose"]
    stem = Path(gpkg).stem
    suffix = ("_controls" if stem.startswith("controls") else "") + \
             ("_reduced" if stem.endswith("_reduced") else "")
    out_prefix = (p["out_prefix"] or Path(tif).stem) + suffix
    if summary_of not in ("mean", "median"):
        raise ValueError(f"summary_of must be 'mean' or 'median', not {summary_of!r}")
 
    # --- Step 1: check the ring definition ----------------------------------
    _check_rings(rings)
    r_max = rings[-1][1]
 
    # --- Step 2: sampling design (the same for every kiln) ------------------
    design = _sampling_design(rings, n_rays, step, include_centre, weighted)
 
    # --- Step 3: load kilns and raster, same coordinate system --------------
    kilns = gpd.read_file(gpkg)
 
    with rasterio.open(tif) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        raster_res = src.res
        raster_nodata = src.nodata
        band_count = src.count
        band_desc = src.descriptions[band - 1] if 1 <= band <= src.count else None
 
    if not 1 <= band <= band_count:
        raise ValueError(f"band = {band}, but the raster only has {band_count} band(s).")
    if raster_crs is None:
        raise ValueError("The raster has no coordinate system.")
    if raster_crs.is_geographic:
        raise ValueError(f"The raster is in degrees ({raster_crs}). Distances in metres need a "
                         "projected CRS such as EPSG:2056. Reproject the raster first.")
 
    if kilns.crs != raster_crs:
        if verbose:
            print(f"Reprojecting kilns from {kilns.crs} to the raster's {raster_crs}")
        kilns = kilns.to_crs(raster_crs)
 
    if verbose:
        print("=" * 70)
        print(f"Raster:        {tif}")
        print(f"Band:          {band} of {band_count} ({band_desc or 'no name'})")
        print(f"CRS:           {raster_crs}")
        print(f"Pixel size:    {raster_res[0]} x {raster_res[1]} m")
        print(f"Nodata value:  {raster_nodata}")
        print(f"Kilns:         {len(kilns)} from {gpkg}")
        print(f"Rays:          {n_rays} (every {360 / n_rays:g}°), sample every {step} m up to {r_max} m")
        print(f"Weighted:      {weighted}")
        print(f"Centre used:   {include_centre}")
        print("Samples per ring (identical for every kiln):")
        for k, (r_in, r_out) in enumerate(rings):
            n = int((design["ring"] == k).sum())
            print(f"   {r_in:>5g}-{r_out:<5g} m: {n} samples")
        if step < raster_res[0]:
            print(f"Note: step ({step} m) is smaller than the pixel size ({raster_res[0]} m), "
                  "so neighbouring samples often read the same pixel.")
        print("=" * 70)
 
    # --- Step 4: read the raster at every sample position of every kiln -----
    ring_labels = [f"{rings[k][0]}-{rings[k][1]}" if k >= 0 else "none" for k in design["ring"]]
    rad = np.radians(np.nan_to_num(design["bearing"]))
    sample_tables = []
 
    with rasterio.open(tif) as src:
        for _, kiln in kilns.iterrows():
            kx, ky = kiln.geometry.x, kiln.geometry.y
 
            # Bearing 0° = north: east offset = d * sin, north offset = d * cos
            sx = kx + design["dist"] * np.sin(rad)
            sy = ky + design["dist"] * np.cos(rad)
 
            values = np.array([v[0] for v in src.sample(zip(sx, sy), indexes=band)],
                              dtype="float64")
 
            if raster_nodata is not None:                      # nodata -> NaN
                values[values == raster_nodata] = np.nan
            outside = ((sx < raster_bounds.left) | (sx > raster_bounds.right) |
                       (sy < raster_bounds.bottom) | (sy > raster_bounds.top))
            values[outside] = np.nan                           # outside raster -> NaN
 
            sample_tables.append(pd.DataFrame({
                "kiln_id": kiln[p["id_column"]],
                "kiln_x": kx,
                "kiln_y": ky,
                "ray": design["ray"],
                "bearing": design["bearing"],
                "distance": design["dist"],
                "ring": ring_labels,
                "ring_index": design["ring"],
                "x": sx,
                "y": sy,
                "value": values,
                "valid": ~np.isnan(values),
                "weight": design["weight"],
            }))
 
    samples = pd.concat(sample_tables, ignore_index=True)
 
    # --- Step 5: statistics per kiln and ring --------------------------------
    rows = []
    for kiln_id, s_kiln in samples.groupby("kiln_id", sort=False):
        row = {"kiln_id": kiln_id,
               "x": s_kiln["kiln_x"].iloc[0],
               "y": s_kiln["kiln_y"].iloc[0]}
 
        for k, (r_in, r_out) in enumerate(rings):
            name = f"{r_in}_{r_out}"
            s_ring = s_kiln[s_kiln["ring_index"] == k]
            s_valid = s_ring[s_ring["valid"]]
 
            n = len(s_ring)
            valid_share = len(s_valid) / n if n else 0.0
 
            if n > 0 and valid_share >= min_valid:
                v = s_valid["value"].to_numpy()
                w = s_valid["weight"].to_numpy()
                mean = np.sum(v * w) / np.sum(w)    # weighted mean (= plain mean if all w = 1)
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
 
    # --- Step 6: overview across all kilns -----------------------------------
    # Each kiln contributes ONE value per ring (its ring mean or median).
    # Note: "median" and "50%" are the same number; both are kept for readability.
    mean_cols = [f"mean_{a}_{b}" for a, b in rings]
    value_cols = [f"{summary_of}_{a}_{b}" for a, b in rings]
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
    summary.columns = [f"{a}-{b} m" for a, b in rings]
 
    # --- Step 7: save ----------------------------------------------------------
    files = {}
    if p["save"]:
        out_dir = Path(p["out_dir"]) / (_run_folder_name(tif, n_rays, step, rings, p["comment"]) + suffix)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "rings":        out_dir / f"{out_prefix}_rings.csv",
            "summary":      out_dir / f"{out_prefix}_summary.csv",
            "samples_csv":  out_dir / f"{out_prefix}_samples.csv",
            "samples_gpkg": out_dir / f"{out_prefix}_samples.gpkg",
        }
        result.to_csv(files["rings"])
        summary.to_csv(files["summary"])
        samples.to_csv(files["samples_csv"], index=False)
        gpd.GeoDataFrame(samples, geometry=gpd.points_from_xy(samples["x"], samples["y"]),
                         crs=raster_crs).to_file(files["samples_gpkg"], layer="samples")
 
    if verbose:
        print(f"\nOverview across all kilns (each kiln = one ring {summary_of}, "
              f"{'weighted' if weighted else 'unweighted'}):")
        print(summary.round(2))
 
        empty = result[mean_cols].isna()
        if empty.any().any():
            print("\nRings left empty (too little valid data):")
            for col in mean_cols:
                ids = list(result.index[empty[col]])
                if ids:
                    print(f"   {col}: kilns {ids}")
 
        if files:
            print(f"\nSaved: {files['rings']}  (one row per kiln)")
            print(f"       {files['summary']}  (overview across all kilns)")
            print(f"       {files['samples_csv']}  ({len(samples)} samples)")
            print(f"       {files['samples_gpkg']}  (same samples as points for QGIS)")
 
    return {"rings": result, "summary": summary, "samples": samples,
            "files": files, "params": p}
 
 
# =============================================================================
# COMMAND LINE
# =============================================================================
if __name__ == "__main__":
    cli_params = {}
    if len(sys.argv) > 1:
        cli_params["tif"] = sys.argv[1]
    if len(sys.argv) > 2:
        cli_params["band"] = int(sys.argv[2])
    kiln_ring_analysis(cli_params)