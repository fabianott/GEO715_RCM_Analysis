"""
Empirical semivariogram of a raster around the kilns
=====================================================
 
What it does
------------
For every kiln, a square window (2 * half_width wide) is read from the raster.
Inside each window, every pixel is compared with the pixels k steps away in
four directions (east, north, north-east, north-west). Half the mean squared
difference of all pairs at a given distance is the semivariance gamma(h):
 
    gamma(h) = 0.5 * mean( (z(x) - z(x + h))^2 )
 
The pairs of all windows are pooled, so the result is one curve for all kilns.
 
How to read the curve
---------------------
nugget  value at very short distances (noise, detail smaller than a pixel)
sill    level where the curve flattens (= roughly the total variance)
range   distance where the curve reaches the sill: points closer than this
        are spatially autocorrelated, points further apart roughly independent
 
Usage in a notebook
-------------------
    from semivariogram import kiln_semivariogram, plot_semivariogram
 
    sv = kiln_semivariogram({"tif": "merged_dem.tif"})
    plot_semivariogram(sv)
 
To compare with control points later: {"gpkg": "controls.gpkg"}.
 
Bootstrap confidence bands
--------------------------
site_sums keeps the pair totals of every window separately (reads the raster
once). bootstrap_semivariogram then draws groups with replacement and computes
gamma again n_boot times; the 2.5 % and 97.5 % values give the 95 % band.
Each kiln is its own group; controls are grouped by kiln_id, so the controls of
one kiln are always drawn together. bootstrap_difference draws the SAME kilns
for both curves (each with its own controls) and gives a band for the ratio
kiln / control: where the band lies entirely below 1, the kiln curve is lower
beyond chance.
 
    settings = {"tif": "merged_dem.tif", "detrend": True}   # same as above
    kiln = site_sums({**settings, "gpkg": "kohlmeiler_2056.gpkg"},
                     group_column="ID_PAROLINI")
    ctrl = site_sums({**settings, "gpkg": "controls_2056.gpkg"},
                     group_column="kiln_id")
 
    ax = plot_band(bootstrap_semivariogram(ctrl), label="controls", color="#4A90C8")
    plot_band(bootstrap_semivariogram(kiln), ax=ax, label="kilns", color="#F68C3A")
 
    ratio = bootstrap_difference(kiln, ctrl)
    plot_band(ratio, color="#D9541A")
 
pip install geopandas rasterio numpy pandas matplotlib
"""
import numpy as np
import pandas as pd

from paths import data
 
DEFAULTS = {
    "gpkg": data("kohlmeiler_2056.gpkg"),  # points around which windows are read
    "tif": data("merged_dem.tif"),         # raster to analyse
    "band": 1,
    "half_width": 50,   # window = 100 m x 100 m around each point
    "max_lag": 40,      # largest distance in metres (keep it below half_width)
    "detrend": False,   # True: remove a fitted plane per window (use if the
                        # raster is absolute elevation, so the slope doesn't
                        # dominate the curve)
}
 
 
# =============================================================================
# CORE: semivariogram of one array
# =============================================================================
def _shifted_pairs(z, dr, dc):
    """Return two views of z whose pixels are (dr, dc) pixels apart (dr >= 0)."""
    rows, cols = z.shape
    if dc >= 0:
        a = z[0:rows - dr, 0:cols - dc]
        b = z[dr:rows, dc:cols]
    else:
        a = z[0:rows - dr, -dc:cols]
        b = z[dr:rows, 0:cols + dc]
    return a, b
 
 
def _remove_plane(z):
    """Subtract a least-squares plane a + b*row + c*col (NaN pixels ignored)."""
    rr, cc = np.indices(z.shape)
    ok = ~np.isnan(z)
    if ok.sum() < 3:
        return z
    A = np.column_stack([np.ones(ok.sum()), rr[ok], cc[ok]])
    coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    return z - (coef[0] + coef[1] * rr + coef[2] * cc)
 
 
def add_pairs(z, res, max_lag, sums):
    """
    Add all pixel pairs of array z up to max_lag metres apart to `sums`,
    a dict {distance rounded to res: [sum of squared differences, number of pairs]}.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]   # E, N, NE, NW (in pixel steps)
    for dr, dc in directions:
        step_len = res * np.hypot(dr, dc)            # 1 pixel, or sqrt(2) on diagonals
        for k in range(1, int(max_lag / step_len) + 1):
            a, b = _shifted_pairs(z, k * dr, k * dc)
            d2 = (a - b) ** 2
            d2 = d2[~np.isnan(d2)]                   # pairs with nodata are skipped
            if d2.size == 0:
                continue
            h = round(k * step_len / res) * res      # bin to the pixel size
            s = sums.setdefault(h, [0.0, 0])
            s[0] += d2.sum()
            s[1] += d2.size
    return sums
 
 
def finish(sums):
    """Turn the accumulated sums into a table: lag, gamma, n_pairs."""
    lags = sorted(sums)
    return pd.DataFrame({
        "lag": lags,
        "gamma": [0.5 * sums[h][0] / sums[h][1] for h in lags],
        "n_pairs": [sums[h][1] for h in lags],
    })
 
 
# =============================================================================
# MAIN: windows around the kilns
# =============================================================================
def kiln_semivariogram(params=None):
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds
 
    params = params or {}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    p = {**DEFAULTS, **params}
    hw = p["half_width"]
 
    points = gpd.read_file(p["gpkg"])
    sums = {}
    with rasterio.open(p["tif"]) as src:
        if points.crs != src.crs:
            points = points.to_crs(src.crs)
        res = src.res[0]
        for geom in points.geometry:
            x, y = geom.x, geom.y
            win = from_bounds(x - hw, y - hw, x + hw, y + hw, src.transform)
            z = (src.read(p["band"], window=win, boundless=True, masked=True)
                    .astype("float64").filled(np.nan))   # nodata / outside -> NaN
            if p["detrend"]:
                z = _remove_plane(z)
            add_pairs(z, res, p["max_lag"], sums)
 
    sv = finish(sums)
    print(f"{len(points)} windows of {2 * hw} x {2 * hw} m, pixel size {res} m")
    return sv
 
 
def plot_semivariogram(sv, ax=None, label=None, color=None, xlim=None, ylim=None, markersize=1.5, lw=0.7):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sv["lag"], sv["gamma"], marker="o", markersize=markersize, lw=lw, color=color, label=label)
    ax.set_xlabel("Distance between pixels (m)")
    ax.set_ylabel("Semivariance (m²)")
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.grid(color="0.92")
    if label:
        ax.legend(frameon=False)
    return ax
 
 
# =============================================================================
# BOOTSTRAP 1: per-site sums (reads the raster once)
# =============================================================================
def site_sums(params=None, group_column="ID_PAROLINI"):
    """
    Returns a dict with
        lags   : distances (m)
        S      : (n_sites, n_lags) sums of squared differences
        N      : (n_sites, n_lags) numbers of pairs
        groups : group label of every site (kiln id)
    """
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds
 
    params = params or {}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    p = {**DEFAULTS, **params}
    hw = p["half_width"]
 
    points = gpd.read_file(p["gpkg"])
    if group_column not in points.columns:
        raise KeyError(f"Column '{group_column}' not in {p['gpkg']}")
 
    per_site = []
    with rasterio.open(p["tif"]) as src:
        if points.crs != src.crs:
            points = points.to_crs(src.crs)
        res = src.res[0]
        for geom in points.geometry:
            x, y = geom.x, geom.y
            win = from_bounds(x - hw, y - hw, x + hw, y + hw, src.transform)
            z = (src.read(p["band"], window=win, boundless=True, masked=True)
                    .astype("float64").filled(np.nan))
            if p["detrend"]:
                z = _remove_plane(z)
            per_site.append(add_pairs(z, res, p["max_lag"], {}))   # own dict per site
 
    return _stack(per_site, points[group_column].to_numpy())
 
 
def _stack(per_site, groups):
    """Turn a list of {lag: [sum, n]} dicts into arrays with a common lag axis."""
    lags = sorted(set().union(*per_site))
    S = np.array([[d.get(h, [0.0, 0])[0] for h in lags] for d in per_site])
    N = np.array([[d.get(h, [0.0, 0])[1] for h in lags] for d in per_site])
    return {"lags": np.array(lags), "S": S, "N": N, "groups": np.asarray(groups)}
 
 
def _group_totals(site, group_ids):
    """Add up S and N per group, in the order of group_ids."""
    idx = {g: i for i, g in enumerate(group_ids)}
    rows = np.array([idx[g] for g in site["groups"]])
    GS = np.zeros((len(group_ids), len(site["lags"])))
    GN = np.zeros_like(GS)
    np.add.at(GS, rows, site["S"])
    np.add.at(GN, rows, site["N"])
    return GS, GN
 
 
# =============================================================================
# BOOTSTRAP 2: band around one curve
# =============================================================================
def bootstrap_semivariogram(site, n_boot=1000, ci=95, seed=0):
    """Resample groups with replacement; returns lag, gamma, low, high."""
    rng = np.random.default_rng(seed)
    group_ids = np.unique(site["groups"])
    GS, GN = _group_totals(site, group_ids)
    n = len(group_ids)
 
    # how often each group is drawn in each repeat -> (n_boot, n_groups)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])
    with np.errstate(invalid="ignore", divide="ignore"):
        gam = 0.5 * (W @ GS) / (W @ GN)
 
    a = (100 - ci) / 2
    return pd.DataFrame({
        "lag": site["lags"],
        "gamma": 0.5 * GS.sum(0) / GN.sum(0),          # curve from all data
        "low": np.nanpercentile(gam, a, axis=0),
        "high": np.nanpercentile(gam, 100 - a, axis=0),
    })
 
 
# =============================================================================
# BOOTSTRAP 3: band around the kiln / control ratio (paired draws)
# =============================================================================
def bootstrap_difference(kiln, ctrl, n_boot=1000, ci=95, seed=0):
    """
    Draws kilns with replacement and uses each drawn kiln together with its
    own controls. Returns lag, ratio (kiln gamma / control gamma), low, high.
    Kilns without controls are left out, so both curves use the same kilns.
    """
    if not np.array_equal(kiln["lags"], ctrl["lags"]):
        raise ValueError("Kiln and control runs used different settings (lags differ).")
    rng = np.random.default_rng(seed)
    group_ids = np.intersect1d(np.unique(kiln["groups"]), np.unique(ctrl["groups"]))
    k_sub = {**kiln, **_subset(kiln, group_ids)}
    c_sub = {**ctrl, **_subset(ctrl, group_ids)}
    KS, KN = _group_totals(k_sub, group_ids)
    CS, CN = _group_totals(c_sub, group_ids)
    n = len(group_ids)
 
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = ((W @ KS) / (W @ KN)) / ((W @ CS) / (W @ CN))
 
    a = (100 - ci) / 2
    out = pd.DataFrame({
        "lag": kiln["lags"],
        "ratio": (KS.sum(0) / KN.sum(0)) / (CS.sum(0) / CN.sum(0)),
        "low": np.nanpercentile(ratio, a, axis=0),
        "high": np.nanpercentile(ratio, 100 - a, axis=0),
    })
    print(f"{n} kilns with controls used")
    return out
 
 
def _subset(site, group_ids):
    keep = np.isin(site["groups"], group_ids)
    return {"S": site["S"][keep], "N": site["N"][keep], "groups": site["groups"][keep]}
 
 
# =============================================================================
# BOOTSTRAP PLOT
# =============================================================================
def plot_band(bs, ax=None, label=None, color=None, markersize=1.5, lw=0.7):
    """Curve with its shaded confidence band (works for both outputs)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    y = "ratio" if "ratio" in bs else "gamma"
    ax.fill_between(bs["lag"], bs["low"], bs["high"], color=color, alpha=0.25, lw=0)
    ax.plot(bs["lag"], bs[y], marker="o", markersize=markersize, lw=lw, color=color, label=label)
    ax.set_xlabel("Distance between pixels (m)")
    if y == "ratio":
        ax.axhline(1, color="0.5", lw=lw, ls="--")
        ax.set_ylabel("Semivariance ratio RCM / control")
    else:
        ax.set_ylabel("Semivariance (m²)")
        ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.grid(color="0.92")
    if label:
        ax.legend(frameon=False)
    return ax
