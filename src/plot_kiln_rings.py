"""
Plots for the output of kiln_ring_analysis.py
==============================================

Reads <raster>_rings.csv (one row per kiln) and draws three panels:

  A  Distribution across kilns   - one box per ring, every kiln as a dot
  B  Profile with distance        - one grey line per kiln, the median of all
                                    kilns in colour, shaded band = 25-75 %
  C  Difference to the first ring - per kiln: ring value minus the value of its
                                    innermost ring (paired comparison)

Usage in a notebook
-------------------
    from plot_kiln_rings import plot_kiln_rings

    params = {
        "rings_csv": "merged_dem_rings.csv",
        "value": "median",
        "y_label": "Canopy height (m)",
    }
    fig = plot_kiln_rings(params)

Any key left out falls back to DEFAULTS below.

Usage from the command line
---------------------------
    python plot_kiln_rings.py                         # uses DEFAULTS
    python plot_kiln_rings.py merged_dem_rings.csv    # any results file

pip install pandas numpy matplotlib
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================
DEFAULTS = {
    "rings_csv": "merged_dem_rings.csv",  # output of kiln_ring_analysis.py
    "value": "mean",                      # plot the ring "mean" or "median" of each kiln
    "y_label": "Canopy height (m)",       # unit of the raster values
    "color": "#23794A",                   # one colour for the highlighted data
    "out_png": None,                      # None -> <csv stem>_<value>_plot.png next to the csv
    "save": True,                         # write the PNG
    "dpi": 200,
    "figsize": (16, 5.2),
    "show": True,                         # call plt.show()
    "seed": 0,                            # fixed seed: dots don't jump between runs
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# =============================================================================
# HELPERS
# =============================================================================
def _read_rings(rings_csv, value):
    """Read the results file and find the rings from the column names."""
    df = pd.read_csv(rings_csv, index_col=0)

    # Columns look like "mean_0_5", "mean_5_10", ... -> rings (0, 5), (5, 10), ...
    rings = []
    for col in df.columns:
        m = re.fullmatch(rf"{value}_([\d.]+)_([\d.]+)", col)
        if m:
            rings.append((float(m.group(1)), float(m.group(2)), col))
    rings.sort()
    if not rings:
        raise ValueError(f"No columns named '{value}_<inner>_<outer>' in {rings_csv}")

    labels = [f"{a:g}-{b:g} m" for a, b, _ in rings]
    values = df[[c for _, _, c in rings]].copy()   # kilns x rings
    values.columns = labels
    return df, rings, labels, values


def _box_with_dots(ax, data, positions, color, rng):
    """Boxplot per column, every kiln as a slightly jittered dot on top."""
    cols = [data[c].dropna().to_numpy() for c in data.columns]
    ax.boxplot(cols, positions=positions, widths=0.5, showfliers=False,
               medianprops=dict(color=color, linewidth=2),
               boxprops=dict(color="0.35"), whiskerprops=dict(color="0.35"),
               capprops=dict(color="0.35"))
    for pos, v in zip(positions, cols):
        jitter = rng.uniform(-0.15, 0.15, v.size)
        ax.scatter(pos + jitter, v, s=14, color=color, alpha=0.45, zorder=3, linewidths=0)
    ax.set_xticks(positions, data.columns)


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def plot_kiln_rings(params=None):
    """
    Draw the three-panel figure for one results file.

    Parameters
    ----------
    params : dict, optional
        Any subset of the keys in DEFAULTS. Unknown keys raise an error.

    Returns
    -------
    matplotlib.figure.Figure
    """
    params = params or {}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown parameter(s): {', '.join(sorted(unknown))}. "
                       f"Allowed: {', '.join(DEFAULTS)}")
    p = {**DEFAULTS, **params}

    rings_csv = p["rings_csv"]
    value = p["value"]
    y_label = p["y_label"]
    color = p["color"]
    out_png = p["out_png"] or Path(rings_csv).with_name(
        Path(rings_csv).stem + f"_{value}_plot.png")

    # --- Step 1: read results ------------------------------------------------
    df, rings, labels, values = _read_rings(rings_csv, value)
    midpoints = np.array([(a + b) / 2 for a, b, _ in rings])

    print(f"{len(df)} RCM, rings: {', '.join(labels)}")
    print(f"RCMs with at least one empty ring: {values.isna().any(axis=1).sum()}")

    # --- Step 2: figure with three panels ------------------------------------
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=p["figsize"])
    rng = np.random.default_rng(p["seed"])

    # A: distribution across kilns per ring
    _box_with_dots(ax_a, values, np.arange(len(labels)), color, rng)
    ax_a.set_title(f"A  Ring {value} across RCMs", loc="left")
    ax_a.set_xlabel("Ring")
    ax_a.set_ylabel(y_label)

    # B: profile with distance
    for _, row in values.iterrows():
        ax_b.plot(midpoints, row.to_numpy(), color="0.6", linewidth=0.7, alpha=0.5)
    q25 = values.quantile(0.25).to_numpy()
    q50 = values.median().to_numpy()
    q75 = values.quantile(0.75).to_numpy()
    ax_b.fill_between(midpoints, q25, q75, color=color, alpha=0.18, linewidth=0,
                      label="25-75 % of RCMs")
    ax_b.plot(midpoints, q50, color=color, linewidth=2.5, marker="o", markersize=7, label="Median of RCMs")
    ax_b.plot([], [], color="0.6", linewidth=0.7, label="Single RCM")
    for _, b, _ in rings:                         # ring limits as light vertical lines
        ax_b.axvline(b, color="0.85", linewidth=0.8, zorder=0)
    ax_b.set_xlim(0, rings[-1][1])
    ax_b.set_title("B  Profile with distance from RCM", loc="left")
    ax_b.set_xlabel("Distance from RCM (m), points at ring centres")
    ax_b.set_ylabel(y_label)
    ax_b.legend(frameon=False, loc="best")

    # C: difference to the innermost ring (paired, per kiln)
    first = values.columns[0]
    diff = values.iloc[:, 1:].sub(values[first], axis=0)
    diff.columns = [f"{c}\nminus {first}" for c in values.columns[1:]]
    if diff.shape[1] > 0:
        _box_with_dots(ax_c, diff, np.arange(diff.shape[1]), color, rng)
    ax_c.axhline(0, color="0.2", linewidth=1)
    ax_c.set_title(f"C  Difference to the {first} ring", loc="left")
    ax_c.set_ylabel(f"Difference in {y_label[0].lower() + y_label[1:]}")
    ax_c.text(0.01, 0.99, "above 0: higher than at the RCM\nbelow 0: lower than at the RCM",
              transform=ax_c.transAxes, va="top", fontsize=8, color="0.35")

    # Common styling: light grid, no top/right frame
    for ax in (ax_a, ax_b, ax_c):
        ax.grid(axis="y", color="0.92")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{Path(rings_csv).stem}  -  {len(df)} RCM, ring {value}s", x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()

    if p["save"]:
        fig.savefig(out_png, dpi=p["dpi"])
        print(f"Saved {out_png}")
    if p["show"]:
        plt.show()

    return fig


# =============================================================================
# COMMAND LINE
# =============================================================================
if __name__ == "__main__":
    cli_params = {}
    if len(sys.argv) > 1:
        cli_params["rings_csv"] = sys.argv[1]
    plot_kiln_rings(cli_params)
