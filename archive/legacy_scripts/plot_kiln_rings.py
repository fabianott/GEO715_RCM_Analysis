"""
Plots for the output of kiln_ring_analysis.py
==============================================

Reads <raster>_rings.csv (one row per kiln) and draws three panels:

  A  Distribution across kilns   - one box per ring, every kiln as a dot
  B  Profile with distance        - one grey line per kiln, the median of all
                                    kilns in colour, shaded band = 25-75 %
  C  Difference to the first ring - per kiln: ring value minus the value of its
                                    innermost ring (paired comparison)

Usage
-----
    python plot_kiln_rings.py                                  # uses RINGS_CSV below
    python plot_kiln_rings.py merged_dem_samples.csv    # any results file

pip install pandas numpy matplotlib
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================
RINGS_CSV = "merged_dem_rings.csv"   # output of kiln_ring_analysis.py
VALUE = "mean"                                # plot the ring "mean" or "median" of each kiln
Y_LABEL = "Canopy height (m)"                 # unit of the raster values
COLOR = "#2f6db3"                             # one colour for the highlighted data
SHOW = True                                   # also open the figure in a window

if len(sys.argv) > 1:
    RINGS_CSV = sys.argv[1]

OUT_PNG = Path(RINGS_CSV).with_name(Path(RINGS_CSV).stem + f"_{VALUE}_plot.png")


# =============================================================================
# STEP 1: Read results and find the rings from the column names
# =============================================================================
df = pd.read_csv(RINGS_CSV, index_col=0)

# Columns look like "mean_0_5", "mean_5_10", ... -> rings (0, 5), (5, 10), ...
rings = []
for col in df.columns:
    m = re.fullmatch(rf"{VALUE}_([\d.]+)_([\d.]+)", col)
    if m:
        rings.append((float(m.group(1)), float(m.group(2)), col))
rings.sort()
if not rings:
    raise SystemExit(f"No columns named '{VALUE}_<inner>_<outer>' in {RINGS_CSV}")

labels = [f"{a:g}-{b:g} m" for a, b, _ in rings]
midpoints = np.array([(a + b) / 2 for a, b, _ in rings])
values = df[[c for _, _, c in rings]]          # kilns x rings
values.columns = labels

print(f"{len(df)} kilns, rings: {', '.join(labels)}")
print(f"Kilns with at least one empty ring: {values.isna().any(axis=1).sum()}")


# =============================================================================
# STEP 2: Figure with three panels
# =============================================================================
fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 5.2))
rng = np.random.default_rng(0)                 # fixed seed: dots don't jump between runs


def box_with_dots(ax, data, positions):
    """Boxplot per column, every kiln as a slightly jittered dot on top."""
    cols = [data[c].dropna().to_numpy() for c in data.columns]
    ax.boxplot(cols, positions=positions, widths=0.5, showfliers=False,
               medianprops=dict(color=COLOR, linewidth=2),
               boxprops=dict(color="0.35"), whiskerprops=dict(color="0.35"),
               capprops=dict(color="0.35"))
    for pos, v in zip(positions, cols):
        jitter = rng.uniform(-0.15, 0.15, v.size)
        ax.scatter(pos + jitter, v, s=14, color=COLOR, alpha=0.45,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(positions, data.columns)


# --- A: distribution across kilns per ring ----------------------------------
pos = np.arange(len(labels))
box_with_dots(ax_a, values, pos)
ax_a.set_title(f"A  Ring {VALUE} across kilns", loc="left", fontweight="bold")
ax_a.set_xlabel("Ring")
ax_a.set_ylabel(Y_LABEL)

# --- B: profile with distance -----------------------------------------------
for _, row in values.iterrows():
    ax_b.plot(midpoints, row.to_numpy(), color="0.6", linewidth=0.7, alpha=0.5)
q25 = values.quantile(0.25).to_numpy()
q50 = values.median().to_numpy()
q75 = values.quantile(0.75).to_numpy()
ax_b.fill_between(midpoints, q25, q75, color=COLOR, alpha=0.18, linewidth=0,
                  label="25-75 % of kilns")
ax_b.plot(midpoints, q50, color=COLOR, linewidth=2.5, marker="o", markersize=7,
          markeredgecolor="white", label="Median of kilns")
ax_b.plot([], [], color="0.6", linewidth=0.7, label="Single kiln")
# Show the ring limits as light vertical lines
for a, b, _ in rings:
    ax_b.axvline(b, color="0.85", linewidth=0.8, zorder=0)
ax_b.set_xlim(0, rings[-1][1])
ax_b.set_title("B  Profile with distance from kiln", loc="left", fontweight="bold")
ax_b.set_xlabel("Distance from kiln (m), points at ring centres")
ax_b.set_ylabel(Y_LABEL)
ax_b.legend(frameon=False, loc="best")

# --- C: difference to the innermost ring (paired, per kiln) -----------------
first = values.columns[0]
diff = values.iloc[:, 1:].sub(values[first], axis=0)
diff.columns = [f"{c}\nminus {first}" for c in values.columns[1:]]
if diff.shape[1] > 0:
    box_with_dots(ax_c, diff, np.arange(diff.shape[1]))
ax_c.axhline(0, color="0.2", linewidth=1)
ax_c.set_title(f"C  Difference to the {first} ring", loc="left", fontweight="bold")
ax_c.set_ylabel(f"Difference in {Y_LABEL[0].lower() + Y_LABEL[1:]}")
ax_c.text(0.01, 0.99, "above 0: higher than at the kiln\nbelow 0: lower than at the kiln",
          transform=ax_c.transAxes, va="top", fontsize=8, color="0.35")

# Common styling: light grid, no top/right frame
for ax in (ax_a, ax_b, ax_c):
    ax.grid(axis="y", color="0.92")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"{Path(RINGS_CSV).stem}  -  {len(df)} kilns, ring {VALUE}s", x=0.01,
             ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
print(f"Saved {OUT_PNG}")

if SHOW:
    plt.show()
