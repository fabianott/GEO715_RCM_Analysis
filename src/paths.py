"""Central paths of the project.

All notebooks and modules use these paths, so the code works no matter
from which folder a notebook is started.

    data/raw/        input data (kilns, park boundary, forest; rasters are not in git)
    data/processed/  derived data (reduced kilns, control points, checks)
    results/         output of the ring analysis, one subfolder per run
    figures/         figures, one subfolder per notebook
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def data(name):
    """Path of a data file by its name: looks in data/processed, then data/raw.

    Files that do not exist yet (e.g. rasters you still have to copy in)
    resolve to data/raw/<name>.
    """
    for folder in (DATA_PROCESSED, DATA_RAW):
        if (folder / name).exists():
            return str(folder / name)
    return str(DATA_RAW / name)
