"""Helper functions for saving figures."""

from pathlib import Path

import matplotlib.pyplot as plt

from paths import FIGURES

# Default folder for save_fig. Each notebook sets its own subfolder:
#     import helpers; helpers.FIG_DIR = FIGURES / "02_semivariogram"
FIG_DIR = FIGURES


def save_fig(name, fig=None, fig_dir=None, dpi=300, formats=("png", "pdf"), **kwargs):
    """Save a figure as PNG and PDF into the figures folder.

    Parameters
    ----------
    name : str
        File name without extension, e.g. "ring_analysis".
    fig : matplotlib.figure.Figure, optional
        Figure to save. If None, the current figure (plt.gcf()) is used.
    fig_dir : str or Path, optional
        Output folder. Defaults to helpers.FIG_DIR.
    dpi : int, optional
        Resolution for raster formats (PNG). Default 300.
    formats : tuple of str, optional
        File formats to save. Default ("png", "pdf").
    **kwargs
        Passed on to fig.savefig().

    Returns
    -------
    list of Path
        Paths of the saved files.
    """
    if fig is None:
        fig = plt.gcf()

    fig_dir = Path(fig_dir if fig_dir is not None else FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)

    kwargs.setdefault("bbox_inches", "tight")

    saved = []
    for ext in formats:
        path = fig_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=dpi, **kwargs)
        saved.append(path)

    return saved
