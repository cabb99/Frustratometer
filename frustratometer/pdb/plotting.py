"""Plotting utilities for distance matrices (dense and sparse).

Four public functions, all using imshow:

- ``plot_distance_map``           – dense L×L, continuous 0-to-vmax colourmap
- ``plot_interaction_map``        – dense L×L, categorical band colours
- ``plot_sparse_distance_map``    – sparse COO -> temporary L×L with NaN gaps
- ``plot_sparse_interaction_map`` – sparse COO -> temporary L×L with NaN gaps

All accept an optional ``PlotConfig`` (pydantic) for visual settings.
Any config field can also be overridden as a keyword argument.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pydantic import BaseModel
from typing import Tuple, Optional

__all__ = [
    'PlotConfig',
    'plot_distance_map',
    'plot_interaction_map',
    'plot_sparse_distance_map',
    'plot_sparse_interaction_map',
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PlotConfig(BaseModel):
    """Visual settings shared by all distance-map plots."""
    vmax: float = 10.0
    cmap: str = 'YlGnBu'
    figsize: Tuple[float, float] = (7, 6)
    title: str = ''
    nan_color: str = 'white'
    # Interaction-band colours
    close_color: str = '#1a1a2e'
    direct_color: str = '#e63946'
    mediated_color: str = '#f4a261'
    electrostatic_color: str = '#457b9d'
    interaction_bg: str = '#e8ecf1'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_config(config: Optional[PlotConfig], **kwargs) -> PlotConfig:
    """Merge *config* with keyword overrides."""
    if config is None:
        config = PlotConfig()
    if kwargs:
        config = config.model_copy(update=kwargs)
    return config


def _sparse_to_dense(sparse_dm) -> np.ndarray:
    """Convert a SparseMatrix (or duck-typed object with row/col/data/shape) to an L×L array with NaN gaps."""
    from .sparse import to_dense
    return to_dense(sparse_dm.row, sparse_dm.col, sparse_dm.data, sparse_dm.shape, fill=np.nan)


def _setup_ax(ax, L: int, title: str):
    """Shared axis decoration."""
    ax.set_xlabel('Residue index')
    ax.set_ylabel('Residue index')
    if title:
        ax.set_title(title)


def _build_interaction_rgba(matrix: np.ndarray, cfg: PlotConfig) -> np.ndarray:
    """Build an RGBA image from band classification.

    Shared by both dense and sparse interaction plots.
    """
    L = matrix.shape[0]
    rgba = np.zeros((L, L, 4))

    bands = [
        (0.0,  4.5, cfg.close_color),
        (4.5,  6.5, cfg.direct_color),
        (6.5,  9.5, cfg.mediated_color),
        (9.5, 40.0, cfg.electrostatic_color),
    ]
    for lo, hi, colour in bands:
        if lo == 0:
            mask = (matrix > 0) & (matrix < hi)
        else:
            mask = (matrix >= lo) & (matrix < hi)
        r, g, b = mcolors.to_rgb(colour)
        rgba[mask] = [r, g, b, 1.0]

    return rgba


def _add_interaction_legend(ax, cfg: PlotConfig):
    """Add legend for the three named interaction bands (not close)."""
    handles = [
        Patch(facecolor=cfg.direct_color,        label='Direct (4.5–6.5 Å)'),
        Patch(facecolor=cfg.mediated_color,       label='Mediated (6.5–9.5 Å)'),
        Patch(facecolor=cfg.electrostatic_color,  label='Electrostatic (9.5–40 Å)'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Public API – dense
# ---------------------------------------------------------------------------

def plot_distance_map(
    distance_matrix: np.ndarray,
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a dense distance matrix with a continuous colourmap (0 -> vmax)."""
    cfg = _resolve_config(config, **kwargs)

    if ax is None:
        _, ax = plt.subplots(figsize=cfg.figsize)

    cmap = plt.get_cmap(cfg.cmap).copy()
    cmap.set_bad(cfg.nan_color)

    im = ax.imshow(distance_matrix, cmap=cmap, vmin=0, vmax=cfg.vmax,
                   origin='lower', interpolation='nearest', aspect='equal')
    _setup_ax(ax, distance_matrix.shape[0], cfg.title)
    plt.colorbar(im, ax=ax, label='Distance (Å)', shrink=0.82)
    return ax


def plot_interaction_map(
    distance_matrix: np.ndarray,
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a dense distance matrix coloured by interaction band."""
    cfg = _resolve_config(config, **kwargs)

    if ax is None:
        _, ax = plt.subplots(figsize=cfg.figsize)

    L = distance_matrix.shape[0]
    rgba = _build_interaction_rgba(distance_matrix, cfg)

    ax.set_facecolor(cfg.interaction_bg)
    ax.imshow(rgba, origin='lower', interpolation='nearest', aspect='equal')
    _setup_ax(ax, L, cfg.title)
    _add_interaction_legend(ax, cfg)
    return ax


# ---------------------------------------------------------------------------
# Public API – sparse (COO -> imshow with NaN for missing)
# ---------------------------------------------------------------------------

def plot_sparse_distance_map(
    sparse_distance_matrix: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a sparse COO distance matrix as an imshow with NaN gaps."""
    cfg = _resolve_config(config, **kwargs)
    matrix = _sparse_to_dense(sparse_distance_matrix)
    return plot_distance_map(matrix, config=cfg, ax=ax)


def plot_sparse_interaction_map(
    sparse_distance_matrix: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot a sparse COO distance matrix coloured by interaction band."""
    cfg = _resolve_config(config, **kwargs)
    matrix = _sparse_to_dense(sparse_distance_matrix)
    return plot_interaction_map(matrix, config=cfg, ax=ax)
