"""Shared, layout-safe plotting helpers for SIGMA analyses."""

from pathlib import Path
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse


FIGURE_SIZES = {
    "single": (3.35, 2.7),
    "single_square": (3.35, 3.35),
    "double": (7.0, 4.2),
    "double_square": (7.0, 7.0),
    "full_page": (7.0, 9.0),
    "slides_wide": (10.0, 4.8),
}


def set_publication_style(font_family="Arial", base_font_size=9):
    """Apply a consistent vector-friendly manuscript style."""
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.titlesize": base_font_size + 2,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": None,
    })


def figure_size(name="double"):
    """Return a standard manuscript figure size."""
    try:
        return FIGURE_SIZES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown figure size {name!r}") from exc


def add_panel_labels(axes, labels=None, x=-0.12, y=1.06, fontsize=11):
    """Add consistent panel labels in axes coordinates."""
    axes = list(axes)
    labels = list(labels or string.ascii_uppercase[:len(axes)])
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=fontsize, fontweight="bold", clip_on=False)


def legend_outside(ax, *, loc="upper left", anchor=(1.02, 1.0), **kwargs):
    """Place a legend outside the data axes."""
    return ax.legend(loc=loc, bbox_to_anchor=anchor, borderaxespad=0,
                     frameon=False, **kwargs)


def shared_colorbar(fig, mappable, axes, *, label=None, location="right", **kwargs):
    """Create one colorbar for a collection of axes."""
    colorbar = fig.colorbar(mappable, ax=list(axes), location=location, **kwargs)
    if label:
        colorbar.set_label(label)
    return colorbar


def style_spatial_axis(ax):
    """Remove decorations that are not meaningful for tissue coordinates."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, path, *, formats=("pdf", "svg"), dpi=300,
                transparent=False, close=False):
    """Save a figure in consistent vector and optional raster formats.

    ``path`` is a stem or a filename.  Layout is expected to be solved by
    constrained layout/GridSpec rather than repaired with ``bbox_inches``.
    """
    path = Path(path)
    stem = path.with_suffix("") if path.suffix else path
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        output = stem.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(output, dpi=dpi, transparent=transparent, facecolor=fig.get_facecolor())
        written.append(output)
    if close:
        plt.close(fig)
    return written


def _spatial_values(adata, key, spatial_key):
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{spatial_key!r}] is required")
    if key not in adata.obs:
        raise KeyError(f"adata.obs[{key!r}] is required")
    return np.asarray(adata.obsm[spatial_key]), adata.obs[key].to_numpy()


def plot_region_probability(adata, *, ax=None, spatial_key="spatial", size=8,
                            cmap="coolwarm", colorbar=True):
    """Plot the inferred continuous SIGMA region probability."""
    xy, values = _spatial_values(adata, "sigma_region_probability", spatial_key)
    ax = ax or plt.subplots(figsize=figure_size("single_square"), constrained_layout=True)[1]
    points = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=size, cmap=cmap,
                        vmin=0, vmax=1, linewidths=0, rasterized=True)
    style_spatial_axis(ax); ax.set_title("SIGMA region probability")
    if colorbar:
        ax.figure.colorbar(points, ax=ax, shrink=.78, label="Region probability")
    return ax


def plot_boundary(adata, *, ax=None, spatial_key="spatial", size=8,
                  boundary_size=10, background="sigma_region_probability"):
    """Plot SIGMA boundary spots over a continuous result field."""
    xy, values = _spatial_values(adata, background, spatial_key)
    if "sigma_boundary" not in adata.obs:
        raise KeyError("adata.obs['sigma_boundary'] is required")
    boundary = adata.obs["sigma_boundary"].to_numpy(bool)
    ax = ax or plt.subplots(figsize=figure_size("single_square"), constrained_layout=True)[1]
    ax.scatter(xy[:, 0], xy[:, 1], c=values, s=size, cmap="coolwarm",
               linewidths=0, rasterized=True)
    ax.scatter(xy[boundary, 0], xy[boundary, 1], s=boundary_size, c="black",
               linewidths=0, label="SIGMA boundary", rasterized=True)
    style_spatial_axis(ax); ax.set_title("SIGMA boundary"); ax.legend(frameon=False)
    return ax


def plot_signed_distance(adata, *, ax=None, spatial_key="spatial", size=8,
                         cmap="coolwarm", colorbar=True):
    """Plot the signed distance to the inferred SIGMA boundary."""
    xy, values = _spatial_values(adata, "sigma_d_signed", spatial_key)
    limit = np.nanmax(np.abs(values))
    ax = ax or plt.subplots(figsize=figure_size("single_square"), constrained_layout=True)[1]
    points = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=size, cmap=cmap,
                        vmin=-limit, vmax=limit, linewidths=0, rasterized=True)
    style_spatial_axis(ax); ax.set_title("Signed distance")
    if colorbar:
        ax.figure.colorbar(points, ax=ax, shrink=.78, label="Signed distance")
    return ax


def plot_interface_feature(adata, feature, *, ax=None, spatial_key="spatial",
                           layer=None, size=8, cmap="viridis", colorbar=True):
    """Plot one feature from ``adata.X`` or an explicitly selected layer."""
    if feature not in adata.var_names:
        raise KeyError(f"Feature {feature!r} was not found in adata.var_names.")
    matrix = adata.layers[layer] if layer is not None else adata.X
    index = int(adata.var_names.get_loc(feature))
    values = matrix[:, index].toarray().ravel() if issparse(matrix) else np.asarray(matrix[:, index]).ravel()
    xy = np.asarray(adata.obsm[spatial_key])
    ax = ax or plt.subplots(figsize=figure_size("single_square"), constrained_layout=True)[1]
    points = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=size, cmap=cmap,
                        linewidths=0, rasterized=True)
    style_spatial_axis(ax); ax.set_title(str(feature))
    if colorbar:
        ax.figure.colorbar(points, ax=ax, shrink=.78, label="Intensity")
    return ax
