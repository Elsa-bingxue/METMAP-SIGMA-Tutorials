"""Validation helpers for the public SIGMA workflow."""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from .preprocessing import get_msi_matrix


REQUIRED_OUTPUT_OBS = (
    "sigma_gaussian_anchor",
    "sigma_region_probability_raw",
    "sigma_region_probability",
    "sigma_inside",
    "sigma_boundary",
    "sigma_d_signed",
)

REQUIRED_OUTPUT_OBSM = (
    "X_sigma_msi",
    "X_sigma_gauss",
    "X_sigma_residual",
    "X_sigma_corrected",
)


def validate_sigma_input(adata, *, anchor_key, representation_key, spatial_key="spatial"):
    """Validate an AnnData object without modifying it."""
    if adata.n_obs < 3:
        raise ValueError("SIGMA requires at least three spatial observations.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{spatial_key!r}] is required")
    xy = np.asarray(adata.obsm[spatial_key])
    if xy.ndim != 2 or xy.shape[0] != adata.n_obs or xy.shape[1] < 2:
        raise ValueError(f"adata.obsm[{spatial_key!r}] must have shape (n_obs, >=2).")
    if not np.all(np.isfinite(xy[:, :2])):
        raise ValueError("Spatial coordinates contain non-finite values.")
    if anchor_key not in adata.obs:
        raise KeyError(f"adata.obs[{anchor_key!r}] is required")
    anchors = np.asarray(adata.obs[anchor_key])
    finite = ~np.asarray(adata.obs[anchor_key].isna())
    observed = set(anchors[finite].tolist())
    if not ({0, 1} <= observed or {False, True} <= observed):
        raise ValueError("Anchor values must contain both 0 (non-tumor) and 1 (tumor); use NaN for unlabeled spots.")
    if representation_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{representation_key!r}] is required by the validated SIGMA model. "
            "SM-only training is not yet exposed through the public API."
        )
    representation = np.asarray(adata.obsm[representation_key])
    if representation.ndim != 2 or representation.shape[0] != adata.n_obs:
        raise ValueError(f"adata.obsm[{representation_key!r}] must have shape (n_obs, n_components).")
    if not np.all(np.isfinite(representation)):
        raise ValueError("The auxiliary representation contains non-finite values.")
    matrix = get_msi_matrix(adata)
    values = matrix.data if issparse(matrix) else np.asarray(matrix)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("The feature matrix is empty or contains non-finite values.")
    if np.any(values < 0):
        raise ValueError("SIGMA log1p preprocessing requires non-negative feature intensities.")
    return adata


def validate_sigma_output(adata):
    """Check the stable output schema produced by the public workflow."""
    missing_obs = [key for key in REQUIRED_OUTPUT_OBS if key not in adata.obs]
    missing_obsm = [key for key in REQUIRED_OUTPUT_OBSM if key not in adata.obsm]
    if missing_obs or missing_obsm:
        raise ValueError(f"Incomplete SIGMA output; missing obs={missing_obs}, obsm={missing_obsm}.")
    probability = adata.obs["sigma_region_probability"].to_numpy(float)
    distance = adata.obs["sigma_d_signed"].to_numpy(float)
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("sigma_region_probability must be finite and lie in [0, 1].")
    if not np.all(np.isfinite(distance)):
        raise ValueError("sigma_d_signed must be finite.")
    return adata
