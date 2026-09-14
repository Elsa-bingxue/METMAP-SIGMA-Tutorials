"""Cross-dataset metrics for quantitative metabolic-interface taxonomy."""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors


def dense_column(matrix, index: int) -> np.ndarray:
    return matrix[:, index].toarray().ravel() if issparse(matrix) else np.asarray(matrix[:, index]).ravel()


def median_neighbor_spacing(xy: np.ndarray) -> float:
    distances, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)
    return float(np.median(distances[:, 1]))


def signed_profile_metrics(
    intensity: np.ndarray,
    signed_distance: np.ndarray,
    spacing: float,
    n_bins: int = 40,
) -> dict:
    """Calculate scale-normalized sharpness and side-specific extension.

    Negative distance is the tumor/region side in the frozen SIGMA outputs;
    positive distance is the stromal/microenvironment side.
    """
    y = np.log1p(np.maximum(np.asarray(intensity, float), 0))
    d = np.asarray(signed_distance, float) / spacing
    finite = np.isfinite(y) & np.isfinite(d)
    y, d = y[finite], d[finite]
    lo, hi = np.percentile(d, [2, 98])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.digitize(d, edges) - 1
    profile = np.array([np.mean(y[idx == k]) if np.sum(idx == k) >= 5 else np.nan for k in range(n_bins)])
    good = np.isfinite(profile)
    filled = np.interp(centers, centers[good], profile[good]) if good.sum() >= 2 else np.zeros(n_bins)
    scaled = (filled - filled.mean()) / (filled.std() + 1e-8)
    near = np.abs(centers) <= min(3.0, np.percentile(np.abs(centers), 40))
    gradient = np.gradient(scaled, centers)
    sharpness = float(np.max(np.abs(gradient[near]))) if near.any() else float(np.max(np.abs(gradient)))
    extent = np.percentile(np.abs(d), 60)
    tumor = y[(d < 0) & (np.abs(d) <= extent)]
    stroma = y[(d > 0) & (np.abs(d) <= extent)]
    stromal_extension = float(np.log2((stroma.mean() + 1e-6) / (tumor.mean() + 1e-6)))
    return {"sharpness": sharpness, "stromal_extension": stromal_extension, "profile": scaled, "profile_x": centers}


def sector_lambda_anisotropy(
    intensity: np.ndarray,
    signed_distance: np.ndarray,
    xy: np.ndarray,
    n_sectors: int = 8,
    min_points: int = 30,
) -> dict:
    """Estimate angular variation of exponential influence range λ."""
    y = np.log1p(np.maximum(np.asarray(intensity, float), 0))
    d = np.abs(np.asarray(signed_distance, float))
    centered = np.asarray(xy, float) - np.median(xy, axis=0)
    angle = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2 * np.pi)
    sector = np.floor(angle / (2 * np.pi / n_sectors)).astype(int)
    lambdas = []
    for k in range(n_sectors):
        use = (sector == k) & np.isfinite(y) & np.isfinite(d)
        if use.sum() < min_points:
            continue
        slope = np.polyfit(d[use], y[use], 1)[0]
        if slope < 0:
            lambdas.append(-1.0 / slope)
    values = np.asarray(lambdas)
    cv = float(values.std() / (values.mean() + 1e-12)) if len(values) >= 3 else np.nan
    return {"anisotropy_cv": cv, "n_valid_sectors": int(len(values))}
