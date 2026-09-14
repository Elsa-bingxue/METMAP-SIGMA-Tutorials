"""Input helpers for user-supplied spatial omics tables."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd


def load_spatial_metabolomics(
    matrix_path,
    coordinates_path,
    anchors_path=None,
    *,
    x_column="x",
    y_column="y",
    anchor_column="anchor",
    anchor_key="sigma_anchor",
):
    """Load spot-by-feature, coordinate, and optional anchor CSV files."""
    matrix = pd.read_csv(Path(matrix_path), index_col=0)
    coordinates = pd.read_csv(Path(coordinates_path), index_col=0)
    if not matrix.index.is_unique or not coordinates.index.is_unique:
        raise ValueError("Spot identifiers must be unique in all input tables.")
    missing = matrix.index.difference(coordinates.index)
    extra = coordinates.index.difference(matrix.index)
    if len(missing) or len(extra):
        raise ValueError(f"Coordinate spot IDs do not match the matrix (missing={list(missing[:5])}, extra={list(extra[:5])}).")
    for column in (x_column, y_column):
        if column not in coordinates:
            raise KeyError(f"Coordinate column {column!r} was not found.")
    result = ad.AnnData(matrix.to_numpy(dtype=float))
    result.obs_names = matrix.index.astype(str)
    result.var_names = matrix.columns.astype(str)
    result.var["feature_type"] = "SM"
    aligned_coordinates = coordinates.loc[matrix.index, [x_column, y_column]]
    result.obsm["spatial"] = aligned_coordinates.to_numpy(dtype=float)
    if anchors_path is not None:
        anchors = pd.read_csv(Path(anchors_path), index_col=0)
        if anchor_column not in anchors:
            raise KeyError(f"Anchor column {anchor_column!r} was not found.")
        missing_anchor = matrix.index.difference(anchors.index)
        if len(missing_anchor):
            raise ValueError(f"Anchor table is missing spot IDs: {list(missing_anchor[:5])}.")
        result.obs[anchor_key] = anchors.loc[matrix.index, anchor_column].to_numpy()
    return result


def load_spatial_transcriptomics(
    matrix_path, coordinates_path, *, x_column="x", y_column="y",
):
    """Load a spot-by-gene table and matching coordinates as AnnData."""
    result = load_spatial_metabolomics(
        matrix_path, coordinates_path, x_column=x_column, y_column=y_column,
    )
    result.var["feature_type"] = "ST"
    return result
