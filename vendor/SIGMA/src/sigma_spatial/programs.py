"""Deterministic construction of signed-distance metabolic programs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import AgglomerativeClustering

from .preprocessing import get_msi_matrix


@dataclass
class ProgramAnalysis:
    assignments: pd.DataFrame
    centroid_profiles: pd.DataFrame
    scores: dict[int, np.ndarray]
    config: dict
    profile_matrix: np.ndarray | None = None

    def save(self, output_dir, *, prefix="sigma"):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.assignments.to_csv(output / f"{prefix}_program_assignments.csv", index=False)
        self.centroid_profiles.to_csv(output / f"{prefix}_program_centroid_profiles.csv", index=False)
        pd.DataFrame(self.scores).to_csv(output / f"{prefix}_program_scores.csv", index=False)
        (output / f"{prefix}_program_provenance.json").write_text(
            json.dumps(self.config, indent=2, sort_keys=True)
        )


def _dense_column(matrix, index):
    value = matrix[:, index]
    return value.toarray().ravel() if issparse(value) else np.asarray(value).ravel()


def _program_matrix(adata):
    """Return the SM-local matrix expected by metabolite ranking indices."""
    for column in ("feature_type", "type"):
        if column in adata.var:
            mask = adata.var[column].astype(str).to_numpy() == "SM"
            if mask.any():
                return adata[:, mask].X
    return get_msi_matrix(adata)


def _index_column(ranking, requested):
    if requested is not None:
        if requested not in ranking:
            raise KeyError(f"ranking has no index column {requested!r}")
        return requested
    for name in ("j", "j_sm", "j_plot"):
        if name in ranking:
            return name
    raise KeyError("ranking must contain one of: j, j_sm, j_plot")


def _ordered_candidates(ranking, *, selection_mode, ranking_col, top_k):
    if selection_mode not in {"reference", "standardized", "lambda_profile"}:
        raise ValueError("selection_mode must be 'reference', 'standardized', or 'lambda_profile'")
    table = ranking.copy().replace([np.inf, -np.inf], np.nan)
    if selection_mode == "standardized":
        if ranking_col is None:
            ranking_col = next((x for x in ("candidate_score", "interface_score", "r2_logI") if x in table), None)
        if ranking_col is None:
            raise KeyError("standardized mode requires ranking_col or a supported score column")
        table = table.sort_values(ranking_col, ascending=False, kind="stable")
    elif selection_mode == "lambda_profile":
        if "r2_logI" not in table:
            raise KeyError("lambda_profile mode requires an r2_logI column")
        primary = ranking_col or "r2_logI"
        if primary not in table:
            raise KeyError(f"lambda_profile ranking has no column {primary!r}")
        order = [primary]
        order += [x for x in ("r2_logI", "boundary_enrichment_ratio") if x in table and x not in order]
        table = table.sort_values(order, ascending=False, kind="stable")
        ranking_col = primary
    # Reference mode deliberately preserves the exact final notebook table order.
    return table.head(int(top_k)).copy(), ranking_col


def discover_programs(
    adata, ranking, *, distance_key="sigma_d_signed", matrix=None,
    n_programs=4, top_k=300, n_bins=40, min_bin_points=5,
    valid_bin_fraction=.80, lower_percentile=2, upper_percentile=98,
    index_col=None, ranking_col=None, selection_mode="standardized",
    random_state=0, adaptive_binning=True, original_notebook=False,
):
    """Build metabolic programs from normalized signed-distance profiles.

    ``reference`` mode preserves the supplied candidate-table order. Therefore
    callers must supply the final frozen notebook table, not an earlier ranking.
    ``standardized`` mode explicitly sorts by ``ranking_col``.
    ``original_notebook=True`` preserves the supplied candidate order and the
    BC515 notebook's bin aggregation and module-score arithmetic. Use it with
    ``selection_mode="lambda_profile"`` and ``adaptive_binning=False``.
    """
    if distance_key not in adata.obs:
        raise KeyError(f"adata.obs[{distance_key!r}] is required")
    if int(n_programs) < 2 or int(n_bins) < 4:
        raise ValueError("n_programs >= 2 and n_bins >= 4 are required")
    if not 0 < valid_bin_fraction <= 1:
        raise ValueError("valid_bin_fraction must be in (0, 1]")
    if original_notebook and (selection_mode != "lambda_profile" or adaptive_binning):
        raise ValueError(
            "original_notebook=True requires selection_mode='lambda_profile' "
            "and adaptive_binning=False"
        )
    matrix = _program_matrix(adata) if matrix is None else matrix
    candidates, used_ranking_col = _ordered_candidates(
        ranking, selection_mode="reference" if original_notebook else selection_mode,
        ranking_col=ranking_col, top_k=top_k
    )
    feature_col = _index_column(candidates, index_col)
    distance = adata.obs[distance_key].to_numpy(float)
    finite_distance = distance[np.isfinite(distance)]
    if len(finite_distance) < int(n_bins) * int(min_bin_points):
        raise ValueError("Too few finite signed-distance observations for requested bins")
    requested_n_bins = int(n_bins)
    edges = np.linspace(
        np.nanpercentile(finite_distance, lower_percentile),
        np.nanpercentile(finite_distance, upper_percentile), requested_n_bins + 1,
    )
    # Some coordinate grids yield repeated signed-distance levels. Reduce the
    # number of equal-width bins only when occupancy would otherwise make the
    # profile undefined. This is deterministic QC, not slice-specific tuning.
    if selection_mode == "lambda_profile" and adaptive_binning:
        for candidate_bins in range(requested_n_bins, 11, -1):
            candidate_edges = np.linspace(
                np.nanpercentile(finite_distance, lower_percentile),
                np.nanpercentile(finite_distance, upper_percentile), candidate_bins + 1,
            )
            candidate_ids = np.digitize(distance, candidate_edges) - 1
            occupancy = np.array([
                np.sum(candidate_ids == b) for b in range(candidate_bins)
            ])
            if np.mean(occupancy >= int(min_bin_points)) >= valid_bin_fraction:
                edges, n_bins = candidate_edges, candidate_bins
                break
    if selection_mode == "lambda_profile":
        # Original manuscript notebooks used all edges followed by ``-1``.
        # Values outside the percentile window remain outside bins 0..n_bins-1.
        bin_id = np.digitize(distance, edges) - 1
    else:
        # Stable endpoint-inclusive behavior retained for newer workflows.
        bin_id = np.digitize(distance, edges[1:-1], right=True)
    profiles, rows = [], []
    required_valid = int(np.ceil(valid_bin_fraction * n_bins))
    for _, row in candidates.iterrows():
        if pd.isna(row[feature_col]):
            continue
        index = int(row[feature_col])
        if not 0 <= index < matrix.shape[1]:
            continue
        values = _dense_column(matrix, index)
        if original_notebook:
            # Aggregate in the input dtype, then construct a float64 profile.
            profile = np.array([
                np.nanmean(values[bin_id == b])
                if np.sum(bin_id == b) >= min_bin_points else np.nan
                for b in range(n_bins)
            ], dtype=float)
        else:
            values = values.astype(float)
            profile = np.array([
                np.nanmean(values[(bin_id == b) & np.isfinite(values)])
                if np.sum((bin_id == b) & np.isfinite(values)) >= min_bin_points else np.nan
                for b in range(n_bins)
            ])
        series = pd.Series(profile)
        if series.notna().sum() < required_valid:
            continue
        profile = series.interpolate(limit_direction="both").to_numpy(float)
        scale = profile.std() if original_notebook else np.nanstd(profile)
        if not np.isfinite(scale) or (not original_notebook and scale < 1e-8):
            continue
        center = profile.mean() if original_notebook else np.nanmean(profile)
        profiles.append((profile - center) / (scale + 1e-6))
        rows.append(row.to_dict())
    if len(profiles) < n_programs:
        raise ValueError(f"Only {len(profiles)} valid profiles for {n_programs} programs")
    profile_matrix = np.vstack(profiles)
    assignments = pd.DataFrame(rows).reset_index(drop=True)
    assignments["sigma_sm_index"] = assignments[feature_col].astype(int)
    assignments["cluster"] = AgglomerativeClustering(
        n_clusters=int(n_programs), linkage="ward"
    ).fit_predict(profile_matrix)
    centers = (edges[:-1] + edges[1:]) / 2
    centroid_rows, scores = [], {}
    for cluster in sorted(assignments.cluster.unique()):
        members = assignments.cluster.to_numpy() == cluster
        centroid = profile_matrix[members].mean(0)
        centroid_rows.extend({
            "program": int(cluster), "signed_distance": float(x),
            "centroid_z": float(y), "n_features": int(members.sum()),
        } for x, y in zip(centers, centroid))
        indices = assignments.loc[members, feature_col].astype(int).to_numpy()
        if original_notebook:
            values = matrix[:, indices]
            values = (values.toarray() if issparse(values) else values).astype(float)
            standardized = (values - values.mean(axis=0, keepdims=True)) / (
                values.std(axis=0, keepdims=True) + 1e-6
            )
            scores[int(cluster)] = standardized.mean(axis=1)
        else:
            values = np.column_stack([_dense_column(matrix, index) for index in indices]).astype(float)
            means, scales = np.nanmean(values, 0), np.nanstd(values, 0)
            scales[scales < 1e-8] = 1
            score = np.nanmean((values - means) / scales, axis=1)
            score_scale = np.nanstd(score)
            scores[int(cluster)] = (score - np.nanmean(score)) / (score_scale + 1e-6)
    config = {
        "schema_version": 1, "selection_mode": selection_mode,
        "ranking_col": used_ranking_col, "index_col": feature_col,
        "distance_key": distance_key, "n_programs": int(n_programs),
        "top_k": int(top_k), "n_bins": int(n_bins),
        "requested_n_bins": requested_n_bins,
        "adaptive_binning": bool(adaptive_binning),
        "original_notebook": bool(original_notebook),
        "min_bin_points": int(min_bin_points),
        "valid_bin_fraction": float(valid_bin_fraction),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "binning": (
            "full_edges_digitize_minus_one" if selection_mode == "lambda_profile"
            else "interior_edges_right_inclusive"
        ),
        "linkage": "ward", "n_input_candidates": int(len(candidates)),
        "n_clustered_features": int(len(assignments)),
        "random_state": int(random_state),
        "clustering_deterministic": True,
    }
    return ProgramAnalysis(
        assignments, pd.DataFrame(centroid_rows), scores, config, profile_matrix
    )
