"""Quality-controlled discovery of interface-associated metabolites/programs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

from .interface_metrics import median_neighbor_spacing
from .preprocessing import resolve_sm_matrix


def _dense(matrix, index):
    value = matrix[:, index]
    return value.toarray().ravel() if issparse(value) else np.asarray(value).ravel()


def _mz_value(name):
    text = str(name).strip()
    for prefix in ("SM_", "mz_", "m/z_", "m/z"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    try:
        return float(text)
    except ValueError:
        return text


def _safe_effect(y, near, far):
    if near.sum() < 5 or far.sum() < 5:
        return np.nan, np.nan
    effect = float(np.nanmean(y[near]) - np.nanmean(y[far]))
    return effect, float(mannwhitneyu(y[near], y[far], alternative="two-sided").pvalue)


def _profile_gradient(y, d_spacing, n_bins=40):
    finite = np.isfinite(y) & np.isfinite(d_spacing)
    if finite.sum() < 20:
        return np.nan, np.nan
    lo, hi = np.nanpercentile(d_spacing[finite], [2, 98])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bins = np.digitize(d_spacing, edges[1:-1], right=True)
    profile = np.array([
        np.nanmean(y[finite & (bins == k)]) if np.sum(finite & (bins == k)) >= 5 else np.nan
        for k in range(n_bins)
    ])
    good = np.isfinite(profile)
    if good.sum() < n_bins * .6:
        return np.nan, np.nan
    profile = np.interp(centers, centers[good], profile[good])
    gradient = np.abs(np.gradient(profile, centers))
    peak = int(np.nanargmax(gradient))
    return float(abs(centers[peak])), float(gradient[peak])


def rank_interface_metabolites(
    adata, *, distance_key="sigma_d_signed", region_key="sigma_inside",
    spatial_key="spatial", near_quantile=.20, far_quantile=.80,
    gradient_window_spacings=3.0, n_neighbors=6, matrix_source="X",
):
    """Rank every SM feature using boundary-specific evidence.

    Intensities are log1p transformed and standardized per feature. Region-
    adjusted delta R2 asks whether proximity to the boundary explains signal
    beyond the binary region label. This function does not alter the AnnData.
    """
    if distance_key not in adata.obs or spatial_key not in adata.obsm:
        raise KeyError("SIGMA signed distance and spatial coordinates are required")
    matrix, names = resolve_sm_matrix(adata, matrix_source)
    d = adata.obs[distance_key].to_numpy(float)
    xy = np.asarray(adata.obsm[spatial_key], float)
    spacing = median_neighbor_spacing(xy)
    d_spacing = d / (spacing + 1e-12)
    proximity = np.exp(-np.abs(d_spacing))
    if region_key in adata.obs:
        region = adata.obs[region_key].to_numpy(float)
    else:
        region = (d < 0).astype(float)
    side_masks = {"negative": d < 0, "positive": d > 0}
    windows = {}
    for side, side_mask in side_masks.items():
        finite = side_mask & np.isfinite(d)
        absolute = np.abs(d[finite])
        near_cut = np.quantile(absolute, near_quantile)
        far_cut = np.quantile(absolute, far_quantile)
        windows[side] = (
            finite & (np.abs(d) <= near_cut),
            finite & (np.abs(d) >= far_cut),
        )
    neighbors = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(xy))).fit(xy)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    edge_left = np.repeat(np.arange(len(xy)), indices.shape[1])
    edge_right = indices.ravel()
    valid_model = np.isfinite(d) & np.isfinite(region)
    x_region = region[valid_model, None]
    x_full = np.column_stack([region[valid_model], proximity[valid_model]])
    rows = []
    for j in range(matrix.shape[1]):
        raw = _dense(matrix, j).astype(float)
        finite_raw = np.isfinite(raw)
        detect_rate = float(np.mean(raw[finite_raw] > 0)) if finite_raw.any() else 0.0
        y = np.log1p(np.maximum(raw, 0))
        scale = np.nanstd(y)
        yz = (y - np.nanmean(y)) / (scale + 1e-8)
        effects, pvalues = {}, {}
        for side, (near, far) in windows.items():
            effects[side], pvalues[side] = _safe_effect(yz, near & np.isfinite(yz), far & np.isfinite(yz))
        best_side = max(effects, key=lambda x: -np.inf if not np.isfinite(effects[x]) else effects[x])
        best_effect, best_p = effects[best_side], pvalues[best_side]
        model_use = valid_model & np.isfinite(yz)
        if model_use.sum() >= 20:
            yr = yz[model_use]
            r2_region = LinearRegression().fit(region[model_use, None], yr).score(region[model_use, None], yr)
            full_x = np.column_stack([region[model_use], proximity[model_use]])
            r2_full = LinearRegression().fit(full_x, yr).score(full_x, yr)
            delta_r2 = float(r2_full - r2_region)
        else:
            delta_r2 = np.nan
        grad_distance, grad_strength = _profile_gradient(yz, d_spacing)
        edge_valid = np.isfinite(yz[edge_left]) & np.isfinite(yz[edge_right])
        if edge_valid.sum() >= 10 and np.nanstd(yz[edge_left][edge_valid]) > 0 and np.nanstd(yz[edge_right][edge_valid]) > 0:
            continuity = float(np.corrcoef(yz[edge_left][edge_valid], yz[edge_right][edge_valid])[0, 1])
        else:
            continuity = np.nan
        rows.append({
            "mz": _mz_value(names[j]), "j": j, "detect_rate": detect_rate,
            "negative_near_minus_far": effects["negative"],
            "positive_near_minus_far": effects["positive"],
            "best_side": best_side, "interface_effect": best_effect,
            "p_value": best_p, "region_adjusted_delta_r2": delta_r2,
            "gradient_peak_distance_spacings": grad_distance,
            "gradient_strength": grad_strength, "spatial_continuity": continuity,
        })
    table = pd.DataFrame(rows)
    table["fdr"] = multipletests(table.p_value.fillna(1), method="fdr_bh")[1]
    evidence = pd.DataFrame({
        "effect": table.interface_effect.rank(pct=True),
        "delta_r2": table.region_adjusted_delta_r2.rank(pct=True),
        "gradient": table.gradient_strength.rank(pct=True),
        "continuity": table.spatial_continuity.rank(pct=True),
    })
    table["interface_score"] = evidence.mean(axis=1, skipna=True)
    table["gradient_near_boundary"] = table.gradient_peak_distance_spacings <= gradient_window_spacings
    supported = (
        (table.detect_rate >= .05) & (table.interface_effect >= .10)
        & (table.region_adjusted_delta_r2 > 0) & table.gradient_near_boundary
        & (table.spatial_continuity > 0)
    )
    high = supported & (table.fdr < .05) & (table.interface_effect >= .30)
    table["evidence_tier"] = np.where(high, "high_confidence", np.where(supported, "supported", "exploratory"))
    table["eligible_representative"] = supported
    table["spot_spacing"] = spacing
    return table.sort_values("interface_score", ascending=False, kind="stable").reset_index(drop=True)


def summarize_interface_programs(assignments, feature_ranking, interface_statistics):
    """Combine feature-level QC with program-level near/far evidence."""
    index_col = "sigma_sm_index" if "sigma_sm_index" in assignments else "j"
    qc = feature_ranking.rename(columns={"interface_score": "standardized_interface_score"})
    merged = assignments.merge(qc, left_on=index_col, right_on="j", how="left", suffixes=("", "_qc"))
    feature_summary = merged.groupby("cluster").agg(
        n_features=(index_col, "size"),
        eligible_fraction=("eligible_representative", "mean"),
        median_interface_score=("standardized_interface_score", "median"),
        median_gradient_distance=("gradient_peak_distance_spacings", "median"),
        median_delta_r2=("region_adjusted_delta_r2", "median"),
    ).reset_index().rename(columns={"cluster": "program"})
    positive = interface_statistics.sort_values("near_minus_far", ascending=False).groupby("program", as_index=False).first()
    summary = feature_summary.merge(
        positive[["program", "orientation", "side_name", "near_minus_far", "fdr"]],
        on="program", how="left",
    )
    components = pd.DataFrame({
        "effect": summary.near_minus_far.rank(pct=True),
        "eligible": summary.eligible_fraction.rank(pct=True),
        "feature_score": summary.median_interface_score.rank(pct=True),
        "delta_r2": summary.median_delta_r2.rank(pct=True),
    })
    summary["program_interface_score"] = components.mean(axis=1)
    return summary.sort_values("program_interface_score", ascending=False).reset_index(drop=True), merged


def select_representative_metabolites(assignments, feature_ranking, *, program, n=5):
    """Select interpretable representatives from one frozen program."""
    index_col = "sigma_sm_index" if "sigma_sm_index" in assignments else "j"
    qc = feature_ranking.rename(columns={"interface_score": "standardized_interface_score"})
    subset = assignments[assignments.cluster.astype(int) == int(program)].merge(
        qc, left_on=index_col, right_on="j", how="left", suffixes=("", "_qc")
    )
    eligible = subset[subset.eligible_representative.fillna(False)].copy()
    pool = eligible if len(eligible) >= n else subset.copy()
    pool["representative_qc"] = np.where(pool.eligible_representative.fillna(False), "pass", "relaxed")
    return pool.sort_values(
        ["eligible_representative", "standardized_interface_score"], ascending=[False, False], kind="stable"
    ).head(int(n)).reset_index(drop=True)
