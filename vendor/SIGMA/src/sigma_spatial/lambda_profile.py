"""Original SIGMA distance-decay ranking used for program discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score

from .interface_selection import _mz_value
from .preprocessing import resolve_sm_matrix


@dataclass(frozen=True)
class LambdaProfilePreset:
    """Frozen post-training settings for one evidence-source workflow."""

    workflow: str
    var_top: int | None = 2000
    min_valid: int = 300
    min_detect_rate: float = 0.0
    min_r2: float = .05
    min_enrichment: float = 1.05
    already_log: bool = False
    ranking_col: str = "r2_logI"
    top_k: int = 300

    def to_dict(self):
        return asdict(self)


LAMBDA_PROFILE_PRESETS = {
    # Same mathematical ranking; presets only reflect the measurement scale
    # and filtering used by the corresponding validated notebook workflow.
    "direct_pathology": LambdaProfilePreset("direct_pathology"),
    "multiomics_inferred": LambdaProfilePreset("multiomics_inferred"),
    "transferred_pathology": LambdaProfilePreset(
        "transferred_pathology", var_top=None, min_r2=.005,
        min_enrichment=1.02, already_log=True,
    ),
    "region_defined_disease": LambdaProfilePreset(
        "region_defined_disease", var_top=None, min_detect_rate=.01,
        min_r2=.005, min_enrichment=1.01, ranking_col="r2_logI",
    ),
}


def get_lambda_profile_preset(workflow: str) -> LambdaProfilePreset:
    """Return a validated workflow preset; never infer one from the result."""
    try:
        return LAMBDA_PROFILE_PRESETS[workflow]
    except KeyError as exc:
        raise ValueError(
            f"workflow must be one of {tuple(LAMBDA_PROFILE_PRESETS)}"
        ) from exc


def _column(matrix, index):
    value = matrix[:, index]
    return value.toarray().ravel() if issparse(value) else np.asarray(value).ravel()


def rank_lambda_profiles(
    adata, *, matrix_source="X", distance_key="sigma_d_signed",
    var_top=2000, min_valid=300, near_quantile=.20, far_quantile=.80,
    min_r2=.05, min_enrichment=1.05, lambda_upper_quantile=.99,
    min_detect_rate=0.0, already_log=False, ranking_col="r2_logI",
):
    """Recompute the manuscript λ/R² feature ranking from SIGMA distance.

    This function uses the current data only. It does not read a frozen ranking,
    program assignment, cluster identity, or representative metabolite list.
    """
    if distance_key not in adata.obs:
        raise KeyError(f"adata.obs[{distance_key!r}] is required")
    matrix, names = resolve_sm_matrix(adata, matrix_source)
    distance = np.abs(adata.obs[distance_key].to_numpy(float))
    if issparse(matrix):
        mean = np.asarray(matrix.mean(0)).ravel()
        variance = np.asarray(matrix.power(2).mean(0)).ravel() - mean ** 2
    else:
        variance = np.var(np.asarray(matrix), axis=0)
    candidates = (
        np.arange(matrix.shape[1]) if var_top is None or var_top >= matrix.shape[1] else
        np.argsort(variance)[::-1][:min(int(var_top), matrix.shape[1])]
    )
    rows = []
    for index in candidates:
        intensity = _column(matrix, int(index)).astype(float)
        valid = np.isfinite(distance) & np.isfinite(intensity)
        d, y = distance[valid], intensity[valid]
        shifted = y - np.nanmin(y) if len(y) else y
        detect_rate = float(np.mean(y > 1e-8)) if len(y) else 0.0
        if (len(d) < int(min_valid) or detect_rate < float(min_detect_rate)
                or np.nanpercentile(shifted, 99) < 1e-8):
            continue
        transformed = shifted if already_log else np.log1p(shifted)
        model = HuberRegressor().fit(d[:, None], transformed)
        slope = float(model.coef_[0])
        if not np.isfinite(slope) or slope >= 0:
            continue
        fit_r2 = float(r2_score(transformed, model.predict(d[:, None])))
        # Each feature uses the quantiles of its own finite observations.
        near, far = np.quantile(d, [near_quantile, far_quantile])
        near_mean = float(np.nanmean(y[d <= near]))
        far_mean = float(np.nanmean(y[d >= far]))
        enrichment = (near_mean + 1e-8) / (far_mean + 1e-8)
        lam = float(-1.0 / slope)
        if (np.isfinite(lam) and np.isfinite(fit_r2) and np.isfinite(enrichment)
                and fit_r2 >= min_r2 and enrichment >= min_enrichment):
            rows.append({
                "j": int(index), "mz": _mz_value(names[index]),
                "lambda": lam, "r2_logI": fit_r2,
                "slope": slope, "intercept": float(model.intercept_),
                "n_valid": int(valid.sum()),
                "boundary_enrichment_ratio": enrichment,
                "near_interface_mean": near_mean,
                "far_interface_mean": far_mean,
                "detect_rate": detect_rate,
            })
    table = pd.DataFrame(rows)
    if len(table):
        if lambda_upper_quantile is not None and 0 < lambda_upper_quantile < 1:
            cutoff = table["lambda"].quantile(lambda_upper_quantile)
            table = table.loc[table["lambda"] <= cutoff].copy()
        table["interface_log2FC_near_vs_far"] = np.log2(
            table["boundary_enrichment_ratio"].clip(lower=1e-8)
        )
        table["interface_score"] = (
            table["r2_logI"].rank(pct=True)
            + table["boundary_enrichment_ratio"].rank(pct=True)
            + table["interface_log2FC_near_vs_far"].rank(pct=True)
        ) / 3.0
        order = (
            ["interface_score", "r2_logI", "boundary_enrichment_ratio"]
            if ranking_col == "interface_score"
            else ["r2_logI", "boundary_enrichment_ratio"]
        )
        table = table.sort_values(order, ascending=False, kind="stable").reset_index(drop=True)
    return table
