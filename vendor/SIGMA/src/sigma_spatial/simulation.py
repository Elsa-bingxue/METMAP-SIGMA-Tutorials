"""Simulation helpers preserved from HBC_515 cells 46 and 49."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


def zscore_vec(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def simulate_interface_features(d_abs, d_signed=None, coords=None, tumor_mask=None,
                                n_positive=100, n_negative=100,
                                n_region_only=300, n_spatial_background=300,
                                n_noise=500, effect_size=1.0, noise_sd=0.5,
                                lam=200.0, random_state=0):
    """Generate the five feature classes used in HBC_515 simulation cell 46."""
    rng = np.random.default_rng(random_state)
    d_abs = np.asarray(d_abs, dtype=float)
    n = len(d_abs)
    if d_signed is None:
        d_signed = d_abs.copy()
    if tumor_mask is None:
        tumor_mask = np.asarray(d_signed) < 0
    tumor_mask = np.asarray(tumor_mask, dtype=bool)
    if coords is None:
        coords = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    coords = np.asarray(coords, dtype=float)
    x_coord, y_coord = zscore_vec(coords[:, 0]), zscore_vec(coords[:, 1])
    features, rows = [], []

    def add_feature(vec, sim_type, direction, true_lambda=np.nan):
        j = len(features)
        vec = np.asarray(vec, dtype=float)
        vec = vec - np.nanmin(vec)
        vec = vec / (np.nanstd(vec) + 1e-8)
        features.append(vec)
        rows.append({"j": j, "mz": f"sim_mz_{j}", "sim_type": sim_type,
                     "true_direction": direction, "true_lambda": true_lambda,
                     "is_boundary_truth": sim_type in ("positive_boundary", "negative_boundary"),
                     "is_positive_truth": sim_type == "positive_boundary",
                     "is_negative_truth": sim_type == "negative_boundary"})

    for _ in range(n_positive):
        local_lam = lam * rng.lognormal(mean=0, sigma=0.25)
        add_feature(effect_size * np.exp(-d_abs / (local_lam + 1e-8)) +
                    rng.normal(0, noise_sd, n), "positive_boundary", "positive", local_lam)
    for _ in range(n_negative):
        local_lam = lam * rng.lognormal(mean=0, sigma=0.25)
        add_feature(effect_size * (1 - np.exp(-d_abs / (local_lam + 1e-8))) +
                    rng.normal(0, noise_sd, n), "negative_boundary", "negative", local_lam)
    for _ in range(n_region_only):
        sign = rng.choice([-1, 1])
        add_feature(effect_size * sign * tumor_mask.astype(float) +
                    rng.normal(0, noise_sd, n), "region_only", "region")
    for _ in range(n_spatial_background):
        angle = rng.uniform(0, 2 * np.pi)
        signal = zscore_vec(np.cos(angle) * x_coord + np.sin(angle) * y_coord)
        add_feature(effect_size * signal + rng.normal(0, noise_sd, n),
                    "spatial_background", "background")
    for _ in range(n_noise):
        add_feature(rng.normal(0, noise_sd, n), "noise", "noise")
    # The original returns a list of records despite documenting a DataFrame.
    return np.vstack(features).T, rows


def recovery_curve_data(df):
    """Return ROC/AP inputs using the exact continuous score from HBC_515 cell 49."""
    truth = df["is_boundary_truth"].astype(int).to_numpy()
    score = np.maximum(df["score_positive"].fillna(-np.inf).to_numpy(),
                       df["score_negative"].fillna(-np.inf).to_numpy())
    fpr, tpr, thresholds = roc_curve(truth, score)
    return {"truth": truth, "score": score, "fpr": fpr, "tpr": tpr,
            "thresholds": thresholds, "auroc": roc_auc_score(truth, score),
            "average_precision": average_precision_score(truth, score)}


def evaluate_simulation_feature_selection(df):
    """Preserve the scalar recovery metrics from HBC_515 cell 49."""
    truth = df["is_boundary_truth"].astype(int).to_numpy()
    selected = df["is_selected_boundary"].astype(int).to_numpy()
    curves = recovery_curve_data(df)
    selected_true = df[df["is_selected_boundary"] & df["is_boundary_truth"]]
    direction_accuracy = (np.mean(selected_true["pred_direction"].to_numpy() ==
                                  selected_true["true_direction"].to_numpy())
                          if len(selected_true) else np.nan)
    pos_recall = recall_score(df["is_positive_truth"].astype(int),
                              (df["pred_direction"] == "positive").astype(int), zero_division=0)
    neg_recall = recall_score(df["is_negative_truth"].astype(int),
                              (df["pred_direction"] == "negative").astype(int), zero_division=0)
    lambda_df = df[(df["true_direction"] == "positive") &
                   np.isfinite(df["true_lambda"]) & np.isfinite(df["lambda_hat"])]
    if len(lambda_df):
        error = np.abs(lambda_df["lambda_hat"] - lambda_df["true_lambda"])
        lambda_mae = error.mean()
        lambda_relative_error = (error / (lambda_df["true_lambda"] + 1e-8)).mean()
    else:
        lambda_mae = lambda_relative_error = np.nan
    return pd.DataFrame([{
        "precision": precision_score(truth, selected, zero_division=0),
        "recall": recall_score(truth, selected, zero_division=0),
        "f1": f1_score(truth, selected, zero_division=0),
        "auroc": curves["auroc"], "auprc": curves["average_precision"],
        "direction_accuracy": direction_accuracy, "positive_recall": pos_recall,
        "negative_recall": neg_recall, "lambda_mae": lambda_mae,
        "lambda_relative_error": lambda_relative_error,
        "n_selected": int(selected.sum()), "n_true_boundary": int(truth.sum()),
        "n_positive_selected": int((df["pred_direction"] == "positive").sum()),
        "n_negative_selected": int((df["pred_direction"] == "negative").sum()),
    }])
