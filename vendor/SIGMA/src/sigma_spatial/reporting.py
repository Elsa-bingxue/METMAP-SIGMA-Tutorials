"""Downstream reporting helpers for validated SIGMA outputs.

These functions do not refit SIGMA or change feature/program assignments.  They
turn a frozen result and a provenance-locked assignment table into report-ready
tables and figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from .plotting import save_figure, style_spatial_axis
from .preprocessing import resolve_sm_matrix


def _assignment_index(assignments):
    for name in ("sigma_sm_index", "j", "j_sm", "j_plot"):
        if name in assignments:
            return name
    raise KeyError("assignments requires an SM-local index column")


def _msi_matrix(adata, matrix_source="X"):
    return resolve_sm_matrix(adata, matrix_source)[0]


def _columns(matrix, indices):
    selected = matrix[:, indices]
    return selected.toarray() if issparse(selected) else np.asarray(selected)


def program_report_data(adata, assignments, *, n_bins=40, min_bin_points=5, matrix_source="X"):
    """Calculate frozen program scores and signed-distance profiles.

    ``assignments`` must contain the preserved ``j``, ``mz`` and ``cluster``
    columns. No clustering or feature selection is performed here.
    """
    table = assignments.copy()
    required = {"mz", "cluster"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"assignments is missing columns: {sorted(missing)}")
    matrix = _msi_matrix(adata, matrix_source)
    index_col = _assignment_index(table)
    distance = adata.obs["sigma_d_signed"].to_numpy(float)
    edges = np.linspace(np.percentile(distance, 2), np.percentile(distance, 98), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    digitized = np.digitize(distance, edges) - 1
    profiles, scores = {}, {}
    for cluster in sorted(table["cluster"].astype(int).unique()):
        subset = table[table["cluster"].astype(int) == cluster]
        indices = subset[index_col].astype(int).to_list()
        values = _columns(matrix, indices).astype(float)
        scaled = (values - values.mean(0, keepdims=True)) / (values.std(0, keepdims=True) + 1e-6)
        score = scaled.mean(1)
        scores[cluster] = (score - score.mean()) / (score.std() + 1e-6)
        profile = np.array([
            np.nanmean(scores[cluster][digitized == b])
            if np.sum(digitized == b) >= min_bin_points else np.nan
            for b in range(n_bins)
        ])
        profiles[cluster] = pd.Series(profile).interpolate(limit_direction="both").to_numpy()
    return {"assignments": table, "centers": centers, "profiles": profiles, "scores": scores}


def representative_metabolites(assignments, *, cluster=None, n=5, score="r2_logI"):
    """Select representatives using a preserved ranking column."""
    table = assignments.copy()
    if cluster is not None:
        table = table[table["cluster"].astype(int) == int(cluster)]
    order = score if score in table.columns else _assignment_index(table)
    return table.sort_values(order, ascending=False).head(n).copy()


def disease_profile_representatives(
    assignments, *, cluster, n=5, max_lambda_quantile=0.90,
    min_r2=0.005, min_enrichment=1.01,
):
    """Select disease-interface representatives using the original HPD logic.

    Selection is deliberately restricted to the already selected Ward program.
    Within that program, retain detected negative-slope distance-decay features,
    remove only the extreme long-lambda tail when enough candidates remain, and
    rank primarily by robust distance-fit R2 followed by near/far enrichment.
    No spatial-hotspot or centroid-correlation score is introduced here.
    """
    table = assignments[assignments["cluster"].astype(int) == int(cluster)].copy()
    if table.empty:
        return table

    enrichment_col = next(
        (name for name in ("boundary_enrichment_ratio", "interface_enrichment_ratio")
         if name in table.columns),
        None,
    )
    required = [name for name in ("r2_logI", "slope", enrichment_col) if name]
    for name in required:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    valid = np.isfinite(table["r2_logI"]) & np.isfinite(table["slope"])
    valid &= table["slope"] < 0
    valid &= table["r2_logI"] >= float(min_r2)
    if enrichment_col is not None:
        valid &= np.isfinite(table[enrichment_col])
        valid &= table[enrichment_col] >= float(min_enrichment)
    table = table.loc[valid].copy()
    if table.empty:
        return table

    if "lambda" in table.columns:
        table["lambda"] = pd.to_numeric(table["lambda"], errors="coerce")
        finite_lambda = table["lambda"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_lambda) >= int(n):
            cutoff = finite_lambda.quantile(float(max_lambda_quantile))
            retained = table[np.isfinite(table["lambda"]) & (table["lambda"] <= cutoff)]
            if len(retained) >= int(n):
                table = retained

    order = ["r2_logI"]
    if enrichment_col is not None:
        order.append(enrichment_col)
    table["representative_selection"] = "original_hpd_distance_profile"
    return table.sort_values(order, ascending=False, kind="stable").head(int(n)).copy()


def spatially_coherent_representatives(
    adata, assignments, report, *, cluster, n=5, matrix_source="X",
    n_neighbors=6, max_lambda_quantile=0.90,
):
    """Rank display features without changing program assignments.

    The preserved distance-profile criteria (R2, interface enrichment and
    lambda) are combined with agreement to the program centroid and local
    spatial continuity. This is intended only for representative figures.
    """
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors

    table = assignments[assignments["cluster"].astype(int) == int(cluster)].copy()
    if table.empty:
        return table
    index_col = _assignment_index(table)
    matrix = _msi_matrix(adata, matrix_source)
    xy = np.asarray(adata.obsm["spatial"], float)
    k = min(int(n_neighbors) + 1, adata.n_obs)
    neighbours = NearestNeighbors(n_neighbors=k).fit(xy).kneighbors(
        xy, return_distance=False
    )[:, 1:]
    centroid = np.asarray(report["scores"][int(cluster)], float)

    rows = []
    for row_index, row in table.iterrows():
        values = _columns(matrix, [int(row[index_col])]).ravel().astype(float)
        finite = np.isfinite(values)
        if finite.sum() < 3 or np.nanstd(values) < 1e-10:
            centroid_r = spatial_r = hotspot_continuity = np.nan
        else:
            centroid_r = spearmanr(values[finite], centroid[finite]).statistic
            filled = values.copy()
            filled[~finite] = np.nanmedian(values[finite])
            z = (filled - filled.mean()) / (filled.std() + 1e-9)
            neighbour_mean = z[neighbours].mean(axis=1)
            spatial_r = float(np.corrcoef(z, neighbour_mean)[0, 1])
            threshold = np.nanquantile(filled, 0.80)
            hotspot = filled >= threshold
            hotspot_continuity = float(
                np.mean(np.any(hotspot[neighbours], axis=1)[hotspot])
            ) if hotspot.any() else np.nan
        rows.append((row_index, centroid_r, spatial_r, hotspot_continuity))

    qc = pd.DataFrame(
        rows, columns=["_row", "centroid_correlation", "spatial_coherence", "hotspot_continuity"]
    ).set_index("_row")
    table = table.join(qc)
    if "lambda" in table and table["lambda"].notna().any():
        cutoff = table["lambda"].quantile(float(max_lambda_quantile))
        retained = table[table["lambda"] <= cutoff]
        if len(retained) >= int(n):
            table = retained

    # A representative spatial map must be locally coherent. Use a relative
    # within-program gate so this display QC adapts to tissue resolution while
    # retaining at least n candidates whenever possible.
    finite_spatial = table["spatial_coherence"].dropna()
    if len(finite_spatial) >= int(n):
        spatial_cutoff = max(0.20, float(finite_spatial.median()))
        retained = table[
            (table["spatial_coherence"] >= spatial_cutoff)
            & (table["centroid_correlation"] > 0)
        ]
        if len(retained) >= int(n):
            table = retained

    metric_weights = {
        "r2_logI": 0.15,
        "boundary_enrichment_ratio": 0.15,
        "centroid_correlation": 0.25,
        "spatial_coherence": 0.35,
        "hotspot_continuity": 0.10,
    }
    available = {key: weight for key, weight in metric_weights.items() if key in table}
    ranks = []
    weights = []
    for key, weight in available.items():
        ranks.append(table[key].rank(pct=True).fillna(0).to_numpy())
        weights.append(weight)
    table["representative_score"] = np.average(ranks, axis=0, weights=weights)
    table["representative_selection"] = "distance_profile_plus_spatial_qc"
    return table.sort_values(
        ["representative_score", "r2_logI"], ascending=False
    ).head(int(n)).copy()


def _format_mz(value):
    """Format numeric and common prefixed m/z feature identifiers."""
    text = str(value).strip()
    for prefix in ("SM_", "mz_", "m/z_", "m/z"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    try:
        return f"{float(text):.4f}"
    except ValueError:
        return str(value)


def plot_program_report(adata, report, output_dir, *, boundary_cluster=3, prefix="HBC515",
                        representative_n=5, representatives=None, matrix_source="X",
                        show_boundary=True, interface_label="SIGMA interface",
                        association_label="interface-associated"):
    """Save program maps, profiles, and boundary-program metabolite maps."""
    output_dir = Path(output_dir)
    xy = np.asarray(adata.obsm["spatial"], float)
    clusters = sorted(report["scores"])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 4)))

    fig, axes = plt.subplots(2, len(clusters), figsize=(2.65 * len(clusters), 5.1),
                             constrained_layout=True)
    vmax = max(np.percentile(np.abs(report["scores"][c]), 98) for c in clusters)
    for column, cluster in enumerate(clusters):
        ax = axes[0, column]
        points = ax.scatter(xy[:, 0], xy[:, 1], c=report["scores"][cluster], s=5,
                            cmap="coolwarm", vmin=-vmax, vmax=vmax, linewidths=0,
                            rasterized=True)
        suffix = f" — {association_label}" if cluster == boundary_cluster else ""
        ax.set_title(f"Program {cluster}{suffix}")
        style_spatial_axis(ax)
        ax = axes[1, column]
        ax.plot(report["centers"], report["profiles"][cluster], color=colors[column])
        ax.axvline(0, color="black", ls="--", lw=.8)
        ax.set_xlabel("Signed distance")
        if column == 0:
            ax.set_ylabel("Module score (z)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(points, ax=list(axes[0]), shrink=.75, label="Module score (z)")
    save_figure(fig, output_dir / f"{prefix}_metabolic_patterns_and_distance_profiles",
                formats=("pdf", "svg", "png"), close=True)

    if representatives is None:
        representatives = representative_metabolites(
            report["assignments"], cluster=boundary_cluster, n=representative_n
        )
    else:
        representatives = representatives.head(int(representative_n)).copy()
    matrix = _msi_matrix(adata, matrix_source)
    index_col = _assignment_index(report["assignments"])
    fig, axes = plt.subplots(1, len(representatives),
                             figsize=(2.25 * len(representatives), 2.65),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, representatives.iterrows()):
        intensity = np.log1p(np.maximum(_columns(matrix, [int(row[index_col])]).ravel(), 0))
        points = ax.scatter(xy[:, 0], xy[:, 1], c=intensity, s=5, cmap="viridis",
                            linewidths=0, rasterized=True)
        if show_boundary and "sigma_boundary" in adata.obs:
            boundary = adata.obs["sigma_boundary"].to_numpy(bool)
            ax.scatter(xy[boundary, 0], xy[boundary, 1], s=.55, c="#D62728",
                       linewidths=0, alpha=.58, rasterized=True)
        ax.set_title(f"m/z {_format_mz(row['mz'])}")
        style_spatial_axis(ax)
        fig.colorbar(points, ax=ax, shrink=.68, label="log(1 + intensity)")
    if show_boundary and "sigma_boundary" in adata.obs and len(axes):
        axes[0].scatter([], [], s=14, c="#D62728", label=interface_label)
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.04), frameon=False,
                       fontsize=7, markerscale=.7)
    save_figure(fig, output_dir / f"{prefix}_boundary_program_{boundary_cluster}_metabolites",
                formats=("pdf", "svg", "png"), close=True)
    return representatives


def plot_bidirectional_program_enrichment(table, output_dir, *, leading_program, prefix="sigma"):
    """Plot program near-versus-far enrichment on both interface sides."""
    plot = table.sort_values(["orientation", "near_minus_far"]).reset_index(drop=True)
    y = np.arange(len(plot))
    colors = []
    for row in plot.itertuples():
        if row.program == leading_program:
            colors.append("#C83E3A")
        elif row.orientation == "negative":
            colors.append("#4C78A8")
        else:
            colors.append("#72A66A")
    fig, ax = plt.subplots(figsize=(6.4, max(3.2, .34 * len(plot) + 1.4)),
                           constrained_layout=True)
    values = plot["near_minus_far"].to_numpy(float)
    ax.axvline(0, color="black", lw=.8)
    ax.hlines(y, 0, values, color=colors, lw=1.6)
    ax.scatter(values, y, c=colors, s=34, edgecolor="white", linewidth=.4, zorder=3)
    labels = [f"Program {row.program} ({row.side_name})" for row in plot.itertuples()]
    ax.set_yticks(y, labels)
    for yy, value, q in zip(y, values, plot["fdr"]):
        stars = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
        if stars:
            ax.annotate(stars, (value, yy), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Interface enrichment (near − far program score)")
    ax.set_title("Interface-associated metabolic programs")
    ax.grid(axis="x", color=".92", lw=.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, Path(output_dir) / f"{prefix}_interface_program_enrichment",
                formats=("pdf", "svg", "png"), close=True)


def plot_integrated_program_summary(
    adata, report, statistics, representatives, output_dir, *, leading_program,
    prefix="sigma", matrix_source="X", interface_label="SIGMA interface",
    annotation_key="annotation", annotation_title="Annotation",
):
    """Combine program patterns, enrichment and representative metabolites."""
    from matplotlib.gridspec import GridSpec

    output_dir = Path(output_dir)
    xy = np.asarray(adata.obsm["spatial"], float)
    clusters = sorted(report["scores"])
    representatives = representatives.head(5)
    # A 30-column common grid keeps the six overview panels, the four profiles,
    # and the five metabolite panels aligned to the same outer margins.
    ncols = 30
    fig = plt.figure(figsize=(17.0, 8.5), constrained_layout=False)
    grid = GridSpec(3, ncols, figure=fig, height_ratios=(2.15, 1.50, 2.05))
    fig.subplots_adjust(left=.065, right=.945, bottom=.065, top=.95,
                        hspace=.56, wspace=.82)
    vmax = max(np.percentile(np.abs(report["scores"][c]), 98) for c in clusters)
    pattern_points = None
    annotation_ax = fig.add_subplot(grid[0, 0:5])
    annotation = (
        adata.obs[annotation_key].astype(str)
        if annotation_key is not None and annotation_key in adata.obs else None
    )
    if annotation is not None:
        categories = list(dict.fromkeys(annotation.tolist()))
        semantic_colors = {
            "tumor_region": "#D95F5F", "tumor_core_like": "#D95F5F",
            "tumor_transition": "#F2A65A", "mixed_transition": "#F2A65A",
            "tme_or_mixed_region": "#67B7C7",
            "invasive_margin_like": "#E3A857",
            "microenvironment_like": "#67B7C7", "neural_like": "#7A83C6",
        }
        fallback = plt.get_cmap("tab10")
        color_map = {
            category: semantic_colors.get(category.lower(), fallback(i % 10))
            for i, category in enumerate(categories)
        }
        annotation_colors = annotation.map(color_map).to_numpy()
        annotation_ax.scatter(xy[:, 0], xy[:, 1], c=annotation_colors, s=4,
                              linewidths=0, rasterized=True)
        for category in categories:
            annotation_ax.scatter([], [], c=[color_map[category]], s=18,
                                  label=str(category).replace("_", " "))
        annotation_ax.legend(loc="upper center", bbox_to_anchor=(.5, -.02),
                             frameon=False, fontsize=6.5,
                             ncol=1 if len(categories) <= 3 else 2)
    annotation_ax.set_title(annotation_title)
    style_spatial_axis(annotation_ax)

    distance_ax = fig.add_subplot(grid[0, 5:10])
    signed = adata.obs["sigma_d_signed"].to_numpy(float)
    dmax = np.nanpercentile(np.abs(signed), 98)
    distance_points = distance_ax.scatter(
        xy[:, 0], xy[:, 1], c=signed, s=4, cmap="coolwarm",
        vmin=-dmax, vmax=dmax, linewidths=0, rasterized=True,
    )
    if "sigma_boundary" in adata.obs:
        boundary = adata.obs["sigma_boundary"].to_numpy(bool)
        distance_ax.scatter(xy[boundary, 0], xy[boundary, 1], s=.65,
                            c="black", linewidths=0, rasterized=True)
    distance_ax.set_title("Signed distance")
    style_spatial_axis(distance_ax)
    # Inset colorbars do not resize their parent axes, preserving column alignment.
    distance_cax = distance_ax.inset_axes([1.025, .17, .035, .66])
    fig.colorbar(distance_points, cax=distance_cax, label="Distance")

    pattern_axes = []
    program_width = 5
    for column, cluster in enumerate(clusters):
        sl = slice(10 + column * program_width, 10 + (column + 1) * program_width)
        ax = fig.add_subplot(grid[0, sl])
        pattern_axes.append(ax)
        pattern_points = ax.scatter(
            xy[:, 0], xy[:, 1], c=report["scores"][cluster], s=4,
            cmap="coolwarm", vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True,
        )
        marker = " *" if int(cluster) == int(leading_program) else ""
        ax.set_title(f"Program {cluster}{marker}")
        style_spatial_axis(ax)
        ax = fig.add_subplot(grid[1, sl])
        ax.plot(report["centers"], report["profiles"][cluster], color="#2C7FB8", lw=1.5)
        ax.axvline(0, color="black", ls="--", lw=.8)
        ax.set_xlabel("Signed distance")
        if column == 0:
            ax.set_ylabel("Program score (z)")
        ax.spines[["top", "right"]].set_visible(False)
    if pattern_points is not None:
        pattern_cax = pattern_axes[-1].inset_axes([1.025, .12, .035, .76])
        fig.colorbar(pattern_points, cax=pattern_cax, label="Program score (z)")

    selected_orientation = statistics.loc[
        statistics["program"].astype(int) == int(leading_program)
    ].sort_values("near_minus_far", ascending=False).iloc[0]["orientation"]
    ax = fig.add_subplot(grid[1, 0:10])
    plot = statistics[statistics["orientation"] == selected_orientation].copy()
    plot = plot.sort_values("program", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot))
    colors = [
        "#C83E3A" if int(row.program) == int(leading_program) else "#91A9C6"
        for row in plot.itertuples()
    ]
    values = plot["near_minus_far"].to_numpy(float)
    ax.axvline(0, color="black", lw=.9, zorder=1)
    ax.hlines(y, 0, values, color=colors, linewidth=2.2)
    ax.scatter(values, y, c=colors, s=48, zorder=3, edgecolor="white", linewidth=.7)
    ax.set_yticks([])
    span = max(.1, float(np.nanmax(np.abs(values))) * 1.18)
    label_x = -.055 * span
    for yy, row in zip(y, plot.itertuples()):
        ax.text(label_x, yy, f"Program {row.program}", ha="right", va="center", fontsize=8)
    for yy, value, q in zip(y, values, plot["fdr"]):
        stars = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
        offset = 4 if value >= 0 else -4
        align = "left" if value >= 0 else "right"
        ax.annotate(stars, (value, yy), xytext=(offset, 0), textcoords="offset points",
                    ha=align, va="center", fontsize=8)
    side_name = plot["side_name"].iloc[0] if len(plot) else selected_orientation
    ax.set_title(f"{side_name.capitalize()} interface enrichment", fontsize=10)
    ax.set_xlabel("Δ interface score")
    ax.set_xlim(min(-.24 * span, float(np.nanmin(values)) * 1.12), span)
    ax.invert_yaxis()
    ax.grid(axis="x", color=".92", lw=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    # Match the active spatial-axis boxes above (rather than only their broader
    # GridSpec slots), so the lollipop begins exactly below Weak annotation.
    lollipop_pos = ax.get_position()
    annotation_pos = annotation_ax.get_position()
    distance_pos = distance_ax.get_position()
    ax.set_position([
        annotation_pos.x0,
        lollipop_pos.y0,
        distance_pos.x1 - annotation_pos.x0,
        lollipop_pos.height,
    ])

    matrix = _msi_matrix(adata, matrix_source)
    index_col = _assignment_index(report["assignments"])
    boundary = adata.obs["sigma_boundary"].to_numpy(bool) if "sigma_boundary" in adata.obs else None
    metabolite_width = 6
    for column, (_, row) in enumerate(representatives.iterrows()):
        sl = slice(column * metabolite_width, (column + 1) * metabolite_width)
        ax = fig.add_subplot(grid[2, sl])
        intensity = np.log1p(np.maximum(_columns(matrix, [int(row[index_col])]).ravel(), 0))
        points = ax.scatter(xy[:, 0], xy[:, 1], c=intensity, s=4, cmap="viridis",
                            linewidths=0, rasterized=True)
        if boundary is not None:
            ax.scatter(xy[boundary, 0], xy[boundary, 1], s=.45, c="#D62728",
                       linewidths=0, alpha=.55, rasterized=True)
        ax.set_title(f"m/z {_format_mz(row['mz'])}")
        style_spatial_axis(ax)
        metabolite_cax = ax.inset_axes([1.015, .16, .035, .68])
        fig.colorbar(points, cax=metabolite_cax)
    save_figure(
        fig, output_dir / f"{prefix}_integrated_interface_program_summary",
        formats=("pdf", "svg", "png"), close=True,
    )


def plot_interface_metric_supplement(
    adata, assignments, representatives, output_dir, *, prefix="sigma",
    matrix_source="X", anisotropy_min_points=100,
):
    """Plot one leading metabolite with its decay, λ context and anisotropy."""
    output_dir = Path(output_dir)
    matrix = _msi_matrix(adata, matrix_source)
    distance = np.abs(adata.obs["sigma_d_signed"].to_numpy(float))
    index_col = _assignment_index(assignments)
    fig = plt.figure(figsize=(14.2, 3.45), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=(1.05, 1.25, 1, 1.05),
                           wspace=.24)

    xy = np.asarray(adata.obsm["spatial"], float)
    boundary = adata.obs["sigma_boundary"].to_numpy(bool) if "sigma_boundary" in adata.obs else None
    # Representatives are already ordered by the workflow-specific interface
    # ranking, so the first row is the single strongest supported example.
    row = representatives.iloc[0]
    ax_map = fig.add_subplot(grid[0, 0])
    intensity = np.log1p(np.maximum(
        _columns(matrix, [int(row[index_col])]).ravel(), 0
    ))
    points = ax_map.scatter(xy[:, 0], xy[:, 1], c=intensity, s=4,
                            cmap="viridis", linewidths=0, rasterized=True)
    if boundary is not None:
        ax_map.scatter(xy[boundary, 0], xy[boundary, 1], s=.5,
                       c="#D62728", linewidths=0, alpha=.6, rasterized=True)
    ax_map.set_title(f"Leading interface metabolite\nm/z {_format_mz(row['mz'])}")
    style_spatial_axis(ax_map)
    fig.colorbar(points, ax=ax_map, shrink=.72, pad=.02,
                 label="log(1 + intensity)")

    ax = fig.add_subplot(grid[0, 2])
    lambda_source = assignments["lambda"] if "lambda" in assignments else pd.Series(dtype=float)
    lambdas = lambda_source.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(lambdas):
        ax.hist(lambdas, bins=min(25, max(8, int(np.sqrt(len(lambdas))))),
                color="#4C78A8", edgecolor="white", linewidth=.5)
    if len(lambdas):
        median = float(np.median(lambdas))
        ax.axvline(median, color="#C83E3A", ls="--", lw=1.4,
                   label=f"Median λ = {median:.1f}")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(.5, .5, "Influence-range estimates\nnot available",
                transform=ax.transAxes, ha="center", va="center", color=".4")
    ax.set_xlabel("Interface influence range λ")
    ax.set_ylabel("Metabolites")
    ax.set_title("Influence-range distribution")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1])
    edges = np.quantile(distance[np.isfinite(distance)], np.linspace(0, 1, 21))
    centers = (edges[:-1] + edges[1:]) / 2
    values = _columns(matrix, [int(row[index_col])]).ravel().astype(float)
    transformed = np.log1p(np.maximum(values - np.nanmin(values), 0))
    bins = np.digitize(distance, edges[1:-1], right=True)
    profile = np.array([
        np.nanmean(transformed[bins == b]) if np.any(bins == b) else np.nan
        for b in range(len(centers))
    ])
    ax.plot(centers, profile, marker="o", ms=3, lw=1.4, color="#2C7FB8")
    ax.set_xlabel("Absolute distance to interface")
    ax.set_ylabel("Mean log(1 + shifted intensity)")
    lambda_value = row.get("lambda", np.nan)
    r2_value = row.get("r2_logI", np.nan)
    metric_text = (
        f"λ={lambda_value:.1f}; R²={r2_value:.3f}"
        if np.isfinite(lambda_value) and np.isfinite(r2_value)
        else "Observed binned profile"
    )
    ax.set_title(f"Distance profile\n{metric_text}")
    ax.spines[["top", "right"]].set_visible(False)

    polar = fig.add_subplot(grid[0, 3], projection="polar")
    candidate = anisotropy_report(
        adata, int(row[index_col]), min_points=int(anisotropy_min_points),
        matrix_source=matrix_source,
    )
    chosen = candidate if len(candidate["table"].dropna(subset=["lambda"])) >= 4 else None
    chosen_row = row
    if chosen is None:
        polar.text(.5, .5, "Insufficient sector-wise fits", transform=polar.transAxes,
                   ha="center", va="center")
        polar.set_axis_off()
    else:
        table = chosen["table"].dropna(subset=["lambda"]).sort_values("sector")
        angles = 2 * np.pi * table["sector"].to_numpy() / 8
        values = table["lambda"].to_numpy(float)
        angles, values = np.r_[angles, angles[0]], np.r_[values, values[0]]
        polar.plot(angles, values, color="#C83E3A", lw=1.6)
        polar.fill(angles, values, color="#C83E3A", alpha=.12)
        polar.plot(angles, np.full_like(values, np.nanmean(values)),
                   color=".4", ls="--", lw=1)
        polar.set_theta_zero_location("N"); polar.set_theta_direction(-1)
        polar.set_title(
            f"Directional influence range\nm/z {_format_mz(chosen_row['mz'])}; "
            f"CV={chosen['anisotropy_cv']:.2f}", y=1.13,
        )
    save_figure(
        fig, output_dir / f"{prefix}_interface_metrics_supplement",
        formats=("pdf", "svg", "png"), close=True,
    )


def anisotropy_report(adata, feature_index, *, n_sectors=8, min_points=200, matrix_source="X"):
    """Reproduce the frozen sector-wise Huber decay and anisotropy CV."""
    xy = np.asarray(adata.obsm["spatial"], float)
    distance = np.abs(adata.obs["sigma_d_signed"].to_numpy(float))
    intensity = _columns(_msi_matrix(adata, matrix_source), [int(feature_index)]).ravel().astype(float)
    center = np.nanmean(xy, axis=0)
    angle = np.mod(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]), 2 * np.pi)
    sectors = np.floor(angle / (2 * np.pi / n_sectors)).astype(int)
    rows = []
    for sector in range(n_sectors):
        use = (sectors == sector) & np.isfinite(distance) & np.isfinite(intensity)
        if use.sum() < min_points:
            rows.append({"sector": sector, "lambda": np.nan, "r2_logI": np.nan, "n": int(use.sum())})
            continue
        shifted = intensity[use] - np.nanmin(intensity[use])
        transformed = np.log1p(shifted)
        model = HuberRegressor().fit(distance[use, None], transformed)
        slope = float(model.coef_[0])
        lam = -1 / slope if slope < 0 else np.nan
        rows.append({"sector": sector, "lambda": lam,
                     "r2_logI": r2_score(transformed, model.predict(distance[use, None])),
                     "n": int(use.sum())})
    table = pd.DataFrame(rows)
    valid = table["lambda"].dropna()
    cv = float(valid.std(ddof=1) / (valid.mean() + 1e-8))
    return {"table": table, "anisotropy_cv": cv, "intensity": intensity,
            "sectors": sectors, "center": center}


def plot_anisotropy_polar(report, output_dir, *, prefix="HBC515"):
    """Save the observed sector ranges against their isotropic null."""
    table = report["table"].dropna(subset=["lambda"]).sort_values("sector")
    angles = 2 * np.pi * table["sector"].to_numpy() / 8
    values = table["lambda"].to_numpy(float)
    angles = np.r_[angles, angles[0]]
    values = np.r_[values, values[0]]
    fig, ax = plt.subplots(figsize=(4.4, 4.2), subplot_kw={"projection": "polar"},
                           constrained_layout=True)
    ax.plot(angles, values, color="#D62728", lw=1.8, label="Observed")
    ax.fill(angles, values, color="#D62728", alpha=.12)
    ax.plot(angles, np.full_like(values, np.nanmean(values)), color="0.4", ls="--",
            lw=1.2, label="Isotropic null")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title(f"Directional influence range λ\nCV = {report['anisotropy_cv']:.2f}",
                 y=1.16, pad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False)
    save_figure(fig, Path(output_dir) / f"{prefix}_anisotropy_polar",
                formats=("pdf", "svg", "png"), close=True)


def score_st_signatures(adata, gene_sets, *, random_state=0, already_log=None):
    """Calculate optional matched-ST signature scores with Scanpy.

    Returns a mapping from score name to arrays aligned to ``adata.obs_names``.
    Missing signatures are skipped when fewer than two genes are present.
    """
    import scanpy as sc

    rna = adata
    for key in ("feature_type", "type"):
        if key in rna.var:
            mask = rna.var[key].astype(str).to_numpy() == "ST"
            if mask.any():
                rna = rna[:, mask]
                break
    rna = rna.copy()
    if "feature_name" in rna.var:
        rna.var_names = rna.var["feature_name"].astype(str).str.upper().to_numpy()
    else:
        rna.var_names = rna.var_names.astype(str).str.replace(r"^ST_", "", regex=True).str.upper()
    rna.var_names_make_unique()
    if "normalized" in rna.layers:
        rna.X = rna.layers["normalized"].copy()
    else:
        maximum = rna.X.max() if issparse(rna.X) else np.nanmax(rna.X)
        inferred_log = maximum <= 50
        use_log = inferred_log if already_log is None else bool(already_log)
        if not use_log:
            sc.pp.normalize_total(rna, target_sum=1e4)
            sc.pp.log1p(rna)
    scores = {}
    for name, genes in gene_sets.items():
        selected = [str(g).upper() for g in genes if str(g).upper() in rna.var_names]
        if len(selected) < 2:
            continue
        sc.tl.score_genes(rna, selected, score_name=name, use_raw=False,
                          random_state=random_state)
        scores[name] = rna.obs[name].to_numpy(float)
    return scores


def stroma_near_far_statistics(
    adata,
    program_scores,
    *,
    signature_scores=None,
    spot_spacing=21.0,
    near_width=5,
    far_start=8,
    annotation_key="annotation",
):
    """Score programs and optional ST signatures on the stromal interface side.

    The defaults preserve the HBC515 analysis: 0 < distance <= 5 spot spacings
    is near and distance >= 8 spot spacings is far. The leading program is
    selected from metabolic programs only, never from RNA signatures.
    """
    distance = adata.obs["sigma_d_signed"].to_numpy(float)
    if annotation_key in adata.obs:
        labels = adata.obs[annotation_key].astype(str).str.lower()
        stroma = labels.str.contains("stroma").to_numpy()
    else:
        stroma = distance > 0
    near = stroma & np.isfinite(distance) & (distance > 0) & (
        distance <= near_width * spot_spacing
    )
    far = stroma & np.isfinite(distance) & (distance >= far_start * spot_spacing)
    combined = {f"cluster_{int(k)}_score_z": v for k, v in program_scores.items()}
    combined.update(dict(signature_scores or {}))
    rows = []
    for name, raw in combined.items():
        values = np.asarray(raw, float)
        near_values = values[near & np.isfinite(values)]
        far_values = values[far & np.isfinite(values)]
        if len(near_values) < 5 or len(far_values) < 5:
            p_value = np.nan
        else:
            p_value = mannwhitneyu(near_values, far_values, alternative="two-sided").pvalue
        near_mean = float(np.nanmean(near_values)) if len(near_values) else np.nan
        far_mean = float(np.nanmean(far_values)) if len(far_values) else np.nan
        rows.append({"score_col": name, "near_mean": near_mean, "far_mean": far_mean,
                     "near_minus_far": near_mean - far_mean, "p_value": p_value,
                     "n_near": len(near_values), "n_far": len(far_values)})
    table = pd.DataFrame(rows)
    table["fdr"] = multipletests(table["p_value"].fillna(1), method="fdr_bh")[1]
    programs = table[table["score_col"].str.startswith("cluster_")]
    leading = int(programs.loc[programs["near_minus_far"].idxmax(), "score_col"].split("_")[1])
    return table, leading


def plot_stroma_near_far_validation(table, output_dir, *, leading_program, prefix="HBC515"):
    """Plot automatic interface-program selection with optional ST support."""
    order = [
        "cluster_0_score_z", "cluster_1_score_z", "cluster_2_score_z", "cluster_3_score_z",
        "epithelial_score", "luminal_secretory_score", "stress_adaptation_score",
        "plasma_cell_score", "immune_leukocyte_score", "ecm_matrix_score",
        "caf_fibroblast_score", "emt_migration_score", "proliferation_score",
    ]
    labels = {
        "epithelial_score": "Epithelial", "luminal_secretory_score": "Luminal secretory",
        "stress_adaptation_score": "Stress adaptation", "plasma_cell_score": "Plasma cell",
        "immune_leukocyte_score": "Immune leukocyte", "ecm_matrix_score": "ECM matrix",
        "caf_fibroblast_score": "CAF / fibroblast", "emt_migration_score": "EMT / migration",
        "proliferation_score": "Proliferation",
    }
    plot = table.set_index("score_col").reindex(order).dropna(subset=["near_minus_far"]).reset_index()
    plot["label"] = [labels.get(x, x.replace("_score_z", "").replace("cluster_", "Cluster "))
                     for x in plot["score_col"]]
    highlight = f"cluster_{int(leading_program)}_score_z"
    colors = ["#C83E3A" if x == highlight else
              ("#91A9C6" if x.startswith("cluster_") else "#9A9A9A")
              for x in plot["score_col"]]
    y = np.arange(len(plot))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    values = plot["near_minus_far"].to_numpy(float)
    ax.axvline(0, color="black", lw=.9)
    ax.barh(y, values, color=colors, height=.62)
    ax.scatter(values, y, c=colors, edgecolor="0.3", linewidth=.4, s=25, zorder=3)
    for yy, value, q in zip(y, values, plot["fdr"]):
        stars = "****" if q < 1e-4 else "***" if q < 1e-3 else "**" if q < .01 else "*" if q < .05 else ""
        if stars:
            ax.annotate(stars, (value, yy), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom")
    ax.set_yticks(y, plot["label"])
    for tick, name in zip(ax.get_yticklabels(), plot["score_col"]):
        if name == highlight:
            tick.set(color="#C83E3A", fontweight="bold")
    row = plot[plot["score_col"] == highlight].iloc[0]
    ax.annotate("dominant interface-associated\nmetabolic program",
                (row["near_minus_far"], y[plot.index[plot["score_col"] == highlight][0]]),
                xytext=(14, -2), textcoords="offset points", va="center", color="#C83E3A")
    ax.set_xlabel("Stroma-side interface enrichment\nΔ mean score, stroma-near − stroma-far")
    ax.set_title("SIGMA identifies a stromal-side interface metabolic program")
    ax.grid(axis="x", color="0.92", lw=.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, Path(output_dir) / f"{prefix}_stroma_side_near_far_validation",
                formats=("pdf", "svg", "png"), close=True)


def plot_program_selection_diagnostics(table, output_dir, *, prefix="sigma"):
    """Plot near-versus-far evidence against feature-level boundary support."""
    required = {
        "program", "near_minus_far", "median_feature_interface_score", "selected",
    }
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Missing selection-diagnostic columns: {sorted(missing)}")
    q = table.copy()
    fig, ax = plt.subplots(figsize=(3.35, 2.75), constrained_layout=True)
    colors = np.where(q["selected"], "#D9534F", "#91A8C4")
    ax.scatter(q["near_minus_far"], q["median_feature_interface_score"],
               c=colors, s=np.where(q["selected"], 42, 28), zorder=3)
    midpoint = float(np.nanmean([
        q["near_minus_far"].min(), q["near_minus_far"].max()
    ]))
    for row in q.itertuples():
        left = row.near_minus_far > midpoint
        ax.annotate(
            f"P{row.program}",
            (row.near_minus_far, row.median_feature_interface_score),
            xytext=(-5, 4) if left else (5, 4), textcoords="offset points",
            ha="right" if left else "left", va="bottom", clip_on=False,
            color="#D9534F" if row.selected else ".25",
            fontweight="bold" if row.selected else "normal", fontsize=7,
        )
    ax.set_xlabel("Near–far enrichment")
    ax.set_ylabel("Median feature interface score")
    ax.set_title("Interface-program selection")
    ax.margins(x=.18, y=.18)
    save_figure(
        fig, Path(output_dir) / f"{prefix}_program_selection_diagnostics",
        formats=("pdf", "svg", "png"), close=True,
    )
