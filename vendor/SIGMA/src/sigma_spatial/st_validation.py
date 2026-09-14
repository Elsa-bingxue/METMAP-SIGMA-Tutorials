"""Independent spatial-transcriptomic validation of SIGMA programs."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from .plotting import save_figure
from .selection import side_near_far_statistics


PD_SIGNATURES = {
    "msn_neuron_score": ["PPP1R1B", "BCL11B", "FOXP1", "FOXP2", "MEIS2", "GPR88", "RGS9", "PDE10A"],
    "msn_d1_score": ["DRD1", "TAC1", "PDYN", "ISL1", "EBF1", "SLC35D3", "CHRM4"],
    "msn_d2_score": ["DRD2", "PENK", "ADORA2A", "GPR6", "OPRM1", "RGS9", "PDE10A"],
    "dopamine_signaling_score": ["DRD1", "DRD2", "DRD3", "DRD5", "PPP1R1B", "PDE10A", "RGS9", "SLC6A3", "TH", "DDC", "MAOA", "MAOB", "COMT"],
    "synaptic_score": ["SYN1", "SYN2", "SNAP25", "SYT1", "GAD1", "GAD2", "DLG4", "CAMK2A"],
    "astrocyte_score": ["GFAP", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "CLU", "VIM"],
    "microglia_score": ["CX3CR1", "P2RY12", "TMEM119", "AIF1", "C1QA", "C1QB", "C1QC", "TYROBP", "LST1", "CTSS", "TREM2", "APOE"],
    "oligodendrocyte_myelin_score": ["MBP", "PLP1", "MOG", "MAG", "MOBP", "CNP", "CLDN11", "OLIG1", "OLIG2"],
    "opc_score": ["PDGFRA", "CSPG4", "VCAN", "OLIG1", "OLIG2", "SOX10"],
    "inflammation_score": ["IL1B", "TNF", "IL6", "CCL2", "CCL3", "CCL4", "NFKBIA", "STAT1", "IRF1", "ISG15"],
    "oxidative_stress_score": ["HMOX1", "NQO1", "SOD1", "SOD2", "GPX1", "GPX4", "PRDX1", "PRDX2", "TXN", "GCLC", "GCLM"],
    "mitochondrial_respiration_score": ["NDUFA1", "NDUFA2", "NDUFA4", "NDUFB8", "SDHA", "SDHB", "UQCRC1", "UQCRC2", "COX4I1", "COX5A", "ATP5F1A", "ATP5F1B"],
    "cellular_stress_score": ["FOS", "JUN", "JUNB", "ATF3", "DUSP1", "DUSP4", "HSPA1A", "HSPA1B", "DNAJB1", "DDIT3"],
}


def program_signature_association(program_scores, signature_scores):
    """Compute program-by-signature Spearman associations with BH FDR."""
    rows = []
    for program, program_values in program_scores.items():
        x = np.asarray(program_values, float)
        for signature, signature_values in signature_scores.items():
            y = np.asarray(signature_values, float)
            if x.shape != y.shape:
                raise ValueError(f"Length mismatch for program {program!r} and {signature!r}")
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 5:
                correlation = p_value = np.nan
            else:
                result = spearmanr(x[valid], y[valid])
                coefficient = getattr(result, "statistic", result.correlation)
                correlation, p_value = float(coefficient), float(result.pvalue)
            rows.append({"program": program, "signature": signature,
                         "correlation": correlation, "p_value": p_value,
                         "n_spots": int(valid.sum())})
    table = pd.DataFrame(rows)
    table["fdr"] = 1.0
    valid = table.p_value.notna()
    if valid.any():
        table.loc[valid, "fdr"] = multipletests(table.loc[valid, "p_value"], method="fdr_bh")[1]
    return table


def side_near_far_signature_statistics(
    signature_scores, signed_distance, *, side_names=None,
    near_quantile=.20, far_quantile=.80, min_group_size=10,
):
    """Apply the generic two-sided near/far test to RNA signatures."""
    table = side_near_far_statistics(
        signature_scores, signed_distance, side_names=side_names,
        near_quantile=near_quantile, far_quantile=far_quantile,
        min_group_size=min_group_size,
    )
    return table.rename(columns={"program": "signature"})


def record_st_validation_provenance(
    adata, *, annotation_signatures=(), validation_signatures=(),
    allow_overlap=False, method="spearman_and_side_near_far",
):
    """Record ST validation inputs and reject silent circular validation."""
    annotation = tuple(map(str, annotation_signatures))
    validation = tuple(map(str, validation_signatures))
    overlap = sorted(set(annotation) & set(validation))
    if overlap and not allow_overlap:
        raise ValueError(f"Annotation and validation signatures overlap: {overlap}")
    payload = {
        "schema_version": 1, "method": method,
        "annotation_signatures": list(annotation),
        "validation_signatures": list(validation),
        "overlap": overlap, "independent_validation": not bool(overlap),
    }
    adata.uns["sigma_st_validation_provenance"] = payload
    return payload


def plot_program_signature_heatmap(table, output_path, *, title="ST validation of metabolic programs"):
    matrix = table.pivot(index="signature", columns="program", values="correlation")
    fig, ax = plt.subplots(figsize=(max(6.4, 1.05 * matrix.shape[1] + 2.8),
                                    max(4.4, .34 * matrix.shape[0] + 1.8)),
                           constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [f"Program {x}" for x in matrix.columns],
                  rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(matrix.shape[0]), [str(x).replace("_score", "").replace("_", " ") for x in matrix.index])
    lookup = table.set_index(["signature", "program"])
    for i, signature in enumerate(matrix.index):
        for j, program in enumerate(matrix.columns):
            q = lookup.loc[(signature, program), "fdr"]
            stars = "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""
            if stars:
                ax.text(j, i, stars, ha="center", va="center", fontsize=7.5)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Metabolic program", fontsize=10)
    ax.set_ylabel("RNA signature", fontsize=10)
    ax.tick_params(axis="both", labelsize=8.5)
    colorbar = fig.colorbar(image, ax=ax, label="Spearman correlation",
                           fraction=.045, pad=.035)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Spearman correlation", fontsize=9)
    save_figure(fig, Path(output_path), formats=("pdf", "png"), close=True)


def plot_signature_enrichment_lollipop(
    table, output_path, *, title="Interface-side ST enrichment", top_n_per_side=None,
    fdr_threshold=.05,
):
    """Plot informative near-versus-far RNA signatures without changing statistics.

    ``top_n_per_side`` only filters the displayed rows. The complete statistics remain
    available in the table returned by the analysis workflow.
    """
    plot = table.copy()
    if fdr_threshold is not None and "fdr" in plot:
        plot = plot.loc[plot["fdr"] < fdr_threshold]
    if top_n_per_side is not None:
        plot = (
            plot.assign(_magnitude=plot["near_minus_far"].abs())
            .sort_values(["side_name", "_magnitude"], ascending=[True, False])
            .groupby("side_name", sort=False, as_index=False)
            .head(int(top_n_per_side))
            .drop(columns="_magnitude")
        )
    plot = plot.sort_values(["side_name", "near_minus_far"]).reset_index(drop=True)
    if plot.empty:
        raise ValueError("No signature enrichment rows passed the plotting filters")
    labels = plot.signature.str.replace("_score", "", regex=False).str.replace("_", " ", regex=False)
    colors = np.where(plot.orientation.eq("negative"), "#3572A5", "#D55E00")
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(7.2, max(3.6, .34 * len(plot) + 1.7)), constrained_layout=True)
    ax.axvline(0, color="black", lw=.8)
    ax.hlines(y, 0, plot.near_minus_far, color=colors, lw=1.5)
    ax.scatter(plot.near_minus_far, y, c=colors, s=30, zorder=3)
    side_labels = plot.side_name.replace({
        "CI / dopamine-low": "CI side",
        "Cd / dopamine-high": "Cd side",
    })
    ax.set_yticks(y, [f"{label} ({side})" for label, side in zip(labels, side_labels)])
    ax.set_xlabel("Near − far signature score", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color=".92")
    save_figure(fig, Path(output_path), formats=("pdf", "png"), close=True)
