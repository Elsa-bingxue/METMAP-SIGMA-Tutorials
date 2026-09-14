"""High-level public entry points for SIGMA."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .pipeline import SIGMA
from .validation import validate_sigma_input, validate_sigma_output


@dataclass
class DownstreamAnalysis:
    """Artifacts returned by :func:`run_downstream_analysis`."""

    programs: object
    interface_statistics: object
    selected_programs: object
    leading_program: object
    representatives: object
    st_associations: object | None = None


@dataclass
class AnalysisResult:
    """Combined core and downstream result returned by :func:`run_analysis`."""

    adata: object
    downstream: DownstreamAnalysis
    workflow: str
    assessment: object | None
    output_dir: Path


def run_sigma(
    adata,
    *,
    anchor_key="sigma_anchor",
    representation_key="X_harmony",
    spatial_key="spatial",
    copy=True,
    n_components=64,
    n_neighbors=15,
    random_state=0,
    device=None,
    **fit_kwargs,
):
    """Run the validated end-to-end SIGMA workflow on an AnnData object.

    Anchors must be encoded as 1 (tumor), 0 (non-tumor), and NaN (unknown).
    The current validated model also requires an auxiliary representation such
    as matched RNA PCs in ``adata.obsm[representation_key]``.
    """
    validate_sigma_input(
        adata, anchor_key=anchor_key, representation_key=representation_key, spatial_key=spatial_key
    )
    result = adata.copy() if copy else adata
    if spatial_key != "spatial":
        result.obsm["spatial"] = result.obsm[spatial_key].copy()
    temporary_key = "_sigma_public_anchor"
    anchor = result.obs[anchor_key].to_numpy()
    labels = ["Tumor" if value == 1 else "Stroma" if value == 0 else None for value in anchor]
    result.obs[temporary_key] = labels
    estimator = SIGMA(n_components=n_components, k=n_neighbors, seed=random_state, device=device)
    estimator.fit(
        result,
        annotation_key=temporary_key,
        tumor_label="Tumor",
        stroma_label="Stroma",
        rna_key=representation_key,
        **fit_kwargs,
    )
    del result.obs[temporary_key]
    validate_sigma_output(result)
    result.uns.setdefault("sigma", {})["public_api"] = {
        "anchor_key": anchor_key,
        "representation_key": representation_key,
        "spatial_key": spatial_key,
    }
    return result


def run_downstream_analysis(
    adata,
    ranking=None,
    output_dir=None,
    *,
    prefix="sigma",
    signature_scores=None,
    annotation_signatures=(),
    side_names=None,
    selection_mode="standardized",
    program_selection="auto",
    workflow="direct_pathology",
    selection_orientation=None,
    ranking_col=None,
    index_col=None,
    top_k=300,
    n_programs=4,
    n_bins=40,
    min_bin_points=5,
    near_quantile=.20,
    far_quantile=.80,
    min_group_size=10,
    fdr_max=.05,
    effect_min=0.0,
    leading_program=None,
    representative_n=5,
    anisotropy_min_points=200,
    anisotropy_min_valid_sectors=6,
    anisotropy_min_r2=.05,
    matrix_source="X",
    interface_label="SIGMA interface",
    association_label="interface-associated",
    annotation_key="annotation",
    annotation_title="Annotation",
    random_state=0,
    deterministic=True,
    program_assignments=None,
    reference_representatives=None,
    report_level="standard",
):
    """Discover, select, validate, and plot SIGMA metabolic programs.

    This is a downstream reporting workflow. It never refits SIGMA and never
    changes graph construction, probability normalization, model parameters,
    boundary thresholds, or the signed-distance definition.
    """
    valid_report_levels = {"none", "standard", "complete", "manuscript"}
    if report_level not in valid_report_levels:
        raise ValueError(f"report_level must be one of {sorted(valid_report_levels)}")
    from .programs import ProgramAnalysis, discover_programs
    from .utils import set_seed
    randomness = set_seed(random_state, deterministic=deterministic)
    from .reporting import (
        anisotropy_report, plot_anisotropy_polar,
        plot_bidirectional_program_enrichment, plot_program_report,
        plot_program_selection_diagnostics,
    )
    from .selection import (
        side_near_far_statistics, select_bidirectional_programs,
        select_leading_program,
    )
    from .st_validation import (
        plot_program_signature_heatmap, plot_signature_enrichment_lollipop,
        program_signature_association, record_st_validation_provenance,
        side_near_far_signature_statistics,
    )

    if output_dir is None:
        raise ValueError("output_dir is required")
    from .interface_selection import (
        rank_interface_metabolites, select_representative_metabolites,
        summarize_interface_programs,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from .preprocessing import resolve_sm_matrix
    sm_matrix, _ = resolve_sm_matrix(adata, matrix_source)
    feature_ranking = rank_interface_metabolites(adata, matrix_source=matrix_source)
    feature_ranking.to_csv(output / f"{prefix}_standardized_feature_ranking.csv", index=False)
    if ranking is None:
        if selection_mode == "lambda_profile":
            from .lambda_profile import get_lambda_profile_preset, rank_lambda_profiles
            preset = get_lambda_profile_preset(workflow)
            ranking = rank_lambda_profiles(
                adata, matrix_source=matrix_source, var_top=preset.var_top,
                min_valid=preset.min_valid,
                min_detect_rate=preset.min_detect_rate, min_r2=preset.min_r2,
                min_enrichment=preset.min_enrichment,
                already_log=preset.already_log, ranking_col=preset.ranking_col,
            )
            ranking.to_csv(output / f"{prefix}_lambda_profile_ranking.csv", index=False)
            ranking_col = preset.ranking_col
            top_k = min(int(top_k), int(preset.top_k))
            (output / f"{prefix}_lambda_profile_preset.json").write_text(
                json.dumps(preset.to_dict(), indent=2, sort_keys=True)
            )
            index_col = "j"
        else:
            ranking = feature_ranking
            selection_mode = "standardized"
            ranking_col = "interface_score"
            index_col = "j"
    if program_assignments is None:
        analysis = discover_programs(
            adata, ranking, selection_mode=selection_mode, ranking_col=ranking_col,
            index_col=index_col, top_k=top_k, n_programs=n_programs, n_bins=n_bins,
            min_bin_points=min_bin_points, matrix=sm_matrix,
            random_state=random_state,
        )
    else:
        import pandas as pd
        from .reporting import program_report_data
        assignments = (
            pd.read_csv(program_assignments)
            if isinstance(program_assignments, (str, Path))
            else program_assignments.copy()
        )
        frozen = program_report_data(
            adata, assignments, n_bins=n_bins,
            min_bin_points=min_bin_points, matrix_source=matrix_source,
        )
        centroid_rows = []
        for program, profile in frozen["profiles"].items():
            count = int((assignments["cluster"].astype(int) == int(program)).sum())
            centroid_rows.extend({
                "program": int(program), "signed_distance": float(distance),
                "centroid_z": float(value), "n_features": count,
            } for distance, value in zip(frozen["centers"], profile))
        analysis = ProgramAnalysis(
            assignments=assignments.reset_index(drop=True),
            centroid_profiles=pd.DataFrame(centroid_rows),
            scores=frozen["scores"],
            config={
                "schema_version": 1, "selection_mode": "reference",
                "program_source": "frozen_assignments",
                "random_state": int(random_state),
                "clustering_performed": False,
                "n_programs": int(assignments["cluster"].nunique()),
                "n_clustered_features": int(len(assignments)),
            },
        )
    analysis.save(output, prefix=prefix)
    distance = adata.obs["sigma_d_signed"].to_numpy(float)
    statistics = side_near_far_statistics(
        analysis.scores, distance, side_names=side_names,
        near_quantile=near_quantile, far_quantile=far_quantile,
        min_group_size=min_group_size,
    )
    selected = select_bidirectional_programs(
        statistics, fdr_max=fdr_max, effect_min=effect_min
    )
    candidates = selected if len(selected) else statistics
    if selection_orientation is None and workflow in {
        "transferred_pathology", "region_defined_disease",
    }:
        # These workflows target programs extending into the positive semantic
        # side recorded in workflow provenance.
        selection_orientation = "positive"
    if selection_orientation is not None:
        oriented = candidates[candidates["orientation"] == selection_orientation]
        if len(oriented):
            candidates = oriented
    finite = candidates[np.isfinite(candidates["near_minus_far"])]
    if finite.empty:
        raise ValueError("No finite interface program enrichment was available")
    program_summary, assignment_qc = summarize_interface_programs(
        analysis.assignments, feature_ranking, statistics
    )
    program_summary.to_csv(output / f"{prefix}_program_interface_summary.csv", index=False)
    assignment_qc.to_csv(output / f"{prefix}_program_assignment_qc.csv", index=False)
    valid_program_selection = {"auto", "near_far", "boundary_localized", "program_interface"}
    if program_selection not in valid_program_selection:
        raise ValueError(
            f"program_selection must be one of {sorted(valid_program_selection)}"
        )
    resolved_program_selection = program_selection
    if resolved_program_selection == "auto":
        if selection_mode == "standardized":
            resolved_program_selection = "program_interface"
        elif workflow == "region_defined_disease":
            resolved_program_selection = "boundary_localized"
        else:
            resolved_program_selection = "near_far"
    if resolved_program_selection == "program_interface":
        automatic_program = program_summary.iloc[0]["program"]
        selection_diagnostics = program_summary.rename(columns={
            "median_interface_score": "median_feature_interface_score",
        }).copy()
        selection_diagnostics["passes_stage1"] = (
            selection_diagnostics["fdr"].le(float(fdr_max))
            & selection_diagnostics["near_minus_far"].gt(float(effect_min))
        )
        selection_diagnostics["selected"] = selection_diagnostics["program"].eq(automatic_program)
        selection_diagnostics["selection_strategy"] = resolved_program_selection
    else:
        automatic_program, selection_diagnostics = select_leading_program(
            statistics, analysis.assignments,
            orientation=selection_orientation,
            strategy=resolved_program_selection,
            fdr_max=fdr_max, effect_min=effect_min,
        )
    selection_diagnostics.to_csv(
        output / f"{prefix}_program_selection_metrics.csv", index=False
    )
    if report_level != "none" and {
        "near_minus_far", "median_feature_interface_score", "selected",
    }.issubset(selection_diagnostics.columns):
        plot_program_selection_diagnostics(
            selection_diagnostics, output, prefix=prefix
        )
    leading_selection = (
        f"automatic_{resolved_program_selection}_{selection_orientation}_side"
        if selection_orientation else f"automatic_{resolved_program_selection}"
    )
    if leading_program is None:
        leading_program = automatic_program
    else:
        available = set(analysis.assignments["cluster"])
        if leading_program not in available:
            raise ValueError(f"leading_program {leading_program!r} is not in {sorted(available)}")
        leading_selection = "explicit_reference_override"
    statistics.to_csv(output / f"{prefix}_interface_program_statistics.csv", index=False)
    selected.to_csv(output / f"{prefix}_selected_interface_programs.csv", index=False)
    if report_level != "none":
        plot_bidirectional_program_enrichment(
            statistics, output, leading_program=leading_program, prefix=prefix
        )
    report = {
        "assignments": analysis.assignments,
        "centers": analysis.centroid_profiles.query("program == @leading_program")["signed_distance"].to_numpy(),
        "profiles": {
            program: group["centroid_z"].to_numpy()
            for program, group in analysis.centroid_profiles.groupby("program", sort=True)
        },
        "scores": analysis.scores,
    }
    if reference_representatives is not None:
        import pandas as pd
        standardized_representatives = (
            pd.read_csv(reference_representatives)
            if isinstance(reference_representatives, (str, Path))
            else reference_representatives.copy()
        )
    else:
        if selection_mode == "standardized":
            standardized_representatives = select_representative_metabolites(
                analysis.assignments, feature_ranking, program=leading_program,
                n=int(representative_n),
            )
        elif selection_mode == "lambda_profile" and workflow == "transferred_pathology":
            # Within the selected non-tumor-side HCC program, display features
            # with the strongest combined decay and interface enrichment.
            from .reporting import representative_metabolites
            standardized_representatives = representative_metabolites(
                analysis.assignments, cluster=leading_program,
                n=int(representative_n), score="interface_score",
            )
        elif selection_mode == "lambda_profile" and workflow == "region_defined_disease":
            # Preserve the original HPD two-stage analysis: Ward clustering of
            # signed-distance profiles first, then distance-fit/enrichment
            # ranking only within the selected disease-interface program.
            from .reporting import disease_profile_representatives
            standardized_representatives = disease_profile_representatives(
                analysis.assignments, cluster=leading_program,
                n=int(representative_n),
            )
        else:
            standardized_representatives = None
    if report_level == "none":
        from .reporting import representative_metabolites
        representatives = (
            standardized_representatives.head(int(representative_n)).copy()
            if standardized_representatives is not None else
            representative_metabolites(
                analysis.assignments, cluster=leading_program,
                n=int(representative_n), score="interface_score",
            )
        )
    else:
        representatives = plot_program_report(
            adata, report, output, boundary_cluster=leading_program, prefix=prefix,
            representative_n=int(representative_n), representatives=standardized_representatives,
            matrix_source=matrix_source,
            interface_label=interface_label,
            association_label=association_label,
        )
    representatives.to_csv(output / f"{prefix}_representative_metabolites.csv", index=False)
    if report_level != "none":
        from .reporting import plot_integrated_program_summary
        plot_integrated_program_summary(
            adata, report, statistics, representatives, output,
            leading_program=leading_program, prefix=prefix,
            matrix_source=matrix_source, interface_label=interface_label,
            annotation_key=annotation_key, annotation_title=annotation_title,
        )
    if report_level in {"complete", "manuscript"}:
        from .reporting import plot_interface_metric_supplement
        plot_interface_metric_supplement(
            adata, analysis.assignments, representatives, output,
            prefix=prefix, matrix_source=matrix_source,
            anisotropy_min_points=max(50, int(anisotropy_min_points // 2)),
        )

    anisotropy_status = "not_requested"
    if report_level in {"complete", "manuscript"} and len(representatives):
        anisotropy_status = "not_run"
        for _, representative in representatives.iterrows():
            representative_index = (
                "sigma_sm_index" if "sigma_sm_index" in representative.index else
                "j_sm" if "j_sm" in representative.index else
                "j_plot" if "j_plot" in representative.index else "j"
            )
            feature_index = int(representative[representative_index])
            try:
                candidate = anisotropy_report(
                    adata, feature_index, min_points=int(anisotropy_min_points),
                    matrix_source=matrix_source,
                )
            except (ValueError, IndexError):
                continue
            valid_sector = candidate["table"]["lambda"].notna() & (
                candidate["table"]["r2_logI"] >= float(anisotropy_min_r2)
            )
            if int(valid_sector.sum()) < int(anisotropy_min_valid_sectors):
                continue
            anisotropy = candidate
            anisotropy["table"].to_csv(output / f"{prefix}_anisotropy.csv", index=False)
            plot_anisotropy_polar(anisotropy, output, prefix=prefix)
            anisotropy_status = f"written for m/z {representative['mz']}"
            break
        else:
            anisotropy_status = "skipped: no representative passed anisotropy QC"

    st_associations = None
    if signature_scores:
        validation_signatures = tuple(map(str, signature_scores))
        record_st_validation_provenance(
            adata, annotation_signatures=annotation_signatures,
            validation_signatures=validation_signatures,
        )
        st_associations = program_signature_association(analysis.scores, signature_scores)
        st_associations.to_csv(output / f"{prefix}_st_program_associations.csv", index=False)
        if report_level != "none":
            plot_program_signature_heatmap(
                st_associations, output / f"{prefix}_st_program_heatmap"
            )
        signature_table = side_near_far_signature_statistics(
            signature_scores, distance, side_names=side_names,
            near_quantile=near_quantile, far_quantile=far_quantile,
            min_group_size=min_group_size,
        )
        signature_table.to_csv(output / f"{prefix}_st_interface_statistics.csv", index=False)
        if report_level != "none":
            plot_signature_enrichment_lollipop(
                signature_table, output / f"{prefix}_st_interface_lollipop"
            )

    summary = {
        "schema_version": 1,
        "leading_program": int(leading_program) if isinstance(leading_program, (int, np.integer)) else str(leading_program),
        "automatic_leading_program": int(automatic_program) if isinstance(automatic_program, (int, np.integer)) else str(automatic_program),
        "leading_program_selection": leading_selection,
        "selection_mode": selection_mode,
        "program_selection": resolved_program_selection,
        "matrix_source": matrix_source,
        "fdr_max": float(fdr_max), "effect_min": float(effect_min),
        "near_quantile": float(near_quantile), "far_quantile": float(far_quantile),
        "n_selected_program_side_pairs": int(len(selected)),
        "st_validation": bool(signature_scores),
        "anisotropy": anisotropy_status,
        "report_level": report_level,
        "randomness": randomness,
    }
    (output / f"{prefix}_downstream_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return DownstreamAnalysis(
        analysis, statistics, selected, leading_program, representatives,
        st_associations,
    )


def run_analysis(
    adata,
    output_dir,
    *,
    prefix="sigma",
    workflow="auto",
    evidence=None,
    report_level="standard",
    selection_mode="lambda_profile",
    program_selection="auto",
    anchor_key="sigma_anchor",
    representation_key="X_harmony",
    spatial_key="spatial",
    matrix_source="X",
    signature_scores=None,
    random_state=0,
    copy=True,
    core_kwargs=None,
    downstream_kwargs=None,
):
    """Run the safest available SIGMA core and downstream workflow.

    ``workflow='auto'`` reads existing workflow provenance or uses explicit
    investigator-supplied ``evidence``. It never guesses pathology evidence
    from whichever route yields the strongest result.
    """
    from .assessment import assess_dataset
    from .workflows import VALID_WORKFLOWS

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assessment = None
    resolved_workflow = workflow
    if workflow == "auto":
        provenance = adata.uns.get("sigma_workflow_provenance", {})
        resolved_workflow = provenance.get("workflow")
        if resolved_workflow is None:
            if evidence is None:
                raise ValueError(
                    "workflow='auto' requires existing sigma_workflow_provenance "
                    "or explicit evidence (for example {'matched_st': True})."
                )
            assessment = assess_dataset(
                adata, annotation_key=None, evidence=evidence, spatial_key=spatial_key,
            )
            resolved_workflow = assessment.recommended_workflow
    if resolved_workflow not in VALID_WORKFLOWS:
        raise ValueError(f"workflow must be 'auto' or one of {VALID_WORKFLOWS}")

    has_sigma = all(key in adata.obs for key in (
        "sigma_region_probability", "sigma_boundary", "sigma_d_signed",
    ))
    core_options = dict(core_kwargs or {})
    if has_sigma:
        result = adata.copy() if copy else adata
        core_route = "precomputed_sigma"
    elif resolved_workflow == "transferred_pathology":
        from .weak_anchor import run_sigma_weak_anchor
        result = run_sigma_weak_anchor(
            adata, anchor_key=anchor_key, spatial_key=spatial_key, copy=copy,
            random_state=random_state, **core_options,
        )
        core_route = "run_sigma_weak_anchor"
    else:
        result = run_sigma(
            adata, anchor_key=anchor_key, representation_key=representation_key,
            spatial_key=spatial_key, copy=copy, random_state=random_state,
            **core_options,
        )
        core_route = "run_sigma"

    result.uns["sigma_workflow_provenance"] = {
        "schema_version": 1,
        "workflow": resolved_workflow,
        "evidence": dict(evidence or {}),
        "core_route": core_route,
        "random_state": int(random_state),
    }
    options = dict(downstream_kwargs or {})
    downstream = run_downstream_analysis(
        result, output_dir=output, prefix=prefix,
        signature_scores=signature_scores, selection_mode=selection_mode,
        program_selection=program_selection,
        workflow=resolved_workflow, matrix_source=matrix_source,
        report_level=report_level, random_state=random_state, **options,
    )
    run_config = {
        "schema_version": 1,
        "package_version": "0.3.1",
        "workflow": resolved_workflow,
        "core_route": core_route,
        "selection_mode": selection_mode,
        "program_selection": program_selection,
        "report_level": report_level,
        "matrix_source": matrix_source,
        "random_state": int(random_state),
        "evidence": dict(evidence or {}),
    }
    result.uns["sigma_analysis_run"] = run_config
    (output / f"{prefix}_run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True)
    )
    return AnalysisResult(result, downstream, resolved_workflow, assessment, output)
