"""SIGMA: spatial interface analysis for spatial metabolomics and transcriptomics."""
from .preprocessing import get_msi_matrix, resolve_sm_matrix, msi_to_embedding
from .boundary import (
    build_boundary_from_binary_mask,
    build_boundary_from_field,
    refine_binary_mask_connectivity,
    signed_distance_from_boundary_points,
)
from .io import load_spatial_metabolomics, load_spatial_transcriptomics
from .input_assessment import InputAssessment, assess_input, inspect_input
from .lambda_profile import (
    LambdaProfilePreset, get_lambda_profile_preset, rank_lambda_profiles,
)
from .validation import validate_sigma_input, validate_sigma_output
from .anchors import (
    AnchorSpec, LEGACY_ANCHOR_SPECS, HIGH_CONFIDENCE_ANCHOR_SPECS,
    CORE_ANCHORED_ANCHOR_SPECS,
    CORE_VS_REST_ANCHOR_SPECS,
    NAMED_REGION_ANCHOR_SPECS,
    build_anchors, anchors_from_preset, summarize_anchors,
)
from .workflows import (
    WorkflowConfig, VALID_WORKFLOWS, DATASET_WORKFLOWS,
    get_workflow_config, record_workflow_provenance,
)

__version__ = "0.3.1+bc515.original"
__all__ = [
    "SIGMA", "run_sigma", "run_analysis", "run_downstream_analysis",
    "AnalysisResult", "DownstreamAnalysis",
    "run_sigma_weak_anchor", "anchors_from_clusters", "load_spatial_metabolomics",
    "load_spatial_transcriptomics", "InputAssessment", "assess_input", "inspect_input",
    "LambdaProfilePreset", "get_lambda_profile_preset", "rank_lambda_profiles",
    "validate_sigma_input", "validate_sigma_output",
    "get_msi_matrix", "resolve_sm_matrix", "msi_to_embedding",
    "build_boundary_from_binary_mask", "build_boundary_from_field",
    "refine_binary_mask_connectivity",
    "signed_distance_from_boundary_points",
    "AnchorSpec", "LEGACY_ANCHOR_SPECS", "HIGH_CONFIDENCE_ANCHOR_SPECS",
    "CORE_ANCHORED_ANCHOR_SPECS",
    "CORE_VS_REST_ANCHOR_SPECS",
    "NAMED_REGION_ANCHOR_SPECS",
    "build_anchors", "anchors_from_preset", "summarize_anchors",
    "WorkflowConfig", "VALID_WORKFLOWS", "DATASET_WORKFLOWS",
    "get_workflow_config", "record_workflow_provenance",
    "program_report_data", "representative_metabolites", "disease_profile_representatives", "plot_program_report",
    "spatially_coherent_representatives",
    "plot_bidirectional_program_enrichment",
    "plot_program_selection_diagnostics",
    "anisotropy_report", "plot_anisotropy_polar",
    "score_st_signatures", "stroma_near_far_statistics", "plot_stroma_near_far_validation",
    "side_near_far_statistics", "select_bidirectional_programs", "select_leading_program",
    "program_signature_association", "side_near_far_signature_statistics",
    "record_st_validation_provenance", "plot_program_signature_heatmap",
    "plot_signature_enrichment_lollipop",
    "PD_SIGNATURES",
    "DatasetAssessment", "assess_dataset", "prepare_anchors",
    "ProgramAnalysis", "discover_programs",
    "rank_interface_metabolites", "summarize_interface_programs",
    "select_representative_metabolites",
    "ReferencePreset", "REFERENCE_PRESETS", "list_reference_datasets",
    "get_reference_preset", "run_reference_analysis",
    "manuscript_manifest", "write_manuscript_manifest", "reproduce_manuscript_figures",
    "AnalysisConfig", "load_analysis_config", "run_analysis_config",
]


def __getattr__(name):
    """Keep light-weight utilities importable without the optional torch stack."""
    if name == "SIGMA":
        from .pipeline import SIGMA
        return SIGMA
    if name in {
        "run_sigma", "run_analysis", "run_downstream_analysis",
        "AnalysisResult", "DownstreamAnalysis",
    }:
        from .api import run_sigma
        from . import api
        return getattr(api, name)
    if name == "run_sigma_weak_anchor":
        from .weak_anchor import run_sigma_weak_anchor
        return run_sigma_weak_anchor
    if name == "anchors_from_clusters":
        from .weak_anchor import anchors_from_clusters
        return anchors_from_clusters
    if name in {
        "program_report_data", "representative_metabolites", "disease_profile_representatives", "spatially_coherent_representatives",
        "plot_program_report",
        "plot_bidirectional_program_enrichment",
        "plot_program_selection_diagnostics",
        "anisotropy_report", "plot_anisotropy_polar",
        "score_st_signatures", "stroma_near_far_statistics", "plot_stroma_near_far_validation",
    }:
        from . import reporting
        return getattr(reporting, name)
    if name in {"side_near_far_statistics", "select_bidirectional_programs", "select_leading_program"}:
        from . import selection
        return getattr(selection, name)
    if name in {
        "manuscript_manifest", "write_manuscript_manifest", "reproduce_manuscript_figures",
    }:
        from . import manuscript
        return getattr(manuscript, name)
    if name in {"AnalysisConfig", "load_analysis_config", "run_analysis_config"}:
        from . import config
        return getattr(config, name)
    if name in {
        "program_signature_association", "side_near_far_signature_statistics",
        "record_st_validation_provenance", "plot_program_signature_heatmap",
        "plot_signature_enrichment_lollipop",
        "PD_SIGNATURES",
    }:
        from . import st_validation
        return getattr(st_validation, name)
    if name in {"DatasetAssessment", "assess_dataset", "prepare_anchors"}:
        from . import assessment
        return getattr(assessment, name)
    if name in {"ProgramAnalysis", "discover_programs"}:
        from . import programs
        return getattr(programs, name)
    if name in {
        "rank_interface_metabolites", "summarize_interface_programs",
        "select_representative_metabolites",
    }:
        from . import interface_selection
        return getattr(interface_selection, name)
    if name in {
        "ReferencePreset", "REFERENCE_PRESETS", "list_reference_datasets",
        "get_reference_preset", "run_reference_analysis",
    }:
        from . import reference
        return getattr(reference, name)
    raise AttributeError(name)
