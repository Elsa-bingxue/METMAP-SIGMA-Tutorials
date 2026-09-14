"""Standard anchor construction with auditable legacy dataset presets.

SIGMA always consumes the same numeric representation: 1 is a positive
anchor, 0 is a negative anchor, and NaN is unknown.  Dataset presets only
describe how existing biological labels are translated into that representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AnchorSpec:
    """Rules translating an annotation column to SIGMA anchors."""

    name: str
    source_column: str
    mode: str
    positive_labels: tuple[str, ...]
    negative_labels: tuple[str, ...] = ()
    uncertain_labels: tuple[str, ...] = ()
    negative_policy: str = "listed"
    description: str = ""


LEGACY_ANCHOR_SPECS = {
    "HLC091": AnchorSpec(
        "HLC091", "annotation", "direct_annotation", ("Tumor",),
        ("Normal cells", "Airway", "Blood vessel", "Lymphocytes"),
        description="Exact binary anchors used by the HLC091 reference notebook.",
    ),
    "HLC276": AnchorSpec(
        "HLC276", "annotation", "direct_annotation", ("Tumor",),
        ("Normal cells",),
        description="Exact binary anchors used by the HLC276 reference notebook.",
    ),
    "ccRCC_Y27T": AnchorSpec(
        "ccRCC_Y27T", "tumor_region_by_marker_in_GT", "marker_informed",
        ("tumor_region",), ("TME_or_mixed_region", "tumor_transition"),
        ("normal_or_mixed_region", "mixed_region"),
        description="Exact GT-region/ST-marker anchor filter in the ccRCC notebook.",
    ),
    "GBM": AnchorSpec(
        "GBM", "gbm_region_by_marker_in_GT", "marker_informed",
        ("tumor_core_like",), negative_policy="complement",
        description="Legacy GBM behavior: every non-core spot is a negative anchor.",
    ),
    "HCC_P1": AnchorSpec(
        "HCC_P1", "annotation", "cluster_inferred", ("tumor",), ("non_tumor",),
        description="SM-cluster/adjacent-section labels stored in the P1 input.",
    ),
    "HCC_P4": AnchorSpec(
        "HCC_P4", "annotation", "cluster_inferred", ("tumor",), ("non_tumor",),
        description="SM-cluster/adjacent-section labels stored in the P4 input.",
    ),
}


HIGH_CONFIDENCE_ANCHOR_SPECS = {
    **{key: value for key, value in LEGACY_ANCHOR_SPECS.items() if key not in {"ccRCC_Y27T", "GBM"}},
    "ccRCC_Y27T": AnchorSpec(
        "ccRCC_Y27T", "tumor_region_by_marker_in_GT", "core_anchored_marker_informed",
        ("tumor_region",), ("TME_or_mixed_region",),
        ("tumor_transition", "normal_or_mixed_region", "mixed_region", "unassigned"),
        description=(
            "Core-anchored ccRCC filter: confident tumor is positive, confident TME is "
            "negative, and transition/ambiguous regions remain unknown."
        ),
    ),
    "GBM": AnchorSpec(
        "GBM", "gbm_region_by_marker_in_GT", "core_anchored_marker_informed",
        ("tumor_core_like",), ("microenvironment_like", "neural_like"),
        ("invasive_margin_like", "mixed_transition", "unassigned"),
        description=(
            "Core-anchored GBM filter: confident core is positive, confident distal "
            "microenvironment/neural regions are negative, and margin/transition remain unknown."
        ),
    ),
}

# Explicit user-facing name for the corrected no-direct-annotation workflow.
# Keep ``high_confidence`` as an alias for backwards compatibility.
CORE_ANCHORED_ANCHOR_SPECS = {
    key: HIGH_CONFIDENCE_ANCHOR_SPECS[key] for key in ("ccRCC_Y27T", "GBM")
}


# Core-versus-rest is the selected main-analysis policy for datasets whose
# biological coordinate is explicitly defined by a marker/H&E-derived core.
# Every non-core region, including transition/mixed labels, is a negative.
CORE_VS_REST_ANCHOR_SPECS = {
    "ccRCC_Y27T": AnchorSpec(
        "ccRCC_Y27T", "tumor_region_by_marker_in_GT", "core_vs_rest",
        ("tumor_region",), negative_policy="complement",
        description="Selected ccRCC main analysis: tumor region versus every other region.",
    ),
    "GBM": AnchorSpec(
        "GBM", "gbm_region_by_marker_in_GT", "core_vs_rest",
        ("tumor_core_like",), negative_policy="complement",
        description="Selected GBM main analysis: tumor core versus every other region.",
    ),
}


# Parkinson's disease sections use an anatomical CI--Cd transition rather
# than a tumour--stroma boundary. Preserve the reference-notebook orientation:
# CI / dopamine-low is positive and Cd / dopamine-high is negative. ACB and
# unassigned regions are deliberately excluded from supervision.
NAMED_REGION_ANCHOR_SPECS = {
    name: AnchorSpec(
        name, "RegionLoupe_str", "named_regions",
        ("CI",), ("Cd",), ("ACB", "NA", "nan", "unk"),
        description=(
            "PD striatal interface: CI is the positive/dopamine-low anchor, "
            "Cd is the negative/dopamine-high anchor, and all other regions are unknown."
        ),
    )
    for name in ("HPD_A1", "HPD_B1", "HPD_C1")
}


def build_anchors(labels, spec: AnchorSpec):
    """Translate labels to 1/0/NaN without guessing unlisted categories."""
    values = np.asarray(labels).astype(str)
    positive = set(map(str, spec.positive_labels))
    negative = set(map(str, spec.negative_labels))
    uncertain = set(map(str, spec.uncertain_labels))
    overlap = (positive & negative) | (positive & uncertain) | (negative & uncertain)
    if overlap:
        raise ValueError(f"Anchor label sets must be disjoint; overlap={sorted(overlap)}")
    if spec.negative_policy not in {"listed", "complement"}:
        raise ValueError("negative_policy must be 'listed' or 'complement'")

    anchors = np.full(values.shape[0], np.nan, dtype=float)
    positive_mask = np.isin(values, list(positive))
    anchors[positive_mask] = 1.0
    if spec.negative_policy == "complement":
        anchors[~positive_mask] = 0.0
    else:
        anchors[np.isin(values, list(negative))] = 0.0
    anchors[np.isin(values, list(uncertain))] = np.nan
    if not np.any(anchors == 1) or not np.any(anchors == 0):
        raise ValueError(f"{spec.name} mapping must produce both positive and negative anchors")
    return anchors


def anchors_from_preset(adata, dataset, *, scheme="legacy", output_key="sigma_anchor"):
    """Add anchors from a frozen preset and record complete provenance."""
    collections = {
        "legacy": LEGACY_ANCHOR_SPECS,
        "high_confidence": HIGH_CONFIDENCE_ANCHOR_SPECS,
        "core_anchored": CORE_ANCHORED_ANCHOR_SPECS,
        "core_vs_rest": CORE_VS_REST_ANCHOR_SPECS,
        "named_regions": NAMED_REGION_ANCHOR_SPECS,
    }
    if scheme not in collections:
        raise ValueError(f"scheme must be one of {sorted(collections)}")
    if dataset not in collections[scheme]:
        raise KeyError(f"No {scheme!r} anchor preset for {dataset!r}")
    spec = collections[scheme][dataset]
    if spec.source_column not in adata.obs:
        raise KeyError(f"adata.obs[{spec.source_column!r}] is required for {dataset}")
    anchors = build_anchors(adata.obs[spec.source_column].to_numpy(), spec)
    adata.obs[output_key] = anchors
    counts = summarize_anchors(anchors)
    adata.uns["sigma_anchor_provenance"] = {
        **asdict(spec), "scheme": scheme, "output_key": output_key, "counts": counts,
    }
    # Workflow provenance is orthogonal to the numerical anchor mapping.
    # Older/custom datasets without a registry entry remain supported.
    try:
        from .workflows import record_workflow_provenance
        record_workflow_provenance(adata, dataset)
    except KeyError:
        pass
    return anchors


def summarize_anchors(anchors):
    """Return portable positive, negative, and unknown counts."""
    values = np.asarray(anchors, dtype=float)
    return {
        "positive": int(np.sum(values == 1)),
        "negative": int(np.sum(values == 0)),
        "unknown": int(np.sum(~np.isfinite(values))),
        "total": int(values.size),
    }
