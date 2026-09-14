"""Evidence-source workflows and dataset registry for SIGMA analyses.

Workflows describe where supervision comes from. They do not select a
different SIGMA mathematical model. ``anchor_policy`` separately records how
the evidence is converted into positive, negative, and unknown anchors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


VALID_WORKFLOWS = (
    "direct_pathology",
    "multiomics_inferred",
    "transferred_pathology",
    "region_defined_disease",
)


@dataclass(frozen=True)
class WorkflowConfig:
    dataset: str
    workflow: str
    anchor_policy: str
    annotation_source: str
    interface_name: str
    positive_semantics: str
    negative_semantics: str
    unknown_semantics: str = "unassigned_or_ambiguous"
    matched_st: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.workflow not in VALID_WORKFLOWS:
            raise ValueError(f"workflow must be one of {VALID_WORKFLOWS}")


DATASET_WORKFLOWS = {
    name: WorkflowConfig(
        name, "direct_pathology", "direct_binary", "same_section_pathology",
        "tumor--microenvironment interface", "tumor", "non-tumor/stroma",
    )
    for name in ("HBC515", "HBC525", "HLC091", "HLC276")
}
DATASET_WORKFLOWS.update({
    "ccRCC_Y27T": WorkflowConfig(
        "ccRCC_Y27T", "multiomics_inferred", "core_vs_rest",
        "ST markers and H&E-informed regional states", "tumor-to-TME transition",
        "tumor epithelial/core-like", "all non-core regions", matched_st=True,
    ),
    "GBM": WorkflowConfig(
        "GBM", "multiomics_inferred", "core_vs_rest",
        "ST markers and H&E-informed regional states", "core-to-margin transition",
        "tumor core-like", "all non-core regions", matched_st=True,
    ),
    "HCC_P1": WorkflowConfig(
        "HCC_P1", "transferred_pathology", "cluster_assisted_transfer",
        "adjacent-section pathology plus SM spatial clustering",
        "tumor--non-tumor interface", "tumor-like", "non-tumor-like",
        notes="No matched ST; pathological anchors are transferred indirectly.",
    ),
    "HCC_P4": WorkflowConfig(
        "HCC_P4", "transferred_pathology", "cluster_assisted_transfer",
        "adjacent-section pathology plus SM spatial clustering",
        "tumor--non-tumor interface", "tumor-like", "non-tumor-like",
        notes="No matched ST; pathological anchors are transferred indirectly.",
    ),
})
for name in ("HPD_A1", "HPD_B1", "HPD_C1"):
    DATASET_WORKFLOWS[name] = WorkflowConfig(
        name, "region_defined_disease", "named_regions",
        "same-section anatomical region labels", "CI--Cd metabolic transition",
        "CI / dopamine-low", "Cd / dopamine-high", "ACB/NA/unknown",
        matched_st=True,
    )


def get_workflow_config(dataset: str) -> WorkflowConfig:
    """Return the frozen evidence-source configuration for a dataset."""
    if dataset not in DATASET_WORKFLOWS:
        raise KeyError(f"No workflow configuration for {dataset!r}")
    return DATASET_WORKFLOWS[dataset]


def record_workflow_provenance(adata, dataset: str, *, extra=None):
    """Record workflow semantics without changing numerical arrays."""
    config = get_workflow_config(dataset)
    payload = asdict(config)
    payload["schema_version"] = 1
    payload.update(dict(extra or {}))
    adata.uns["sigma_workflow_provenance"] = payload
    return payload
