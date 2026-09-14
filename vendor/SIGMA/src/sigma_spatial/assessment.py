"""Preflight assessment and generic anchor preparation for new datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping, Sequence
import numpy as np

from .anchors import AnchorSpec, build_anchors, summarize_anchors
from .workflows import VALID_WORKFLOWS


@dataclass(frozen=True)
class DatasetAssessment:
    recommended_workflow: str
    recommended_anchor_policy: str
    checks: dict
    warnings: tuple[str, ...]
    evidence: dict

    @property
    def ready(self):
        return bool(self.checks.get("has_spatial") and self.checks.get("has_matrix"))

    def to_dict(self):
        return asdict(self)

    def print_report(self):
        print(f"Recommended workflow: {self.recommended_workflow}")
        print(f"Recommended anchor policy: {self.recommended_anchor_policy}")
        for key, passed in self.checks.items():
            print(f"[{'OK' if passed else 'FAIL'}] {key}")
        for warning in self.warnings:
            print(f"[WARNING] {warning}")


def _recommend(evidence):
    flags = {
        key: bool(evidence.get(key, False)) for key in (
            "same_section_pathology", "matched_st",
            "adjacent_section_pathology", "region_defined",
        )
    }
    if sum(flags.values()) == 0:
        raise ValueError("Declare at least one annotation-evidence source")
    if flags["region_defined"]:
        return "region_defined_disease", "named_regions"
    if flags["same_section_pathology"]:
        return "direct_pathology", "direct_binary"
    if flags["adjacent_section_pathology"]:
        return "transferred_pathology", "cluster_assisted_transfer"
    if flags["matched_st"]:
        return "multiomics_inferred", "high_confidence_three_state"
    raise AssertionError("Unreachable evidence combination")


def assess_dataset(
    adata, *, annotation_key=None, evidence: Mapping[str, bool],
    spatial_key="spatial",
):
    """Validate required arrays and recommend a workflow from declared evidence.

    Evidence must be supplied by the investigator; it is never guessed from a
    column name or from whichever workflow produces the strongest result.
    """
    workflow, policy = _recommend(dict(evidence))
    has_spatial = spatial_key in adata.obsm
    spatial_valid = False
    if has_spatial:
        xy = np.asarray(adata.obsm[spatial_key])
        spatial_valid = xy.ndim == 2 and xy.shape[0] == adata.n_obs and xy.shape[1] >= 2 and np.isfinite(xy[:, :2]).all()
    has_matrix = adata.X is not None and adata.shape[0] > 0 and adata.shape[1] > 0
    has_annotation = annotation_key is not None and annotation_key in adata.obs
    warnings = []
    if annotation_key is not None and not has_annotation:
        warnings.append(f"annotation_key={annotation_key!r} was not found")
    if has_annotation:
        missing = float(adata.obs[annotation_key].isna().mean())
        if missing > 0:
            warnings.append(f"{missing:.1%} of annotation values are missing")
    if workflow == "multiomics_inferred" and not evidence.get("matched_st", False):
        warnings.append("multiomics_inferred requires matched ST evidence")
    checks = {
        "has_spatial": has_spatial, "spatial_is_finite_n_by_2": spatial_valid,
        "has_matrix": has_matrix, "has_annotation": has_annotation,
    }
    assessment = DatasetAssessment(workflow, policy, checks, tuple(warnings), dict(evidence))
    adata.uns["sigma_dataset_assessment"] = assessment.to_dict()
    return assessment


def prepare_anchors(
    adata, *, annotation_key: str,
    positive_labels: Sequence[str], negative_labels: Sequence[str] = (),
    unknown_labels: Sequence[str] = (), workflow: str,
    anchor_policy: str, output_key="sigma_anchor",
    negative_policy="listed", interface_name=None, copy=False,
):
    """Prepare auditable 1/0/NaN anchors for a new dataset.

    Unlisted labels remain unknown under ``negative_policy='listed'``. Using
    ``negative_policy='complement'`` must be an explicit investigator choice.
    """
    if workflow not in VALID_WORKFLOWS:
        raise ValueError(f"workflow must be one of {VALID_WORKFLOWS}")
    if annotation_key not in adata.obs:
        raise KeyError(f"adata.obs[{annotation_key!r}] is required")
    result = adata.copy() if copy else adata
    spec = AnchorSpec(
        name="custom", source_column=annotation_key, mode=anchor_policy,
        positive_labels=tuple(map(str, positive_labels)),
        negative_labels=tuple(map(str, negative_labels)),
        uncertain_labels=tuple(map(str, unknown_labels)),
        negative_policy=negative_policy,
        description="User-declared anchor mapping for a new dataset.",
    )
    anchors = build_anchors(result.obs[annotation_key].to_numpy(), spec)
    result.obs[output_key] = anchors
    result.uns["sigma_anchor_provenance"] = {
        **asdict(spec), "output_key": output_key,
        "counts": summarize_anchors(anchors), "custom_mapping": True,
    }
    result.uns["sigma_workflow_provenance"] = {
        "schema_version": 1, "dataset": "custom",
        "workflow": workflow, "anchor_policy": anchor_policy,
        "annotation_source": annotation_key,
        "interface_name": interface_name or "user-defined interface",
        "negative_policy": negative_policy,
    }
    return result if copy else anchors
