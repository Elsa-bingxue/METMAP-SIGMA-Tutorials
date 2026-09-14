"""Safe modality inspection and routing for user-supplied spatial omics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class InputAssessment:
    modalities: tuple[str, ...]
    has_spatial: bool
    has_anchors: bool
    has_sigma_result: bool
    recommended_entry_point: str
    supports_metabolic_sigma: bool
    messages: tuple[str, ...]

    def to_dict(self):
        return asdict(self)

    def print_report(self):
        print(f"Modalities: {', '.join(self.modalities) or 'undeclared'}")
        print(f"Recommended entry point: {self.recommended_entry_point}")
        for message in self.messages:
            print(f"- {message}")


def inspect_input(
    adata, *, modality="auto", feature_type_key="feature_type",
    spatial_key="spatial", anchor_key="sigma_anchor",
):
    """Inspect modality and recommend a safe SIGMA public entry point."""
    declared = str(modality).upper()
    if declared not in {"AUTO", "SM", "ST", "JOINT"}:
        raise ValueError("modality must be 'auto', 'SM', 'ST', or 'joint'")
    if declared == "AUTO":
        if feature_type_key not in adata.var:
            modalities = ()
        else:
            values = {str(x).upper() for x in adata.var[feature_type_key].dropna().unique()}
            modalities = tuple(x for x in ("SM", "ST") if x in values)
    elif declared == "JOINT":
        modalities = ("SM", "ST")
    else:
        modalities = (declared,)
    has_spatial = spatial_key in adata.obsm
    if has_spatial:
        xy = np.asarray(adata.obsm[spatial_key])
        has_spatial = bool(xy.ndim == 2 and xy.shape[0] == adata.n_obs and xy.shape[1] >= 2)
    has_anchors = anchor_key in adata.obs
    has_sigma = all(key in adata.obs for key in (
        "sigma_region_probability", "sigma_boundary", "sigma_d_signed",
    ))
    messages = []
    if not modalities:
        entry, supported = "declare_modality", False
        messages.append("No feature_type declaration was found; modality was not guessed.")
    elif modalities == ("ST",):
        entry, supported = "st_interface_validation", False
        messages.append("ST-only data can characterize an interface but cannot produce metabolite programs.")
    elif has_sigma:
        entry, supported = "run_downstream_analysis", True
        messages.append("A complete SIGMA coordinate is present; proceed to downstream reporting.")
    elif modalities == ("SM",):
        entry, supported = "run_sigma_weak_anchor", True
        messages.append("SM-only analysis requires explicit anchors; weak-anchor fitting is currently HCC-reference validated.")
    else:
        entry, supported = "run_sigma", True
        messages.append("Joint SM+ST uses SM as the target and an ST representation for auxiliary alignment.")
    if not has_spatial:
        messages.append("Valid coordinates are required in adata.obsm['spatial'].")
    if not has_anchors and not has_sigma and supported:
        messages.append("Prepare explicit 1/0/NaN anchors before fitting SIGMA.")
    result = InputAssessment(
        modalities, has_spatial, has_anchors, has_sigma, entry, supported,
        tuple(messages),
    )
    adata.uns["sigma_input_assessment"] = result.to_dict()
    return result


def assess_input(adata, **kwargs):
    """Alias for :func:`inspect_input` used by the quick-start workflow."""
    return inspect_input(adata, **kwargs)
