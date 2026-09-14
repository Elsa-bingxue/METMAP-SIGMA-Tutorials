"""Frozen manuscript-compatible downstream presets.

Reference mode reproduces the published analysis logic from a frozen ranking
distributed with the corresponding reference data bundle. It never silently
falls back to standardized discovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReferencePreset:
    dataset: str
    matrix_source: str
    leading_program: int | None
    ranking_col: str | None = None
    index_col: str = "j"
    top_k: int = 300
    n_programs: int = 4
    n_bins: int = 40
    near_quantile: float = .20
    far_quantile: float = .80
    random_state: int = 0


REFERENCE_PRESETS = {
    "HBC515": ReferencePreset("HBC515", "msi_uns", 3),
    "HBC525": ReferencePreset("HBC525", "msi_uns", 0),
    "HLC091": ReferencePreset("HLC091", "msi_uns", 0),
    "HLC276": ReferencePreset("HLC276", "msi_uns", 2),
    "ccRCC_Y27T": ReferencePreset("ccRCC_Y27T", "X", 0),
    "GBM": ReferencePreset("GBM", "X", 1),
    "HCC_P1": ReferencePreset("HCC_P1", "X", 3, near_quantile=.25, far_quantile=.75),
    "HCC_P4": ReferencePreset("HCC_P4", "X", 2, near_quantile=.25, far_quantile=.75),
    "HPD_A1": ReferencePreset("HPD_A1", "X", 3, ranking_col="r2_logI"),
    "HPD_B1": ReferencePreset("HPD_B1", "X", 2, ranking_col="r2_logI"),
    "HPD_C1": ReferencePreset("HPD_C1", "X", 1, ranking_col="r2_logI"),
}


def list_reference_datasets():
    """Return datasets with frozen manuscript-compatible settings."""
    return tuple(REFERENCE_PRESETS)


def get_reference_preset(dataset):
    try:
        return REFERENCE_PRESETS[str(dataset)]
    except KeyError as exc:
        raise KeyError(f"No reference preset for {dataset!r}") from exc


def run_reference_analysis(
    adata, *, dataset, ranking=None, ranking_path=None,
    assignments=None, assignments_path=None, output_dir,
    representatives=None, representatives_path=None,
    prefix=None, deterministic=True,
):
    """Reproduce a frozen reference downstream analysis.

    The ranking is an explicit scientific input and should come from the
    versioned SIGMA reference-data bundle. Requiring it prevents accidental
    replacement by a newly computed generalized ranking.
    """
    if (ranking is None) == (ranking_path is None):
        raise ValueError("Provide exactly one of ranking or ranking_path")
    if (assignments is None) == (assignments_path is None):
        raise ValueError("Provide exactly one of assignments or assignments_path")
    if ranking is None:
        ranking = pd.read_csv(Path(ranking_path))
    preset = get_reference_preset(dataset)
    if preset.ranking_col is not None:
        if preset.ranking_col not in ranking:
            raise KeyError(
                f"Reference ranking for {dataset} requires {preset.ranking_col!r}"
            )
        ranking = ranking.sort_values(
            preset.ranking_col, ascending=False, kind="stable"
        ).reset_index(drop=True)
    if assignments is None:
        assignments = pd.read_csv(Path(assignments_path))
    if representatives is not None and representatives_path is not None:
        raise ValueError("Provide at most one of representatives or representatives_path")
    if representatives is None and representatives_path is not None:
        representatives = pd.read_csv(Path(representatives_path))
    from .api import run_downstream_analysis
    from .workflows import get_workflow_config

    workflow = get_workflow_config(dataset)
    sides = {
        "negative": workflow.positive_semantics,
        "positive": workflow.negative_semantics,
    }
    result = run_downstream_analysis(
        adata,
        ranking=ranking,
        output_dir=output_dir,
        prefix=prefix or dataset,
        side_names=sides,
        selection_mode="reference",
        ranking_col=preset.ranking_col,
        index_col=preset.index_col,
        top_k=preset.top_k,
        n_programs=preset.n_programs,
        n_bins=preset.n_bins,
        near_quantile=preset.near_quantile,
        far_quantile=preset.far_quantile,
        leading_program=preset.leading_program,
        matrix_source=preset.matrix_source,
        interface_label=workflow.interface_name,
        association_label="reference interface-associated",
        random_state=preset.random_state,
        deterministic=deterministic,
        program_assignments=assignments,
        reference_representatives=representatives,
    )
    Path(output_dir, f"{prefix or dataset}_reference_preset.json").write_text(
        __import__("json").dumps(asdict(preset), indent=2, sort_keys=True)
    )
    return result
