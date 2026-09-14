"""Versioned helpers for reproducing the SIGMA manuscript figure set.

The package contains the orchestration and frozen program manifest, whereas
large reference AnnData objects and histology images remain in the separately
distributed SIGMA reference-data project.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


MANUSCRIPT_VERSION = "manuscript-v1"
MANUSCRIPT_PROGRAMS = {
    "HBC515": 3, "HBC525": 0,
    "HLC091": 0, "HLC276": 2,
    "ccRCC_Y27T": 0, "GBM": 1,
    "HCC_P1": 3, "HCC_P4": 2,
    "HPD_A1": 3, "HPD_B1": 2, "HPD_C1": 1,
}
FIGURE_BUILDERS = {
    "Fig2": "analysis/polish_overleaf_fig2_extended1.py",
    "Fig3": "analysis/polish_overleaf_fig3_breast.py",
    "Fig4": "analysis/polish_overleaf_fig4_lung.py",
    "Fig5": "analysis/polish_overleaf_fig5_ccrcc_gbm.py",
    "Fig6": "analysis/polish_overleaf_fig6_hcc.py",
    "Fig7": "analysis/polish_overleaf_fig7_cross_cancer.py",
    "Fig8": "analysis/polish_overleaf_fig8_pd.py",
}


def manuscript_manifest() -> dict:
    """Return the frozen program and figure-builder manifest."""
    return {
        "schema_version": 1,
        "manuscript_version": MANUSCRIPT_VERSION,
        "package_version": "0.3.1",
        "random_seed": 0,
        "selected_programs": dict(MANUSCRIPT_PROGRAMS),
        "figure_builders": dict(FIGURE_BUILDERS),
        "reference_data_required": True,
    }


def write_manuscript_manifest(path) -> Path:
    """Write the reproducibility manifest as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manuscript_manifest(), indent=2, sort_keys=True))
    return destination


def reproduce_manuscript_figures(project_root, *, figures=None) -> list[Path]:
    """Run the versioned figure builders in a complete SIGMA data project.

    This function does not download restricted or large reference data. The
    supplied project must already contain the manuscript analysis scripts and
    reference inputs named by those scripts.
    """
    root = Path(project_root).resolve()
    requested = list(FIGURE_BUILDERS) if figures is None else list(figures)
    unknown = sorted(set(requested).difference(FIGURE_BUILDERS))
    if unknown:
        raise ValueError(f"Unknown manuscript figures: {unknown}")
    scripts = [root / FIGURE_BUILDERS[name] for name in requested]
    missing = [str(path) for path in scripts if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The SIGMA reference-data project is incomplete; missing builders: "
            + ", ".join(missing)
        )
    environment = os.environ.copy()
    environment.setdefault("SIGMA_SKIP_TIFF", "1")
    with tempfile.TemporaryDirectory(prefix="sigma-mpl-") as mpl_config:
        environment["MPLCONFIGDIR"] = mpl_config
        for script in scripts:
            subprocess.run(
                [sys.executable, str(script)], cwd=root, env=environment, check=True,
            )
    return [root / "Overleaf_Paper" / "Figure" / f"{name}.pdf" for name in requested]
