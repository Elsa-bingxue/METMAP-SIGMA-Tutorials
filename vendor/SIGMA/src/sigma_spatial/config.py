"""Configuration-driven SIGMA analysis entry points.

JSON configurations provide a stable bridge between command-line runs,
notebooks and manuscript reproduction without embedding local paths in code.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Validated, resolved inputs for one SIGMA analysis."""

    input_h5ad: Path
    output_dir: Path
    options: dict
    source: Path


_RUN_KEYS = {
    "prefix", "workflow", "evidence", "report_level", "selection_mode",
    "program_selection", "anchor_key", "representation_key", "spatial_key",
    "matrix_source", "random_state", "copy", "core_kwargs", "downstream_kwargs",
}


def load_analysis_config(path) -> AnalysisConfig:
    """Load and validate a portable JSON analysis configuration.

    Relative input and output paths are resolved against the configuration
    file directory, not the caller's working directory.
    """
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("analysis configuration must contain a JSON object")
    missing = {"input_h5ad", "output_dir"}.difference(payload)
    if missing:
        raise ValueError(f"analysis configuration is missing {sorted(missing)}")
    unknown = set(payload).difference(_RUN_KEYS | {"schema_version", "input_h5ad", "output_dir"})
    if unknown:
        raise ValueError(f"unknown analysis configuration keys: {sorted(unknown)}")
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported analysis configuration schema_version")
    base = source.parent
    input_h5ad = Path(payload["input_h5ad"]).expanduser()
    output_dir = Path(payload["output_dir"]).expanduser()
    if not input_h5ad.is_absolute():
        input_h5ad = (base / input_h5ad).resolve()
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()
    options = {key: payload[key] for key in _RUN_KEYS if key in payload}
    options.setdefault("random_state", 0)
    options.setdefault("report_level", "complete")
    options.setdefault("selection_mode", "lambda_profile")
    options.setdefault("program_selection", "auto")
    return AnalysisConfig(input_h5ad, output_dir, options, source)


def run_analysis_config(path):
    """Run :func:`sigma_spatial.run_analysis` from a JSON configuration."""
    config = load_analysis_config(path)
    if not config.input_h5ad.is_file():
        raise FileNotFoundError(f"input AnnData not found: {config.input_h5ad}")
    import anndata as ad
    from .api import run_analysis

    result = run_analysis(
        ad.read_h5ad(config.input_h5ad), config.output_dir, **config.options,
    )
    destination = config.output_dir / "analysis_config.resolved.json"
    destination.write_text(json.dumps({
        "schema_version": 1,
        "source_config": str(config.source),
        "input_h5ad": str(config.input_h5ad),
        "output_dir": str(config.output_dir),
        **config.options,
    }, indent=2, sort_keys=True))
    return result
