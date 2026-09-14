"""Command-line entry points for SIGMA analysis and manuscript reproduction."""
from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma-omics")
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="run SIGMA and generate a standard result set")
    report.add_argument("input_h5ad")
    report.add_argument("output_dir")
    report.add_argument("--prefix", default="sigma")
    report.add_argument("--workflow", default="auto")
    report.add_argument("--report-level", choices=("none", "standard", "complete", "manuscript"), default="complete")
    report.add_argument("--program-selection", choices=("auto", "near_far", "boundary_localized", "program_interface"), default="auto")
    report.add_argument("--seed", type=int, default=0)
    report.add_argument("--same-section-pathology", action="store_true")
    report.add_argument("--matched-st", action="store_true")
    report.add_argument("--adjacent-section-pathology", action="store_true")
    report.add_argument("--region-defined", action="store_true")

    configured = sub.add_parser(
        "run-config", help="run SIGMA from a portable JSON configuration"
    )
    configured.add_argument("config_json")

    manifest = sub.add_parser("write-manuscript-manifest", help="write the frozen manuscript manifest")
    manifest.add_argument("output_json")

    reproduce = sub.add_parser("reproduce-manuscript", help="run manuscript figure builders in a reference-data project")
    reproduce.add_argument("project_root")
    reproduce.add_argument("--figures", nargs="*", default=None)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write-manuscript-manifest":
        from .manuscript import write_manuscript_manifest
        print(write_manuscript_manifest(args.output_json))
        return 0
    if args.command == "reproduce-manuscript":
        from .manuscript import reproduce_manuscript_figures
        for path in reproduce_manuscript_figures(args.project_root, figures=args.figures):
            print(path)
        return 0
    if args.command == "run-config":
        from .config import run_analysis_config
        result = run_analysis_config(args.config_json)
        print(result.output_dir)
        return 0
    if args.command == "report":
        import anndata as ad
        from .api import run_analysis
        evidence = {
            "same_section_pathology": args.same_section_pathology,
            "matched_st": args.matched_st,
            "adjacent_section_pathology": args.adjacent_section_pathology,
            "region_defined": args.region_defined,
        }
        evidence = {key: value for key, value in evidence.items() if value}
        result = run_analysis(
            ad.read_h5ad(args.input_h5ad), Path(args.output_dir),
            prefix=args.prefix, workflow=args.workflow,
            evidence=evidence or None, report_level=args.report_level,
            program_selection=args.program_selection, random_state=args.seed,
        )
        print(result.output_dir)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
