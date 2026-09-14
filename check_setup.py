"""Check the course environment and BC-515 input before training."""

from pathlib import Path
import argparse
import hashlib
import importlib
import importlib.metadata
import os
import sys

HERE = Path(__file__).resolve().parent
EXPECTED_SHA256 = "1f96d6f1aaa36c117ccd7fb94b172a1c1435585d54279e8cffb93df0299c7204"
MODULES = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
    "scikit-learn": "sklearn", "matplotlib": "matplotlib", "anndata": "anndata",
    "torch": "torch", "torch-geometric": "torch_geometric", "scanpy": "scanpy",
    "statsmodels": "statsmodels", "zarr": "zarr", "threadpoolctl": "threadpoolctl",
    "ipykernel": "ipykernel", "nbformat": "nbformat", "nbclient": "nbclient",
    "nbconvert": "nbconvert", "jupyterlab": "jupyterlab",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get(
        "BC515_DATA", HERE / "data/BC_515_Section_1.h5ad")))
    parser.add_argument("--batch", action="store_true",
                        help="Check terminal execution without requiring the JupyterLab interface.")
    args = parser.parse_args()
    errors = []
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 11):
        errors.append("Use Python 3.11 for the recorded course environment.")

    os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".cache/numba"))
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    for distribution, module in MODULES.items():
        if args.batch and distribution == "jupyterlab":
            continue
        try:
            importlib.import_module(module)
            print(f"  {distribution}: {importlib.metadata.version(distribution)}")
        except Exception as exc:
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")

    for location in ("vendor/MET-MAP/src/multi_gaston/multi_gaston.py",
                     "vendor/SIGMA/src/sigma_spatial/pipeline.py"):
        if not (HERE / location).is_file():
            errors.append(f"Missing bundled library file: {location}")

    input_path = args.data.expanduser()
    if not input_path.is_file():
        errors.append("Course data not found. Follow DATA.md, or pass --data PATH.")
    else:
        digest = hashlib.sha256()
        with input_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        print(f"Input SHA-256: {actual}")
        if actual != EXPECTED_SHA256:
            errors.append("Input checksum differs from the saved reference run.")

    if errors:
        print("\nResolve these items before running the notebooks:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nEnvironment imports and input checksum passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
