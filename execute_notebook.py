"""Execute the teaching notebooks with this interpreter and save their outputs."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
for key, value in {
    "MPLCONFIGDIR": str(HERE / ".mplconfig"),
    "IPYTHONDIR": str(HERE / ".ipython"),
    "JUPYTER_RUNTIME_DIR": str(HERE / ".jupyter/runtime"),
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(key, value)
for key in ("MPLCONFIGDIR", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR"):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def run(path):
    """Stop at the first failed cell, retaining outputs for inspection."""
    path = Path(path).resolve()
    notebook = nbformat.read(path, as_version=4)
    kernel_dir = HERE / ".jupyter/kernels/bc515-tutorial"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (BC-515 tutorial)",
        "language": "python",
    }), encoding="utf-8")
    manager = KernelManager(
        kernel_name="bc515-tutorial",
        kernel_spec_manager=KernelSpecManager(
            kernel_dirs=[str(kernel_dir.parent)], ensure_native_kernel=False
        ),
    )
    started = time.perf_counter()

    def completed(cell, cell_index, **kwargs):
        nbformat.write(notebook, path)
        elapsed = time.perf_counter() - started
        print(f"{path.name}: cell {cell_index + 1}/{len(notebook.cells)} ({elapsed:.1f} s)", flush=True)

    client = NotebookClient(
        notebook, km=manager, timeout=7200, allow_errors=False,
        resources={"metadata": {"path": str(HERE)}},
        on_cell_executed=completed,
    )
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(notebook, path)
    nbformat.validate(notebook)
    html, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(notebook)
    path.with_suffix(".html").write_text(html, encoding="utf-8")
    print(f"Saved {path.name} and {path.with_suffix('.html').name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="+", help="Notebook filenames to execute, in order")
    arguments = parser.parse_args()
    for name in arguments.notebook:
        run(name)
