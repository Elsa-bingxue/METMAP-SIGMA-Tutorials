# MET-MAP and SIGMA tutorials

Four executed Jupyter notebooks use the same BC-515 spatial metabolomics sample to compare a library implementation with an implementation written in the notebook. Code comments, explanations, and lecture slides are in English.

## Start here

| Method | Import the library | Write the method from scratch |
|---|---|---|
| MET-MAP | [01_METMAP_library.ipynb](01_METMAP_library.ipynb) | [03_METMAP_from_scratch.ipynb](03_METMAP_from_scratch.ipynb) |
| SIGMA | [02_SIGMA_library.ipynb](02_SIGMA_library.ipynb) | [04_SIGMA_from_scratch.ipynb](04_SIGMA_from_scratch.ipynb) |

Open a notebook on GitHub to read its saved figures and tables. No installation or data download is needed to inspect those results. For offline reading, download the repository and double-click a notebook's accompanying `.html` file. The HTML files include the saved figures and tables and open in a web browser. Use the `.ipynb` files in JupyterLab to edit or execute code.

The [English lecture slides](slides/METMAP_SIGMA_BC515_Original_Workflow_Course.pptx) introduce both methods and the reproduction steps. Read the [slide notes](SLIDE_NOTES.md), particularly the explanation of the SIGMA distance profiles on slide 21.

## Run the notebooks

### 1. Download the repository

Use **Code → Download ZIP** on GitHub and extract the folder, or clone the repository. Open a terminal in the extracted repository folder.

### 2. Create a Python 3.11 environment

Using Conda or Miniforge:

```bash
conda create -n metmap-sigma python=3.11 -y
conda activate metmap-sigma
python -m pip install -r requirements.txt
```

Alternatively, create a virtual environment with an existing Python 3.11 installation. The scientific package versions in `requirements.txt` match the environment used for the saved results. JupyterLab supplies the notebook interface. The examples run on CPU; a GPU is not required.

### 3. Download the course data

Download [BC_515_Section_1.h5ad](https://github.com/Elsa-bingxue/METMAP-SIGMA-Tutorials/releases/download/bc515-data-v1/BC_515_Section_1.h5ad) from the GitHub Release and place it here:

```text
data/BC_515_Section_1.h5ad
```

The file is 113,322,821 bytes (about 108 MiB). The repository ZIP does not include this large data file; download it separately using the link above. See [DATA.md](DATA.md) for its source, required contents, and SHA-256 checksum. A different file with a similar name may not contain the same preprocessing or annotations.

### 4. Check the environment and start JupyterLab

```bash
python check_setup.py
python -m jupyterlab
```

Open a notebook and select **Kernel → Restart Kernel and Run All Cells**. Run the two notebooks for each method to generate results for their comparison cells. A useful reading order is **01 → 03 → 02 → 04**. Each notebook trains its model from the input data; saved outputs are not used as a substitute for training.

For terminal execution, `python check_setup.py --batch` checks the dependencies without requiring the JupyterLab interface:

```bash
python execute_notebook.py 01_METMAP_library.ipynb 03_METMAP_from_scratch.ipynb
python execute_notebook.py 02_SIGMA_library.ipynb 04_SIGMA_from_scratch.ipynb
```

Execution writes new outputs into the notebooks and creates files under `results/`. Keep a copy of the downloaded notebooks if you want to retain the supplied reference outputs. Runtime depends on the computer; the examples train for 3,000 MET-MAP epochs and 1,000 SIGMA epochs.

If the data are stored elsewhere, set the `BC515_DATA` environment variable to the full path before starting Python or JupyterLab. You can also pass that path to `python check_setup.py --data PATH` to check it.

## What is reproduced

**MET-MAP:** a fixed-parameter application to BC-515, using the upstream `multi_gaston` library and a matching implementation in the notebook. It does not reproduce the original paper's liver or intestine figures or its full parameter search.

**SIGMA:** the original BC-515 training and downstream workflow. The notebooks fit intensity decay against absolute distance, rank features by fit R² and boundary enrichment, group the leading profiles into four metabolic programs, and select one representative from each program. The library and from-scratch implementations use the same settings.

The original workflow defines decay length as λ = −1 / slope from a robust fit of `log1p(intensity − minimum intensity)` against absolute distance. Profiles used for program clustering retain signed distance. The two distance conventions answer different questions and appear separately in the notebooks.

The SIGMA library snapshot includes documented compatibility changes that restore the original notebook's initialization order and numerical operations. Read [ORIGINAL_PROTOCOL.md](ORIGINAL_PROTOCOL.md) for the source settings and the verification scope. The notebooks cover model training, boundary analysis, decay fits, and metabolic programs; the original project's simulation benchmarks and downstream RNA enrichment analyses are outside this lesson.

The current SIGMA run gives 751 inside spots, 838 boundary spots and 312 retained decay features. Its library and from-scratch implementations match in all 21 paired checks, and both match separately executed original code in 62 checks. See the [verification record](reference/verification.json) and [original-code comparison](reference/original_protocol_comparison.csv).

Saved outputs let students inspect the completed runs. Final comparison cells check agreement between the two implementations for the same input, seed, and environment. Agreement between implementations does not establish independent biological validation or guarantee bitwise equality across operating systems. An m/z value alone does not establish chemical identity.

## Source snapshots

Small source snapshots are included so the library notebooks use the code supplied with the course:

| Source | Version or commit | Location |
|---|---|---|
| [MET-MAP](https://github.com/raphael-group/MET-MAP) | `43c9386ffe7d5b2f0f66d64d4820358c3cd8a11c` | `vendor/MET-MAP/src/multi_gaston/` |
| [SIGMA](https://github.com/Elsa-bingxue/SIGMA) | Based on 0.3.1, commit `4fdd7444bf4bb69792dac2142c0f042959b638e3`, with original-protocol compatibility changes | `vendor/SIGMA/src/sigma_spatial/` |

MET-MAP retains its [BSD 3-Clause license](vendor/MET-MAP/LICENSE); SIGMA retains its [MIT license](vendor/SIGMA/LICENSE). MET-MAP source files are unchanged. SIGMA compatibility changes are documented in `ORIGINAL_PROTOCOL.md`. The from-scratch notebooks implement the method operations directly using general scientific Python and PyTorch tools.

## Troubleshooting

- **Missing data:** download the AnnData file from the [BC-515 data release](https://github.com/Elsa-bingxue/METMAP-SIGMA-Tutorials/releases/tag/bc515-data-v1) and place it in `data/`. Downloading the repository ZIP does not download the data.
- **Missing package:** activate the environment used for installation and start JupyterLab from that terminal.
- **Wrong Python version:** these examples use Python 3.11. Create the environment above rather than installing into a system Python environment.
- **Parity files not found:** run the counterpart notebook, then rerun the final comparison cell.
- **Different numerical results:** first compare the input checksum, package versions, seed, and CPU settings. Small numerical differences can arise across platforms.
