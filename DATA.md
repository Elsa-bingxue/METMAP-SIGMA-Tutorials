# BC-515 course data

Re-running the notebooks requires the processed `BC_515_Section_1.h5ad` supplied for this course.

**[Download BC_515_Section_1.h5ad](https://github.com/Elsa-bingxue/METMAP-SIGMA-Tutorials/releases/download/bc515-data-v1/BC_515_Section_1.h5ad)** (113,322,821 bytes; about 108 MiB).

The file is hosted in the [BC-515 data release](https://github.com/Elsa-bingxue/METMAP-SIGMA-Tutorials/releases/tag/bc515-data-v1). It is not included in the repository ZIP or a Git clone, so download it separately.

Place it at `data/BC_515_Section_1.h5ad`, or set the `BC515_DATA` environment variable to its path. Do not substitute a raw source-study file without reproducing the preprocessing and checking that the required fields match.

## Expected input

| Field | Meaning | Expected shape |
|---|---|---|
| `adata.uns['msi']` | Spatial metabolomics intensities | 3,508 spots × 2,086 m/z features |
| `adata.uns['mz_features']` | m/z feature labels | 2,086 features |
| `adata.obsm['spatial']` | Spatial coordinates | 3,508 × 2 |
| `adata.obsm['X_harmony']` | Matched RNA representation | At least 32 columns; the first 32 are used |
| `adata.obs['annotation']` | Weak pathological annotations | Stroma: 2,707; Tumor: 724; Unlabelled: 77 |
| `adata.X` | RNA expression, not metabolomics | 3,508 × 15,953 |

Expected SHA-256:

```text
1f96d6f1aaa36c117ccd7fb94b172a1c1435585d54279e8cffb93df0299c7204
```

Run `python check_setup.py` to verify the file before training. A checksum mismatch means the input differs from the saved reference run, even if the dimensions agree.

## Study source

The SIGMA reference-data documentation attributes BC-515 and BC-525 to Godfrey et al., *Angewandte Chemie International Edition* (2025), [doi:10.1002/anie.202502028](https://doi.org/10.1002/anie.202502028). The course file contains processed, matched spatial metabolomics and transcriptomics with annotations. The source-study citation does not itself supply a verified download link for this exact course file.

Consult the source study for data access and reuse terms. The software licenses under `vendor/` do not license the dataset. The `data/` input files are ignored by Git; the processed course input is distributed as a GitHub Release asset.
