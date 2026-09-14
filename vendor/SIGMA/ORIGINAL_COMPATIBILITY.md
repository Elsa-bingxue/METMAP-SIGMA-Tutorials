# BC515 notebook compatibility

This teaching copy reports version `0.3.1+bc515.original`. It starts from the
SIGMA 0.3.1 source and restores the operations in `HBC_515_SIGMA.ipynb` for the
BC515 example. The installed PyPI package and the upstream repository are not
modified by this copy.

## Core model

The RNA projection is initialized before the residual reconstruction head, as
in original cell 10. This order matters because both layers consume the seeded
Torch random sequence. The Torch seed is reset immediately before model
construction. The positive-class BCE weight retains the dtype inferred from the
NumPy count ratio, which is float64 in the BC515 workflow.

Both diffusion functions divide each neighbor sum by its degree before
multiplying by the diffusion coefficient. This preserves the original float32
arithmetic grouping. Region probabilities use the NumPy sigmoid expression in
cell 12. Boundary construction and distances use the original spatial
coordinates converted directly to float64.

These changes preserve the original operations. They do not guarantee identical
saved results across different Torch, NumPy, BLAS, or hardware versions. The
original notebook selected CUDA when available and did not save a complete
runtime specification.

## Distance-decay ranking

`rank_lambda_profiles` fits the original Huber regression of
`log1p(y - min(y))` against absolute interface distance. The BC515 tutorial uses
2,000 variance-selected features, at least 300 finite observations, a negative
slope, R² of at least 0.05, near/far enrichment of at least 1.05, and removal of
the upper 1% of retained lambda values. Ranking uses R² and then enrichment.

The 20th- and 80th-percentile distance thresholds are computed separately for
each feature after excluding its missing observations. The returned table
includes `near_interface_mean` and `far_interface_mean`. Additional generic
score columns remain available but do not determine the original BC515 ranking.

## Metabolic programs

The original workflow first computes eight-sector anisotropy for the first 300
lambda-ranked features, then sorts by R² and anisotropy. Pass that ordered table
to the library with:

```python
analysis = discover_programs(
    adata,
    ordered_candidates,
    matrix=X_msi,
    selection_mode="lambda_profile",
    original_notebook=True,
    adaptive_binning=False,
    n_programs=4,
    top_k=300,
    n_bins=40,
    min_bin_points=5,
    valid_bin_fraction=0.80,
    lower_percentile=2,
    upper_percentile=98,
)
```

This option preserves the supplied candidate order, uses all 41 bin edges with
`np.digitize(distance, edges) - 1`, and computes bin means in the input intensity
dtype before forming float64 profiles. Profiles require at least 80% defined
bins. Missing bins are interpolated in both directions; each profile is then
standardized with `std + 1e-6` before Ward clustering.

Module scores follow original cells 41–42: standardize each raw-intensity
feature with `std + 1e-6`, then average the features in each cluster. The average
is not standardized a second time. `analysis.profile_matrix` exposes the exact
cluster input in assignment order. The default `original_notebook=False`
retains the library's general program workflow.

## Numerical checks

The compatibility functions were compared directly with function and class
definitions extracted from the original notebook. On the same BC515 inputs,
mapped model initialization, one Adam update, embedding and label diffusion,
lambda ranking, 300 × 40 profiles, cluster assignments, and module scores
matched exactly. A ranking check with missing measurements also matched. These checks isolate code
agreement; they do not substitute the original saved field for a newly trained
result.

Original notebook cell numbers in this document are zero-based.
