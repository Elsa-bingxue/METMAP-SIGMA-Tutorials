# Original BC-515 SIGMA protocol

The source for this lesson is the instructor's `HBC_515_SIGMA.ipynb`. The library notebook imports a bundled `sigma_spatial` snapshot; the from-scratch notebook defines the method operations directly. Both follow the original protocol below.

## Core model

| Operation | Setting |
|---|---|
| MSI input | `uns['msi']`; all 3,508 spots and 2,086 features |
| Embedding | log1p, RobustScaler 10–90%, SVD with 64 components, feature standardization |
| Spatial graph | 15 neighbors; Gaussian weights |
| Embedding diffusion | 10 iterations, alpha 0.9 |
| Label reference field | 50 iterations, alpha 0.85, fixed labeled anchors |
| Network | Two high-pass blocks, 64 to 128 to 64; dropout 0.1 |
| Output initialization | RNA projection, residual reconstruction, then region classifier |
| Training | Seed 0, CPU, 1,000 epochs, Adam learning rate 0.001, weight decay 0.0001 |
| Loss | Smooth L1 reconstruction + 0.05 weighted region BCE + 0.05 RNA MSE |
| RNA target | First 32 columns of `X_harmony` |
| Region | Min–max normalized sigmoid score; inside at score ≥ 0.5 |
| Boundary | Both same-side and opposite-side spots among 10 neighbors |
| Signed distance | Distance to the nearest boundary spot; negative inside |

The compatibility changes preserve the original output-layer initialization order, diffusion arithmetic grouping, and NumPy sigmoid calculation. These details affect reproducibility even when the nominal architecture and seed match. The source coordinates supply boundary distances in their input units.

## Decay ranking

1. Select the 2,000 features with highest raw-intensity variance.
2. Fit `log1p(I − min(I)) = intercept + slope × abs(distance)` with Huber regression.
3. Require at least 300 finite observations and a negative slope; set λ = −1 / slope.
4. Require fit R² ≥ 0.05 and near/far mean-intensity ratio ≥ 1.05. Near and far use the 20th and 80th absolute-distance percentiles.
5. Remove λ above the 99th percentile among retained candidates. Rank by R², then enrichment, in descending order.

## Metabolic programs

Take the first 300 ranked features and calculate decay anisotropy in eight sectors around the tissue center. Sector fits require at least 150 observations. The candidate table follows the original ordering by R² and anisotropy.

Build profiles from mean raw intensity in 40 equal-width signed-distance bins between the 2nd and 98th distance percentiles. A bin requires at least five spots; a feature requires at least 80% defined bins. Interpolate missing bins in both directions, z-score each feature profile, and use Ward clustering with four groups.

For each group, remove its highest 10% of λ values when valid candidates remain, then select the feature with the highest R², using enrichment to break ties. Module scores average the member features' spot-level intensity z-scores without standardizing that average again.

## Numerical comparison

The historical source notebook recorded 752 inside spots and 834 boundary spots, with 310 retained decay features. Its four program sizes were 33, 121, 134, and 12. Historical representatives were m/z 817.534520, 818.540767, 887.561462, and 328.057840. These values describe the saved original run.

The revised notebooks recompute the analysis from the input. The current run gives 751 inside spots, 838 boundary spots and 312 retained features. Program sizes are 30, 97, 161 and 12; all four representative m/z values match the historical selection. Both implementations match an independently executed original-code reference in all 62 listed comparisons. See [original_protocol_comparison.csv](reference/original_protocol_comparison.csv). Their outputs and the course verification records report the new run. Re-running the original core code in the same current environment supplies an additional reference beyond the library/from-scratch comparison. Historical package versions and hardware were not fully captured, so matching the original computation does not by itself guarantee every historical floating-point value.
