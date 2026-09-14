# Notes for the English lecture slides

The presentation in `slides/` follows the supplied course template and uses the revised original-protocol SIGMA notebooks.

## Slide 21: representatives of the four metabolic programs

The top row shows one m/z feature from each of the four metabolic programs. Each point is a spatial spot. Representatives follow the original within-program lambda filter and ranking by fit R² and enrichment.

The bottom row shows intensity against absolute distance to the boundary, in the input coordinate units. The plot uses 30 quantile bins and the original display normalization. Lambda comes from the raw-intensity decay fit; the displayed trend retains that slope while adapting the intercept to the normalized intensity scale. Spot-level error bars describe variation within this section.

Program clustering uses signed-distance profiles, while decay-length fitting uses absolute distance. Negative signed distance denotes the inferred inside region. Absolute distance pools both sides to measure the range of the decay pattern.

An m/z value is a feature label rather than a confirmed compound identity. The figures summarize one tissue section; they do not estimate variation between biological replicates.
