import numpy as np
from scipy.stats import spearmanr


def boundary_enrichment_score(values, d_signed, width):
    """Mean interface signal relative to the remaining tissue."""
    x = np.asarray(values, dtype=float)
    d = np.asarray(d_signed, dtype=float)
    near = np.abs(d) <= float(width)
    valid = np.isfinite(x) & np.isfinite(d)
    if (near & valid).sum() < 3 or ((~near) & valid).sum() < 3:
        return np.nan
    return float(np.nanmean(x[near & valid]) - np.nanmean(x[(~near) & valid]))


def distance_association(values, d_signed):
    x = np.asarray(values, dtype=float)
    d = np.asarray(d_signed, dtype=float)
    valid = np.isfinite(x) & np.isfinite(d)
    if valid.sum() < 3:
        return np.nan, np.nan
    return spearmanr(np.abs(d[valid]), x[valid])
