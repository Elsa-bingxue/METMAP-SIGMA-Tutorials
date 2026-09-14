import numpy as np
from sklearn.neighbors import NearestNeighbors


def gaussian_knn_graph(xy, k=15, sigma=None, symmetrize=False):
    """Build the Gaussian-weighted spatial kNN graph used by SIGMA.

    The reference notebooks use directed ``i -> neighbour`` edges.  The
    ``symmetrize`` option is retained for experiments, but is deliberately
    disabled by default to preserve the published notebook behaviour.
    """
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("xy must have shape (n_spots, >=2).")
    k = min(int(k), xy.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(xy)
    dist, idx = nn.kneighbors(xy)
    dist, idx = dist[:, 1:], idx[:, 1:]
    if sigma is None:
        sigma = float(np.median(dist[dist > 0]))
    sigma = max(float(sigma), 1e-12)
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
    src = np.repeat(np.arange(xy.shape[0]), k)
    dst = idx.reshape(-1)
    ew = w.reshape(-1)
    if symmetrize:
        src0, dst0, ew0 = src, dst, ew
        src = np.concatenate([src0, dst0])
        dst = np.concatenate([dst0, src0])
        ew = np.concatenate([ew0, ew0])
        # merge duplicate edges by maximum weight
        key = src.astype(np.int64) * xy.shape[0] + dst.astype(np.int64)
        order = np.argsort(key)
        key, src, dst, ew = key[order], src[order], dst[order], ew[order]
        starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
        src = src[starts]
        dst = dst[starts]
        ew = np.maximum.reduceat(ew, starts)
    return np.vstack([src, dst]).astype(np.int64), ew.astype(np.float32), sigma


def gaussian_label_smoothing(edge_index, edge_weight, y, labeled_mask,
                             n_iter=50, alpha=0.85):
    """Diffuse binary anchors while clamping labeled nodes each iteration."""
    y = np.asarray(y)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    n = y.shape[0]
    field = np.zeros(n, dtype=np.float32)
    field[labeled_mask] = y[labeled_mask].astype(np.float32)
    anchor = field.copy()
    src, dst = np.asarray(edge_index)
    degree = np.zeros(n, dtype=np.float32)
    np.add.at(degree, src, edge_weight)
    degree = np.maximum(degree, 1e-12)
    for _ in range(int(n_iter)):
        message = np.zeros(n, dtype=np.float32)
        np.add.at(message, src, edge_weight * field[dst])
        updated = alpha * (message / degree) + (1.0 - alpha) * anchor
        updated[labeled_mask] = anchor[labeled_mask]
        field = updated
    return field


def smooth_embedding(edge_index, edge_weight, e, n_iter=10, alpha=0.9):
    """Spatial diffusion of an embedding on a weighted graph."""
    e = np.asarray(e, dtype=np.float32)
    f = e.copy()
    src, dst = edge_index
    deg = np.zeros(e.shape[0], dtype=np.float32)
    np.add.at(deg, src, edge_weight)
    deg = np.maximum(deg, 1e-12)
    for _ in range(int(n_iter)):
        msg = np.zeros_like(f)
        np.add.at(msg, src, edge_weight[:, None] * f[dst])
        f = alpha * (msg / deg[:, None]) + (1.0 - alpha) * e
    return f
