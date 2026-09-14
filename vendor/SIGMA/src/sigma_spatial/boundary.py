import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def _knn_adjacency(xy, k_nn):
    xy = np.asarray(xy, dtype=float)
    k_nn = min(int(k_nn), xy.shape[0] - 1)
    if k_nn < 1:
        raise ValueError("At least two spatial observations are required.")
    idx = NearestNeighbors(n_neighbors=k_nn + 1).fit(xy).kneighbors(
        xy, return_distance=False
    )[:, 1:]
    rows = np.repeat(np.arange(xy.shape[0]), k_nn)
    graph = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, idx.ravel())),
        shape=(xy.shape[0], xy.shape[0]),
    )
    return graph.maximum(graph.T)


def refine_binary_mask_connectivity(
    xy, inside_mask, *, k_nn=10, min_component_size=10, max_hole_size=10
):
    """Remove tiny islands and fill tiny holes on the spatial kNN graph.

    This is a morphology-only robustness option. It does not alter the
    probability field or threshold and it deliberately preserves multiple
    sufficiently large components.
    """
    inside = np.asarray(inside_mask, dtype=bool).copy()
    graph = _knn_adjacency(xy, k_nn)

    def component_sizes(mask):
        nodes = np.flatnonzero(mask)
        if nodes.size == 0:
            return nodes, np.empty(0, dtype=int), np.empty(0, dtype=int)
        n_components, labels = connected_components(
            graph[nodes][:, nodes], directed=False, return_labels=True
        )
        return nodes, labels, np.bincount(labels, minlength=n_components)

    nodes, labels, sizes = component_sizes(inside)
    if int(min_component_size) > 1 and nodes.size:
        inside[nodes[sizes[labels] < int(min_component_size)]] = False

    outside_nodes, outside_labels, outside_sizes = component_sizes(~inside)
    if int(max_hole_size) > 0 and outside_nodes.size:
        small = outside_sizes[outside_labels] <= int(max_hole_size)
        # Only fill components fully enclosed by inside neighbours. Components
        # touching the tissue edge are retained as exterior background.
        xy_arr = np.asarray(xy, dtype=float)
        edge = (
            np.isclose(xy_arr[:, 0], np.nanmin(xy_arr[:, 0]))
            | np.isclose(xy_arr[:, 0], np.nanmax(xy_arr[:, 0]))
            | np.isclose(xy_arr[:, 1], np.nanmin(xy_arr[:, 1]))
            | np.isclose(xy_arr[:, 1], np.nanmax(xy_arr[:, 1]))
        )
        touches_edge = np.bincount(
            outside_labels, weights=edge[outside_nodes].astype(int),
            minlength=len(outside_sizes),
        ) > 0
        fill = small & ~touches_edge[outside_labels]
        inside[outside_nodes[fill]] = True
    return inside


def build_boundary_from_binary_mask(xy, inside_mask, k_nn=10):
    """Return the reference notebook's same-and-opposite-neighbour boundary."""
    xy = np.asarray(xy, dtype=float)
    inside = np.asarray(inside_mask, dtype=bool)
    k_nn = min(int(k_nn), xy.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1:]
    same = np.any(inside[idx] == inside[:, None], axis=1)
    opposite = np.any(inside[idx] != inside[:, None], axis=1)
    return same & opposite


def build_boundary_from_field(
    xy, field, level=0.5, k_nn=10, *, boundary_mode="legacy",
    min_component_size=10, max_hole_size=10,
):
    """Threshold a continuous field and construct the reference boundary."""
    field = np.asarray(field, dtype=float)
    if not np.all(np.isfinite(field)):
        raise ValueError("The boundary-defining field contains non-finite values.")
    inside = field >= float(level)
    if boundary_mode == "robust":
        inside = refine_binary_mask_connectivity(
            xy, inside, k_nn=k_nn,
            min_component_size=min_component_size,
            max_hole_size=max_hole_size,
        )
    elif boundary_mode != "legacy":
        raise ValueError("boundary_mode must be 'legacy' or 'robust'")
    boundary = build_boundary_from_binary_mask(xy, inside, k_nn=k_nn)
    return boundary, inside


def signed_distance_from_boundary_points(xy, boundary_mask, inside_mask):
    """Distance to nearest boundary point; tumor/inside side negative, outside positive."""
    xy = np.asarray(xy, dtype=float)
    b = np.asarray(boundary_mask, dtype=bool)
    inside = np.asarray(inside_mask, dtype=bool)
    if not np.any(b):
        raise ValueError("No boundary points were detected.")
    nn = NearestNeighbors(n_neighbors=1).fit(xy[b])
    dist = nn.kneighbors(xy, return_distance=True)[0][:, 0]
    return np.where(inside, -dist, dist)
