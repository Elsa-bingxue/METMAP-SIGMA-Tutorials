import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler


def get_msi_matrix(adata, layer=None):
    """Return an n_spots x n_features MSI matrix from AnnData."""
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found in adata.layers")
        return adata.layers[layer]
    if "msi" in adata.uns:
        x = adata.uns["msi"]
        if hasattr(x, "shape") and x.shape[0] == adata.n_obs:
            return x
    if "raw" in adata.layers:
        x = adata.layers["raw"]
        if hasattr(x, "shape") and x.shape[0] == adata.n_obs:
            return x
    return adata.X


def resolve_sm_matrix(adata, matrix_source="X"):
    """Resolve an explicit SM matrix without silently changing preprocessing.

    Joint objects are subset by ``feature_type``/``type == 'SM'`` before a
    matrix or layer is selected. ``msi_uns`` is assumed to already be SM-only.
    """
    source = str(matrix_source)
    if source == "sm_var":
        if "SM_features" not in adata.uns:
            raise KeyError("adata.uns['SM_features'] is required for matrix_source='sm_var'")
        requested = np.asarray(adata.uns["SM_features"]).astype(str)
        var_names = np.asarray(adata.var_names).astype(str)
        if len(np.unique(var_names)) != len(var_names):
            raise ValueError("adata.var_names must be unique for matrix_source='sm_var'")
        lookup = {name: j for j, name in enumerate(var_names)}
        missing = [name for name in requested if name not in lookup]
        if missing:
            raise KeyError(
                f"{len(missing)} SM_features are absent from adata.var_names; "
                f"first missing feature: {missing[0]!r}"
            )
        indices = np.asarray([lookup[name] for name in requested], dtype=int)
        matrix = adata[:, indices].X
        mz = adata.uns.get("SM_mz")
        names = np.asarray(mz if mz is not None else requested).astype(str)
        if len(names) != matrix.shape[1]:
            raise ValueError("adata.uns['SM_mz'] does not match adata.uns['SM_features']")
        return matrix, names
    if source == "msi_uns":
        if "msi" not in adata.uns:
            raise KeyError("adata.uns['msi'] is required for matrix_source='msi_uns'")
        matrix = adata.uns["msi"]
        if matrix.shape[0] != adata.n_obs:
            raise ValueError("adata.uns['msi'] has incompatible spot count")
        names = np.asarray(adata.uns.get("mz_features", [f"mz_{j}" for j in range(matrix.shape[1])])).astype(str)
        if len(names) != matrix.shape[1]:
            raise ValueError("adata.uns['mz_features'] does not match adata.uns['msi']")
        return matrix, names
    mask = None
    for key in ("feature_type", "type"):
        if key in adata.var:
            candidate = adata.var[key].astype(str).to_numpy() == "SM"
            if candidate.any():
                mask = candidate
                break
    view = adata[:, mask] if mask is not None else adata
    if source == "X":
        matrix = view.X
    elif source == "raw":
        if "raw" not in view.layers:
            raise KeyError("adata.layers['raw'] is required for matrix_source='raw'")
        matrix = view.layers["raw"]
    elif source.startswith("layer:"):
        layer = source.split(":", 1)[1]
        if layer not in view.layers:
            raise KeyError(f"adata.layers[{layer!r}] is required")
        matrix = view.layers[layer]
    else:
        raise ValueError(
            "matrix_source must be 'X', 'raw', 'msi_uns', 'sm_var', or 'layer:<name>'"
        )
    names = view.var.get("feature_name", view.var_names.to_series()).astype(str).to_numpy()
    return matrix, names


def msi_to_embedding(x, n_components=64, random_state=0):
    """log1p + robust scaling (dense only) + TruncatedSVD + z-score."""
    max_components = min(x.shape) - 1
    k = min(int(n_components), max_components)
    if k < 1:
        raise ValueError("MSI matrix is too small for SVD.")
    if issparse(x):
        xlog = x.copy()
        xlog.data = np.log1p(xlog.data)
        e = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(xlog)
    else:
        xlog = np.log1p(np.asarray(x, dtype=np.float32))
        xs = RobustScaler(quantile_range=(10, 90)).fit_transform(xlog)
        e = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(xs)
    e = np.asarray(e, dtype=np.float32)
    return (e - e.mean(0)) / (e.std(0) + 1e-6)
