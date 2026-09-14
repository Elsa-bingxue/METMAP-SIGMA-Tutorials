import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from .preprocessing import get_msi_matrix, msi_to_embedding
from .graph import gaussian_knn_graph, gaussian_label_smoothing, smooth_embedding
from .model import SpectralResidualNet
from .boundary import build_boundary_from_field, signed_distance_from_boundary_points
from .utils import set_seed


class SIGMA:
    """Minimal end-to-end SIGMA estimator.

    Core workflow only: MSI embedding -> spatial graph -> Gaussian low-pass prior ->
    spectral residual learning -> region field -> boundary -> signed distance.
    """
    def __init__(self, n_components=64, k=15, seed=0, device=None):
        self.n_components = n_components
        self.k = k
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def fit(self, adata, annotation_key="annotation", tumor_label="Tumor", stroma_label="Stroma",
            rna_key="X_harmony", z_dim=32, epochs=1000, lr=1e-3,
            lambda_sup=0.05, beta_rna=0.05, boundary_k=10,
            boundary_mode="legacy", boundary_min_component_size=10,
            boundary_max_hole_size=10):
        set_seed(self.seed)
        if "spatial" not in adata.obsm:
            raise KeyError("adata.obsm['spatial'] is required")
        if annotation_key not in adata.obs:
            raise KeyError(f"adata.obs[{annotation_key!r}] is required")
        if rna_key not in adata.obsm:
            raise KeyError(f"adata.obsm[{rna_key!r}] is required")

        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        ann = adata.obs[annotation_key].astype(str).to_numpy()
        x_msi = get_msi_matrix(adata)
        e = msi_to_embedding(x_msi, self.n_components, self.seed)
        edge_index, edge_w, sigma = gaussian_knn_graph(xy, self.k, symmetrize=False)
        e_gauss = smooth_embedding(edge_index, edge_w, e)
        r_target = e - e_gauss

        y = np.full(adata.n_obs, -1, dtype=np.int64)
        y[ann == tumor_label] = 1
        y[ann == stroma_label] = 0
        mask = y >= 0
        if not np.any(y[mask] == 1) or not np.any(y[mask] == 0):
            raise ValueError("Both tumor and stroma anchors are required.")
        f_gauss = gaussian_label_smoothing(
            edge_index, edge_w, y, mask, n_iter=50, alpha=0.85
        )

        z = np.asarray(adata.obsm[rna_key][:, :z_dim], dtype=np.float32)
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        data = Data(
            x=torch.tensor(e_gauss, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_w, dtype=torch.float32),
        ).to(self.device)
        r_t = torch.tensor(r_target, dtype=torch.float32, device=self.device)
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device)
        idx = torch.tensor(np.where(mask)[0], dtype=torch.long, device=self.device)
        y_t = torch.tensor(y[mask], dtype=torch.float32, device=self.device)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model_ = SpectralResidualNet(e.shape[1], hidden=128, out_dim=e.shape[1], r_dim=e.shape[1], z_dim=z.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=lr, weight_decay=1e-4)
        pos, neg = (y[mask] == 1).sum(), (y[mask] == 0).sum()
        # Preserve the dtype inferred from the NumPy count ratio in cell 10.
        bce_weight = torch.tensor([neg / max(pos, 1)], device=self.device)

        self.loss_history_ = []
        for _ in range(int(epochs)):
            self.model_.train(); opt.zero_grad()
            out = self.model_(data.x, data.edge_index, data.edge_attr)
            loss_rec = F.smooth_l1_loss(out["r"], r_t)
            loss_sup = F.binary_cross_entropy_with_logits(out["logit"][idx], y_t, pos_weight=bce_weight)
            loss_rna = F.mse_loss(out["z_hat"], z_t)
            loss = loss_rec + lambda_sup * loss_sup + beta_rna * loss_rna
            loss.backward(); opt.step()
            self.loss_history_.append(float(loss.detach().cpu()))

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(data.x, data.edge_index, data.edge_attr)
            r = out["r"].cpu().numpy()
            logit = out["logit"].detach().cpu().numpy()
            p_raw = 1.0 / (1.0 + np.exp(-logit))
            p = (p_raw - np.nanmin(p_raw)) / (
                np.nanmax(p_raw) - np.nanmin(p_raw) + 1e-8
            )

        xy = np.asarray(adata.obsm["spatial"], dtype=float)
        boundary, inside = build_boundary_from_field(
            xy, p, level=0.5, k_nn=boundary_k,
            boundary_mode=boundary_mode,
            min_component_size=boundary_min_component_size,
            max_hole_size=boundary_max_hole_size,
        )
        d_signed = signed_distance_from_boundary_points(xy, boundary, inside)

        adata.obsm["X_sigma_msi"] = e
        adata.obsm["X_sigma_gauss"] = e_gauss
        adata.obsm["X_sigma_residual"] = r
        adata.obsm["X_sigma_corrected"] = e_gauss + r
        adata.obs["sigma_gaussian_anchor"] = f_gauss
        adata.obs["sigma_region_probability_raw"] = p_raw
        adata.obs["sigma_region_probability"] = p
        adata.obs["sigma_inside"] = inside
        adata.obs["sigma_boundary"] = boundary
        adata.obs["sigma_d_signed"] = d_signed
        adata.uns["sigma"] = {
            "sigma": sigma, "k": self.k, "seed": self.seed,
            "boundary_mode": boundary_mode,
            "boundary_min_component_size": int(boundary_min_component_size),
            "boundary_max_hole_size": int(boundary_max_hole_size),
        }
        self.adata_ = adata
        return self
