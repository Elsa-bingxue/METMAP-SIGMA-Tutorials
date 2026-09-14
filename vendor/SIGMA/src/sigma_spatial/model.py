import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GraphLowPass(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, edge_weight=None):
        n = x.size(0)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=n)
        row, col = edge_index
        # Preserve the normalization used in all reference notebooks.
        deg = degree(col, n, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class SpectralHighPassBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lowpass = GraphLowPass()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        hp = x - self.lowpass(x, edge_index, edge_weight)
        return self.dropout(F.relu(self.norm(self.lin(hp))))


class SpectralResidualNet(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, r_dim=None, z_dim=32, use_region_head=True):
        super().__init__()
        r_dim = in_dim if r_dim is None else r_dim
        self.hp1 = SpectralHighPassBlock(in_dim, hidden)
        self.hp2 = SpectralHighPassBlock(hidden, out_dim)
        # The BC515 notebook initializes the RNA projection before reconstruction.
        self.rna_head = nn.Linear(out_dim, z_dim)
        self.residual_head = nn.Linear(out_dim, r_dim)
        self.use_region_head = use_region_head
        self.region_head = nn.Linear(out_dim, 1) if use_region_head else None

    def forward(self, x, edge_index, edge_weight=None):
        h = self.hp2(self.hp1(x, edge_index, edge_weight), edge_index, edge_weight)
        out = {"r": self.residual_head(h), "z_hat": self.rna_head(h), "h": h}
        if self.region_head is not None:
            out["logit"] = self.region_head(h).squeeze(-1)
        return out
