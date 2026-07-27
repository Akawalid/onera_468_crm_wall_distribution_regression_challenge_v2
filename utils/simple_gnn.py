"""Simple 2-layer mean-aggregation GNN (GraphSAGE-style) -- a deliberately minimal baseline
for finding the point where GNNs start beating the pointwise baselines (utils/base_models.py's
XGBoost/LightGBM, the conditions-only GlobalMLP in data_exploration_splitv2_v2.ipynb) on this
challenge, before paying for utils/offset_gnn.py's much heavier offset-attention architecture.

No coordinate-offset attention, no per-edge MLP, no edge chunking or gradient checkpointing:
mean-aggregating node features over kNN neighbors only ever materializes node-level tensors
(no (num_edges, hidden_dim) intermediates), so a single full-graph forward pass over this
mesh's ~260k nodes fits comfortably without the memory tricks OffsetGNN needs.
"""

import torch
import torch.nn as nn

from utils.offset_gnn import build_hybrid_graph


class SimpleGraphConv(nn.Module):
    """One mean-aggregation layer: h_i' = W_self x_i + W_neigh * mean_{j in N(i)} x_j."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)
        neigh_sum = torch.zeros(n, x.size(1), device=x.device, dtype=x.dtype)
        neigh_sum.index_add_(0, dst, x[src])
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones(src.size(0), device=x.device, dtype=x.dtype))
        neigh_mean = neigh_sum / deg.clamp_min(1.0).unsqueeze(-1)
        return self.lin_self(x) + self.lin_neigh(neigh_mean)


class SimpleGNN(nn.Module):
    """`n_layers` of SimpleGraphConv + ReLU + Dropout, then a linear decoder to out_dim.

    edge_index should NOT have self-loops (see build_knn_graph): SimpleGraphConv already has a
    separate lin_self term for each node's own features (standard GraphSAGE-style mean
    aggregator), so a self-loop edge would double-count them inside the neighbor-mean too.

    dropout is applied after each hidden layer's activation (not after the final decoder,
    same convention as GlobalMLP in data_exploration_splitv2_v2.ipynb) -- combined with L2
    weight decay on the optimizer (see train_simple_gnn.py), this is this model's only defense
    against overfitting given how few training simulations (252) there are relative to how many
    times the model sees each one over 100 epochs.
    """

    def __init__(self, in_dim, hidden_dim, out_dim=1, n_layers=2, dropout=0.0):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(
            SimpleGraphConv(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = self.dropout(self.act(conv(h, edge_index)))
        return self.decoder(h)


def build_knn_graph(pos, k=10):
    """Plain kNN graph, no self-loops (see SimpleGraphConv). Reuses offset_gnn.build_hybrid_graph
    (there's no stored mesh connectivity for this dataset either, same as train_offset_gnn.py) --
    same call as that file's load_static_graph makes, so this produces the exact same graph and
    can share its "_noselfloop" edge cache.
    """
    mesh_edge_index = torch.zeros((2, 0), dtype=torch.long)
    return build_hybrid_graph(pos, mesh_edge_index, k=k, add_self_loops=False)
