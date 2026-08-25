"""Distance-weighted GNN (GraphSAGE-style, with LayerNorm + residuals) -- a still-lightweight
baseline for finding the point where GNNs start beating the pointwise baselines
(utils/base_models.py's XGBoost/LightGBM, the conditions-only GlobalMLP in
data_exploration_splitv2_v2.ipynb) on this challenge, before paying for
utils/offset_gnn.py's much heavier offset-attention architecture.

Still no per-edge MLP, no learned attention, no edge chunking or gradient checkpointing: the
per-edge weight is a fixed geometric quantity (inverse distance) computed once from the static
mesh, not a function of learned features, so aggregation is still a single weighted scatter-sum
per layer -- node-level tensors only, no (num_edges, hidden_dim) intermediates.
"""

import torch
import torch.nn as nn

from utils.offset_gnn import build_hybrid_graph


def compute_inverse_distance_weights(pos, edge_index, eps=1e-6):
    """Per-edge weight 1/(||pos[dst]-pos[src]||+eps), normalized to sum to 1 over each dst's
    incoming edges -- a fixed (non-learned) geometric quantity, so it only needs computing once
    for this dataset's static mesh and can be reused across every layer, every simulation, and
    every epoch (unlike OffsetGNN's theta_mlp(offset), which is learned and depends on the
    current layer's weights).

    Replaces the plain unweighted mean SimpleGraphConv used to use: a neighbor 1cm away no
    longer counts the same as one 10cm away, which matters most exactly where node density
    varies most sharply -- e.g. near shocks -- and is where OffsetGNN's much heavier attention
    earns most of its advantage over a plain mean-aggregator.
    """
    src, dst = edge_index[0], edge_index[1]
    n = pos.size(0)
    dist = (pos[dst] - pos[src]).norm(dim=-1)
    w = 1.0 / (dist + eps)
    w_sum = torch.zeros(n, device=pos.device, dtype=pos.dtype)
    w_sum.index_add_(0, dst, w)
    return w / w_sum[dst].clamp_min(eps)


class SimpleGraphConv(nn.Module):
    """One distance-weighted aggregation layer:
    h_i' = W_self x_i + W_neigh * sum_{j in N(i)} edge_weight_ij * x_j
    (edge_weight already normalized to sum to 1 per destination node, see
    compute_inverse_distance_weights, so this is a weighted mean, not a weighted sum).
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        src, dst = edge_index[0], edge_index[1]
        n = x.size(0)
        weighted = torch.zeros(n, x.size(1), device=x.device, dtype=x.dtype)
        weighted.index_add_(0, dst, edge_weight.unsqueeze(-1) * x[src])
        return self.lin_self(x) + self.lin_neigh(weighted)


class SimpleGNN(nn.Module):
    """`n_layers` of SimpleGraphConv -> LayerNorm -> ReLU -> Dropout, with a residual add on
    every layer past the first (the first layer changes dimensionality in_dim -> hidden_dim, so
    it has nothing shape-compatible to add against; every later layer is hidden_dim ->
    hidden_dim and residual-adds its input, both to ease gradient flow at n_layers > 2 and so
    each layer only has to learn a refinement instead of re-deriving the full representation).
    A linear decoder maps the final hidden state to out_dim.

    edge_index should NOT have self-loops (see build_knn_graph): SimpleGraphConv already has a
    separate lin_self term for each node's own features (standard GraphSAGE-style aggregator),
    so a self-loop edge would double-count them inside the neighbor aggregate too.

    dropout (default 0.2, matching GlobalMLP in data_exploration_splitv2_v2.ipynb) plus L2
    weight decay on the optimizer (see train_simple_gnn.py) are this model's defense against
    overfitting given how few training simulations (252) there are relative to how many times
    the model sees each one over 100 epochs.
    """

    def __init__(self, in_dim, hidden_dim, out_dim=1, n_layers=3, dropout=0.2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(
            SimpleGraphConv(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(n_layers))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            out = self.dropout(self.act(norm(conv(h, edge_index, edge_weight))))
            h = h + out if i > 0 else out
        return self.decoder(h)


def build_knn_graph(pos, k=10):
    """Plain kNN graph, no self-loops (see SimpleGraphConv). Reuses offset_gnn.build_hybrid_graph
    (there's no stored mesh connectivity for this dataset either, same as train_offset_gnn.py) --
    same call as that file's load_static_graph makes, so this produces the exact same graph and
    can share its "_noselfloop" edge cache (at the same k).
    """
    mesh_edge_index = torch.zeros((2, 0), dtype=torch.long)
    return build_hybrid_graph(pos, mesh_edge_index, k=k, add_self_loops=False)
