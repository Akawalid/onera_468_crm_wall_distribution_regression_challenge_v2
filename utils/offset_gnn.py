"""Offset-based graph convolution for mesh graphs (paper section 4.2, PointConv-inspired).

Reference: https://openreview.net/pdf/7ba472d6d5b9fcca692adb74f72c0d642c837f00.pdf
Code released alongside the paper: https://github.com/SolarisAdams/ML4CFD-Offset-based-Graph-Convolution

This module is a from-scratch, standalone reimplementation of the architecture described in
the paper (it does not use their code). Data pipeline glue (loading this project's meshes,
wiring up an optimizer/training loop) is intentionally left out since the CRM dataset in this
repo is a wall point cloud without the velocity/pressure/turbulent-viscosity volume fields the
paper targets -- adapt `build_offset_gnn_models` / `build_hybrid_graph` to your own graphs.
"""

import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, coalesce


def _mlp(dims, out_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or out_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OffsetGraphConv(MessagePassing):
    """One offset-based graph convolution layer.

    For each edge (j -> i): offset o_ij = pos_i - pos_j, dynamic kernel weights
    theta_ij = MLP_theta(o_ij), and inverse-distance attention
    alpha_ij = softmax_j(-||o_ij||^2). The message is alpha_ij * (theta_ij ⊙ h_j).
    Node update: MLP_out(MLP_self(h_i) ⊕ sum_j alpha_ij * (theta_ij ⊙ h_j)).
    """

    def __init__(self, in_dim, out_dim, coord_dim=2):
        super().__init__(aggr='add', flow='source_to_target')
        self.theta_mlp = _mlp([coord_dim, out_dim, in_dim])
        self.self_mlp = _mlp([in_dim, out_dim, out_dim])
        self.out_mlp = _mlp([in_dim + out_dim, out_dim, out_dim])

    def forward(self, x, pos, edge_index):
        aggregated = self.propagate(edge_index, x=x, pos=pos)
        return self.out_mlp(torch.cat([self.self_mlp(x), aggregated], dim=-1))

    def message(self, x_j, pos_i, pos_j, index, ptr, size_i):
        offset = pos_i - pos_j
        theta = self.theta_mlp(offset)
        raw_scores = -offset.pow(2).sum(dim=-1)
        alpha = softmax(raw_scores, index, ptr, size_i)
        return alpha.unsqueeze(-1) * (theta * x_j)


class OffsetGNN(nn.Module):
    """Two offset-based graph conv layers with skip connections + an MLP decoder head."""

    def __init__(self, in_dim, hidden_dim, out_dim, coord_dim=2, decoder_layers=4):
        super().__init__()
        self.conv1 = OffsetGraphConv(in_dim, hidden_dim, coord_dim)
        self.conv2 = OffsetGraphConv(hidden_dim, hidden_dim, coord_dim)
        self.skip1 = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.decoder = _mlp([hidden_dim] * decoder_layers + [out_dim])

    def forward(self, x, pos, edge_index):
        h1 = torch.relu(self.conv1(x, pos, edge_index) + self.skip1(x))
        h2 = torch.relu(self.conv2(h1, pos, edge_index) + h1)
        return self.decoder(h2)


def build_offset_gnn_models(in_dim, coord_dim=2, velocity_hidden=256, turbulence_hidden=128):
    """Three dedicated models: velocity (2 components), pressure, turbulent viscosity."""
    return {
        'velocity': OffsetGNN(in_dim, velocity_hidden, out_dim=2, coord_dim=coord_dim),
        'pressure': OffsetGNN(in_dim, velocity_hidden, out_dim=1, coord_dim=coord_dim),
        'turbulent_viscosity': OffsetGNN(in_dim, turbulence_hidden, out_dim=1, coord_dim=coord_dim),
    }


def build_hybrid_graph(pos, mesh_edge_index, k=10):
    """Augment mesh connectivity with kNN edges to smooth over mesh-segmentation jumps.

    Uses sklearn's NearestNeighbors (brute-force/tree on CPU) rather than torch-cluster,
    which isn't installed here. Fine for typical mesh sizes; swap in a GPU kNN for very
    large point clouds.
    """
    pos_np = pos.detach().cpu().numpy()
    n = pos_np.shape[0]
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(pos_np).kneighbors(pos_np)
    dst = torch.as_tensor(indices[:, 1:].reshape(-1), dtype=torch.long)  # drop self-match
    src = torch.arange(n).repeat_interleave(k)
    knn_edges = torch.stack([src, dst])
    edge_index = torch.cat([mesh_edge_index, knn_edges], dim=1)
    return coalesce(edge_index, num_nodes=n)


def boundary_aware_sample_weights(is_surface, is_near_wall, surface_factor=8.0, near_wall_factor=2.0):
    """Per-node sampling weights: surface nodes 8x, near-wall nodes 2x, others 1x."""
    weights = torch.ones_like(is_surface, dtype=torch.float32)
    weights[is_near_wall] = near_wall_factor
    weights[is_surface] = surface_factor
    return weights


def add_gaussian_noise(x, std=0.01):
    return x + torch.randn_like(x) * std if std > 0 else x


def surface_weighted_loss(pred, target, is_surface, surface_weight=2.0):
    """MSE with extra weight on surface nodes as a stand-in for the paper's tailored loss."""
    weights = torch.where(is_surface, surface_weight, 1.0)
    return (weights * (pred - target).pow(2)).mean()
