"""Offset-based graph convolution -- ported from the paper's actual released implementation
(github.com/SolarisAdams/ML4CFD-Offset-based-Graph-Convolution, submission/OffsetGNN.py:
CoordConv/CoordGNN), not from the paper text's section 4.2 formula. Their code and the paper's
written formula disagree on the attention function -- this module matches the code.

Concretely, vs. an earlier version of this file that implemented the paper's text literally:
  - attention is softmax(1/(||o_ij||_1 + eps)) (their actual code), not softmax(-||o_ij||^2)
    (what the paper text states).
  - ELU activations, not ReLU.
  - the per-layer fusion MLP is 4 Linear layers, not 2.
  - a self-loop edge is added to the graph, so a node's own (zero-offset) contribution competes
    inside the same softmax as its real neighbors, on top of the separate self_mlp branch.
  - an extra Linear projection of the ORIGINAL raw input features is concatenated (not added)
    right before the final decoder, alongside the usual residual adds between the two conv layers.
  - additive Gaussian noise is injected inside the model's forward() itself, gated by
    self.training, instead of by the training script -- eval-time is noise-free by construction
    (model.eval() disables it) rather than by remembering to pass noise_std=0 externally.

Not ported: their DGL-based stochastic neighbor-sampling training loop (MultiLayerNeighborSampler,
mini-batches of ~256 seed nodes). This module keeps this project's own memory-safe strategy
(edge-chunked + gradient-checkpointed full-graph aggregation, see train_offset_gnn.py) instead of
adding a DGL dependency and reimplementing their custom balanced sampler.

Because 1/(||o||_1+eps) is unbounded as offset->0 (unlike the paper-text Gaussian, which is
bounded <=1), softmax over it needs a proper numerically-stable two-pass max-subtraction --
naive exp() would overflow, especially with self-loops (offset=0 gives the largest possible
score, 1/eps).
"""

import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import coalesce

DEFAULT_EDGE_CHUNK_SIZE = 300_000
SOFTMAX_EPS = 0.001  # matches the reference implementation's dist_fn="eculidean" default


def _fusion_mlp(in_dim, hidden_dim, out_dim):
    """4 Linear layers w/ ELU, matching CoordConv.mlp / CoordGNN.out1 in the reference code."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _kernel_mlp(coord_dim, hidden_dim, out_dim):
    """2-layer MLP, matching CoordConv.kernel / mlp_self in the reference code."""
    return nn.Sequential(
        nn.Linear(coord_dim, hidden_dim), nn.ELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _l1_inverse_max_scores(pos, edge_index, eps, chunk_size):
    """Per-destination-node max of 1/(||o||_1+eps), for numerically-stable softmax.

    No gradient needed here -- softmax(x) == softmax(x - c) for any per-group constant c, so the
    max used purely for numerical stability can be computed under no_grad. Cheap relative to the
    main pass (no MLP calls), so it doesn't need checkpointing.
    """
    n = pos.size(0)
    max_score = torch.full((n,), float('-inf'), device=pos.device, dtype=pos.dtype)
    with torch.no_grad():
        num_edges = edge_index.size(1)
        for start in range(0, num_edges, chunk_size):
            chunk = edge_index[:, start:start + chunk_size]
            src, dst = chunk[0], chunk[1]
            offset = pos[dst] - pos[src]
            score = 1.0 / (offset.abs().sum(dim=-1) + eps)
            max_score.scatter_reduce_(0, dst, score, reduce='amax', include_self=True)
    return max_score


def _edge_chunk_terms(x, pos, edge_chunk, theta_mlp, max_score, eps):
    src, dst = edge_chunk[0], edge_chunk[1]
    offset = pos[dst] - pos[src]
    theta = theta_mlp(offset)
    score = 1.0 / (offset.abs().sum(dim=-1) + eps)
    exp_score = torch.exp(score - max_score[dst])
    numerator = exp_score.unsqueeze(-1) * theta * x[src]
    return numerator, exp_score


class OffsetGraphConv(nn.Module):
    """One CoordConv-equivalent layer: inverse-L1-distance softmax attention over
    kernel-weighted neighbor features, fused with a self-transform via a 4-layer MLP.

    Edges are processed in chunks under gradient checkpointing so peak memory stays bounded
    by `edge_chunk_size` regardless of total edge count or hidden_dim.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, coord_dim=3, eps=SOFTMAX_EPS):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.theta_mlp = _kernel_mlp(coord_dim, hidden_dim, in_dim)
        self.self_mlp = _kernel_mlp(in_dim, hidden_dim, hidden_dim)
        self.fusion_mlp = _fusion_mlp(in_dim + hidden_dim, hidden_dim, out_dim)

    def forward(self, x, pos, edge_index, edge_chunk_size=DEFAULT_EDGE_CHUNK_SIZE):
        n, device, dtype = x.size(0), x.device, x.dtype
        max_score = _l1_inverse_max_scores(pos, edge_index, self.eps, edge_chunk_size)

        numerator = torch.zeros(n, self.in_dim, device=device, dtype=dtype)
        denominator = torch.zeros(n, device=device, dtype=dtype)
        num_edges = edge_index.size(1)
        for start in range(0, num_edges, edge_chunk_size):
            edge_chunk = edge_index[:, start:start + edge_chunk_size]
            chunk_num, chunk_denom = checkpoint(
                _edge_chunk_terms, x, pos, edge_chunk, self.theta_mlp, max_score, self.eps,
                use_reentrant=False,
            )
            dst = edge_chunk[1]
            numerator.index_add_(0, dst, chunk_num)
            denominator.index_add_(0, dst, chunk_denom)

        aggregated = numerator / denominator.clamp_min(1e-12).unsqueeze(-1)
        return self.fusion_mlp(torch.cat([aggregated, self.self_mlp(x)], dim=-1))


class OffsetGNN(nn.Module):
    """CoordGNN-equivalent tower (one branch): 2 conv layers with pre-activation residuals,
    a fresh raw-input projection concatenated before the decoder, and training-time-only
    additive Gaussian noise on the input features (matches CoordGNN.forward's `if self.training`
    noise injection, gated automatically by model.eval()/.train()).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, coord_dim=3, noise_std=1e-5, eps=SOFTMAX_EPS):
        super().__init__()
        self.noise_std = noise_std
        self.conv1 = OffsetGraphConv(in_dim, hidden_dim, hidden_dim, coord_dim, eps)
        self.conv2 = OffsetGraphConv(hidden_dim, hidden_dim, hidden_dim, coord_dim, eps)
        self.skip_in = nn.Linear(in_dim, hidden_dim)      # matches reference's skip1
        self.skip_concat = nn.Linear(in_dim, hidden_dim)  # matches reference's skip4
        self.act = nn.ELU()
        self.decoder = _fusion_mlp(hidden_dim * 2, hidden_dim, out_dim)

    def forward(self, x, pos, edge_index, edge_chunk_size=DEFAULT_EDGE_CHUNK_SIZE):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # Checkpoint each conv call (not just its internal edge loop) and the decoder -- the
        # 4-layer fusion MLP (vs. the 2-layer one this replaced) makes the NODE-level activations
        # (self_mlp/fusion_mlp outputs, one (num_nodes, hidden_dim) tensor per layer) big enough
        # to OOM a small GPU on their own if retained for backward, on top of the already-chunked
        # edge-level ones.
        layer1_out = checkpoint(self.conv1, x, pos, edge_index, edge_chunk_size, use_reentrant=False) + self.skip_in(x)
        h = self.act(layer1_out)
        h = checkpoint(self.conv2, h, pos, edge_index, edge_chunk_size, use_reentrant=False) + layer1_out
        h = torch.cat([h, self.skip_concat(x)], dim=-1)
        h = self.act(h)
        return checkpoint(self.decoder, h, use_reentrant=False)


def build_offset_gnn_models(in_dim, coord_dim=3, velocity_hidden=256, turbulence_hidden=128):
    """Three dedicated models: velocity (2 components), pressure, turbulent viscosity.

    noise_std values match the reference's args.noise1/noise2/noise3 defaults (velocity, pressure,
    turbulent viscosity respectively).
    """
    return {
        'velocity': OffsetGNN(in_dim, velocity_hidden, out_dim=2, coord_dim=coord_dim, noise_std=1e-5),
        'pressure': OffsetGNN(in_dim, velocity_hidden, out_dim=1, coord_dim=coord_dim, noise_std=0.0),
        'turbulent_viscosity': OffsetGNN(in_dim, turbulence_hidden, out_dim=1, coord_dim=coord_dim, noise_std=1e-5),
    }


def build_hybrid_graph(pos, mesh_edge_index, k=10, add_self_loops=True):
    """Augment mesh connectivity with kNN edges (+ optional self-loops), matching the reference's
    dgl.knn_graph(k=10) + mesh edges + dgl.add_self_loop() graph construction.

    Uses sklearn's NearestNeighbors (brute-force/tree on CPU) rather than torch-cluster/dgl's GPU
    kNN, which aren't installed here. Fine for typical mesh sizes; swap in a GPU kNN for very
    large point clouds.
    """
    pos_np = pos.detach().cpu().numpy()
    n = pos_np.shape[0]
    _, indices = NearestNeighbors(n_neighbors=k + 1).fit(pos_np).kneighbors(pos_np)
    # indices[i, 1:] are node i's own k nearest neighbors (col 0 is i itself). Aggregation
    # happens into `dst` (see OffsetGraphConv's index_add_ on edge_index[1]), so node i must
    # be the target receiving from its own neighbors: edges are (src=neighbor_of_i, dst=i).
    src = torch.as_tensor(indices[:, 1:].reshape(-1), dtype=torch.long)
    dst = torch.arange(n).repeat_interleave(k)
    knn_edges = torch.stack([src, dst])
    edge_index = torch.cat([mesh_edge_index, knn_edges], dim=1)
    if add_self_loops:
        self_loops = torch.arange(n).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
    return coalesce(edge_index, num_nodes=n)


def estimate_softmax_eps(pos, edge_index):
    """Estimate a dataset-appropriate eps for the inverse-distance softmax, from the median
    ||o||_1 among real (non-self-loop) edges.

    The reference implementation's eps=0.001 is a magic constant implicitly calibrated to their
    2D airfoil dataset's mesh spacing -- on this project's 3D CRM mesh it's ~50x too small
    (median real-neighbor score ~20 vs. eps=0.001's self-loop score of 1000), which matters a lot
    if self-loops are enabled (verified empirically: it makes the self-loop win the softmax
    ~100% of the time, silently zeroing out all real spatial aggregation -- see
    train_offset_gnn.py's decision to disable self-loops for this dataset instead of rescaling
    eps further, since this mesh's ~68x nearest-neighbor-distance spread between p10 and p90
    means no single global eps avoids self-loop domination for a majority of nodes anyway).
    Even without self-loops, eps still sets how sharply the softmax differentiates near vs. far
    real neighbors, so it's worth calibrating to the data rather than reusing 0.001 blindly.
    """
    src, dst = edge_index[0], edge_index[1]
    non_self = src != dst
    l1 = (pos[dst[non_self]] - pos[src[non_self]]).abs().sum(dim=-1)
    return float(l1.median())


def boundary_aware_sample_weights(is_surface, is_near_wall, surface_factor=8.0, near_wall_factor=2.0):
    """Per-node sampling weights: surface nodes 8x, near-wall nodes 2x, others 1x."""
    weights = torch.ones_like(is_surface, dtype=torch.float32)
    weights[is_near_wall] = near_wall_factor
    weights[is_surface] = surface_factor
    return weights
