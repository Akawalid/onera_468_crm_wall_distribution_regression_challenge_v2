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
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import coalesce

DEFAULT_EDGE_CHUNK_SIZE = 300_000


def _mlp(dims, out_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or out_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _edge_chunk_terms(x, pos, edge_chunk, theta_mlp):
    """Unnormalized softmax numerator/denominator terms for one chunk of edges.

    softmax(-||o_ij||^2) is expressed as exp(raw_ij) / sum_k exp(raw_ik) rather than via
    torch_geometric.utils.softmax, so it can be accumulated chunk-by-chunk with plain
    index_add_ instead of needing the whole edge set materialized for a single group-softmax.
    raw_ij = -||o_ij||^2 <= 0, so exp(raw_ij) in (0, 1] -- no max-subtraction needed for stability.
    """
    src, dst = edge_chunk[0], edge_chunk[1]
    offset = pos[dst] - pos[src]
    theta = theta_mlp(offset)
    exp_raw = torch.exp(-offset.pow(2).sum(dim=-1))
    numerator = exp_raw.unsqueeze(-1) * theta * x[src]
    return numerator, exp_raw


class OffsetGraphConv(nn.Module):
    """One offset-based graph convolution layer.

    For each edge (j -> i): offset o_ij = pos_i - pos_j, dynamic kernel weights
    theta_ij = MLP_theta(o_ij), and inverse-distance attention
    alpha_ij = softmax_j(-||o_ij||^2). The message is alpha_ij * (theta_ij ⊙ h_j).
    Node update: MLP_out(MLP_self(h_i) ⊕ sum_j alpha_ij * (theta_ij ⊙ h_j)).

    Edges are processed in chunks under gradient checkpointing so peak memory stays bounded
    by `edge_chunk_size` regardless of total edge count or hidden_dim -- a full 260k-node,
    k=10 graph at hidden_dim=256 would otherwise materialize O(2.6M x 256) tensors per layer.
    """

    def __init__(self, in_dim, out_dim, coord_dim=2):
        super().__init__()
        self.in_dim = in_dim
        self.theta_mlp = _mlp([coord_dim, out_dim, in_dim])
        self.self_mlp = _mlp([in_dim, out_dim, out_dim])
        self.out_mlp = _mlp([in_dim + out_dim, out_dim, out_dim])

    def forward(self, x, pos, edge_index, edge_chunk_size=DEFAULT_EDGE_CHUNK_SIZE):
        n, device, dtype = x.size(0), x.device, x.dtype
        numerator = torch.zeros(n, self.in_dim, device=device, dtype=dtype)
        denominator = torch.zeros(n, device=device, dtype=dtype)

        num_edges = edge_index.size(1)
        for start in range(0, num_edges, edge_chunk_size):
            edge_chunk = edge_index[:, start:start + edge_chunk_size]
            chunk_num, chunk_denom = checkpoint(
                _edge_chunk_terms, x, pos, edge_chunk, self.theta_mlp, use_reentrant=False
            )
            dst = edge_chunk[1]
            numerator.index_add_(0, dst, chunk_num)
            denominator.index_add_(0, dst, chunk_denom)

        aggregated = numerator / denominator.clamp_min(1e-12).unsqueeze(-1)
        return self.out_mlp(torch.cat([self.self_mlp(x), aggregated], dim=-1))


class OffsetGNN(nn.Module):
    """Two offset-based graph conv layers with skip connections + an MLP decoder head."""

    def __init__(self, in_dim, hidden_dim, out_dim, coord_dim=2, decoder_layers=4):
        super().__init__()
        self.conv1 = OffsetGraphConv(in_dim, hidden_dim, coord_dim)
        self.conv2 = OffsetGraphConv(hidden_dim, hidden_dim, coord_dim)
        self.skip1 = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        self.decoder = _mlp([hidden_dim] * decoder_layers + [out_dim])

    def forward(self, x, pos, edge_index, edge_chunk_size=DEFAULT_EDGE_CHUNK_SIZE):
        h1 = torch.relu(self.conv1(x, pos, edge_index, edge_chunk_size) + self.skip1(x))
        h2 = torch.relu(self.conv2(h1, pos, edge_index, edge_chunk_size) + h1)
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
    # indices[i, 1:] are node i's own k nearest neighbors (col 0 is i itself). Aggregation
    # happens into `dst` (see OffsetGraphConv's index_add_ on edge_index[1]), so node i must
    # be the target receiving from its own neighbors: edges are (src=neighbor_of_i, dst=i).
    src = torch.as_tensor(indices[:, 1:].reshape(-1), dtype=torch.long)
    dst = torch.arange(n).repeat_interleave(k)
    knn_edges = torch.stack([src, dst])
    edge_index = torch.cat([mesh_edge_index, knn_edges], dim=1)
    return coalesce(edge_index, num_nodes=n)


def add_gaussian_noise(x, std=0.01):
    return x + torch.randn_like(x) * std if std > 0 else x


def surface_weighted_loss(pred, target, is_surface, surface_weight=2.0):
    """MSE with extra weight on surface nodes as a stand-in for the paper's tailored loss."""
    weights = torch.where(is_surface, surface_weight, 1.0)
    return (weights * (pred - target).pow(2)).mean()





# """Train the offset-based graph convolution GNN (utils/offset_gnn.py) on the CRM wall dataset.

# Adapts the architecture from paper section 4.2 to this project's actual data
# (data/splitv2/*.npy), which differs from what the paper targets:
#   - 3D wall points (x, y, z) instead of a 2D volume mesh -> coord_dim=3.
#   - No stored mesh connectivity at all -> the graph is kNN-only (k=10), not a
#     kNN augmentation of existing mesh edges.
#   - A single scalar target per point (density-like field) instead of the
#     paper's velocity/pressure/turbulent-viscosity triplet -> one OffsetGNN
#     head instead of three.
#   - Every point is already a wall point, so the paper's surface-vs-interior
#     boundary-aware sampling doesn't apply here.

# Component (wing/pylon/fuselage/nacelle) one-hot is still fed in as a node feature so the
# model can tell them apart, but utils/base_models.py's KL_WEIGHTS is deliberately NOT used to
# weight the training loss (an earlier version of this script did that -- since every component
# gets a similar 0.2-0.3 weight, it wasn't achieving any differential emphasis, it was just
# uniformly shrinking the loss/gradient to ~25% of plain MSE, i.e. a stealth smaller learning
# rate for no benefit). KL_WEIGHTS is still used, as elsewhere in this repo, inside evaluate()'s
# KLw metric at evaluation time -- that's its actual intended purpose.

# The (x, y, z) coordinates and surface normals are identical across all 252
# simulations in this dataset (fixed CRM mesh, only Minf/AoA/Pi vary), so the
# kNN graph and the normal/component node features are built once and cached,
# then reused for every simulation and every epoch.
# """

# import argparse
# import json
# import os
# import sys
# import time

# import numpy as np
# import torch

# # Make `utils` importable as a package regardless of the process's cwd at launch --
# # `python -m utils.train_offset_gnn` only resolves the `utils` package if the repo root
# # happens to be the current working directory, which isn't guaranteed under Slurm/sbatch.
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.base_models import evaluate
# from utils.offset_gnn import OffsetGNN, build_hybrid_graph, add_gaussian_noise, DEFAULT_EDGE_CHUNK_SIZE

# NWALLP = 260774
# COL_MINF, COL_AOA, COL_PI = 6, 7, 8
# DATA_DIR = 'data/'


# def load_static_graph(data_dir, k, cache_dir):
#     """Coordinates/normals/components are identical across all sims (fixed mesh) -- build once."""
#     os.makedirs(cache_dir, exist_ok=True)
#     edge_cache = os.path.join(cache_dir, f'knn_edge_index_k{k}.pt')

#     X = np.load(data_dir + 'splitv2/train_data.npy', mmap_mode='r')
#     coords = torch.from_numpy(np.asarray(X[:NWALLP, :3])).float()
#     normals = torch.from_numpy(np.asarray(X[:NWALLP, 3:6])).float()

#     if os.path.exists(edge_cache):
#         edge_index = torch.load(edge_cache)
#     else:
#         print(f'Building kNN (k={k}) graph over {NWALLP} nodes...', flush=True)
#         t0 = time.time()
#         mesh_edge_index = torch.zeros((2, 0), dtype=torch.long)  # no stored mesh connectivity
#         edge_index = build_hybrid_graph(coords, mesh_edge_index, k=k)
#         print(f'  done in {time.time() - t0:.1f}s, {edge_index.shape[1]} edges', flush=True)
#         torch.save(edge_index, edge_cache)

#     component_labels = np.load(os.path.join(data_dir, 'component_labels_unique.npy'))
#     with open(os.path.join(data_dir, 'component_map.json')) as f:
#         component_map = {int(k_): v for k_, v in json.load(f).items()}
#     n_components = len(component_map)
#     component_onehot = torch.zeros(NWALLP, n_components)
#     for cid, cname in component_map.items():
#         component_onehot[torch.from_numpy(component_labels == cid), cid] = 1.0

#     static_feats = torch.cat([normals, component_onehot], dim=1)  # (NWALLP, 3 + n_components)
#     return coords, edge_index, static_feats, component_labels, component_map


# def load_split(data_dir, split):
#     X = np.load(data_dir + f'splitv2/{split}_data.npy', mmap_mode='r')
#     y = np.load(data_dir + f'splitv2/{split}_labels.npy')
#     weights_path = data_dir + f'splitv2/{split}_weights.npy'
#     weights = np.load(weights_path) if os.path.exists(weights_path) else None
#     n_sims = X.shape[0] // NWALLP
#     # one row per sim is enough to get (Minf, AoA, Pi) -- conditions are constant within a sim,
#     # so this avoids paging in the full 260774-row block just to read 3 numbers from it.
#     conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1][:n_sims])
#     return y, weights, n_sims, conds


# def make_node_features(static_feats, cond, noise_std=0.0):
#     cond_broadcast = cond.expand(static_feats.shape[0], -1)
#     x = torch.cat([static_feats, cond_broadcast], dim=1)
#     return add_gaussian_noise(x, std=noise_std)


# def predict_split(model, n_sims, conds, coords, edge_index, static_feats, device, edge_chunk_size, cond_mean, cond_std):
#     """Run the model over every simulation in a split and return flat (n_sims*NWALLP,) predictions."""
#     model.eval()
#     preds = np.empty(n_sims * NWALLP, dtype=np.float32)
#     with torch.no_grad():
#         for i in range(n_sims):
#             cond = torch.from_numpy((conds[i] - cond_mean) / cond_std).float().to(device)
#             x = make_node_features(static_feats, cond)
#             preds[i * NWALLP:(i + 1) * NWALLP] = model(x, coords, edge_index, edge_chunk_size).squeeze(-1).cpu().numpy()
#     model.train()
#     return preds


# def main():
#     p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
#     p.add_argument('--data_dir', default=DATA_DIR)
#     p.add_argument('--k', type=int, default=10, help='kNN neighbors per node.')
#     p.add_argument('--hidden_dim', type=int, default=256)
#     p.add_argument('--epochs', type=int, default=20)
#     p.add_argument('--lr', type=float, default=1e-3)
#     p.add_argument('--noise_std', type=float, default=0.01, help='additive Gaussian noise on node features during training.')
#     p.add_argument('--edge_chunk_size', type=int, default=DEFAULT_EDGE_CHUNK_SIZE,
#                    help='edges processed per gradient-checkpointed chunk in OffsetGraphConv; '
#                         'lower this if you hit CUDA OOM, raise it (or set to a huge number) for more speed on big GPUs.')
#     p.add_argument('--max_train_sims', type=int, default=None, help='cap number of training sims per epoch (for quick smoke tests).')
#     p.add_argument('--eval_every', type=int, default=5)
#     p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
#     p.add_argument('--out_prefix', default='offset_gnn')
#     p.add_argument('--seed', type=int, default=0)
#     args = p.parse_args()

#     torch.manual_seed(args.seed)
#     device = torch.device(args.device)
#     cache_dir = os.path.join(args.data_dir, 'cache')

#     print('Loading static graph/features...', flush=True)
#     coords, edge_index, static_feats, component_labels, component_map = load_static_graph(
#         args.data_dir, args.k, cache_dir
#     )
#     coords, edge_index = coords.to(device), edge_index.to(device)
#     static_feats = static_feats.to(device)
#     comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

#     y_train, train_weights, n_train, train_conds = load_split(args.data_dir, 'train')
#     y_test1, test1_weights, n_test1, test1_conds = load_split(args.data_dir, 'test_phase1')
#     print(f'n_train={n_train}  n_test1={n_test1}', flush=True)

#     # standardize (Minf, AoA, Pi) using train statistics, same convention as the rest of this
#     # repo (gp_pod_model.py / train.py both fit a StandardScaler on train_conds).
#     cond_mean = train_conds.mean(axis=0)
#     cond_std = train_conds.std(axis=0)
#     cond_std[cond_std == 0] = 1.0

#     sigma_ref = 0.01 * float(np.mean(y_train))
#     in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)
#     model = OffsetGNN(in_dim, args.hidden_dim, out_dim=1, coord_dim=3).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     n_sims_per_epoch = min(args.max_train_sims, n_train) if args.max_train_sims else n_train

#     for epoch in range(args.epochs):
#         sim_order = np.random.permutation(n_train)[:n_sims_per_epoch]
#         epoch_loss = 0.0
#         t0 = time.time()
#         for i in sim_order:
#             cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
#             x = make_node_features(static_feats, cond, noise_std=args.noise_std)
#             target = torch.from_numpy(np.asarray(y_train[i * NWALLP:(i + 1) * NWALLP])).float().to(device)
#             sim_weight = float(train_weights[i]) if train_weights is not None else 1.0

#             optimizer.zero_grad()
#             pred = model(x, coords, edge_index, args.edge_chunk_size).squeeze(-1)
#             loss = sim_weight * (pred - target).pow(2).mean()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         print(f'epoch {epoch+1}/{args.epochs}  loss={epoch_loss / len(sim_order):.5f}  '
#               f'({time.time() - t0:.1f}s)', flush=True)

#         if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
#             preds1 = predict_split(model, n_test1, test1_conds, coords, edge_index, static_feats,
#                                     device, args.edge_chunk_size, cond_mean, cond_std)
#             w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
#             res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref)
#             print(f'  test_phase1  R2={res["r2"]:.4f}  worst_rMAE={res["worst_rmae"]:.4f}  '
#                   f'mean_rMAE={res["mean_rmae"]:.4f}  mean_KLw={res["mean_klw"]:.4f}  '
#                   f'max_KLw={res["max_klw"]:.4f}', flush=True)
#             torch.save(model.state_dict(), f'{args.out_prefix}_model.pt')

#     torch.save(model.state_dict(), f'{args.out_prefix}_model.pt')
#     print(f'Saved: {args.out_prefix}_model.pt', flush=True)


# if __name__ == '__main__':
#     main()

