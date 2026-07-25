"""Train the offset-based graph convolution GNN (utils/offset_gnn.py) on the CRM wall dataset.

Adapts the architecture from paper section 4.2 to this project's actual data
(data/splitv2/*.npy), which differs from what the paper targets:
  - 3D wall points (x, y, z) instead of a 2D volume mesh -> coord_dim=3.
  - No stored mesh connectivity at all -> the graph is kNN-only (k=10), not a
    kNN augmentation of existing mesh edges.
  - A single scalar target per point (density-like field) instead of the
    paper's velocity/pressure/turbulent-viscosity triplet -> one OffsetGNN
    head instead of three.
  - Every point is already a wall point, so the paper's surface-vs-interior
    boundary-aware sampling doesn't apply here. Its place is taken by the
    existing per-component weighting (utils/base_models.py KL_WEIGHTS) as the
    "surface-specific regularization" analogue.

The (x, y, z) coordinates and surface normals are identical across all 252
simulations in this dataset (fixed CRM mesh, only Minf/AoA/Pi vary), so the
kNN graph and the normal/component node features are built once and cached,
then reused for every simulation and every epoch.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from utils.base_models import KL_WEIGHTS, evaluate
from utils.offset_gnn import OffsetGNN, build_hybrid_graph, add_gaussian_noise

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
DATA_DIR = 'data/'


def load_static_graph(data_dir, k, cache_dir):
    """Coordinates/normals/components are identical across all sims (fixed mesh) -- build once."""
    os.makedirs(cache_dir, exist_ok=True)
    edge_cache = os.path.join(cache_dir, f'knn_edge_index_k{k}.pt')

    X = np.load(data_dir + 'splitv2/train_data.npy', mmap_mode='r')
    coords = torch.from_numpy(np.asarray(X[:NWALLP, :3])).float()
    normals = torch.from_numpy(np.asarray(X[:NWALLP, 3:6])).float()

    if os.path.exists(edge_cache):
        edge_index = torch.load(edge_cache)
    else:
        print(f'Building kNN (k={k}) graph over {NWALLP} nodes...', flush=True)
        t0 = time.time()
        mesh_edge_index = torch.zeros((2, 0), dtype=torch.long)  # no stored mesh connectivity
        edge_index = build_hybrid_graph(coords, mesh_edge_index, k=k)
        print(f'  done in {time.time() - t0:.1f}s, {edge_index.shape[1]} edges', flush=True)
        torch.save(edge_index, edge_cache)

    component_labels = np.load(os.path.join(data_dir, 'component_labels_unique.npy'))
    with open(os.path.join(data_dir, 'component_map.json')) as f:
        component_map = {int(k_): v for k_, v in json.load(f).items()}
    n_components = len(component_map)
    component_onehot = torch.zeros(NWALLP, n_components)
    for cid, cname in component_map.items():
        component_onehot[torch.from_numpy(component_labels == cid), cid] = 1.0

    node_weight = torch.zeros(NWALLP)
    for cid, cname in component_map.items():
        node_weight[torch.from_numpy(component_labels == cid)] = KL_WEIGHTS.get(cname, 1.0)

    static_feats = torch.cat([normals, component_onehot], dim=1)  # (NWALLP, 3 + n_components)
    return coords, edge_index, static_feats, node_weight, component_labels, component_map


def load_split(data_dir, split):
    X = np.load(data_dir + f'splitv2/{split}_data.npy', mmap_mode='r')
    y = np.load(data_dir + f'splitv2/{split}_labels.npy')
    weights_path = data_dir + f'splitv2/{split}_weights.npy'
    weights = np.load(weights_path) if os.path.exists(weights_path) else None
    n_sims = X.shape[0] // NWALLP
    return X, y, weights, n_sims


def make_node_features(static_feats, cond, noise_std=0.0):
    cond_broadcast = cond.expand(static_feats.shape[0], -1)
    x = torch.cat([static_feats, cond_broadcast], dim=1)
    return add_gaussian_noise(x, std=noise_std)


def weighted_mse(pred, target, node_weight, sim_weight=1.0):
    return sim_weight * (node_weight * (pred - target).pow(2)).mean()


def predict_split(model, X, n_sims, coords, edge_index, static_feats, device):
    """Run the model over every simulation in a split and return flat (n_sims*NWALLP,) predictions."""
    model.eval()
    preds = np.empty(n_sims * NWALLP, dtype=np.float32)
    with torch.no_grad():
        for i in range(n_sims):
            block = np.asarray(X[i * NWALLP:(i + 1) * NWALLP])
            cond = torch.from_numpy(block[0, COL_MINF:COL_PI + 1]).float().to(device)
            x = make_node_features(static_feats, cond)
            preds[i * NWALLP:(i + 1) * NWALLP] = model(x, coords, edge_index).squeeze(-1).cpu().numpy()
    model.train()
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--k', type=int, default=10, help='kNN neighbors per node.')
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--noise_std', type=float, default=0.01, help='additive Gaussian noise on node features during training.')
    p.add_argument('--max_train_sims', type=int, default=None, help='cap number of training sims per epoch (for quick smoke tests).')
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_prefix', default='offset_gnn')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    print('Loading static graph/features...', flush=True)
    coords, edge_index, static_feats, node_weight, component_labels, component_map = load_static_graph(
        args.data_dir, args.k, cache_dir
    )
    coords, edge_index = coords.to(device), edge_index.to(device)
    static_feats, node_weight = static_feats.to(device), node_weight.to(device)
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    X_train, y_train, train_weights, n_train = load_split(args.data_dir, 'train')
    X_test1, y_test1, test1_weights, n_test1 = load_split(args.data_dir, 'test_phase1')
    print(f'n_train={n_train}  n_test1={n_test1}', flush=True)

    sigma_ref = 0.01 * float(np.mean(y_train))
    in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)
    model = OffsetGNN(in_dim, args.hidden_dim, out_dim=1, coord_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_sims_per_epoch = min(args.max_train_sims, n_train) if args.max_train_sims else n_train

    for epoch in range(args.epochs):
        sim_order = np.random.permutation(n_train)[:n_sims_per_epoch]
        epoch_loss = 0.0
        t0 = time.time()
        for i in sim_order:
            block = np.asarray(X_train[i * NWALLP:(i + 1) * NWALLP])
            cond = torch.from_numpy(block[0, COL_MINF:COL_PI + 1]).float().to(device)
            x = make_node_features(static_feats, cond, noise_std=args.noise_std)
            target = torch.from_numpy(np.asarray(y_train[i * NWALLP:(i + 1) * NWALLP])).float().to(device)
            sim_weight = float(train_weights[i]) if train_weights is not None else 1.0

            optimizer.zero_grad()
            pred = model(x, coords, edge_index).squeeze(-1)
            loss = weighted_mse(pred, target, node_weight, sim_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'epoch {epoch+1}/{args.epochs}  loss={epoch_loss / len(sim_order):.5f}  '
              f'({time.time() - t0:.1f}s)', flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            preds1 = predict_split(model, X_test1, n_test1, coords, edge_index, static_feats, device)
            w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
            res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref)
            print(f'  test_phase1  R2={res["r2"]:.4f}  worst_rMAE={res["worst_rmae"]:.4f}  '
                  f'mean_rMAE={res["mean_rmae"]:.4f}  mean_KLw={res["mean_klw"]:.4f}  '
                  f'max_KLw={res["max_klw"]:.4f}', flush=True)

    torch.save(model.state_dict(), f'{args.out_prefix}_model.pt')
    print(f'Saved: {args.out_prefix}_model.pt', flush=True)


if __name__ == '__main__':
    main()
