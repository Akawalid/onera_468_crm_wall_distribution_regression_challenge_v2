"""Train the offset-based graph convolution GNN (utils/offset_gnn.py, ported from the reference
implementation's CoordConv/CoordGNN) on the CRM wall dataset.

Adapts the architecture to this project's actual data (data/splitv2/*.npy), which differs from
what the paper targets:
  - 3D wall points (x, y, z) instead of a 2D volume mesh -> coord_dim=3.
  - No stored mesh connectivity at all -> the graph is kNN-only (k=10, + self-loops), not a
    kNN augmentation of existing mesh edges.
  - A single scalar target per point (density-like field) instead of the
    paper's velocity/pressure/turbulent-viscosity triplet -> one OffsetGNN
    head instead of three.
  - Every point is already a wall point, so the paper's surface-vs-interior
    boundary-aware sampling doesn't apply here.
  - The reference trains via DGL stochastic neighbor sampling (mini-batches of ~256 seed nodes,
    15 neighbors/layer); this script keeps full-graph training instead, relying on
    OffsetGraphConv's edge-chunked gradient checkpointing to stay within GPU memory.

Component (wing/pylon/fuselage/nacelle) one-hot is still fed in as a node feature so the
model can tell them apart, but utils/base_models.py's KL_WEIGHTS is deliberately NOT used to
weight the training loss (an earlier version of this script did that -- since every component
gets a similar 0.2-0.3 weight, it wasn't achieving any differential emphasis, it was just
uniformly shrinking the loss/gradient to ~25% of plain MSE, i.e. a stealth smaller learning
rate for no benefit). KL_WEIGHTS is still used, as elsewhere in this repo, inside evaluate()'s
KLw metric at evaluation time -- that's its actual intended purpose.

The (x, y, z) coordinates and surface normals are identical across all 252
simulations in this dataset (fixed CRM mesh, only Minf/AoA/Pi vary), so the
kNN graph and the normal/component node features are built once and cached,
then reused for every simulation and every epoch.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Make `utils` importable as a package regardless of the process's cwd at launch --
# `python -m utils.train_offset_gnn` only resolves the `utils` package if the repo root
# happens to be the current working directory, which isn't guaranteed under Slurm/sbatch.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_models import evaluate
from utils.offset_gnn import OffsetGNN, build_hybrid_graph, estimate_softmax_eps, DEFAULT_EDGE_CHUNK_SIZE

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
DATA_DIR = 'data/'


def load_static_graph(data_dir, k, cache_dir):
    """Coordinates/normals/components are identical across all sims (fixed mesh) -- build once."""
    os.makedirs(cache_dir, exist_ok=True)
    edge_cache = os.path.join(cache_dir, f'knn_edge_index_k{k}_noselfloop.pt')

    X = np.load(data_dir + 'splitv2/train_data.npy', mmap_mode='r')
    coords = torch.from_numpy(np.asarray(X[:NWALLP, :3])).float()
    normals = torch.from_numpy(np.asarray(X[:NWALLP, 3:6])).float()

    if os.path.exists(edge_cache):
        edge_index = torch.load(edge_cache)
    else:
        print(f'Building kNN (k={k}) graph over {NWALLP} nodes...', flush=True)
        t0 = time.time()
        mesh_edge_index = torch.zeros((2, 0), dtype=torch.long)  # no stored mesh connectivity
        # add_self_loops=False: verified empirically that with this mesh's 67.5x density spread
        # (p10 to p90 nearest-neighbor distance), a self-loop's inverse-distance score dominates
        # the softmax for the majority of nodes regardless of how eps is calibrated -- it would
        # zero out real spatial aggregation for most of the graph. See estimate_softmax_eps.
        edge_index = build_hybrid_graph(coords, mesh_edge_index, k=k, add_self_loops=False)
        print(f'  done in {time.time() - t0:.1f}s, {edge_index.shape[1]} edges', flush=True)
        torch.save(edge_index, edge_cache)

    component_labels = np.load(os.path.join(data_dir, 'component_labels_unique.npy'))
    with open(os.path.join(data_dir, 'component_map.json')) as f:
        component_map = {int(k_): v for k_, v in json.load(f).items()}
    n_components = len(component_map)
    component_onehot = torch.zeros(NWALLP, n_components)
    for cid, cname in component_map.items():
        component_onehot[torch.from_numpy(component_labels == cid), cid] = 1.0

    static_feats = torch.cat([normals, component_onehot], dim=1)  # (NWALLP, 3 + n_components)
    return coords, edge_index, static_feats, component_labels, component_map


def load_split(data_dir, split):
    X = np.load(data_dir + f'splitv2/{split}_data.npy', mmap_mode='r')
    y = np.load(data_dir + f'splitv2/{split}_labels.npy')
    weights_path = data_dir + f'splitv2/{split}_weights.npy'
    weights = np.load(weights_path) if os.path.exists(weights_path) else None
    n_sims = X.shape[0] // NWALLP
    # one row per sim is enough to get (Minf, AoA, Pi) -- conditions are constant within a sim,
    # so this avoids paging in the full 260774-row block just to read 3 numbers from it.
    conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1][:n_sims])
    return y, weights, n_sims, conds


def make_node_features(static_feats, cond):
    cond_broadcast = cond.expand(static_feats.shape[0], -1)
    return torch.cat([static_feats, cond_broadcast], dim=1)


def predict_split(model, n_sims, conds, coords, edge_index, static_feats, device, edge_chunk_size, cond_mean, cond_std):
    """Run the model over every simulation in a split and return flat (n_sims*NWALLP,) predictions."""
    model.eval()
    preds = np.empty(n_sims * NWALLP, dtype=np.float32)
    with torch.no_grad():
        for i in range(n_sims):
            cond = torch.from_numpy((conds[i] - cond_mean) / cond_std).float().to(device)
            x = make_node_features(static_feats, cond)
            preds[i * NWALLP:(i + 1) * NWALLP] = model(x, coords, edge_index, edge_chunk_size).squeeze(-1).cpu().numpy()
    model.train()
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--k', type=int, default=10, help='kNN neighbors per node.')
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--noise_std', type=float, default=1e-5,
                   help='additive Gaussian noise on node features during training, applied inside '
                        "OffsetGNN.forward (gated by self.training) -- matches the reference "
                        'implementation\'s noise1/noise3 defaults (1e-5), not this script\'s old ad-hoc 0.01.')
    p.add_argument('--edge_chunk_size', type=int, default=DEFAULT_EDGE_CHUNK_SIZE,
                   help='edges processed per gradient-checkpointed chunk in OffsetGraphConv; '
                        'lower this if you hit CUDA OOM, raise it (or set to a huge number) for more speed on big GPUs.')
    p.add_argument('--max_train_sims', type=int, default=None, help='cap number of training sims per epoch (for quick smoke tests).')
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_prefix', default='offset_gnn')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume_from', default=None,
                   help='path to a checkpoint (saved by this script) to resume training from -- '
                        'restores model + optimizer state and continues the epoch count.')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    print('Loading static graph/features...', flush=True)
    coords, edge_index, static_feats, component_labels, component_map = load_static_graph(
        args.data_dir, args.k, cache_dir
    )
    coords, edge_index = coords.to(device), edge_index.to(device)
    static_feats = static_feats.to(device)
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    y_train, train_weights, n_train, train_conds = load_split(args.data_dir, 'train')
    y_test1, test1_weights, n_test1, test1_conds = load_split(args.data_dir, 'test_phase1')
    print(f'n_train={n_train}  n_test1={n_test1}', flush=True)

    # standardize (Minf, AoA, Pi) using train statistics, same convention as the rest of this
    # repo (gp_pod_model.py / train.py both fit a StandardScaler on train_conds).
    cond_mean = train_conds.mean(axis=0)
    cond_std = train_conds.std(axis=0)
    cond_std[cond_std == 0] = 1.0

    sigma_ref = 0.01 * float(np.mean(y_train))
    in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)
    softmax_eps = estimate_softmax_eps(coords, edge_index)
    print(f'softmax_eps (data-calibrated, see estimate_softmax_eps docstring) = {softmax_eps:.5f}', flush=True)
    model = OffsetGNN(in_dim, args.hidden_dim, out_dim=1, coord_dim=3,
                       noise_std=args.noise_std, eps=softmax_eps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume_from:
        print(f'Resuming from {args.resume_from}...', flush=True)
        checkpoint = torch.load(args.resume_from, map_location=device)
        if 'model' in checkpoint:  # new checkpoint format: {'model', 'optimizer', 'epoch'}
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'  restored model + optimizer, resuming at epoch {start_epoch + 1}', flush=True)
        else:  # legacy format: a bare model state_dict, no optimizer/epoch to restore
            model.load_state_dict(checkpoint)
            print('  restored model only (legacy checkpoint format, no optimizer/epoch state) '
                  '-- optimizer momentum and epoch count restart from scratch', flush=True)

    n_sims_per_epoch = min(args.max_train_sims, n_train) if args.max_train_sims else n_train

    for epoch in range(start_epoch, args.epochs):
        sim_order = np.random.permutation(n_train)[:n_sims_per_epoch]
        epoch_loss = 0.0
        t0 = time.time()
        for i in sim_order:
            cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
            x = make_node_features(static_feats, cond)
            target = torch.from_numpy(np.asarray(y_train[i * NWALLP:(i + 1) * NWALLP])).float().to(device)
            sim_weight = float(train_weights[i]) if train_weights is not None else 1.0

            optimizer.zero_grad()
            pred = model(x, coords, edge_index, args.edge_chunk_size).squeeze(-1)
            loss = sim_weight * (pred - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'epoch {epoch+1}/{args.epochs}  loss={epoch_loss / len(sim_order):.5f}  '
              f'({time.time() - t0:.1f}s)', flush=True)

        # Save every epoch (cheap relative to an epoch's training time) so a crash or interruption
        # loses at most one epoch of progress, and --resume_from can pick back up from here.
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                   f'{args.out_prefix}_model.pt')

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            preds1 = predict_split(model, n_test1, test1_conds, coords, edge_index, static_feats,
                                    device, args.edge_chunk_size, cond_mean, cond_std)
            w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
            res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref)
            print(f'  test_phase1  R2={res["r2"]:.4f}  worst_rMAE={res["worst_rmae"]:.4f}  '
                  f'mean_rMAE={res["mean_rmae"]:.4f}  mean_KLw={res["mean_klw"]:.4f}  '
                  f'max_KLw={res["max_klw"]:.4f}', flush=True)

    print(f'Saved: {args.out_prefix}_model.pt', flush=True)


if __name__ == '__main__':
    main()
