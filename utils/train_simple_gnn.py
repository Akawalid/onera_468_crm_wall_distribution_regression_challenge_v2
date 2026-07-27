"""Train the simple mean-aggregation GNN (utils/simple_gnn.py) on the CRM wall dataset.

Structured to mirror train_offset_gnn.py (same data loading, same node features, same
evaluate() metrics printed at the same cadence) so its results are directly comparable --
the point of this script is to find where the much heavier OffsetGNN actually earns its cost
over a minimal GNN baseline, and where a GNN of any kind earns its cost over the pointwise
baselines in utils/base_models.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Make `utils` importable as a package regardless of the process's cwd at launch, same reasoning
# as train_offset_gnn.py (see that file's comment -- matters under Slurm/sbatch).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_models import evaluate
from utils.simple_gnn import SimpleGNN, build_knn_graph

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
DATA_DIR = 'data/'


def load_static_graph(data_dir, k, cache_dir):
    """Coordinates/normals/components are identical across all sims (fixed mesh) -- build once.

    Same cache file as train_offset_gnn.py's load_static_graph (knn_edge_index_k{k}_noselfloop.pt):
    build_knn_graph makes the identical no-self-loop kNN graph, so the two scripts can share it.
    """
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
        edge_index = build_knn_graph(coords, k=k)
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
    conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1][:n_sims])
    return y, weights, n_sims, conds


def make_node_features(static_feats, cond):
    cond_broadcast = cond.expand(static_feats.shape[0], -1)
    return torch.cat([static_feats, cond_broadcast], dim=1)


def predict_split(model, n_sims, conds, edge_index, static_feats, device, cond_mean, cond_std):
    """Run the model over every simulation in a split and return flat (n_sims*NWALLP,) predictions."""
    model.eval()
    preds = np.empty(n_sims * NWALLP, dtype=np.float32)
    with torch.no_grad():
        for i in range(n_sims):
            cond = torch.from_numpy((conds[i] - cond_mean) / cond_std).float().to(device)
            x = make_node_features(static_feats, cond)
            preds[i * NWALLP:(i + 1) * NWALLP] = model(x, edge_index).squeeze(-1).cpu().numpy()
    model.train()
    return preds


def sim_loss(model, edge_index, static_feats, cond, y, weight, device):
    x = make_node_features(static_feats, cond)
    target = torch.from_numpy(np.asarray(y)).float().to(device)
    pred = model(x, edge_index).squeeze(-1)
    return weight * (pred - target).pow(2).mean()


def compute_val_loss(model, val_idx, train_conds, train_weights, y_train, edge_index, static_feats,
                      cond_mean, cond_std, device):
    """Mean per-sim weighted MSE over the held-out validation sims -- same loss as training,
    just with dropout disabled (model.eval()) and no gradient, so it reflects the model's actual
    (non-stochastic) predictions rather than a dropout-noised training-time estimate.
    """
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in val_idx:
            cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
            weight = float(train_weights[i]) if train_weights is not None else 1.0
            loss = sim_loss(model, edge_index, static_feats, cond,
                             y_train[i * NWALLP:(i + 1) * NWALLP], weight, device)
            total += loss.item()
    model.train()
    return total / len(val_idx)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--k', type=int, default=10, help='kNN neighbors per node.')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.2,
                   help='dropout after each hidden layer, same default as GlobalMLP in '
                        'data_exploration_splitv2_v2.ipynb.')
    p.add_argument('--weight_decay', type=float, default=1e-5, help='L2 penalty on the optimizer (Adam).')
    p.add_argument('--val_frac', type=float, default=0.1,
                   help='fraction of train sims held out for early stopping, same VAL_FRAC '
                        'convention as train_mlp_with_kl.py -- NOT test_phase1, so the reported '
                        'test metric stays unbiased by the stopping decision.')
    p.add_argument('--patience', type=int, default=15,
                   help='stop if val loss has not improved for this many consecutive epochs.')
    p.add_argument('--max_train_sims', type=int, default=None, help='cap number of training sims per epoch (for quick smoke tests).')
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_prefix', default='utils/simple_gnn')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    print('Loading static graph/features...', flush=True)
    coords, edge_index, static_feats, component_labels, component_map = load_static_graph(
        args.data_dir, args.k, cache_dir
    )
    edge_index = edge_index.to(device)
    static_feats = static_feats.to(device)
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    y_train, train_weights, n_train, train_conds = load_split(args.data_dir, 'train')
    y_test1, test1_weights, n_test1, test1_conds = load_split(args.data_dir, 'test_phase1')
    print(f'n_train={n_train}  n_test1={n_test1}', flush=True)

    # standardize (Minf, AoA, Pi) using train statistics, same convention as train_offset_gnn.py
    # and the rest of this repo (gp_pod_model.py / train.py both fit a StandardScaler on train_conds).
    cond_mean = train_conds.mean(axis=0)
    cond_std = train_conds.std(axis=0)
    cond_std[cond_std == 0] = 1.0

    sigma_ref = 0.01 * float(np.mean(y_train))
    in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)
    model = SimpleGNN(in_dim, args.hidden_dim, out_dim=1, n_layers=args.n_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Held out from TRAIN only (never test_phase1/2) -- used purely for the early-stopping
    # decision, same VAL_FRAC/PATIENCE convention as train_mlp_with_kl.py.
    n_val = max(1, int(round(args.val_frac * n_train)))
    perm0 = np.random.permutation(n_train)
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
    print(f'train/val split: {len(tr_idx)} train  {len(val_idx)} val (patience={args.patience})', flush=True)

    n_sims_per_epoch = min(args.max_train_sims, len(tr_idx)) if args.max_train_sims else len(tr_idx)

    best_val = float('inf')
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(args.epochs):
        sim_order = np.random.permutation(tr_idx)[:n_sims_per_epoch]
        epoch_loss = 0.0
        t0 = time.time()
        for i in sim_order:
            cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
            sim_weight = float(train_weights[i]) if train_weights is not None else 1.0

            optimizer.zero_grad()
            loss = sim_loss(model, edge_index, static_feats, cond,
                             y_train[i * NWALLP:(i + 1) * NWALLP], sim_weight, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = compute_val_loss(model, val_idx, train_conds, train_weights, y_train,
                                     edge_index, static_feats, cond_mean, cond_std, device)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f'epoch {epoch+1}/{args.epochs}  loss={epoch_loss / len(sim_order):.5f}  '
              f'val_loss={val_loss:.5f}  best_val={best_val:.5f} (epoch {best_epoch+1})  '
              f'({time.time() - t0:.1f}s)', flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            preds1 = predict_split(model, n_test1, test1_conds, edge_index, static_feats,
                                    device, cond_mean, cond_std)
            w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
            res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref)
            print(f'  test_phase1  R2={res["r2"]:.4f}  worst_rMAE={res["worst_rmae"]:.4f}  '
                  f'mean_rMAE={res["mean_rmae"]:.4f}  mean_KLw={res["mean_klw"]:.4f}  '
                  f'max_KLw={res["max_klw"]:.4f}', flush=True)

        if bad_epochs >= args.patience:
            print(f'early stopping at epoch {epoch+1} (best epoch {best_epoch+1}, val_loss={best_val:.5f})', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds1 = predict_split(model, n_test1, test1_conds, edge_index, static_feats,
                            device, cond_mean, cond_std)
    w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
    res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref)
    print(f'best model (epoch {best_epoch+1})  test_phase1  R2={res["r2"]:.4f}  '
          f'worst_rMAE={res["worst_rmae"]:.4f}  mean_rMAE={res["mean_rmae"]:.4f}  '
          f'mean_KLw={res["mean_klw"]:.4f}  max_KLw={res["max_klw"]:.4f}', flush=True)

    torch.save(model.state_dict(), f'{args.out_prefix}_model.pt')
    print(f'Saved: {args.out_prefix}_model.pt', flush=True)


if __name__ == '__main__':
    main()
