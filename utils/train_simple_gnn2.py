"""gnn_simple2: same SimpleGNN architecture as train_simple_gnn.py, but trained with a
differentiable KLw loss (soft-histogram KL divergence against the reference distribution)
instead of plain weighted MSE -- directly optimizing the challenge's primary metric, the same way
train_mlp_with_kl_v2.py does for the MLP baseline.

Uses the same P/Q decoupling as train_mlp_with_kl_v2.py (see that file's module docstring for the
full reasoning): P's histogram range is normalized by *this simulation's own std* (sigma_y), while
Q's absolute width stays pinned to the fixed global sigma_ref (0.01 * mean(y_train)) regardless of
sigma_y -- so a naturally noisier simulation doesn't get an automatically more lenient target.

Reuses load_static_graph / load_split / make_node_features / predict_split straight from
train_simple_gnn.py so the graph, features, and model are identical to that script's; only the
loss (and the val-loss used for early stopping) differ. Periodic test_phase1 evaluation still
goes through base_models.evaluate (the histogram-range bug there is already fixed), so the
numbers printed during training are trustworthy.

Cost note: the soft-histogram itself is O(n_points x n_bins) = 260774 x 200 per simulation per
step (vs O(n_points) for plain MSE) -- comparable in scale to what train_mlp_with_kl_v2.py already
does per batch element, so it should fit the same GPU allocation, just slower per simulation than
train_simple_gnn.py's MSE loss.

Usage:
    python utils/train_simple_gnn2.py --split_dir splitv3 \
        --out_prefix /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/simple_gnn2
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_models import evaluate
from utils.simple_gnn import SimpleGNN
from utils.train_simple_gnn import (
    NWALLP, DATA_DIR, load_static_graph, load_split, make_node_features, predict_split,
)

N_BINS     = 200
TAU        = 10.0 / N_BINS
KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}


def make_klw_sim_loss(w_pts, centers, sigma_ref, tau=TAU):
    """Differentiable per-simulation KLw loss. P's range is normalized by sigma_y (this
    simulation's own std); Q is rebuilt per call so its absolute width stays pinned to the global
    sigma_ref regardless of sigma_y (re-expressed in sigma_y-normalized units as sigma_ref/sigma_y)
    -- see module docstring / train_mlp_with_kl_v2.py for why this decoupling matters."""
    def klw_sim_loss(y_pred, y_true, sigma_y, weight):
        eps  = (y_pred - y_true) / sigma_y
        d    = (eps.unsqueeze(1) - centers.view(1, -1)) / tau
        soft = torch.softmax(-0.5 * d * d, dim=1)          # (n_points, n_bins)
        p    = torch.einsum('p,pb->b', w_pts, soft)          # component-weighted histogram, (n_bins,)
        p    = torch.clamp(p, min=1e-10)
        p    = p / p.sum()

        q_width = sigma_ref / sigma_y
        q = torch.exp(-0.5 * (centers / q_width) ** 2)
        q = torch.clamp(q, min=1e-10)
        q = q / q.sum()

        return weight * (p * (p.log() - q.log())).sum()
    return klw_sim_loss


def compute_val_loss_klw(model, val_idx, train_conds, train_weights, y_train, sigma_train,
                          edge_index, edge_weight, static_feats, cond_mean, cond_std,
                          klw_sim_loss, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in val_idx:
            cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
            x = make_node_features(static_feats, cond)
            target = torch.from_numpy(np.asarray(y_train[i * NWALLP:(i + 1) * NWALLP])).float().to(device)
            weight = float(train_weights[i]) if train_weights is not None else 1.0
            pred = model(x, edge_index, edge_weight).squeeze(-1)
            loss = klw_sim_loss(pred, target, float(sigma_train[i]), weight)
            total += loss.item()
    model.train()
    return total / len(val_idx)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--split_dir', default='splitv2',
                   help='subfolder of data_dir holding {train,test_phase1,test_phase2}_{data,labels,weights}.npy.')
    p.add_argument('--k', type=int, default=20, help='kNN neighbors per node.')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--val_frac', type=float, default=0.1,
                   help='fraction of train sims held out for early stopping, drawn from TRAIN only.')
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--max_train_sims', type=int, default=None)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_prefix', default='utils/simple_gnn2')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    print('Loading static graph/features...', flush=True)
    coords, edge_index, edge_weight, static_feats, component_labels, component_map = load_static_graph(
        args.data_dir, args.k, cache_dir, split_dir=args.split_dir
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    static_feats = static_feats.to(device)
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    y_train, train_weights, n_train, train_conds = load_split(args.data_dir, 'train', split_dir=args.split_dir)
    y_test1, test1_weights, n_test1, test1_conds = load_split(args.data_dir, 'test_phase1', split_dir=args.split_dir)
    print(f'n_train={n_train}  n_test1={n_test1}', flush=True)

    cond_mean = train_conds.mean(axis=0)
    cond_std = train_conds.std(axis=0)
    cond_std[cond_std == 0] = 1.0

    mean_global = float(np.mean(y_train))
    sigma_ref = 0.01 * mean_global
    sigma_train = np.asarray(y_train).reshape(n_train, NWALLP).std(axis=1) + 1e-6
    print(f'global train mean = {mean_global:.4f}  sigma_ref (fixed Q width) = {sigma_ref:.5f}', flush=True)

    sw = np.zeros(NWALLP, dtype=np.float32)
    for cname, mask in comp_masks.items():
        sw[mask] = KL_WEIGHTS.get(cname, 0.0)
    w_pts = torch.tensor(sw / sw.sum(), device=device)

    edges = torch.linspace(-5.0, 5.0, N_BINS + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    klw_sim_loss = make_klw_sim_loss(w_pts, centers, sigma_ref)

    sigma_ref_eval = sigma_ref  # same convention base_models.evaluate expects

    in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)
    model = SimpleGNN(in_dim, args.hidden_dim, out_dim=1, n_layers=args.n_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            x = make_node_features(static_feats, cond)
            target = torch.from_numpy(np.asarray(y_train[i * NWALLP:(i + 1) * NWALLP])).float().to(device)
            sim_weight = float(train_weights[i]) if train_weights is not None else 1.0

            optimizer.zero_grad()
            pred = model(x, edge_index, edge_weight).squeeze(-1)
            loss = klw_sim_loss(pred, target, float(sigma_train[i]), sim_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = compute_val_loss_klw(model, val_idx, train_conds, train_weights, y_train, sigma_train,
                                         edge_index, edge_weight, static_feats, cond_mean, cond_std,
                                         klw_sim_loss, device)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f'epoch {epoch+1}/{args.epochs}  train KLw={epoch_loss / len(sim_order):.5f}  '
              f'val KLw={val_loss:.5f}  best_val={best_val:.5f} (epoch {best_epoch+1})  '
              f'({time.time() - t0:.1f}s)', flush=True)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            preds1 = predict_split(model, n_test1, test1_conds, edge_index, edge_weight, static_feats,
                                    device, cond_mean, cond_std)
            w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
            res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref_eval)
            print(f'  test_phase1  R2={res["r2"]:.4f}  worst_rMAE={res["worst_rmae"]:.4f}  '
                  f'mean_rMAE={res["mean_rmae"]:.4f}  mean_KLw={res["mean_klw"]:.4f}  '
                  f'max_KLw={res["max_klw"]:.4f}', flush=True)

        if bad_epochs >= args.patience:
            print(f'early stopping at epoch {epoch+1} (best epoch {best_epoch+1}, val_loss={best_val:.5f})', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds1 = predict_split(model, n_test1, test1_conds, edge_index, edge_weight, static_feats,
                            device, cond_mean, cond_std)
    w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
    res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref_eval)
    print(f'best model (epoch {best_epoch+1})  test_phase1  R2={res["r2"]:.4f}  '
          f'worst_rMAE={res["worst_rmae"]:.4f}  mean_rMAE={res["mean_rmae"]:.4f}  '
          f'mean_KLw={res["mean_klw"]:.4f}  max_KLw={res["max_klw"]:.4f}', flush=True)

    torch.save(model.state_dict(), f'{args.out_prefix}_model.pt')
    print(f'Saved: {args.out_prefix}_model.pt', flush=True)


if __name__ == '__main__':
    main()
