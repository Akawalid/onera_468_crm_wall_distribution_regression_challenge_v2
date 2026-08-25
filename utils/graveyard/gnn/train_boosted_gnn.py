"""Sequentially (boosting-style) fit the BoostedGNN ensemble (utils/boosted_gnn.py) on the CRM
wall dataset.

Reuses utils/train_simple_gnn.py's data pipeline verbatim (same static graph/features, same
standardization, same evaluate() metrics) so results are directly comparable to SimpleGNN and
OffsetGNN -- this script only implements the staged fitting loop on top of it.

Each stage k is trained for --stage_epochs epochs to predict the residual y - F_{k-1}(x) left by
every earlier stage (already frozen), exactly like gradient boosting fits a new tree to the
current residual -- except the weak learner here is a tiny one-layer graph conv instead of a
tree. Once a stage's inner epoch loop finishes, its parameters are frozen for good and folded
into the running ensemble sum via `shrinkage` (default 0.1, the standard GBM learning rate) before
moving to the next stage. The natural "iteration" unit for boosting is a stage, not an epoch --
metrics are logged/plotted per stage (after each stage's fit), not per inner epoch.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_models import evaluate
from utils.boosted_gnn import BoostedGNN
from utils.train_simple_gnn import load_static_graph, load_split, make_node_features

NWALLP = 260774
DATA_DIR = 'data/'


def predict_split(model, n_sims, conds, edge_index, edge_weight, static_feats, device, cond_mean, cond_std, up_to=None):
    model.eval()
    preds = np.empty(n_sims * NWALLP, dtype=np.float32)
    with torch.no_grad():
        for i in range(n_sims):
            cond = torch.from_numpy((conds[i] - cond_mean) / cond_std).float().to(device)
            x = make_node_features(static_feats, cond)
            preds[i * NWALLP:(i + 1) * NWALLP] = model(x, edge_index, edge_weight, up_to=up_to).squeeze(-1).cpu().numpy()
    model.train()
    return preds


def compute_val_loss(model, val_idx, train_conds, train_weights, y_train, edge_index, edge_weight,
                      static_feats, cond_mean, cond_std, device, up_to=None):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in val_idx:
            cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
            weight = float(train_weights[i]) if train_weights is not None else 1.0
            x = make_node_features(static_feats, cond)
            target = torch.from_numpy(np.asarray(y_train[i * NWALLP:(i + 1) * NWALLP])).float().to(device)
            pred = model(x, edge_index, edge_weight, up_to=up_to).squeeze(-1)
            total += (weight * (pred - target).pow(2).mean()).item()
    model.train()
    return total / len(val_idx)


def fit_stage(model, stage, stage_idx, args, tr_idx, val_idx, train_conds, train_weights, y_train,
              edge_index, edge_weight, static_feats, cond_mean, cond_std, device):
    """Train `stage` (already appended to model but not yet frozen) for --stage_epochs epochs to
    predict the residual left by all earlier (frozen) stages. Returns the best (lowest held-out
    val loss) state_dict for this stage alone, mirroring train_simple_gnn.py's early-stopping.
    """
    optimizer = torch.optim.Adam(stage.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_sims_per_epoch = min(args.max_train_sims, len(tr_idx)) if args.max_train_sims else len(tr_idx)

    best_val, best_state, bad_epochs = float('inf'), None, 0
    for epoch in range(args.stage_epochs):
        sim_order = np.random.permutation(tr_idx)[:n_sims_per_epoch]
        epoch_loss = 0.0
        for i in sim_order:
            cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
            sim_weight = float(train_weights[i]) if train_weights is not None else 1.0
            x = make_node_features(static_feats, cond)
            target = torch.from_numpy(np.asarray(y_train[i * NWALLP:(i + 1) * NWALLP])).float().to(device)

            with torch.no_grad():
                # sum of all EARLIER (frozen) stages + bias -- the current ensemble before this
                # stage's own (shrinkage-scaled) contribution is added.
                f_prev = model(x, edge_index, edge_weight, up_to=stage_idx).squeeze(-1)
            residual = target - f_prev

            optimizer.zero_grad()
            stage_pred = stage(x, edge_index, edge_weight).squeeze(-1)
            loss = sim_weight * (stage_pred - residual).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = compute_val_loss(model, val_idx, train_conds, train_weights, y_train, edge_index,
                                     edge_weight, static_feats, cond_mean, cond_std, device, up_to=stage_idx + 1)
        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in stage.state_dict().items()}
        else:
            bad_epochs += 1

        print(f'    stage {stage_idx+1} epoch {epoch+1}/{args.stage_epochs}  '
              f'loss={epoch_loss/len(sim_order):.5f}  val_loss={val_loss:.5f}  '
              f'best_val={best_val:.5f}', flush=True)
        if bad_epochs >= args.stage_patience:
            print(f'    stage {stage_idx+1}: early stop at epoch {epoch+1}', flush=True)
            break

    if best_state is not None:
        stage.load_state_dict(best_state)
    return best_val


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--k', type=int, default=20, help='kNN neighbors per node.')
    p.add_argument('--num_stages', type=int, default=20, help='K, the number of weak learners to boost.')
    p.add_argument('--shrinkage', type=float, default=0.1, help='learning-rate-like scale applied to each stage when summed into the ensemble.')
    p.add_argument('--stage_epochs', type=int, default=15, help='max epochs to fit each individual stage.')
    p.add_argument('--stage_patience', type=int, default=5, help='early-stop a stage\'s own fitting if its val loss stalls this many epochs.')
    p.add_argument('--boosting_patience', type=int, default=5,
                   help='stop adding new stages if the held-out val loss of the whole ensemble '
                        'has not improved for this many consecutive stages.')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--max_train_sims', type=int, default=None, help='cap number of training sims per epoch (for quick smoke tests).')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_prefix', default='utils/boosted_gnn')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume_from', default=None, help='checkpoint (saved by this script) to resume boosting from.')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    print('Loading static graph/features...', flush=True)
    coords, edge_index, edge_weight, static_feats, component_labels, component_map = load_static_graph(
        args.data_dir, args.k, cache_dir
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    static_feats = static_feats.to(device)
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    y_train, train_weights, n_train, train_conds = load_split(args.data_dir, 'train')
    y_test1, test1_weights, n_test1, test1_conds = load_split(args.data_dir, 'test_phase1')
    print(f'n_train={n_train}  n_test1={n_test1}', flush=True)

    cond_mean = train_conds.mean(axis=0)
    cond_std = train_conds.std(axis=0)
    cond_std[cond_std == 0] = 1.0
    sigma_ref = 0.01 * float(np.mean(y_train))
    in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)

    n_val = max(1, int(round(args.val_frac * n_train)))
    perm0 = np.random.permutation(n_train)
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
    print(f'train/val split: {len(tr_idx)} train  {len(val_idx)} val', flush=True)

    model = BoostedGNN(in_dim, out_dim=1, shrinkage=args.shrinkage).to(device)
    model.bias.data.fill_(float(np.mean(y_train)))  # F_0 = mean(y), same init as classic GBM

    start_stage = 0
    if args.resume_from:
        print(f'Resuming from {args.resume_from}...', flush=True)
        checkpoint = torch.load(args.resume_from, map_location=device)
        for _ in range(checkpoint['num_stages']):
            model.add_stage()
        model.load_state_dict(checkpoint['model'])
        for i in range(checkpoint['num_stages']):
            model.freeze_stage(i)
        start_stage = checkpoint['num_stages']
        print(f'  restored {start_stage} completed stages, resuming at stage {start_stage + 1}', flush=True)

    best_ensemble_val = float('inf')
    bad_stages = 0

    for stage_idx in range(start_stage, args.num_stages):
        stage = model.add_stage().to(device)
        t0 = time.time()
        fit_stage(model, stage, stage_idx, args, tr_idx, val_idx, train_conds, train_weights, y_train,
                  edge_index, edge_weight, static_feats, cond_mean, cond_std, device)
        model.freeze_stage(stage_idx)

        ensemble_val = compute_val_loss(model, val_idx, train_conds, train_weights, y_train, edge_index,
                                         edge_weight, static_feats, cond_mean, cond_std, device, up_to=stage_idx + 1)

        preds1 = predict_split(model, n_test1, test1_conds, edge_index, edge_weight, static_feats,
                                device, cond_mean, cond_std, up_to=stage_idx + 1)
        w1 = test1_weights if test1_weights is not None else np.ones(n_test1)
        res = evaluate(np.asarray(y_test1), preds1, w1, NWALLP, comp_masks, sigma_ref)
        print(f'stage {stage_idx+1}/{args.num_stages}  ensemble_val_loss={ensemble_val:.5f}  '
              f'({time.time()-t0:.1f}s)  test_phase1  R2={res["r2"]:.4f}  '
              f'worst_rMAE={res["worst_rmae"]:.4f}  mean_rMAE={res["mean_rmae"]:.4f}  '
              f'mean_KLw={res["mean_klw"]:.4f}  max_KLw={res["max_klw"]:.4f}', flush=True)

        torch.save({'model': model.state_dict(), 'num_stages': stage_idx + 1}, f'{args.out_prefix}_model.pt')

        if ensemble_val < best_ensemble_val - 1e-6:
            best_ensemble_val, bad_stages = ensemble_val, 0
        else:
            bad_stages += 1
        if bad_stages >= args.boosting_patience:
            print(f'boosting early stop after stage {stage_idx+1} '
                  f'(val loss stalled for {args.boosting_patience} stages)', flush=True)
            break

    print(f'Saved: {args.out_prefix}_model.pt', flush=True)


if __name__ == '__main__':
    main()
