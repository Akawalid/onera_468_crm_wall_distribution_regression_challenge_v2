"""Optuna hyperparameter search for utils/simple_gnn.py's SimpleGNN.

Reuses the exact data loading / node-feature / loss functions from train_simple_gnn.py (so the
model being tuned is identical to the one that script trains) -- only the search loop and epoch
budget differ: shorter per trial, with pruning, and no periodic test_phase1 evaluation (tuning
stays on the internal train/val split, held out from TRAIN only, same convention as
train_simple_gnn.py/train_mlp_with_kl.py -- test_phase1/2 stay untouched during the search so
they remain an honest check afterward).

--k is fixed per run, not searched: it changes the kNN graph itself (and its on-disk cache), so
varying it across trials would make trials non-comparable and defeat the point of caching the
graph once. Rerun this script with a different --k if you want to compare k values too.

Each trial samples hidden_dim/n_layers/lr/dropout/weight_decay, trains with early stopping, and
reports the best validation loss to Optuna's MedianPruner, which kills clearly unpromising trials
early. --storage persists the study to sqlite so it survives a job timeout and can be resumed or
inspected later (e.g. with optuna-dashboard).

Usage:
    python utils/tune_simple_gnn.py --n_trials 50 --epochs 60 --patience 10 \
        --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \
        --split_dir splitv3 --study_name simple_gnn_splitv3 \
        --storage sqlite:///utils/optuna_simple_gnn_splitv3.db

Once you have a winning trial, plug its params into sbatch_simple_gnn.sh's
--hidden_dim/--n_layers/--lr/--dropout/--weight_decay for the real run (with the full epoch
budget and periodic test_phase1 evaluation train_simple_gnn.py already does).
"""

import argparse
import gc
import os
import sys

import numpy as np
import optuna
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.simple_gnn import SimpleGNN
from utils.train_simple_gnn import load_static_graph, load_split, sim_loss, compute_val_loss

NWALLP = 260774


def run_trial(trial, args, static, train_data, device):
    edge_index, edge_weight, static_feats, cond_mean, cond_std = static
    y_train, train_weights, n_train, train_conds = train_data

    hidden_dim   = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    n_layers     = trial.suggest_int('n_layers', 2, 5)
    lr           = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout      = trial.suggest_float('dropout', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    torch.manual_seed(args.seed)
    in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)
    model = SimpleGNN(in_dim, hidden_dim, out_dim=1, n_layers=n_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Wrapped in try/finally: PyTorch's CUDA caching allocator doesn't return freed memory to the
    # OS between trials, and consecutive trials build differently-shaped models (hidden_dim/
    # n_layers both vary), which fragments that cached pool over a long study -- observed this
    # crash the whole run with a CUDA OOM around trial 9. Explicitly deleting the model/optimizer
    # and calling empty_cache() here (every exit path: normal return, pruned, or OOM) keeps each
    # trial's memory from leaking into the next one instead of just hoping GC gets to it in time.
    try:
        # Same VAL_FRAC convention as train_simple_gnn.py/train_mlp_with_kl.py: held out from
        # TRAIN only, never test_phase1/2, and reseeded per trial off args.seed so every trial
        # sees the same split (isolates the effect of the hyperparameters, not the split).
        n_val = max(1, int(round(args.val_frac * n_train)))
        rng = np.random.default_rng(args.seed)
        perm0 = rng.permutation(n_train)
        val_idx, tr_idx = perm0[:n_val], perm0[n_val:]

        best_val, bad_epochs = float('inf'), 0
        for epoch in range(args.epochs):
            model.train()
            for i in rng.permutation(tr_idx):
                cond = torch.from_numpy((train_conds[i] - cond_mean) / cond_std).float().to(device)
                weight = float(train_weights[i]) if train_weights is not None else 1.0
                optimizer.zero_grad()
                loss = sim_loss(model, edge_index, edge_weight, static_feats, cond,
                                 y_train[i * NWALLP:(i + 1) * NWALLP], weight, device)
                loss.backward()
                optimizer.step()

            val_loss = compute_val_loss(model, val_idx, train_conds, train_weights, y_train,
                                         edge_index, edge_weight, static_feats, cond_mean, cond_std, device)

            if val_loss < best_val - 1e-6:
                best_val, bad_epochs = val_loss, 0
            else:
                bad_epochs += 1

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if bad_epochs >= args.patience:
                break

        return best_val
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default='data/')
    p.add_argument('--split_dir', default='splitv2')
    p.add_argument('--k', type=int, default=20, help='kNN neighbors per node, fixed for the whole study (see module docstring).')
    p.add_argument('--epochs', type=int, default=60,
                   help='per-trial epoch budget -- shorter than train_simple_gnn.py default (100) since this runs many trials.')
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--n_trials', type=int, default=50)
    p.add_argument('--timeout', type=int, default=None, help='optional wall-clock budget in seconds for the whole study.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--study_name', default='simple_gnn')
    p.add_argument('--storage', default=None,
                   help='e.g. sqlite:///utils/optuna_simple_gnn.db -- persists trials so the study survives a job timeout and can be resumed/inspected afterward.')
    args = p.parse_args()

    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    print('Loading static graph/features (shared across all trials, k fixed)...', flush=True)
    _coords, edge_index, edge_weight, static_feats, _component_labels, _component_map = load_static_graph(
        args.data_dir, args.k, cache_dir, split_dir=args.split_dir)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    static_feats = static_feats.to(device)

    y_train, train_weights, n_train, train_conds = load_split(args.data_dir, 'train', split_dir=args.split_dir)
    cond_mean = train_conds.mean(axis=0)
    cond_std = train_conds.std(axis=0)
    cond_std[cond_std == 0] = 1.0
    print(f'n_train={n_train}', flush=True)

    static = (edge_index, edge_weight, static_feats, cond_mean, cond_std)
    train_data = (y_train, train_weights, n_train, train_conds)

    def objective(trial):
        return run_trial(trial, args, static, train_data, device)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='minimize', study_name=args.study_name,
                                 storage=args.storage, load_if_exists=True, pruner=pruner)
    # catch=(...,): an individual trial hitting CUDA OOM (run_trial's finally already cleans up
    # the memory) gets marked FAILED and the study moves on to the next trial, instead of an
    # uncaught OutOfMemoryError propagating out of study.optimize() and killing the whole job --
    # that's what happened at trial 9 previously, losing every trial after it.
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                    catch=(torch.OutOfMemoryError,))

    print('\nBest trial:', flush=True)
    print(f'  value (val loss): {study.best_trial.value:.5f}', flush=True)
    for name, val in study.best_trial.params.items():
        print(f'  {name}: {val}', flush=True)
    print(f'\n{len(study.trials)} trials total '
          f'({sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)} pruned, '
          f'{sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)} completed)', flush=True)


if __name__ == '__main__':
    main()
