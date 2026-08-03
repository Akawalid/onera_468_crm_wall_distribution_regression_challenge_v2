"""Print the current standings of an Optuna study without needing a GPU -- just reads the sqlite
file tune_simple_gnn.py has been writing to. Safe to run on a login node while a tuning job is
still running elsewhere (Optuna's sqlite backend handles concurrent readers fine).

Note: the ranking/value shown is the tuner's objective, validation MSE (see tune_simple_gnn.py's
docstring) -- NOT the challenge's KLw metric. Trials run after tune_simple_gnn.py started
attaching R2/wrMAE/KLw as trial user_attrs (computed on the internal validation split, never
test_phase1/2) will show those alongside the MSE below; older trials from before that change
won't have them. Either way, no trial saves a model checkpoint (memory-cleanup deletes it at the
end of every trial) and these val-split numbers are not the same thing as the real test_phase1/2
metrics -- that still requires retraining the winning config through train_simple_gnn.py, which
this script prints the command for.

Usage:
    python utils/summarize_optuna_study.py --study_name simple_gnn_splitv3 \
        --storage sqlite:////data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_simple_gnn_splitv3.db --top 10
"""

import argparse

import optuna

DB_PATH = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_simple_gnn_splitv3.db'


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--study_name', default='simple_gnn_splitv3')
    p.add_argument('--storage', default=f'sqlite:///{DB_PATH}')
    p.add_argument('--top', type=int, default=10, help='how many best COMPLETE trials to list.')
    args = p.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    trials = study.trials

    by_state = {}
    for t in trials:
        by_state.setdefault(t.state.name, []).append(t)

    print(f'Study: {args.study_name}  ({len(trials)} trials total)')
    for state, ts in sorted(by_state.items(), key=lambda kv: -len(kv[1])):
        print(f'  {state:10} {len(ts)}')

    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print('\nNo completed trials yet.')
        return

    complete.sort(key=lambda t: t.value)  # study direction is 'minimize' (validation MSE)

    print(f'\nTop {min(args.top, len(complete))} trials by validation MSE (lower is better):')
    for rank, t in enumerate(complete[:args.top], start=1):
        print(f'\n  #{rank}  trial {t.number}  value={t.value:.6f}  '
              f'duration={t.duration}')
        for name, val in t.params.items():
            print(f'      {name}: {val}')
        if t.user_attrs:
            print(f'      [internal val]  R2={t.user_attrs.get("val_r2", float("nan")):.4f}  '
                  f'worst_rMAE={t.user_attrs.get("val_worst_rmae", float("nan")):.4f}  '
                  f'mean_rMAE={t.user_attrs.get("val_mean_rmae", float("nan")):.4f}  '
                  f'mean_KLw={t.user_attrs.get("val_mean_klw", float("nan")):.4f}  '
                  f'max_KLw={t.user_attrs.get("val_max_klw", float("nan")):.4f}')

    best = complete[0]
    print(f'\nBest trial: {best.number}  value={best.value:.6f}')
    print('Retrain it for real (honest KLw on test_phase1, saved checkpoint) with:')
    print(f'    python utils/train_simple_gnn.py \\')
    print(f'        --data_dir /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/ \\')
    print(f'        --split_dir splitv3 --k 20 \\')
    print(f'        --hidden_dim {best.params["hidden_dim"]} --n_layers {best.params["n_layers"]} \\')
    print(f'        --lr {best.params["lr"]} --dropout {best.params["dropout"]} \\')
    print(f'        --weight_decay {best.params["weight_decay"]} \\')
    print(f'        --epochs 100 --eval_every 5 \\')
    print(f'        --out_prefix /data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/simple_gnn_tuned')


if __name__ == '__main__':
    main()
