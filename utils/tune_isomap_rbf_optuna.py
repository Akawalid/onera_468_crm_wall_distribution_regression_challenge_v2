"""
Optuna hyperparameter search for IsoMap+RBF (paper_isomap_rbf.py). That script fixes most of its
hyperparameters by hand: N_COMPONENTS=3 was picked from a visual Frobenius-norm stress-curve check
(not cross-validated), ISOMAP_K=15 was picked only to keep the neighbor graph connected, RBF_KERNEL
was never varied from 'thin_plate_spline', and RBFInterpolator's smoothing was left at its default
0 (exact interpolation through every training point -- no regularization at all in the p->z step).
This script tunes all of those, plus the backmapping k, against real held-out CV performance.

backmap_k is folded into this SAME outer search instead of paper_isomap_rbf.py's own nested
leave-10-out CV (select_backmap_k): that inner CV is the same statistical exercise as this
script's outer CV (repeatedly holding out data, comparing candidate values by held-out accuracy),
so running it nested inside every outer fold of every trial would be redundant nested
cross-validation -- expensive for no real benefit. Tuning k at the same level as everything else
is cheaper and answers the identical question.

Same leave-two-consecutive-Machs-out CV as tune_pod_gp.py/tune_pod_gp_optuna.py, within the
training set only -- test_phase1/test_phase2 are never touched during the search, only for the one
final evaluation. Each trial reports its running mean_KL after every CV fold, so Optuna's pruner
can abandon a clearly bad configuration partway through its 8 folds.

Reuses paper_isomap_rbf.py's official metrics (evaluate_phase, print_result, knn_backmap,
compute_R2) directly via import.

Usage:
    python utils/tune_isomap_rbf_optuna.py --n_trials 150 --timeout 10000 \\
        --study_name isomap_rbf_splitv3 \\
        --storage sqlite:////data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_isomap_rbf.db
"""

import argparse
import os

# Must happen before numpy/scipy/sklearn are imported anywhere (including transitively via
# paper_isomap_rbf below) -- see tune_pod_gp.py's identical block for the full oversubscription
# reasoning. Isomap/NearestNeighbors/RBFInterpolator all have their own internal parallelism (or
# BLAS calls) that can oversubscribe a many-core cluster node the same way.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys

import numpy as np
import optuna
from scipy.interpolate import RBFInterpolator
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_isomap_rbf as pir
from tune_pod_gp import mach_fold_splits

# Polyharmonic-family kernels only (linear/cubic/quintic/thin_plate_spline): these are scale-free,
# unlike multiquadric/gaussian/inverse_* which need an extra `epsilon` shape parameter -- keeping
# to the scale-free family avoids a second, conditional hyperparameter.
RBF_KERNEL_CHOICES = ['linear', 'thin_plate_spline', 'cubic', 'quintic']


def fit_predict_isomap_rbf_tunable(train_conds_sc, Y_train, query_conds_sc,
                                    n_components, isomap_k, rbf_kernel, smoothing, backmap_k,
                                    seed):
    isomap = Isomap(n_neighbors=isomap_k, n_components=n_components)
    Z_train = isomap.fit_transform(Y_train)

    rbf = RBFInterpolator(train_conds_sc, Z_train, kernel=rbf_kernel, smoothing=smoothing)
    Z_pred = rbf(query_conds_sc)

    y_pred = pir.knn_backmap(Z_pred, Z_train, Y_train, backmap_k)
    return y_pred.reshape(-1)


def run_trial(trial, train_conds, Y_train, train_weights, comp_masks, seed):
    n_components = trial.suggest_int('n_components', 2, 8)
    isomap_k = trial.suggest_int('isomap_k', 8, 40)
    rbf_kernel = trial.suggest_categorical('rbf_kernel', RBF_KERNEL_CHOICES)
    smoothing = trial.suggest_float('smoothing', 1e-8, 1e-1, log=True)
    backmap_k = trial.suggest_int('backmap_k', 2, 12)

    print(f'trial {trial.number}: n_components={n_components}  isomap_k={isomap_k}  '
          f'rbf_kernel={rbf_kernel}  smoothing={smoothing:.2e}  backmap_k={backmap_k}', flush=True)

    kl_values = []
    for fold_i, (tr_idx, val_idx, label) in enumerate(mach_fold_splits(train_conds)):
        scaler = StandardScaler()
        tr_conds_sc = scaler.fit_transform(train_conds[tr_idx])
        val_conds_sc = scaler.transform(train_conds[val_idx])

        try:
            y_pred = fit_predict_isomap_rbf_tunable(
                tr_conds_sc, Y_train[tr_idx], val_conds_sc,
                n_components, isomap_k, rbf_kernel, smoothing, backmap_k, seed)
        except (np.linalg.LinAlgError, ValueError) as e:
            # e.g. an ill-conditioned RBF system for some kernel/smoothing combinations -- fail
            # this trial cleanly rather than letting the whole study crash.
            print(f'  trial {trial.number}  fold {fold_i}: fit failed ({e}), pruning trial', flush=True)
            raise optuna.TrialPruned()

        y_true = Y_train[val_idx].reshape(-1)
        w_val = train_weights[val_idx]
        res = pir.evaluate_phase(y_true, y_pred, w_val, comp_masks)
        kl_values.append(res['mean_KL'])

        running_mean = float(np.mean(kl_values))
        trial.report(running_mean, fold_i)
        print(f'  trial {trial.number}  fold {fold_i} ({label})  KL={res["mean_KL"]:.4f}  '
              f'running mean={running_mean:.4f}', flush=True)
        if trial.should_prune():
            print(f'  trial {trial.number}: pruned after fold {fold_i}', flush=True)
            raise optuna.TrialPruned()

    return float(np.mean(kl_values))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=pir.DATA_DIR)
    p.add_argument('--split_dir', default=pir.SPLIT_DIR)
    p.add_argument('--n_trials', type=int, default=150)
    p.add_argument('--timeout', type=int, default=None,
                   help='optional wall-clock budget in seconds for the whole study.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--study_name', default='isomap_rbf')
    p.add_argument('--storage', default=None,
                   help='e.g. sqlite:////data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_isomap_rbf.db '
                        '-- persists trials so the study survives a job timeout and can be resumed/inspected afterward.')
    args = p.parse_args()

    print('Loading data...', flush=True)
    data = pir.load_split(args.data_dir, args.split_dir)
    train_conds, Y_train = data['train_conds'], data['Y_train']
    comp_masks = data['comp_masks']
    train_weights = np.load(os.path.join(args.data_dir, args.split_dir, 'train_weights.npy'))

    n_folds = len(np.unique(train_conds[:, 0])) - 1
    print(f'n_train={data["n_train"]}  CV folds={n_folds} (leave-2-consecutive-Machs-out)', flush=True)

    def objective(trial):
        return run_trial(trial, train_conds, Y_train, train_weights, comp_masks, args.seed)

    pruner = optuna.pruners.PercentilePruner(percentile=75.0, n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction='minimize', study_name=args.study_name,
                                 storage=args.storage, load_if_exists=True, pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print('\nBest trial:', flush=True)
    print(f'  value (CV mean_KL): {study.best_trial.value:.5f}', flush=True)
    for name, val in study.best_trial.params.items():
        print(f'  {name}: {val}', flush=True)
    print(f'\n{len(study.trials)} trials total '
          f'({sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)} pruned, '
          f'{sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)} completed)', flush=True)

    # Final refit on the FULL training set with the best hyperparameters, evaluated once on the
    # real (never touched during tuning) phase1/phase2 test sets.
    best = study.best_trial.params
    print('\nRefitting on the full training set with the best hyperparameters...', flush=True)
    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    y_pred1 = fit_predict_isomap_rbf_tunable(
        train_conds_sc, Y_train, test1_conds_sc,
        best['n_components'], best['isomap_k'], best['rbf_kernel'], best['smoothing'],
        best['backmap_k'], args.seed)
    y_pred2 = fit_predict_isomap_rbf_tunable(
        train_conds_sc, Y_train, test2_conds_sc,
        best['n_components'], best['isomap_k'], best['rbf_kernel'], best['smoothing'],
        best['backmap_k'], args.seed)

    res1 = pir.evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], comp_masks)
    res2 = pir.evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], comp_masks)
    pir.print_result('IsoMap+RBF (Optuna-tuned)', 'Phase 1 (interpolation)', res1)
    pir.print_result('IsoMap+RBF (Optuna-tuned)', 'Phase 2 (extrapolation)', res2)

    out_prefix = os.path.join(args.data_dir, 'paper_isomap_rbf_optuna')   # iceberg/shared drive, not utils/
    try:
        np.save(f'{out_prefix}_tuned_y_pred1.npy', y_pred1)
        np.save(f'{out_prefix}_tuned_y_pred2.npy', y_pred2)
        print(f'\nSaved: {out_prefix}_tuned_y_pred1.npy, {out_prefix}_tuned_y_pred2.npy', flush=True)
    except OSError as e:
        print(f'\n[WARN] Could not save predictions: {e}', flush=True)


if __name__ == '__main__':
    main()
