"""
Optuna hyperparameter search for POD+GP (paper_pod_gp.py) -- a wider, Bayesian-optimized version
of tune_pod_gp.py's small fixed 3x3 grid. Directly tunes the number of retained POD modes (not
just a coarse variance threshold), the GP kernel's Matern smoothness, its regularization nugget,
and -- new relative to tune_pod_gp.py -- the kernel's own length-scale/noise-level upper bounds.

Why the bounds are tunable here: paper_pod_gp.py's make_kernel() comment documents a manual
experiment where widening those bounds uniformly made held-out phase 1 R2 drop (0.980->0.956) and
mean_KL nearly triple, because Pi only has 3 distinct values in training -- the GP's own marginal-
likelihood optimizer is poorly identified along that dimension and drifts to bad values once given
more room. That was one manual guess in one direction. Letting Optuna search the bounds themselves,
scored by actual held-out CV performance (not the GP's internal likelihood), is a more principled
way to answer "how much room should this kernel have" than picking a single value by hand.

Same leave-two-consecutive-Machs-out CV as tune_pod_gp.py, within the training set only --
test_phase1/test_phase2 are never touched during the search, only for the one final evaluation.
Each trial reports its running mean_KL after every CV fold, so Optuna's pruner can abandon a
clearly bad configuration partway through its 8 folds instead of always paying for all of them.

Optuna's default sampler is TPE (Tree-structured Parzen Estimator) -- notably, this is literally
the algorithm Peter et al. 2025 used for their own hyperparameter search ("Bayesian optimization
using the Tree-structured Parzen Estimator algorithm implemented in the Optuna framework").

Usage:
    python utils/tune_pod_gp_optuna.py --n_trials 100 --timeout 10000 \\
        --study_name pod_gp_splitv3 \\
        --storage sqlite:////data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_pod_gp.db
"""

import argparse
import os

# Must happen before numpy/scipy/sklearn/joblib are imported anywhere (including transitively via
# paper_pod_gp below) -- see tune_pod_gp.py's identical block for the full oversubscription
# reasoning (64 joblib worker processes each also trying to multithread their own BLAS calls).
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys

import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_pod_gp as pg
from tune_pod_gp import mach_fold_splits

# Matern nu restricted to sklearn's closed-form values (0.5, 1.5, 2.5) plus inf (-> RBF/squared-
# exponential): arbitrary non-standard nu falls back to the general Bessel-function formula, which
# is slower and can be numerically fragile. np.inf itself isn't Optuna/sqlite-friendly as a
# categorical value, so it's keyed by string and mapped back to the real float here.
NU_CHOICES = {'0.5': 0.5, '1.5': 1.5, '2.5': 2.5, 'inf': np.inf}


def make_kernel_tunable(n_dims, nu, length_scale_upper, noise_level_upper):
    """ Same structure as paper_pod_gp.py's make_kernel(), but with tunable upper bounds (lower
    bounds kept fixed at that script's own, empirically-good values: 1e-2 for length_scale,
    1e-12 for noise_level). """
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, length_scale_upper), nu=nu)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-12, noise_level_upper)))


def fit_one_gp_tunable(conds_sc, z_col, kernel, alpha, n_restarts, seed):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts,
                                   normalize_y=True, random_state=seed)
    gp.fit(conds_sc, z_col)
    return gp


def fit_predict_pod_gp_tunable(train_conds_sc, Y_train, query_conds_sc, n_modes, nu,
                                length_scale_upper, noise_level_upper, alpha, n_restarts, seed):
    n_train = Y_train.shape[0]
    n_components = min(n_modes, n_train - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    Z_train = pca.fit_transform(Y_train)   # mean-subtracted internally by PCA
    components = pca.components_
    mean_field = pca.mean_

    n_dims = train_conds_sc.shape[1]
    gps = Parallel(n_jobs=-1)(
        delayed(fit_one_gp_tunable)(
            train_conds_sc, Z_train[:, m],
            make_kernel_tunable(n_dims, nu, length_scale_upper, noise_level_upper),
            alpha, n_restarts, seed)
        for m in range(n_components)
    )
    Z_pred = np.column_stack([gp.predict(query_conds_sc) for gp in gps])
    y_pred = mean_field[None, :] + Z_pred @ components   # exact linear POD decoder
    return y_pred.reshape(-1)


def run_trial(trial, train_conds, Y_train, train_weights, comp_masks, n_restarts, seed):
    n_modes = trial.suggest_int('n_modes', 10, 150)
    nu_key = trial.suggest_categorical('kernel_nu', list(NU_CHOICES.keys()))
    nu = NU_CHOICES[nu_key]
    length_scale_upper = trial.suggest_float('length_scale_upper', 10.0, 1000.0, log=True)
    noise_level_upper = trial.suggest_float('noise_level_upper', 0.01, 10.0, log=True)
    alpha = trial.suggest_float('alpha', 1e-10, 1e-4, log=True)

    print(f'trial {trial.number}: n_modes={n_modes}  nu={nu_key}  '
          f'length_scale_upper={length_scale_upper:.2f}  noise_level_upper={noise_level_upper:.4f}  '
          f'alpha={alpha:.2e}', flush=True)

    kl_values = []
    for fold_i, (tr_idx, val_idx, label) in enumerate(mach_fold_splits(train_conds)):
        scaler = StandardScaler()
        tr_conds_sc = scaler.fit_transform(train_conds[tr_idx])
        val_conds_sc = scaler.transform(train_conds[val_idx])

        y_pred = fit_predict_pod_gp_tunable(tr_conds_sc, Y_train[tr_idx], val_conds_sc,
                                             n_modes, nu, length_scale_upper, noise_level_upper,
                                             alpha, n_restarts, seed)
        y_true = Y_train[val_idx].reshape(-1)
        w_val = train_weights[val_idx]
        res = pg.evaluate_phase(y_true, y_pred, w_val, comp_masks)
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
    p.add_argument('--data_dir', default=pg.DATA_DIR)
    p.add_argument('--split_dir', default=pg.SPLIT_DIR)
    p.add_argument('--n_restarts', type=int, default=2,
                   help="GP optimizer restarts per mode during the search (cheaper than the final "
                        "refit, which uses paper_pod_gp.py's own N_RESTARTS).")
    p.add_argument('--n_trials', type=int, default=100)
    p.add_argument('--timeout', type=int, default=None,
                   help='optional wall-clock budget in seconds for the whole study.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--study_name', default='pod_gp')
    p.add_argument('--storage', default=None,
                   help='e.g. sqlite:////data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/optuna_pod_gp.db '
                        '-- persists trials so the study survives a job timeout and can be resumed/inspected afterward.')
    args = p.parse_args()

    print('Loading data...', flush=True)
    data = pg.load_split(args.data_dir, args.split_dir)
    train_conds, Y_train = data['train_conds'], data['Y_train']
    comp_masks = data['comp_masks']
    train_weights = np.load(os.path.join(args.data_dir, args.split_dir, 'train_weights.npy'))

    n_folds = len(np.unique(train_conds[:, 0])) - 1
    print(f'n_train={data["n_train"]}  CV folds={n_folds} (leave-2-consecutive-Machs-out)', flush=True)

    def objective(trial):
        return run_trial(trial, train_conds, Y_train, train_weights, comp_masks,
                          args.n_restarts, args.seed)

    # n_warmup_steps=2 (vs tune_simple_gnn.py's 15): folds are the "steps" here, and there are only
    # 8 of them total, so a trial gets to run a couple of folds before it's eligible for pruning.
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
    # real (never touched during tuning) phase1/phase2 test sets -- same convention as
    # tune_pod_gp.py, using paper_pod_gp.py's own (higher) N_RESTARTS for this one fit.
    best = study.best_trial.params
    nu = NU_CHOICES[best['kernel_nu']]
    print('\nRefitting on the full training set with the best hyperparameters...', flush=True)
    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    y_pred1 = fit_predict_pod_gp_tunable(train_conds_sc, Y_train, test1_conds_sc,
                                          best['n_modes'], nu, best['length_scale_upper'],
                                          best['noise_level_upper'], best['alpha'],
                                          pg.N_RESTARTS, args.seed)
    y_pred2 = fit_predict_pod_gp_tunable(train_conds_sc, Y_train, test2_conds_sc,
                                          best['n_modes'], nu, best['length_scale_upper'],
                                          best['noise_level_upper'], best['alpha'],
                                          pg.N_RESTARTS, args.seed)

    res1 = pg.evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], comp_masks)
    res2 = pg.evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], comp_masks)
    pg.print_result('POD+GP (Optuna-tuned)', 'Phase 1 (interpolation)', res1)
    pg.print_result('POD+GP (Optuna-tuned)', 'Phase 2 (extrapolation)', res2)

    out_prefix = os.path.join(args.data_dir, 'paper_pod_gp_optuna')   # iceberg/shared drive, not utils/
    try:
        np.save(f'{out_prefix}_tuned_y_pred1.npy', y_pred1)
        np.save(f'{out_prefix}_tuned_y_pred2.npy', y_pred2)
        print(f'\nSaved: {out_prefix}_tuned_y_pred1.npy, {out_prefix}_tuned_y_pred2.npy', flush=True)
    except OSError as e:
        print(f'\n[WARN] Could not save predictions: {e}', flush=True)


if __name__ == '__main__':
    main()
