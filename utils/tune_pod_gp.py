"""
Hyperparameter tuning for POD+GP (paper_pod_gp.py): searches over the POD variance threshold
(how many modes get kept) and the GP kernel's Matern smoothness (nu), using leave-two-
consecutive-Machs-out cross-validation *within the training set only* -- test_phase1/test_phase2
are never touched during the search, only for the one final evaluation at the end. Same CV
convention as bundle/starting_kit/kit_utils/data.py's mach_fold_splits (every training simulation
is validated on exactly once).

Reuses paper_pod_gp.py's exact POD/GP fitting code (choose_n_modes, fit_one_gp, ALPHA) and
official metrics (evaluate_phase, print_result) directly via import, so what gets tuned here is
exactly what's deployed in paper_pod_gp.py -- not a re-derived copy that could silently drift out
of sync.

Primary tuning objective: mean_KL (minimize) -- the challenge's primary metric per
bundle/starting_kit/kit_utils/metrics.py's own docstring ("Primary metric: KLw (mean_KL), lower is
better"). R2/wrMAE/score are also tracked and printed for every candidate.

CPU-only. The full grid (measured on a 12-core laptop: ~5.4 min per config x fold-count at
n_restarts=2) is ~45-50 min there -- intended to run on the cluster instead (see
sbatch_tune_pod_gp.sh), where more cores let the per-mode GP parallelism (see the thread-pinning
note below) finish each config much faster.
"""

import os

# Must happen before numpy/scipy/sklearn/joblib are imported anywhere (including transitively via
# paper_pod_gp below) -- BLAS thread counts are fixed at library load time. Without this, each of
# joblib.Parallel(n_jobs=-1)'s worker PROCESSES (paper_pod_gp.fit_one_gp's parallelism, one per POD
# mode) would ALSO try to multithread its own BLAS calls across every core on the node -- fine on a
# 12-core laptop, but on a 64-core cluster node that's up to 64 processes x 64 BLAS threads each,
# i.e. massive oversubscription and a much SLOWER run than a single-threaded-per-worker one. Pin
# each worker to 1 BLAS thread and let joblib alone own the parallelism.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_pod_gp as pg

EPS_MACH = 1e-6

VARIANCE_THRESHOLDS = [0.97, 0.99, 0.995]
KERNEL_NUS = [1.5, 2.5, np.inf]   # inf -> Matern reduces to the RBF/squared-exponential kernel
N_RESTARTS_CV = 3                 # fewer restarts during the search than the final refit, for speed
N_RESTARTS_FINAL = 5              # matches paper_pod_gp.py's own default for the full-data refit
MAX_COMPONENTS = 150
SEED = 0


def mach_fold_splits(conds, epsilon=EPS_MACH):
    """ Leave-two-consecutive-Machs-out CV folds -- same convention as
    bundle/starting_kit/kit_utils/data.py's mach_fold_splits. """
    mach_values = np.unique(conds[:, 0])
    for i in range(len(mach_values) - 1):
        m0, m1 = mach_values[i], mach_values[i + 1]
        val_mask = (np.abs(conds[:, 0] - m0) < epsilon) | (np.abs(conds[:, 0] - m1) < epsilon)
        yield np.flatnonzero(~val_mask), np.flatnonzero(val_mask), f'{m0:.2f}+{m1:.2f}'


def fit_predict_pod_gp(train_conds_sc, Y_train, query_conds_sc, variance_threshold, kernel_nu,
                        n_restarts, max_components=MAX_COMPONENTS, seed=SEED):
    n_train = Y_train.shape[0]
    n_components = min(max_components, n_train - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    Z_train_full = pca.fit_transform(Y_train)
    k, _ = pg.choose_n_modes(pca, variance_threshold)
    Z_train = Z_train_full[:, :k]
    components = pca.components_[:k]
    mean_field = pca.mean_

    gps = Parallel(n_jobs=-1)(
        delayed(pg.fit_one_gp)(train_conds_sc, Z_train[:, m], train_conds_sc.shape[1],
                                kernel_nu, pg.ALPHA, n_restarts, seed)
        for m in range(k)
    )
    Z_pred = np.column_stack([gp.predict(query_conds_sc) for gp in gps])
    y_pred = mean_field[None, :] + Z_pred @ components
    return y_pred.reshape(-1), k


def cv_score(train_conds, Y_train, train_weights, comp_masks, variance_threshold, kernel_nu,
             n_restarts):
    """ Average mean_KL/R2/wrMAE/score across leave-2-Machs-out CV folds. """
    kl_list, r2_list, wrmae_list, score_list, k_list = [], [], [], [], []
    for tr_idx, val_idx, _label in mach_fold_splits(train_conds):
        scaler = StandardScaler()
        tr_conds_sc = scaler.fit_transform(train_conds[tr_idx])
        val_conds_sc = scaler.transform(train_conds[val_idx])
        y_pred, k = fit_predict_pod_gp(tr_conds_sc, Y_train[tr_idx], val_conds_sc,
                                        variance_threshold, kernel_nu, n_restarts)
        y_true = Y_train[val_idx].reshape(-1)
        w_val = train_weights[val_idx]
        res = pg.evaluate_phase(y_true, y_pred, w_val, comp_masks)
        kl_list.append(res['mean_KL'])
        r2_list.append(res['R2'])
        wrmae_list.append(res['wrMAE'])
        score_list.append(res['score'])
        k_list.append(k)
    return dict(mean_KL=float(np.mean(kl_list)), R2=float(np.mean(r2_list)),
                wrMAE=float(np.mean(wrmae_list)), score=float(np.mean(score_list)),
                k=float(np.mean(k_list)))


def main():
    print('Loading data...', flush=True)
    data = pg.load_split(pg.DATA_DIR, pg.SPLIT_DIR)
    train_conds, Y_train, n_train = data['train_conds'], data['Y_train'], data['n_train']
    comp_masks = data['comp_masks']

    train_weights = np.load(os.path.join(pg.DATA_DIR, pg.SPLIT_DIR, 'train_weights.npy'))

    n_folds = len(np.unique(train_conds[:, 0])) - 1
    print(f'n_train={n_train}  CV folds={n_folds} (leave-2-consecutive-Machs-out)', flush=True)
    print(f'Grid: variance_threshold in {VARIANCE_THRESHOLDS}, kernel_nu in {KERNEL_NUS} '
          f'({len(VARIANCE_THRESHOLDS) * len(KERNEL_NUS)} configs x {n_folds} folds = '
          f'{len(VARIANCE_THRESHOLDS) * len(KERNEL_NUS) * n_folds} fits)\n', flush=True)

    results = []
    t_start = time.time()
    for vt in VARIANCE_THRESHOLDS:
        for nu in KERNEL_NUS:
            t0 = time.time()
            res = cv_score(train_conds, Y_train, train_weights, comp_masks, vt, nu, N_RESTARTS_CV)
            dt = time.time() - t0
            res.update(variance_threshold=vt, kernel_nu=nu)
            results.append(res)
            nu_label = 'RBF(inf)' if np.isinf(nu) else f'{nu}'
            print(f'  var_thresh={vt:<6}  nu={nu_label:<9}  modes~={res["k"]:.0f}  '
                  f'mean_KL={res["mean_KL"]:.4f}  R2={res["R2"]:.4f}  wrMAE={res["wrMAE"]:.4f}  '
                  f'score={res["score"]:.4f}  ({dt:.0f}s)', flush=True)

    print(f'\nTotal tuning time: {time.time() - t_start:.0f}s', flush=True)

    best = min(results, key=lambda r: r['mean_KL'])
    print(f"\nBest by mean_KL (primary metric): variance_threshold={best['variance_threshold']}  "
          f"kernel_nu={best['kernel_nu']}  "
          f"(CV mean_KL={best['mean_KL']:.4f}, R2={best['R2']:.4f}, wrMAE={best['wrMAE']:.4f}, "
          f"~{best['k']:.0f} modes)", flush=True)

    # Final refit on the FULL training set with the best hyperparameters, evaluated once on the
    # real (never touched during tuning) phase1/phase2 test sets.
    print('\nRefitting on the full training set with the best hyperparameters...', flush=True)
    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    y_pred1, k_final = fit_predict_pod_gp(train_conds_sc, Y_train, test1_conds_sc,
                                           best['variance_threshold'], best['kernel_nu'],
                                           N_RESTARTS_FINAL)
    y_pred2, _ = fit_predict_pod_gp(train_conds_sc, Y_train, test2_conds_sc,
                                     best['variance_threshold'], best['kernel_nu'],
                                     N_RESTARTS_FINAL)
    print(f'Final model uses {k_final} modes.', flush=True)

    res1 = pg.evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], comp_masks)
    res2 = pg.evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], comp_masks)
    pg.print_result('POD+GP (tuned)', 'Phase 1 (interpolation)', res1)
    pg.print_result('POD+GP (tuned)', 'Phase 2 (extrapolation)', res2)

    try:
        np.save(f'{pg.OUT_PREFIX}_tuned_y_pred1.npy', y_pred1)
        np.save(f'{pg.OUT_PREFIX}_tuned_y_pred2.npy', y_pred2)
        print(f'\nSaved: {pg.OUT_PREFIX}_tuned_y_pred1.npy, {pg.OUT_PREFIX}_tuned_y_pred2.npy',
              flush=True)
    except OSError as e:
        print(f'\n[WARN] Could not save predictions: {e}', flush=True)


if __name__ == '__main__':
    main()
