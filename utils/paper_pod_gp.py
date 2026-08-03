"""
POD + Gaussian Process baseline: (Minf, AoA, Pi) -> full wall rho field, via mean-subtracted
POD/PCA + one independent, anisotropic (ARD) Matern GP per retained mode.

Replaces the IsoMap+RBF baseline (paper_isomap_rbf.py) as this project's second field regressor.
Reasoning: kNN backmapping (what IsoMap+RBF uses to go from latent space back to the full field)
produces a convex combination of training fields. That means (a) it can never exceed the range of
what's in the training set -- fatal for phase 2 extrapolation -- and (b) it smears shocks: fields
with a shock at different chordwise positions average into a blurred double-shock that matches no
real solution. POD's decoder is exact and linear (mean_field + Z @ components, no
neighbor-averaging), and a GP regularizes instead of interpolating exactly through control points
like the RBF did, while also giving a predictive variance per point -- useful for flagging where
phase 2 is extrapolating.

This is ported from this repo's existing utils/gp_pod_model.py, whose core POD+GP fitting logic is
already correct (sklearn's PCA always mean-subtracts internally; passing length_scale as a
per-dimension array already makes sklearn's Matern kernel anisotropic/ARD, no extra work needed)
-- but that script (a) is hardcoded to the stale splitv2 split, not the current splitv3 every
other model here uses, and (b) evaluates with a hand-rolled KL/rMAE convention that doesn't match
bundle/scoring_program/scoring.py, so its numbers were never comparable to the sibling paper_*.py
scripts. This version fixes both: splitv3 + the same official-metric functions as
paper_global_mlp.py / paper_isomap_rbf.py.

Cheap, CPU-only: GP regression here costs O(n_train^3), not O(n_train * NWALLP) -- everything
operates on the 324-row training set, one GP per POD mode, fit in parallel across CPU cores.
"""

import json
import os
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
EPS = 1e-6

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
KL_N_BINS = 200

DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'
SPLIT_DIR = 'splitv3'

VARIANCE_THRESHOLD = 0.99   # keep the smallest number of POD modes reaching this cumulative
                             # explained variance (same convention as gp_pod_model.py / the
                             # paper's own POD+RBF baseline, section 5.3)
MAX_COMPONENTS = 150         # PCA is first fit with this many components (capped at n_train - 1)
KERNEL_NU = 2.5              # Matern smoothness (0.5, 1.5, 2.5, or inf=RBF)
ALPHA = 1e-8                 # numerical nugget added to the GP kernel diagonal
N_RESTARTS = 5                # random restarts of the GP hyperparameter optimizer, per mode
N_JOBS = -1
SEED = 0

OUT_PREFIX = 'utils/paper_pod_gp'


# ---------------------------------------------------------------------------
# Official metrics -- ported verbatim from bundle/scoring_program/scoring.py, identical to
# paper_global_mlp.py / paper_isomap_rbf.py so all three are directly comparable.
# ---------------------------------------------------------------------------

def compute_R2(y, yhat, confidence_per_case):
    y_blocks = y.reshape(-1, NWALLP)
    yhat_blocks = yhat.reshape(-1, NWALLP)
    ymean = np.mean(y)
    sq_err = ((y_blocks - yhat_blocks) ** 2).sum(axis=1)
    sq_dev = ((y_blocks - ymean) ** 2).sum(axis=1)
    SSE = float(np.dot(confidence_per_case, sq_err))
    SSD = float(np.dot(confidence_per_case, sq_dev))
    if SSD < EPS:
        return 0.0
    return 1.0 - SSE / SSD


def compute_wrMAE(y, yhat, confidence_per_case):
    y_blocks = y.reshape(-1, NWALLP)
    yhat_blocks = yhat.reshape(-1, NWALLP)
    mask = confidence_per_case >= 1.0
    if not np.any(mask):
        raise ValueError('No high-confidence conditions to compute wrMAE.')
    mean_abs_diff = np.mean(np.abs(y_blocks - yhat_blocks), axis=1)
    mean_abs_y = np.maximum(np.mean(np.abs(y_blocks), axis=1), EPS)
    relMAE = mean_abs_diff / mean_abs_y
    candidates = np.flatnonzero(mask)
    iworst_local = int(np.argmax(relMAE[mask]))
    iworst = int(candidates[iworst_local])
    return iworst, float(relMAE[iworst])


def _residual_kl(y_true_case, y_pred_case, comp_masks, sigma_ref, n_bins=KL_N_BINS):
    eps = y_pred_case - y_true_case
    sigma_y = float(y_true_case.std()) + EPS
    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = KL_WEIGHTS[cname]
    lim = 5.0 * sigma_y
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
    p = np.clip(p * dx, 1e-10, None)
    p /= p.sum()
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = np.exp(-0.5 * (bin_centers / sigma_ref) ** 2) / (sigma_ref * np.sqrt(2.0 * np.pi)) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_mean_KL(y, yhat, confidence_per_case, comp_masks, sigma_ref):
    y_blocks = y.reshape(-1, NWALLP)
    yhat_blocks = yhat.reshape(-1, NWALLP)
    valid_idx = np.flatnonzero(confidence_per_case >= 1.0)
    if valid_idx.size == 0:
        raise ValueError('No high-confidence conditions to compute mean_KL.')
    kl_values = np.array([
        _residual_kl(y_blocks[i], yhat_blocks[i], comp_masks, sigma_ref)
        for i in valid_idx
    ])
    iworst_local = int(np.argmax(kl_values))
    iworst = int(valid_idx[iworst_local])
    return float(kl_values.mean()), iworst, float(kl_values[iworst_local])


def evaluate_phase(y, yhat, weights, comp_masks):
    sigma_ref = max(0.01 * float(np.mean(y)), EPS)
    R2 = compute_R2(y, yhat, weights)
    iworst_wrmae, wrMAE = compute_wrMAE(y, yhat, weights)
    mean_kl, iworst_kl, worst_kl = compute_mean_KL(y, yhat, weights, comp_masks, sigma_ref)
    score = 5.0 * R2 + 5.0 * (1.0 - wrMAE)
    return dict(R2=R2, wrMAE=wrMAE, iworst_wrmae=iworst_wrmae,
                mean_KL=mean_kl, iworst_kl=iworst_kl, worst_kl=worst_kl, score=score)


def print_result(name, phase_label, res):
    print(f'\n{name} -- {phase_label}')
    print(f'  R2      : {res["R2"]:.6f}')
    print(f'  wrMAE   : {res["wrMAE"]:.6f}  (worst sim idx {res["iworst_wrmae"]})')
    print(f'  mean_KL : {res["mean_KL"]:.6f}  (worst sim idx {res["iworst_kl"]}, KL={res["worst_kl"]:.6f})')
    print(f'  score   : {res["score"]:.6f}', flush=True)


# ---------------------------------------------------------------------------
# Data loading -- identical to paper_global_mlp.py / paper_isomap_rbf.py
# ---------------------------------------------------------------------------

def load_split(data_dir, split_dir=SPLIT_DIR):
    base = os.path.join(data_dir, split_dir)

    X_train = np.load(os.path.join(base, 'train_data.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(base, 'train_labels.npy'))
    X_test1 = np.load(os.path.join(base, 'test_phase1_data.npy'), mmap_mode='r')
    y_test1 = np.load(os.path.join(base, 'test_phase1_labels.npy'))
    w_test1 = np.load(os.path.join(base, 'test_phase1_weights.npy'))
    X_test2 = np.load(os.path.join(base, 'test_phase2_data.npy'), mmap_mode='r')
    y_test2 = np.load(os.path.join(base, 'test_phase2_labels.npy'))
    w_test2 = np.load(os.path.join(base, 'test_phase2_weights.npy'))

    component_labels = np.load(os.path.join(base, 'component_labels_unique.npy'))
    with open(os.path.join(base, 'component_map.json')) as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    n_train = X_train.shape[0] // NWALLP
    n_test1 = X_test1.shape[0] // NWALLP
    n_test2 = X_test2.shape[0] // NWALLP

    train_conds = np.asarray(X_train[::NWALLP, COL_MINF:COL_PI + 1])
    test1_conds = np.asarray(X_test1[::NWALLP, COL_MINF:COL_PI + 1])
    test2_conds = np.asarray(X_test2[::NWALLP, COL_MINF:COL_PI + 1])

    Y_train = y_train.reshape(n_train, NWALLP)

    return dict(
        train_conds=train_conds, Y_train=Y_train, n_train=n_train,
        test1_conds=test1_conds, y_test1=y_test1, w_test1=w_test1, n_test1=n_test1,
        test2_conds=test2_conds, y_test2=y_test2, w_test2=w_test2, n_test2=n_test2,
        comp_masks=comp_masks,
    )


# ---------------------------------------------------------------------------
# POD + one ARD GP per mode -- same design as gp_pod_model.py
# ---------------------------------------------------------------------------

def make_kernel(n_dims, nu):
    # Tried widening these from (1e-2,1e2)/(1e-12,1e-1) after a first fit hit both ceilings
    # repeatedly. Result: held-out phase 1 R2 dropped 0.980->0.956 and mean_KL nearly tripled
    # (1.48->3.41) -- worse, not better, and warnings just relocated to the new bounds instead of
    # disappearing. Diagnosis: Pi only takes 3 distinct values in training, so the marginal
    # likelihood is nearly flat along that length-scale, poorly identified by n=324 gridded (not
    # randomly scattered) points -- these bounds are acting as a useful regularizing prior here,
    # not blocking a genuinely better optimum. Reverted to gp_pod_model.py's original bounds.
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, 1e2), nu=nu)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-12, 1e-1)))


def fit_one_gp(conds_sc, z_col, n_dims, nu, alpha, n_restarts, seed):
    gp = GaussianProcessRegressor(
        kernel=make_kernel(n_dims, nu),
        alpha=alpha,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=seed,
    )
    gp.fit(conds_sc, z_col)
    return gp


def choose_n_modes(pca, variance_threshold):
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, variance_threshold) + 1)
    k = min(k, len(cum))
    return k, float(cum[k - 1])


def main():
    print('Loading data...', flush=True)
    data = load_split(DATA_DIR, SPLIT_DIR)
    train_conds, Y_train, n_train = data['train_conds'], data['Y_train'], data['n_train']
    print(f'n_train={n_train}  n_test1={data["n_test1"]}  n_test2={data["n_test2"]}', flush=True)

    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    n_components = min(MAX_COMPONENTS, n_train - 1)
    print(f'Fitting PCA ({n_components} components, capped by n_train-1={n_train - 1})...', flush=True)
    pca = PCA(n_components=n_components, random_state=SEED)
    Z_train_full = pca.fit_transform(Y_train)   # mean-subtracted internally by PCA

    k, cum_var = choose_n_modes(pca, VARIANCE_THRESHOLD)
    print(f'Keeping {k}/{n_components} modes to reach {VARIANCE_THRESHOLD:.4f} explained variance '
          f'(actual cumulative = {cum_var:.5f})', flush=True)

    Z_train = Z_train_full[:, :k]
    components = pca.components_[:k]        # (k, NWALLP)
    mean_field = pca.mean_                   # (NWALLP,)

    print(f'Fitting {k} independent ARD Gaussian Processes '
          f'(Matern nu={KERNEL_NU}, {N_RESTARTS} restarts, n_jobs={N_JOBS})...', flush=True)
    t0 = time.time()
    gps = Parallel(n_jobs=N_JOBS)(
        delayed(fit_one_gp)(train_conds_sc, Z_train[:, m], train_conds_sc.shape[1],
                             KERNEL_NU, ALPHA, N_RESTARTS, SEED)
        for m in range(k)
    )
    print(f'GP fitting done in {time.time() - t0:.1f}s', flush=True)

    def predict_phase(conds_sc, n_sims):
        Z_mean = np.empty((n_sims, k))
        Z_std = np.empty((n_sims, k))
        for m, gp in enumerate(gps):
            mean_m, std_m = gp.predict(conds_sc, return_std=True)
            Z_mean[:, m], Z_std[:, m] = mean_m, std_m
        y_pred = mean_field[None, :] + Z_mean @ components          # exact linear POD decoder
        # diagonal approximation: Var[y_j] = sum_m components[m,j]^2 * Var[z_m]
        var_pred = (Z_std ** 2) @ (components ** 2)
        return y_pred.reshape(-1), np.sqrt(var_pred).reshape(-1)

    print('Predicting phase 1...', flush=True)
    y_pred1, std_pred1 = predict_phase(test1_conds_sc, data['n_test1'])
    print('Predicting phase 2...', flush=True)
    y_pred2, std_pred2 = predict_phase(test2_conds_sc, data['n_test2'])

    # Metrics first, saving last -- same fix as paper_global_mlp.py: a save failure must never
    # cost results that are already computed.
    res1 = evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], data['comp_masks'])
    res2 = evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], data['comp_masks'])
    print_result('POD+GP', 'Phase 1 (interpolation)', res1)
    print_result('POD+GP', 'Phase 2 (extrapolation)', res2)

    print(f'\nMean predictive std (phase 1, all points): {std_pred1.mean():.6f}')
    print(f'Mean predictive std (phase 2, all points): {std_pred2.mean():.6f}', flush=True)

    try:
        np.save(f'{OUT_PREFIX}_y_pred1.npy', y_pred1)
        np.save(f'{OUT_PREFIX}_y_pred2.npy', y_pred2)
        np.save(f'{OUT_PREFIX}_std_pred1.npy', std_pred1)
        np.save(f'{OUT_PREFIX}_std_pred2.npy', std_pred2)
        print(f'\nSaved: {OUT_PREFIX}_y_pred1.npy, {OUT_PREFIX}_y_pred2.npy, '
              f'{OUT_PREFIX}_std_pred1.npy, {OUT_PREFIX}_std_pred2.npy', flush=True)
    except OSError as e:
        print(f'\n[WARN] Could not save predictions: {e}', flush=True)

    try:
        joblib.dump({
            'gps': gps, 'k': k, 'n_components_fit': n_components,
            'pca_mean': mean_field, 'pca_components': components,
            'cond_scaler_mean': cond_scaler.mean_, 'cond_scaler_scale': cond_scaler.scale_,
            'variance_threshold': VARIANCE_THRESHOLD, 'cum_var': cum_var,
            'kernel_nu': KERNEL_NU, 'alpha': ALPHA,
        }, f'{OUT_PREFIX}_model.joblib')
        print(f'Saved: {OUT_PREFIX}_model.joblib', flush=True)
    except OSError as e:
        print(f'[WARN] Could not save model: {e}', flush=True)


if __name__ == '__main__':
    main()
