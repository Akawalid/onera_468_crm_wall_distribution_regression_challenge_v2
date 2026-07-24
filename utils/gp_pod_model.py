"""
Gaussian Process (genuinely Bayesian) baseline: (Minf, AoA, Pi) -> full wall
rho field, via PCA/POD + one independent GP per retained mode.

--------------------------------------------------------------------------
Why PCA, and why one GP per mode instead of a pointwise GP
--------------------------------------------------------------------------
Exact GP regression costs O(n^3) in the number of TRAINING ROWS (Cholesky
of the kernel matrix), so the row count is the whole design question here.

  - A pointwise regressor -- (x, y, z, Minf, AoA, Pi) -> rho, one row per
    wall point, the approach used for the XGBoost baseline in
    klw_boosting.py / the pointwise section of train.py -- turns every wall
    point of every simulation into its own row: n_train * nwallp ~
    250 * 260,774 ~ 6.5*10^7 rows. Light-years past what exact GP can
    factor; you'd need sparse/inducing-point variational GPs (a much bigger
    undertaking), and it still wouldn't exploit the fact that the field's
    variation across conditions is low-rank to begin with.

  - The full-field regressors elsewhere in this repo (ExtraTrees, and the
    POD + HistGradientBoosting baseline in train.py) instead treat one
    SIMULATION as one row: n_train rows (a few hundred), nwallp targets.
    That fits exact GP's budget trivially (n ~ 250-450) -- but naively
    that's still nwallp = 260,774 independent single-output GPs to fit and
    store, most of them modeling pure noise.

So: PCA on the training fields (same idea as train.py's POD baseline),
keep only the modes needed to explain --variance_threshold of the
variance, and fit one small GP per mode: (Minf, AoA, Pi) -> mode
coefficient. Reconstruction is mean_field + sum_k z_k * component_k.
Being an actual Bayesian model, each GP also returns a predictive
variance per mode; these are combined (assuming independence across
modes -- a diagonal approximation, the same one used to go from PCA
component variances to reconstructed variance in PCA-whitening) into a
per-wall-point predictive std, which none of the point-estimate baselines
in this repo (KNN, the KLw-loss MLP, XGBoost, ExtraTrees) give you.

--------------------------------------------------------------------------
Where to run this
--------------------------------------------------------------------------
Locally, on CPU -- no cluster/GPU needed. sklearn's GaussianProcessRegressor
doesn't use a GPU anyway, and the whole fit operates on a compact
(n_train x n_modes) matrix, not the full (n_train x nwallp) field, so it's
nothing like the memory/compute footprint of the soft-histogram MLP or
pointwise XGBoost training in this repo. With n_train ~ a few hundred rows
and ~100-150 modes, each GP fit (few restarts of an O(n^3) optimization,
n ~ a few hundred) takes a couple of seconds; fit in parallel across CPU
cores (--n_jobs) and the whole thing finishes in a few minutes.
"""

import argparse
import json
import time

import numpy as np
import joblib
from joblib import Parallel, delayed
from scipy.stats import norm, entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

NWALLP   = 260774
COL_MINF = 6
COL_AOA  = 7
COL_PI   = 8
DATA_DIR = '/home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/FILES_RHO_ALL_POINTS_reduitfloat32/'

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
N_BINS     = 200

SIGMA_SCALE = None  # set from data in main(), same convention as the other utils/*.py scripts


# ---------------------------------------------------------------------------
# Metrics -- identical formulas to bundle/starting_kit/kit_utils/metrics.py,
# duplicated locally (same convention as klw_boosting.py / train_mlp_with_kl.py)
# ---------------------------------------------------------------------------

def residual_kl_normal(y_true, y_pred, sigma_ref_frac=0.1, n_bins=N_BINS):
    eps       = y_pred - y_true
    sigma_s   = SIGMA_SCALE
    sigma_ref = sigma_ref_frac * sigma_s
    lim  = 5.0 * sigma_s
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx   = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, density=True)
    p    = np.clip(p * dx, 1e-10, None)
    p   /= p.sum()
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q    = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q    = np.clip(q, 1e-10, None)
    q   /= q.sum()
    kl   = float(entropy(p, q))
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)),
            'bias': float(np.mean(eps) / sigma_s), 'spread': float(np.std(eps) / sigma_s)}


def residual_kl_weighted(y_true, y_pred, comp_masks, comp_weights, sigma_ref_frac=0.1, n_bins=N_BINS):
    eps       = y_pred - y_true
    sigma_s   = SIGMA_SCALE
    sigma_ref = sigma_ref_frac * sigma_s
    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = comp_weights.get(cname, 0.0)
    lim  = 5.0 * sigma_s
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx   = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
    p    = np.clip(p * dx, 1e-10, None)
    p   /= p.sum()
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()
    kl = float(entropy(p, q))
    bias   = float(np.average(eps, weights=sample_weight)) / sigma_s
    spread = float(np.sqrt(np.average((eps - eps.mean()) ** 2, weights=sample_weight))) / sigma_s
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)), 'bias': bias, 'spread': spread}


def compute_R2(y, yhat, confidence_pointwise):
    ymean = np.mean(y)
    SSE   = np.sum(confidence_pointwise * (y - yhat) ** 2)
    SSD   = np.sum(confidence_pointwise * (y - ymean) ** 2)
    return float(1.0 - SSE / SSD)


def compute_worst_rMAE(y, yhat, confidence_per_case):
    rMAE_list, idx_list = [], []
    for l in range(len(confidence_per_case)):
        if confidence_per_case[l] < 1.0:
            continue
        ycase    = y[l * NWALLP:(l + 1) * NWALLP]
        yhatcase = yhat[l * NWALLP:(l + 1) * NWALLP]
        rMAE_list.append(np.mean(np.abs(ycase - yhatcase)) / np.mean(np.abs(ycase)))
        idx_list.append(l)
    rMAE_arr    = np.array(rMAE_list)
    iworst_local = int(np.argmax(rMAE_arr))
    return idx_list[iworst_local], float(rMAE_arr[iworst_local])


def evaluate_phase(y, y_pred, weights, n_sims, comp_masks):
    Y, Yh = y.reshape(n_sims, NWALLP), y_pred.reshape(n_sims, NWALLP)
    confidence_pointwise = np.repeat(weights, NWALLP)
    iworst, worst_rMAE = compute_worst_rMAE(y, y_pred, weights)
    r2_global = compute_R2(y, y_pred, confidence_pointwise)
    comp_kl   = {c: [None] * n_sims for c in KL_WEIGHTS}
    kl_w      = np.full(n_sims, np.nan)
    valid_idx = np.where(weights == 1.0)[0]
    for i in valid_idx:
        yc, yhatc = Y[i], Yh[i]
        for cname, mask in comp_masks.items():
            if cname not in KL_WEIGHTS:
                continue
            comp_kl[cname][i] = residual_kl_normal(yc[mask], yhatc[mask])
        kl_w[i] = residual_kl_weighted(yc, yhatc, comp_masks, KL_WEIGHTS)['kl']
    return dict(Y=Y, Yh=Yh, iworst=iworst, worst_rMAE=worst_rMAE, r2=r2_global,
                kl=kl_w, comp_kl=comp_kl)


def print_phase_summary(label, res):
    i_mae = res['iworst']
    i_kl  = int(np.nanargmax(res['kl']))
    rMAE_at_worst_kl = np.mean(np.abs(res['Y'][i_kl] - res['Yh'][i_kl])) / np.mean(np.abs(res['Y'][i_kl]))
    print(f'\n{label}')
    print(f'  R2 global: {res["r2"]:.4f}')
    print(f'  worst rMAE: sim {i_mae}  rMAE={res["worst_rMAE"]:.4f}  KL={res["kl"][i_mae]:.4f}  score={1.0 / (1.0 + res["kl"][i_mae]):.4f}')
    print(f'  worst KL  : sim {i_kl}  KL={res["kl"][i_kl]:.4f}  score={1.0 / (1.0 + res["kl"][i_kl]):.4f}  rMAE={rMAE_at_worst_kl:.4f}')
    print(f'\n  KL by component')
    print(f'  {"component":<10}  {"mean KL":>9}  {"max KL":>9}  {"mean score":>10}  {"mean bias":>10}  {"mean spread":>11}')
    for cname in KL_WEIGHTS:
        vals = [v for v in res['comp_kl'][cname] if v is not None]
        kl_c = np.array([v['kl'] for v in vals])
        sc_c = np.array([v['score'] for v in vals])
        bi_c = np.array([v['bias'] for v in vals])
        sp_c = np.array([v['spread'] for v in vals])
        print(f'  {cname:<10}  {kl_c.mean():>9.4f}  {kl_c.max():>9.4f}  {sc_c.mean():>10.4f}  {bi_c.mean():>10.4f}  {sp_c.mean():>11.4f}')
    kl_global    = res['kl'][~np.isnan(res['kl'])]
    score_global = 1.0 / (1.0 + kl_global)
    print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}', flush=True)


# ---------------------------------------------------------------------------
# GP-per-mode fitting
# ---------------------------------------------------------------------------

def make_kernel(n_dims, nu):
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


# ---------------------------------------------------------------------------
# Data loading -- same conventions as the other utils/*.py scripts
# ---------------------------------------------------------------------------

def load_data(data_dir):
    print('Loading data...', flush=True)
    X_train = np.load(data_dir + 'splitv2/train_data.npy', mmap_mode='r')
    y_train = np.load(data_dir + 'splitv2/train_labels.npy')
    X_test1 = np.load(data_dir + 'splitv2/test_phase1_data.npy', mmap_mode='r')
    y_test1 = np.load(data_dir + 'splitv2/test_phase1_labels.npy')
    X_test2 = np.load(data_dir + 'splitv2/test_phase2_data.npy', mmap_mode='r')
    y_test2 = np.load(data_dir + 'splitv2/test_phase2_labels.npy')

    component_labels = np.load(data_dir + 'component_labels_unique.npy')
    with open(data_dir + 'component_map.json') as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    n_train = X_train.shape[0] // NWALLP
    n_test1 = X_test1.shape[0] // NWALLP
    n_test2 = X_test2.shape[0] // NWALLP
    print(f'n_train={n_train}  n_test1={n_test1}  n_test2={n_test2}', flush=True)

    train_conds = np.asarray(X_train[::NWALLP, COL_MINF:COL_PI + 1])
    test1_conds = np.asarray(X_test1[::NWALLP, COL_MINF:COL_PI + 1])
    test2_conds = np.asarray(X_test2[::NWALLP, COL_MINF:COL_PI + 1])

    test1_weights = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
    test2_weights = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

    Y_train = y_train.reshape(n_train, NWALLP)

    return dict(
        train_conds=train_conds, Y_train=Y_train,
        test1_conds=test1_conds, y_test1=y_test1, test1_weights=test1_weights, n_test1=n_test1,
        test2_conds=test2_conds, y_test2=y_test2, test2_weights=test2_weights, n_test2=n_test2,
        comp_masks=comp_masks,
    )


def main():
    global SIGMA_SCALE

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--variance_threshold', type=float, default=0.99,
                   help='keep the smallest number of PCA modes whose cumulative explained variance '
                        'reaches this fraction.')
    p.add_argument('--max_components', type=int, default=150,
                   help='PCA is first fit with this many components (capped at n_train - 1); '
                        '--variance_threshold then decides how many of them get a GP.')
    p.add_argument('--kernel_nu', type=float, default=2.5, help='Matern kernel smoothness (0.5, 1.5, 2.5, or inf=RBF).')
    p.add_argument('--alpha', type=float, default=1e-8, help='numerical nugget added to the GP kernel diagonal.')
    p.add_argument('--n_restarts', type=int, default=5, help='random restarts of the GP hyperparameter optimizer, per mode.')
    p.add_argument('--n_jobs', type=int, default=-1, help='parallel workers across modes (-1 = all cores).')
    p.add_argument('--out_prefix', default='gp_pod')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    data = load_data(args.data_dir)
    train_conds, Y_train = data['train_conds'], data['Y_train']
    n_train = Y_train.shape[0]

    mean_global = float(np.mean(Y_train))
    SIGMA_SCALE = 10.0 * 0.01 * mean_global
    print(f'global train mean = {mean_global:.4f}  sigma_scale = {SIGMA_SCALE:.5f}', flush=True)

    scaler = StandardScaler()
    train_conds_sc = scaler.fit_transform(train_conds)
    test1_conds_sc = scaler.transform(data['test1_conds'])
    test2_conds_sc = scaler.transform(data['test2_conds'])

    n_components = min(args.max_components, n_train - 1)
    print(f'Fitting PCA ({n_components} components, capped by n_train-1={n_train - 1})...', flush=True)
    pca = PCA(n_components=n_components, random_state=args.seed)
    Z_train_full = pca.fit_transform(Y_train)

    k, cum_var = choose_n_modes(pca, args.variance_threshold)
    print(f'Keeping {k}/{n_components} modes to reach {args.variance_threshold:.4f} explained variance '
          f'(actual cumulative = {cum_var:.5f})', flush=True)

    Z_train = Z_train_full[:, :k]
    components = pca.components_[:k]        # (k, nwallp)
    mean_field = pca.mean_                   # (nwallp,)

    print(f'Fitting {k} independent Gaussian Processes '
          f'(Matern nu={args.kernel_nu}, {args.n_restarts} restarts, n_jobs={args.n_jobs})...', flush=True)
    t0 = time.time()
    gps = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_one_gp)(train_conds_sc, Z_train[:, m], train_conds_sc.shape[1],
                             args.kernel_nu, args.alpha, args.n_restarts, args.seed)
        for m in range(k)
    )
    print(f'GP fitting done in {time.time() - t0:.1f}s', flush=True)

    def predict_phase(conds_sc, n_sims):
        Z_mean = np.empty((n_sims, k))
        Z_std  = np.empty((n_sims, k))
        for m, gp in enumerate(gps):
            mean_m, std_m = gp.predict(conds_sc, return_std=True)
            Z_mean[:, m], Z_std[:, m] = mean_m, std_m
        y_pred = mean_field[None, :] + Z_mean @ components          # (n_sims, nwallp)
        # diagonal approximation: Var[y_j] = sum_m components[m,j]^2 * Var[z_m]
        var_pred = (Z_std ** 2) @ (components ** 2)                 # (n_sims, nwallp)
        return y_pred.reshape(-1), np.sqrt(var_pred).reshape(-1)

    print('Predicting phase 1...', flush=True)
    y_pred1, std_pred1 = predict_phase(test1_conds_sc, data['n_test1'])
    print('Predicting phase 2...', flush=True)
    y_pred2, std_pred2 = predict_phase(test2_conds_sc, data['n_test2'])

    np.save(f'{args.out_prefix}_y_pred1.npy', y_pred1)
    np.save(f'{args.out_prefix}_y_pred2.npy', y_pred2)
    np.save(f'{args.out_prefix}_std_pred1.npy', std_pred1)
    np.save(f'{args.out_prefix}_std_pred2.npy', std_pred2)

    joblib.dump({
        'gps': gps, 'k': k, 'n_components_fit': n_components,
        'pca_mean': mean_field, 'pca_components': components,
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
        'variance_threshold': args.variance_threshold, 'cum_var': cum_var,
        'kernel_nu': args.kernel_nu, 'alpha': args.alpha,
    }, f'{args.out_prefix}_model.joblib')
    print(f'Saved: {args.out_prefix}_model.joblib, {args.out_prefix}_y_pred{{1,2}}.npy, '
          f'{args.out_prefix}_std_pred{{1,2}}.npy', flush=True)

    res1 = evaluate_phase(data['y_test1'], y_pred1, data['test1_weights'], data['n_test1'], data['comp_masks'])
    res2 = evaluate_phase(data['y_test2'], y_pred2, data['test2_weights'], data['n_test2'], data['comp_masks'])
    for label, res in [('Phase 1 (GP + POD)', res1), ('Phase 2 (GP + POD)', res2)]:
        print_phase_summary(label, res)

    print(f'\nMean predictive std (phase 1, all points): {std_pred1.mean():.6f}')
    print(f'Mean predictive std (phase 2, all points): {std_pred2.mean():.6f}')


if __name__ == '__main__':
    main()
