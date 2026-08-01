"""
IsoMap+RBF baseline, reproduced from Peter et al. 2025 ("ONERA's CRM WBPN database for machine
learning activities..."), section 5.4:

  1. IsoMap embeds the training wall-field snapshots {y_i} in R^n_p onto a low-dimensional
     manifold R^r (r=3 in the paper, matching the 3 physical flow parameters Minf/AoA/Pi).
  2. The r latent coordinates z are regressed as functions of the flow conditions p using RBF
     ("a well-established numerical toolbox" -- the paper cites SMT [37]; SMT isn't installed in
     this repo/cluster venv, so scipy.interpolate.RBFInterpolator is used instead -- it implements
     the same RBF-regression math and is the standard substitute).
  3. IsoMap has no natural backmapping, so kNN interpolation (same inverse-distance formula as the
     paper's input-space kNN baseline, section 5.2 / Eq. 5) is used in the *latent* space instead:
     given a predicted z*, the k nearest training z's are found and their wall fields are
     inverse-distance-weighted together.

The paper predicts 4 fields (Cp, Cfx, Cfy, Cfz); this repo only has `rho` (density) extracted into
data/splitv3, so that's the target here -- same target as every other model in this repo, which is
what makes the results comparable.

This is a cheap, CPU-only model (everything operates on the 324-row training set, not the
85M-row pointwise data) -- unlike paper_global_mlp.py it runs fine on a laptop.
"""

import json
import os

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
EPS = 1e-6

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
KL_N_BINS = 200

DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'
SPLIT_DIR = 'splitv3'

N_COMPONENTS = 3          # r: paper finds this matches the 3 physical flow parameters
ISOMAP_K = 15              # IsoMap's own neighbor-graph size (not stated in the paper); smallest
                            # value found to keep the neighbor graph on this training set fully
                            # connected -- n_neighbors=10 leaves 2 disconnected components, which
                            # sklearn silently patches by connecting them at a large distance
RBF_KERNEL = 'thin_plate_spline'
K_CANDIDATES = range(2, 13)   # paper's exact candidate range for the backmapping k
N_REPEATS = 20
N_HOLDOUT = 10             # paper: "repeatedly removing only ten randomly selected wall distributions"
SEED = 0

OUT_PREFIX = 'utils/paper_isomap_rbf'


# ---------------------------------------------------------------------------
# Official metrics -- ported verbatim from bundle/scoring_program/scoring.py
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
# Data loading
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
# kNN backmapping in latent space -- paper's Eq. 5 (inverse-distance weighting),
# applied to z-space neighbors instead of p-space neighbors.
# ---------------------------------------------------------------------------

def knn_backmap(Z_query, Z_pool, Y_pool, k):
    k = min(k, len(Z_pool))
    nn = NearestNeighbors(n_neighbors=k).fit(Z_pool)
    dist, idx = nn.kneighbors(Z_query)
    dist = np.maximum(dist, 1e-12)
    w = 1.0 / dist
    w /= w.sum(axis=1, keepdims=True)
    y_pred = np.empty((Z_query.shape[0], Y_pool.shape[1]), dtype=np.float64)
    for i in range(Z_query.shape[0]):
        y_pred[i] = w[i] @ Y_pool[idx[i]]
    return y_pred


def select_backmap_k(Z_train, Y_train, n_repeats=N_REPEATS, n_holdout=N_HOLDOUT,
                      k_candidates=K_CANDIDATES, seed=SEED):
    """
    Paper's procedure (section 5.2/5.4): repeatedly hold out ~10 random training snapshots and
    measure the accuracy of the kNN-backmapped prediction for each candidate k, then keep the best
    on average. IsoMap+RBF are fit once on the full training set beforehand (RBFInterpolator
    interpolates its training points exactly at zero smoothing, so a held-out point's already-fit
    latent coordinate Z_train[i] is what a fresh p->z regression would also predict for it) -- this
    CV loop isolates and tunes only the backmapping step's k, which is what the paper's "best
    number of neighbors k" search is actually about.
    """
    rng = np.random.default_rng(seed)
    n_train = Z_train.shape[0]
    scores = {k: [] for k in k_candidates}

    for _ in range(n_repeats):
        holdout_idx = rng.choice(n_train, size=min(n_holdout, n_train - 1), replace=False)
        keep_mask = np.ones(n_train, dtype=bool)
        keep_mask[holdout_idx] = False
        keep_idx = np.flatnonzero(keep_mask)

        Z_holdout = Z_train[holdout_idx]
        Y_holdout = Y_train[holdout_idx]

        for k in k_candidates:
            y_pred = knn_backmap(Z_holdout, Z_train[keep_idx], Y_train[keep_idx], k)
            r2 = compute_R2(Y_holdout.reshape(-1), y_pred.reshape(-1),
                             np.ones(len(holdout_idx)))
            scores[k].append(r2)

    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    best_k = max(mean_scores, key=mean_scores.get)
    return best_k, mean_scores


def main():
    print('Loading data...', flush=True)
    data = load_split(DATA_DIR, SPLIT_DIR)
    train_conds, Y_train, n_train = data['train_conds'], data['Y_train'], data['n_train']
    print(f'n_train={n_train}  n_test1={data["n_test1"]}  n_test2={data["n_test2"]}', flush=True)

    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    print(f'\nIsoMap reconstruction-error curve (paper: r=3 matches the 3 physical parameters):')
    for r in range(2, 7):
        iso_probe = Isomap(n_neighbors=ISOMAP_K, n_components=r)
        iso_probe.fit(Y_train)
        print(f'  r={r}  reconstruction_error={iso_probe.reconstruction_error():.4f}', flush=True)

    print(f'\nFitting IsoMap (n_neighbors={ISOMAP_K}, n_components={N_COMPONENTS}) on training fields...', flush=True)
    isomap = Isomap(n_neighbors=ISOMAP_K, n_components=N_COMPONENTS)
    Z_train = isomap.fit_transform(Y_train)

    print(f'Fitting RBFInterpolator (kernel={RBF_KERNEL}) for p -> z...', flush=True)
    rbf = RBFInterpolator(train_conds_sc, Z_train, kernel=RBF_KERNEL)

    print(f'\nSelecting backmapping k via {N_REPEATS}x leave-{N_HOLDOUT}-out CV, '
          f'k in {list(K_CANDIDATES)}...', flush=True)
    best_k, mean_scores = select_backmap_k(Z_train, Y_train)
    for k, s in mean_scores.items():
        marker = '  <-- best' if k == best_k else ''
        print(f'  k={k:2d}  mean R2={s:.4f}{marker}')
    print(f'Selected k={best_k}', flush=True)

    Z_pred1 = rbf(test1_conds_sc)
    Z_pred2 = rbf(test2_conds_sc)

    print('\nPredicting phase 1...', flush=True)
    y_pred1 = knn_backmap(Z_pred1, Z_train, Y_train, best_k).reshape(-1)
    print('Predicting phase 2...', flush=True)
    y_pred2 = knn_backmap(Z_pred2, Z_train, Y_train, best_k).reshape(-1)

    joblib.dump({
        'isomap': isomap, 'rbf': rbf, 'Z_train': Z_train, 'Y_train': Y_train,
        'cond_scaler_mean': cond_scaler.mean_, 'cond_scaler_scale': cond_scaler.scale_,
        'best_k': best_k, 'n_components': N_COMPONENTS, 'isomap_k': ISOMAP_K,
        'rbf_kernel': RBF_KERNEL,
    }, f'{OUT_PREFIX}_model.joblib')
    np.save(f'{OUT_PREFIX}_y_pred1.npy', y_pred1)
    np.save(f'{OUT_PREFIX}_y_pred2.npy', y_pred2)
    print(f'Saved: {OUT_PREFIX}_model.joblib, {OUT_PREFIX}_y_pred1.npy, {OUT_PREFIX}_y_pred2.npy', flush=True)

    res1 = evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], data['comp_masks'])
    res2 = evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], data['comp_masks'])
    print_result('IsoMap+RBF', 'Phase 1 (interpolation)', res1)
    print_result('IsoMap+RBF', 'Phase 2 (extrapolation)', res2)


if __name__ == '__main__':
    main()
