"""
Global MLP baseline, reproduced from Peter et al. 2025 ("ONERA's CRM WBPN database for machine
learning activities..."), section 5.1: a field-by-field regressor mapping the 3 flow conditions
(Minf, AoA, Pi) directly to the entire wall field in one shot -- hidden layers (75, 120, 1226,
16490), trained for 200 epochs. The paper predicts 4 fields (Cp, Cfx, Cfy, Cfz); this repo only
has `rho` (density) extracted into data/splitv3, so that's the target here -- same target as
every other model in this repo (train_mlp_with_kl.py, gp_pod_model.py, ...), which is what makes
the results comparable.

Kept literal to the paper: no mean-field residual skip (unlike this repo's own GlobalMLP class in
train_mlp_with_kl.py), no architecture-shrinking safety net. The final Linear(16490, NWALLP) layer
alone is ~4.3B parameters (~17GB fp32) -- this is why train_mlp_with_kl.py shrank the same
architecture to (128,256,512) previously. This script does not shrink it; run on a cluster GPU
node, not a laptop.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
EPS = 1e-6

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
KL_N_BINS = 200

DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'
SPLIT_DIR = 'splitv3'

HIDDEN = (75, 120, 1226, 16490)  # paper's exact sizes, section 5.1
N_EPOCHS = 200                   # paper's stated global-model training budget
BATCH = 32
LR = 1e-3
SEED = 0

OUT_PREFIX = 'utils/paper_global_mlp'


# ---------------------------------------------------------------------------
# Official metrics -- ported verbatim from bundle/scoring_program/scoring.py
# so results are directly comparable to every other model scored the same way.
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
# Model -- plain feed-forward, paper's exact hidden sizes, no residual skip.
# ---------------------------------------------------------------------------

class GlobalMLP(nn.Module):
    def __init__(self, hidden, n_out):
        super().__init__()
        layers, d = [], 3
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.01)]
            d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, c):
        return self.net(c)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    print('Loading data...', flush=True)
    data = load_split(DATA_DIR, SPLIT_DIR)
    train_conds, Y_train, n_train = data['train_conds'], data['Y_train'], data['n_train']
    print(f'n_train={n_train}  n_test1={data["n_test1"]}  n_test2={data["n_test2"]}', flush=True)

    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    y_mean = float(Y_train.mean())
    y_std = float(Y_train.std()) + EPS
    Y_train_sc = (Y_train - y_mean) / y_std

    C_train = torch.tensor(train_conds_sc, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train_sc, dtype=torch.float32, device=device)

    model = GlobalMLP(HIDDEN, NWALLP).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model parameters: {n_params:,}', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f'Training Global MLP (hidden={HIDDEN}, {N_EPOCHS} epochs, batch={BATCH})...', flush=True)
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n_train, device=device)
        model.train()
        tot_loss = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            pred = model(C_train[idx])
            loss = loss_fn(pred, Y_train_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            elapsed = time.time() - t0
            print(f'epoch {epoch:4d}  train MSE (scaled) = {tot_loss / n_train:.6f}  '
                  f'elapsed = {elapsed:.1f}s', flush=True)

    print(f'Training done in {time.time() - t0:.1f}s', flush=True)

    model.eval()
    with torch.no_grad():
        C1 = torch.tensor(test1_conds_sc, dtype=torch.float32, device=device)
        C2 = torch.tensor(test2_conds_sc, dtype=torch.float32, device=device)
        y_pred1 = (model(C1).cpu().numpy() * y_std + y_mean).astype(np.float64).reshape(-1)
        y_pred2 = (model(C2).cpu().numpy() * y_std + y_mean).astype(np.float64).reshape(-1)

    torch.save({'state_dict': model.state_dict(),
                'cond_scaler_mean': cond_scaler.mean_, 'cond_scaler_scale': cond_scaler.scale_,
                'y_mean': y_mean, 'y_std': y_std,
                'hidden': HIDDEN, 'nwallp': NWALLP},
               f'{OUT_PREFIX}_model.pt')
    np.save(f'{OUT_PREFIX}_y_pred1.npy', y_pred1)
    np.save(f'{OUT_PREFIX}_y_pred2.npy', y_pred2)
    print(f'Saved: {OUT_PREFIX}_model.pt, {OUT_PREFIX}_y_pred1.npy, {OUT_PREFIX}_y_pred2.npy', flush=True)

    res1 = evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], data['comp_masks'])
    res2 = evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], data['comp_masks'])
    print_result('Global MLP', 'Phase 1 (interpolation)', res1)
    print_result('Global MLP', 'Phase 2 (extrapolation)', res2)


if __name__ == '__main__':
    main()
