import json
import numpy as np
import xgboost as xgb
from scipy.stats import norm, entropy

nwallp   = 260774
COL_MINF = 6
COL_AOA  = 7
COL_PI   = 8
DATA_DIR = '/home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/FILES_RHO_ALL_POINTS_reduitfloat32/'

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}

N_BINS         = 200
SIGMA_REF      = 0.1
TAU            = 10.0 / N_BINS
POINTS_PER_SIM = 20000
N_ROUNDS       = 400
PATIENCE       = 40
LR             = 0.02
MAX_DEPTH      = 7
LAMBDA_MSE     = 0.1
HESS_FLOOR     = 1e-9
SEED           = 0

SIGMA_SCALE = None

rng = np.random.default_rng(SEED)


def residual_kl_normal(y_true, y_pred, sigma_ref_frac=0.1, n_bins=200):
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


def residual_kl_weighted(y_true, y_pred, comp_masks, comp_weights, sigma_ref_frac=0.1, n_bins=200):
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
        ycase    = y[l * nwallp:(l + 1) * nwallp]
        yhatcase = yhat[l * nwallp:(l + 1) * nwallp]
        rMAE_list.append(np.mean(np.abs(ycase - yhatcase)) / np.mean(np.abs(ycase)))
        idx_list.append(l)
    rMAE_arr     = np.array(rMAE_list)
    iworst_local = int(np.argmax(rMAE_arr))
    return idx_list[iworst_local], float(rMAE_arr[iworst_local])


def evaluate_phase(y, y_pred, weights, n_sims, comp_masks):
    Y, Yh = y.reshape(n_sims, nwallp), y_pred.reshape(n_sims, nwallp)
    confidence_pointwise = np.repeat(weights, nwallp)
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


class KLWObjective:
    def __init__(self, y_rows, sim_ids, point_w, sigma_s, centers, log_q, n_sims):
        self.y = y_rows
        self.sim_ids = sim_ids
        self.w = point_w
        self.sigma = sigma_s
        self.c = centers
        self.log_q = log_q
        self.n_sims = n_sims
        self.slices = [np.where(sim_ids == s)[0] for s in range(n_sims)]
        self.last_kl = None

    def __call__(self, preds, dtrain):
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        kls = np.zeros(self.n_sims)
        for s in range(self.n_sims):
            idx = self.slices[s]
            eps = (preds[idx] - self.y[idx]) / self.sigma
            w = self.w[idx]
            w = w / w.sum()
            d = (eps[:, None] - self.c[None, :]) / TAU
            logits = -0.5 * d * d
            logits -= logits.max(axis=1, keepdims=True)
            sfm = np.exp(logits)
            sfm /= sfm.sum(axis=1, keepdims=True)
            p = w @ sfm
            p = np.clip(p, 1e-10, None)
            p /= p.sum()
            r = np.log(p) - self.log_q
            kls[s] = float(np.sum(p * r))
            dl = -d / TAU
            A = np.sum(sfm * dl, axis=1)
            B = sfm * (dl - A[:, None])
            g_eps = w * (B @ r)
            h_eps = (w * w) * np.sum(B * B / p[None, :], axis=1)
            n_pts = len(idx)
            grad[idx] = n_pts * g_eps / self.sigma
            hess[idx] = n_pts * h_eps / (self.sigma * self.sigma)
        self.last_kl = kls
        eps_all = preds - self.y
        grad += LAMBDA_MSE * eps_all
        hess += LAMBDA_MSE
        hess = np.maximum(hess, HESS_FLOOR)
        return grad, hess


def main():
    global SIGMA_SCALE

    print('Loading data...', flush=True)
    X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
    y_train = np.load(DATA_DIR + 'splitv2/train_labels.npy')
    X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
    y_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_labels.npy')
    X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')
    y_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_labels.npy')

    component_labels = np.load(DATA_DIR + 'component_labels_unique.npy')
    with open(DATA_DIR + 'component_map.json') as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    train_conds = X_train[::nwallp, COL_MINF:COL_PI + 1]
    test1_conds = X_test1[::nwallp, COL_MINF:COL_PI + 1]
    test2_conds = X_test2[::nwallp, COL_MINF:COL_PI + 1]

    n_train = X_train.shape[0] // nwallp
    n_test1 = X_test1.shape[0] // nwallp
    n_test2 = X_test2.shape[0] // nwallp
    print(f'n_train={n_train}  n_test1={n_test1}  n_test2={n_test2}', flush=True)

    mean_global = float(np.mean(y_train))
    SIGMA_SCALE = 10.0 * 0.01 * mean_global
    print(f'global train mean = {mean_global:.4f}', flush=True)

    test1_weights = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
    test2_weights = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

    Y = y_train.reshape(n_train, nwallp)
    mean_field = Y.mean(axis=0)
    std_field = Y.std(axis=0)

    sw = np.zeros(nwallp, dtype=np.float64)
    for cname, mask in comp_masks.items():
        sw[mask] = KL_WEIGHTS.get(cname, 0.0)

    X_ref = X_train[:nwallp]
    static = np.column_stack([
        X_ref[:, :3], X_ref[:, 3:6], mean_field, std_field,
    ]).astype(np.float32)

    print('Assembling training rows...', flush=True)
    keep = np.where(sw > 0.0)[0]
    rows_per_sim = min(POINTS_PER_SIM, len(keep))
    F_list, y_list, id_list, w_list, m_list = [], [], [], [], []
    for s in range(n_train):
        pts = rng.choice(keep, size=rows_per_sim, replace=False)
        f_conds = np.tile(train_conds[s].astype(np.float32), (rows_per_sim, 1))
        F_list.append(np.hstack([static[pts], f_conds]))
        y_list.append(Y[s, pts].astype(np.float64))
        id_list.append(np.full(rows_per_sim, s, dtype=np.int32))
        w_list.append(sw[pts])
        m_list.append(mean_field[pts].astype(np.float64))
    F_train = np.vstack(F_list)
    y_rows = np.concatenate(y_list)
    sim_ids = np.concatenate(id_list)
    point_w = np.concatenate(w_list)
    base_margin = np.concatenate(m_list)
    print(f'train matrix: {F_train.shape}', flush=True)

    edges = np.linspace(-5.0, 5.0, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    q = norm.pdf(centers, loc=0.0, scale=SIGMA_REF)
    q = np.clip(q / q.sum(), 1e-10, None)
    log_q = np.log(q)

    dtrain = xgb.DMatrix(F_train, label=y_rows)
    dtrain.set_base_margin(base_margin)

    objective = KLWObjective(y_rows, sim_ids, point_w, SIGMA_SCALE, centers, log_q, n_train)

    params = {
        'max_depth': MAX_DEPTH,
        'eta': LR,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1e-6,
        'lambda': 1.0,
        'tree_method': 'hist',
        'nthread': -1,
        'seed': SEED,
    }

    print('Training XGBoost with soft histogram KLw objective (Newton, GN hessian)...', flush=True)
    booster = None
    best_kl = float('inf')
    best_raw = None
    best_it = -1
    bad = 0
    for it in range(N_ROUNDS):
        booster = xgb.train(params, dtrain, num_boost_round=1, obj=objective, xgb_model=booster)
        dtrain.set_base_margin(base_margin)
        kl_mean = float(objective.last_kl.mean())
        if kl_mean < best_kl - 1e-4:
            best_kl = kl_mean
            best_raw = booster.save_raw(raw_format='ubj')
            best_it = it
            bad = 0
        else:
            bad += 1
        if it % 10 == 0 or it == N_ROUNDS - 1:
            print(f'round {it:4d}  train KLw mean = {kl_mean:.4f}  max = {objective.last_kl.max():.4f}  best = {best_kl:.4f} (round {best_it})', flush=True)
        if bad >= PATIENCE:
            print(f'early stopping at round {it} (best round {best_it}, KLw {best_kl:.4f})', flush=True)
            break
    if best_raw is not None:
        booster = xgb.Booster(model_file=None)
        booster.load_model(bytearray(best_raw))

    def predict_phase(conds, n_sims):
        y_pred = np.empty(n_sims * nwallp, dtype=np.float64)
        for s in range(n_sims):
            f_conds = np.tile(conds[s].astype(np.float32), (nwallp, 1))
            F = np.hstack([static, f_conds])
            dm = xgb.DMatrix(F)
            dm.set_base_margin(mean_field)
            y_pred[s * nwallp:(s + 1) * nwallp] = booster.predict(dm)
        return y_pred

    print('Predicting phase 1...', flush=True)
    y_pred1 = predict_phase(test1_conds, n_test1)
    print('Predicting phase 2...', flush=True)
    y_pred2 = predict_phase(test2_conds, n_test2)

    np.save('y_pred1_klwboost.npy', y_pred1)
    np.save('y_pred2_klwboost.npy', y_pred2)
    booster.save_model('klwboost.json')
    print('Saved: klwboost.json', flush=True)

    res1 = evaluate_phase(y_test1, y_pred1, test1_weights, n_test1, comp_masks)
    res2 = evaluate_phase(y_test2, y_pred2, test2_weights, n_test2, comp_masks)
    for label, res in [('Phase 1 (XGBoost, KLw objective)', res1), ('Phase 2 (XGBoost, KLw objective)', res2)]:
        print_phase_summary(label, res)


if __name__ == '__main__':
    main()