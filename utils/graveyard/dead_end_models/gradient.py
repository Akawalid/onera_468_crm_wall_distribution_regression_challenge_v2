import numpy as np
import json
import lightgbm as lgb
from scipy.stats import norm, entropy
from sklearn.neighbors import NearestNeighbors

nwallp = 260774
COL_MINF = 6
COL_AOA = 7
COL_PI = 8
DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'

K_NEIGHBORS = 8
POINTS_PER_SIM = 20000
PREDICT_CHUNK = 2000000
SEED = 0

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}

rng = np.random.default_rng(SEED)

print('Loading data...', flush=True)
X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
y_train = np.load(DATA_DIR + 'splitv2/train_labels.npy')
X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
y_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_labels.npy')
X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')
y_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_labels.npy')

train_conds = X_train[::nwallp, COL_MINF:COL_PI + 1]
test1_conds = X_test1[::nwallp, COL_MINF:COL_PI + 1]
test2_conds = X_test2[::nwallp, COL_MINF:COL_PI + 1]

n_train = X_train.shape[0] // nwallp
n_test1 = X_test1.shape[0] // nwallp
n_test2 = X_test2.shape[0] // nwallp

test1_weights = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
test2_weights = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

GLOBAL_MEAN_RHO = float(np.mean(y_train))
SIGMA_REF_GLOBAL = 0.01 * GLOBAL_MEAN_RHO

component_labels = np.load(DATA_DIR + 'component_labels_unique.npy')
with open(DATA_DIR + 'component_map.json') as f:
    component_map = {int(k): v for k, v in json.load(f).items()}

comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}


def residual_kl_normal(y_true, y_pred, sigma_ref, n_bins=200):
    eps = y_pred - y_true
    sigma_y = y_true.std() + 1e-12
    lim = 5.0 * sigma_y
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, density=True)
    p = np.clip(p * dx, 1e-10, None)
    p /= p.sum()
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()
    kl = float(entropy(p, q))
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)),
            'bias': float(np.mean(eps) / sigma_y), 'spread': float(np.std(eps) / sigma_y)}


def residual_kl_weighted(y_true, y_pred, comp_masks_arg, comp_weights, sigma_ref, n_bins=200):
    eps = y_pred - y_true
    sigma_y = y_true.std() + 1e-12
    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks_arg.items():
        sample_weight[mask] = comp_weights.get(cname, 0.0)
    lim = 5.0 * sigma_y
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
    p = np.clip(p * dx, 1e-10, None)
    p /= p.sum()
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()
    kl = float(entropy(p, q))
    bias = float(np.average(eps, weights=sample_weight)) / sigma_y
    spread = float(np.sqrt(np.average((eps - eps.mean()) ** 2, weights=sample_weight))) / sigma_y
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)), 'bias': bias, 'spread': spread}


def compute_R2(y, yhat, confidence_pointwise):
    ymean = np.mean(y)
    SSE = np.sum(confidence_pointwise * (y - yhat) ** 2)
    SSD = np.sum(confidence_pointwise * (y - ymean) ** 2)
    return float(1.0 - SSE / SSD)


def compute_worst_rMAE(y, yhat, confidence_per_case):
    rMAE_list, idx_list = [], []
    for l in range(len(confidence_per_case)):
        if confidence_per_case[l] < 1.0:
            continue
        ycase = y[l * nwallp:(l + 1) * nwallp]
        yhatcase = yhat[l * nwallp:(l + 1) * nwallp]
        rMAE_list.append(np.mean(np.abs(ycase - yhatcase)) / np.mean(np.abs(ycase)))
        idx_list.append(l)
    rMAE_arr = np.array(rMAE_list)
    iworst_local = int(np.argmax(rMAE_arr))
    return idx_list[iworst_local], float(rMAE_arr[iworst_local])


def _bootstrap_ci(values, stat_func=np.max, n_boot=1000, ci=95, rng_arg=None):
    rng_local = rng_arg or np.random.default_rng()
    n = len(values)
    idx = rng_local.integers(0, n, size=(n_boot, n))
    boot_stats = stat_func(values[idx], axis=1)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_stats, [alpha, 100 - alpha])
    return float(lo), float(hi)


def evaluate_phase(y, y_pred, weights, n_sims, nwallp_arg, sigma_ref):
    Y, Yh = y.reshape(n_sims, nwallp_arg), y_pred.reshape(n_sims, nwallp_arg)
    confidence_pointwise = np.repeat(weights, nwallp_arg)

    iworst, worst_rMAE = compute_worst_rMAE(y, y_pred, weights)
    r2_global = compute_R2(y, y_pred, confidence_pointwise)

    comp_r2 = {}
    comp_r2_persim = {c: np.full(n_sims, np.nan) for c in KL_WEIGHTS}
    comp_rMAE = {c: np.full(n_sims, np.nan) for c in KL_WEIGHTS}
    comp_kl = {c: [None] * n_sims for c in KL_WEIGHTS}
    kl_w = np.full(n_sims, np.nan)
    valid_idx = np.where(weights == 1.0)[0]

    for cname, mask in comp_masks.items():
        if cname not in KL_WEIGHTS:
            continue
        full_mask = np.tile(mask, n_sims)
        comp_r2[cname] = compute_R2(y[full_mask], y_pred[full_mask], confidence_pointwise[full_mask])

    for i in valid_idx:
        yc, yhatc = Y[i], Yh[i]
        for cname, mask in comp_masks.items():
            if cname not in KL_WEIGHTS:
                continue
            ycm, yhatcm = yc[mask], yhatc[mask]
            comp_rMAE[cname][i] = np.mean(np.abs(ycm - yhatcm)) / np.mean(np.abs(ycm))
            comp_r2_persim[cname][i] = compute_R2(ycm, yhatcm, np.ones_like(ycm))
            comp_kl[cname][i] = residual_kl_normal(ycm, yhatcm, sigma_ref)
        kl_w[i] = residual_kl_weighted(yc, yhatc, comp_masks, KL_WEIGHTS, sigma_ref)['kl']

    return dict(Y=Y, Yh=Yh, iworst=iworst, worst_rMAE=worst_rMAE, r2=r2_global, kl=kl_w,
                comp_rMAE=comp_rMAE, comp_r2=comp_r2, comp_r2_persim=comp_r2_persim, comp_kl=comp_kl)


def _print_table(title, comp_dict_pooled, comp_dict_persim, fmt, n_boot=1000, ci=95):
    print(f'\n  {title}')
    print(f'  {"component":<10}  {"pooled":>9}  {f"{ci}% CI":>19}')
    print(f'  {"─"*10}  {"─"*9}  {"─"*19}')
    for cname in KL_WEIGHTS:
        vals = comp_dict_persim[cname]
        vals = vals[~np.isnan(vals)]
        lo, hi = _bootstrap_ci(vals, np.mean, n_boot=n_boot, ci=ci)
        print(f'  {cname:<10}  {comp_dict_pooled[cname]:>9{fmt}}  [{lo:>7{fmt}}, {hi:>7{fmt}}]')


def _print_table_per_sim(title, comp_dict, fmt, n_boot=1000, ci=95):
    print(f'\n  {title}')
    print(f'  {"component":<10}  {"worst":>9}  {f"{ci}% CI":>19}')
    print(f'  {"─"*10}  {"─"*9}  {"─"*19}')
    for cname in KL_WEIGHTS:
        vals = comp_dict[cname]
        vals = vals[~np.isnan(vals)]
        worst = np.max(vals)
        lo, hi = _bootstrap_ci(vals, np.max, n_boot=n_boot, ci=ci)
        print(f'  {cname:<10}  {worst:>9{fmt}}  [{lo:>7{fmt}}, {hi:>7{fmt}}]')


def _print_kl_table(comp_kl, n_boot=1000, ci=95):
    print(f'\n  KL by component')
    print(f'  {"component":<10}  {"mean KL":>9}  {f"{ci}% CI":>19}  {"max KL":>9}  {"mean score":>10}  {"mean bias":>10}  {"mean spread":>11}')
    print(f'  {"─"*10}  {"─"*9}  {"─"*19}  {"─"*9}  {"─"*10}  {"─"*10}  {"─"*11}')
    for cname in KL_WEIGHTS:
        vals = [v for v in comp_kl[cname] if v is not None]
        kl_c, sc_c, bi_c, sp_c = (np.array([v[k] for v in vals]) for k in ('kl', 'score', 'bias', 'spread'))
        lo, hi = _bootstrap_ci(kl_c, np.mean, n_boot=n_boot, ci=ci)
        print(f'  {cname:<10}  {kl_c.mean():>9.4f}  [{lo:>7.4f}, {hi:>7.4f}]  {kl_c.max():>9.4f}  '
              f'{sc_c.mean():>10.4f}  {bi_c.mean():>10.4f}  {sp_c.mean():>11.4f}')


def print_phase_summary(label, res):
    i_mae = res['iworst']
    i_kl = int(np.nanargmax(res['kl']))
    rMAE_at_worst_kl = np.mean(np.abs(res['Y'][i_kl] - res['Yh'][i_kl])) / np.mean(np.abs(res['Y'][i_kl]))
    print(f'\n{label}')
    print(f'  worst rMAE: sim {i_mae}  rMAE={res["worst_rMAE"]:.4f}  KL={res["kl"][i_mae]:.4f}  score={1.0/(1.0+res["kl"][i_mae]):.4f}')
    print(f'  worst KL  : sim {i_kl}  KL={res["kl"][i_kl]:.4f}  score={1.0/(1.0+res["kl"][i_kl]):.4f}  rMAE={rMAE_at_worst_kl:.4f}')
    _print_table_per_sim('rMAE by component (worst + bootstrap CI)', res['comp_rMAE'], '.4f')
    _print_table('R2 by component (pooled)', res['comp_r2'], res['comp_r2_persim'], '.4f')
    _print_kl_table(res['comp_kl'])
    kl_global = res['kl'][~np.isnan(res['kl'])]
    score_global = 1.0 / (1.0 + kl_global)
    print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}', flush=True)


def compute_point_gradients(coords, values, k=K_NEIGHBORS):
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(coords)
    _, nbr_idx = nn.kneighbors(coords)
    nbr_idx = nbr_idx[:, 1:]
    grads = np.zeros((coords.shape[0], 3), dtype=np.float64)
    dP = coords[nbr_idx] - coords[:, None, :]
    dV = values[nbr_idx] - values[:, None]
    for i in range(coords.shape[0]):
        A = dP[i]
        b = dV[i]
        g, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        grads[i] = g
    return grads


def build_static_features():
    coords = X_train[:nwallp, :3].astype(np.float64)
    normals = X_train[:nwallp, 3:6].astype(np.float64)
    Y = y_train.reshape(n_train, nwallp)
    mean_field = Y.mean(axis=0)
    std_field = Y.std(axis=0)

    print('Computing spatial gradient of the mean field...', flush=True)
    grad_mean = compute_point_gradients(coords, mean_field)
    grad_mag = np.linalg.norm(grad_mean, axis=1)

    print('Computing spatial gradient of the std field...', flush=True)
    grad_std = compute_point_gradients(coords, std_field)
    grad_std_mag = np.linalg.norm(grad_std, axis=1)

    static = np.column_stack([
        coords, normals,
        mean_field, std_field,
        grad_mean, grad_mag,
        grad_std_mag,
    ]).astype(np.float32)
    return static, mean_field.astype(np.float64)


def assemble_features(static, conds, sim_indices, point_indices):
    f_static = static[point_indices]
    f_conds = conds[sim_indices].astype(np.float32)
    return np.hstack([f_static, f_conds])


def build_train_set(static, mean_field):
    rows_per_sim = min(POINTS_PER_SIM, nwallp)
    Xs, ys = [], []
    Y = y_train.reshape(n_train, nwallp)
    for s in range(n_train):
        pts = rng.choice(nwallp, size=rows_per_sim, replace=False)
        sim_idx = np.full(rows_per_sim, s)
        Xs.append(assemble_features(static, train_conds, sim_idx, pts))
        ys.append((Y[s, pts] - mean_field[pts]).astype(np.float32))
    return np.vstack(Xs), np.concatenate(ys)


def predict_phase(model, static, conds, n_sims, mean_field):
    y_pred = np.empty(n_sims * nwallp, dtype=np.float64)
    all_pts = np.arange(nwallp)
    for s in range(n_sims):
        sim_idx = np.full(nwallp, s)
        F = assemble_features(static, conds, sim_idx, all_pts)
        for start in range(0, nwallp, PREDICT_CHUNK):
            stop = min(start + PREDICT_CHUNK, nwallp)
            y_pred[s * nwallp + start:s * nwallp + stop] = (
                model.predict(F[start:stop]) + mean_field[start:stop]
            )
    return y_pred


def main():
    static, mean_field = build_static_features()

    print('Assembling training set...', flush=True)
    F_train, t_train = build_train_set(static, mean_field)
    print(f'train matrix: {F_train.shape}', flush=True)

    print('Training LightGBM...', flush=True)
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        num_leaves=255,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(F_train, t_train)

    feat_names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'mean_rho', 'std_rho',
                  'grad_x', 'grad_y', 'grad_z', 'grad_mag', 'grad_std_mag',
                  'Minf', 'AoA', 'Pi']
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    print('\nFeature importances:')
    for j in order:
        print(f'  {feat_names[j]:<14} {imp[j]}')

    print('\nPredicting phase 1...', flush=True)
    y_pred1 = predict_phase(model, static, test1_conds, n_test1, mean_field)
    print('Predicting phase 2...', flush=True)
    y_pred2 = predict_phase(model, static, test2_conds, n_test2, mean_field)

    np.save('y_pred1_gradlgbm.npy', y_pred1)
    np.save('y_pred2_gradlgbm.npy', y_pred2)

    res1 = evaluate_phase(y_test1, y_pred1, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
    res2 = evaluate_phase(y_test2, y_pred2, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)
    print_phase_summary('Phase 1 (LightGBM pointwise + gradient features)', res1)
    print_phase_summary('Phase 2 (LightGBM pointwise + gradient features)', res2)


if __name__ == '__main__':
    main()