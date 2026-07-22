import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, entropy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

nwallp           = 260774   # wall points per simulation
COL_MINF         = 6        # Mach number column
COL_AOA          = 7        # angle of attack column
COL_PI           = 8        # tunnel pressure ratio column
# DATA_DIR         = 'data/'

DATA_DIR         = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'

#=========================================================
#=========================================================
#=========================================================

print('Loading data...')
X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
y_train = np.load(DATA_DIR + 'splitv2/train_labels.npy')
X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
y_test1 = np.load(DATA_DIR+'splitv2/test_phase1_labels.npy')
X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')
y_test2 = np.load(DATA_DIR+'splitv2/test_phase2_labels.npy')

train_conds = X_train[::nwallp, COL_MINF:COL_PI+1]
test1_conds = X_test1[::nwallp, COL_MINF:COL_PI+1]
test2_conds = X_test2[::nwallp, COL_MINF:COL_PI+1]

n_train = X_train.shape[0] // nwallp
n_test1 = X_test1.shape[0] // nwallp
n_test2 = X_test2.shape[0] // nwallp

#meth1
test1_weights = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
test2_weights = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

# #meth2, what is the difference? à voir après...
# epsilon=10e-6
# confidenceScore = 1.
# if (AoA<=-10.+epsilon) or (AoA>=10.-epsilon) : confidenceScore = .5

scaler         = StandardScaler()
train_conds_sc = scaler.fit_transform(train_conds)
test1_conds_sc = scaler.transform(test1_conds)
test2_conds_sc = scaler.transform(test2_conds)

# print('Training KNN...')
# knn = KNeighborsRegressor(n_neighbors=5, algorithm='auto', n_jobs=-1)
# knn.fit(train_conds_sc, y_train.reshape(n_train, nwallp))

# print('Predicting phase 1...')
# y_pred1 = knn.predict(test1_conds_sc).reshape(-1)
# print('Predicting phase 2...')
# y_pred2 = knn.predict(test2_conds_sc).reshape(-1)

GLOBAL_MEAN_RHO = float(np.mean(y_train))
SIGMA_REF_GLOBAL = 0.01 * GLOBAL_MEAN_RHO

#=========================================================
#=========================================================
#=========================================================


component_labels = np.load(DATA_DIR + 'component_labels_unique.npy')
with open(DATA_DIR + 'component_map.json') as f:
    component_map = {int(k): v for k, v in json.load(f).items()}

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}


#=========================================================
#=========================================================
#=========================================================

def residual_kl_normal(y_true, y_pred, sigma_ref, n_bins=200):
    """
    KL(p_eps || N(0, sigma_ref)) where sigma_ref = sigma_ref_frac * sigma_y.
    Returns KL, score=1/(1+KL), normalised bias and spread.
    """
    eps       = y_pred - y_true
    sigma_y   = y_true.std() + 1e-12

    lim  = 5.0 * sigma_y
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx   = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, density=True)
    p    = np.clip(p * dx, 1e-10, None)
    p   /= p.sum()

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q    = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q    = np.clip(q, 1e-10, None)
    q   /= q.sum()

    kl    = float(entropy(p, q))
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)),
            'bias': float(np.mean(eps) / sigma_y), 'spread': float(np.std(eps) / sigma_y)}

def residual_kl_weighted(y_true, y_pred, comp_masks, comp_weights, sigma_ref, n_bins=200):
    """
    KL(p_eps || N(0, sigma_ref)) on residuals pooled across components,
    with each point weighted by its component weight (so wing/pylon points
    count more than fuselage/nacelle points in the SAME histogram).
    Single shared sigma_y -> comparable across all simulations.
    """
    eps      = y_pred - y_true
    sigma_y  = y_true.std() + 1e-12
    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = comp_weights.get(cname, 0.0)

    lim  = 5.0 * sigma_y
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
    bias   = float(np.average(eps, weights=sample_weight)) / sigma_y
    spread = float(np.sqrt(np.average((eps - eps.mean())**2, weights=sample_weight))) / sigma_y
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)), 'bias': bias, 'spread': spread}

def compute_R2(y, yhat, confidence_pointwise):
    """Confidence-weighted R^2 score."""
    ymean = np.mean(y)
    SSE   = np.sum(confidence_pointwise * (y - yhat) ** 2)    # weighted residual sum of squares
    SSD   = np.sum(confidence_pointwise * (y - ymean) ** 2)   # weighted total sum of squares
    return float(1.0 - SSE / SSD)


def compute_worst_rMAE(y, yhat, confidence_per_case):
    """
    Compute the per-case relative MAE (rMAE) on high-confidence cases only,
    then return the index and value of the worst (max) one.
    """
    rMAE_list, idx_list = [], []
    for l in range(len(confidence_per_case)):
        if confidence_per_case[l] < 1.0:
            continue  # skip low-confidence cases entirely
        ycase    = y[l * nwallp:(l + 1) * nwallp]
        yhatcase = yhat[l * nwallp:(l + 1) * nwallp]
        rMAE_list.append(np.mean(np.abs(ycase - yhatcase)) / np.mean(np.abs(ycase)))
        idx_list.append(l)
    rMAE_arr    = np.array(rMAE_list)
    iworst_local = int(np.argmax(rMAE_arr))  # worst = highest rMAE
    return idx_list[iworst_local], float(rMAE_arr[iworst_local])

#=========================================================
#=========================================================
#=========================================================

# ---------------------------------------------------------------------------
# Bootstrap confidence interval, resampled directly from existing predictions
# (no re-prediction, so this stays cheap)
# ---------------------------------------------------------------------------

def _bootstrap_ci(values, stat_func=np.max, n_boot=1000, ci=95, rng=None):
    """Percentile bootstrap CI for stat_func over an existing 1-D array of values."""
    rng = rng or np.random.default_rng()
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))          # n_boot resamples of size n
    boot_stats = stat_func(values[idx], axis=1)          # stat computed per resample
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_stats, [alpha, 100 - alpha])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-phase evaluation: pooled R2 / worst rMAE, plus per-component rMAE and KL
# ---------------------------------------------------------------------------

def     evaluate_phase(y, y_pred, weights, n_sims, nwallp, sigma_ref):
    """Pooled worst-rMAE/R2 (challenge metrics) plus per-simulation component KL and R2."""
    Y, Yh = y.reshape(n_sims, nwallp), y_pred.reshape(n_sims, nwallp)
    confidence_pointwise = np.repeat(weights, nwallp)

    iworst, worst_rMAE = compute_worst_rMAE(y, y_pred, weights)
    r2_global = compute_R2(y, y_pred, confidence_pointwise)

    comp_r2         = {}                                             # pooled R2 per component
    comp_r2_persim  = {c: np.full(n_sims, np.nan) for c in KL_WEIGHTS}  # per-sim R2, for bootstrap
    comp_rMAE = {c: np.full(n_sims, np.nan) for c in KL_WEIGHTS}
    comp_kl   = {c: [None] * n_sims for c in KL_WEIGHTS}
    kl_w      = np.full(n_sims, np.nan)
    valid_idx = np.where(weights == 1.0)[0]

    # pooled R2 per component, across all simulations at once
    for cname, mask in comp_masks.items():
        if cname not in KL_WEIGHTS:
            continue
        full_mask = np.tile(mask, n_sims)
        comp_r2[cname] = compute_R2(y[full_mask], y_pred[full_mask], confidence_pointwise[full_mask])

    # per-simulation, per-component rMAE, R2 and KL (only for valid/high-confidence sims)
    for i in valid_idx:
        yc, yhatc = Y[i], Yh[i]
        for cname, mask in comp_masks.items():
            if cname not in KL_WEIGHTS:
                continue
            ycm, yhatcm = yc[mask], yhatc[mask]
            comp_rMAE[cname][i]      = np.mean(np.abs(ycm - yhatcm)) / np.mean(np.abs(ycm))
            comp_r2_persim[cname][i] = compute_R2(ycm, yhatcm, np.ones_like(ycm))  # unweighted, single sim
            comp_kl[cname][i]        = residual_kl_normal(ycm, yhatcm, sigma_ref)
        kl_w[i] = residual_kl_weighted(yc, yhatc, comp_masks, KL_WEIGHTS, sigma_ref)['kl']

    return dict(Y=Y, Yh=Yh, iworst=iworst, worst_rMAE=worst_rMAE, r2=r2_global, kl=kl_w,
                comp_rMAE=comp_rMAE, comp_r2=comp_r2, comp_r2_persim=comp_r2_persim, comp_kl=comp_kl)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_table(title, comp_dict_pooled, comp_dict_persim, fmt, n_boot=1000, ci=95):
    """Print pooled value per component plus a bootstrap CI from per-sim values (e.g. R2)."""
    print(f'\n  {title}')
    print(f'  {"component":<10}  {"pooled":>9}  {f"{ci}% CI":>19}')
    print(f'  {"─"*10}  {"─"*9}  {"─"*19}')
    for cname in KL_WEIGHTS:
        vals = comp_dict_persim[cname]
        vals = vals[~np.isnan(vals)]           # drop low-confidence / skipped sims
        lo, hi = _bootstrap_ci(vals, np.mean, n_boot=n_boot, ci=ci)
        print(f'  {cname:<10}  {comp_dict_pooled[cname]:>9{fmt}}  [{lo:>7{fmt}}, {hi:>7{fmt}}]')


def _print_table_per_sim(title, comp_dict, fmt, n_boot=1000, ci=95):
    """
    Print the worst (max) per-simulation value per component, with a
    bootstrap confidence interval around that worst value.
    """
    print(f'\n  {title}')
    print(f'  {"component":<10}  {"worst":>9}  {f"{ci}% CI":>19}')
    print(f'  {"─"*10}  {"─"*9}  {"─"*19}')
    for cname in KL_WEIGHTS:
        vals = comp_dict[cname]
        vals = vals[~np.isnan(vals)]           # drop low-confidence sims (left as NaN)
        worst  = np.max(vals)
        lo, hi = _bootstrap_ci(vals, np.max, n_boot=n_boot, ci=ci)
        print(f'  {cname:<10}  {worst:>9{fmt}}  [{lo:>7{fmt}}, {hi:>7{fmt}}]')


def _print_kl_table(comp_kl, n_boot=1000, ci=95):
    """Print mean/max KL, mean score, mean bias, mean spread per component, with a CI on mean KL."""
    print(f'\n  KL by component')
    print(f'  {"component":<10}  {"mean KL":>9}  {f"{ci}% CI":>19}  {"max KL":>9}  {"mean score":>10}  {"mean bias":>10}  {"mean spread":>11}')
    print(f'  {"─"*10}  {"─"*9}  {"─"*19}  {"─"*9}  {"─"*10}  {"─"*10}  {"─"*11}')
    for cname in KL_WEIGHTS:
        vals = [v for v in comp_kl[cname] if v is not None]  # drop skipped (low-confidence) sims
        kl_c, sc_c, bi_c, sp_c = (np.array([v[k] for v in vals]) for k in ('kl', 'score', 'bias', 'spread'))
        lo, hi = _bootstrap_ci(kl_c, np.mean, n_boot=n_boot, ci=ci)
        print(f'  {cname:<10}  {kl_c.mean():>9.4f}  [{lo:>7.4f}, {hi:>7.4f}]  {kl_c.max():>9.4f}  '
              f'{sc_c.mean():>10.4f}  {bi_c.mean():>10.4f}  {sp_c.mean():>11.4f}')


def print_phase_summary(label, res):
    """Print the full summary block for one phase (worst-case + component tables)."""
    i_mae = res['iworst']                       # sim index with the globally worst rMAE
    i_kl  = int(np.nanargmax(res['kl']))         # sim index with the worst weighted KL

    # single-value rMAE for the worst-KL sim (not the aggregated worst_rMAE)
    rMAE_at_worst_kl = np.mean(np.abs(res['Y'][i_kl] - res['Yh'][i_kl])) / np.mean(np.abs(res['Y'][i_kl]))

    print(f'\n{label}')
    print(f'  worst rMAE: sim {i_mae}  rMAE={res["worst_rMAE"]:.4f}  KL={res["kl"][i_mae]:.4f}  score={1.0/(1.0+res["kl"][i_mae]):.4f}')
    print(f'  worst KL  : sim {i_kl}  KL={res["kl"][i_kl]:.4f}  score={1.0/(1.0+res["kl"][i_kl]):.4f}  rMAE={rMAE_at_worst_kl:.4f}')

    _print_table_per_sim('rMAE by component (worst + bootstrap CI)', res['comp_rMAE'], '.4f')
    _print_table('R2 by component (pooled)', res['comp_r2'], res['comp_r2_persim'], '.4f')
    _print_kl_table(res['comp_kl'])

    # drop sims where KL is NaN (skipped low-confidence sims) before aggregating
    kl_global    = res['kl'][~np.isnan(res['kl'])]
    score_global = 1.0 / (1.0 + kl_global)
    print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')


#=========================================================
#=========================================================
#=========================================================


# res1 = evaluate_phase(y_test1, y_pred1, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
# res2 = evaluate_phase(y_test2, y_pred2, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)

# for label, res in [('Phase 1', res1), ('Phase 2', res2)]:
#     print_phase_summary(label, res)

# import os
# import lightgbm as lgb

# N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

from sklearn.preprocessing import StandardScaler

# scale all 9 input columns (coords + Minf/AoA/Pi), fit on train only
scaler_pointwise = StandardScaler()
X_train_sc = scaler_pointwise.fit_transform(X_train)
X_test1_sc = scaler_pointwise.transform(X_test1)
X_test2_sc = scaler_pointwise.transform(X_test2)

# print('Training LightGBM (pointwise)...')
# lgb_model = lgb.LGBMRegressor(
#     n_estimators=500,
#     num_leaves=255,
#     max_depth=-1,
#     learning_rate=0.05,
#     min_child_samples=50,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     n_jobs=N_JOBS,
#     random_state=0,
#     verbose=-1,
# )
# lgb_model.fit(X_train_sc, y_train)

# print('Predicting phase 1...')
# y_pred1_lgb = lgb_model.predict(X_test1_sc)
# print('Predicting phase 2...')
# y_pred2_lgb = lgb_model.predict(X_test2_sc)

# res1_lgb = evaluate_phase(y_test1, y_pred1_lgb, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
# res2_lgb = evaluate_phase(y_test2, y_pred2_lgb, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)

# for label, res in [('Phase 1 (LightGBM pointwise)', res1_lgb), ('Phase 2 (LightGBM pointwise)', res2_lgb)]:
#     print(f'\n{label}')
#     _print_kl_table(res['comp_kl'])
#     kl_global    = res['kl'][~np.isnan(res['kl'])]
#     score_global = 1.0 / (1.0 + kl_global)
#     print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')
    
    
# import os
# from catboost import CatBoostRegressor

# N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

# print('Training CatBoost (pointwise)...')
# cat_model = CatBoostRegressor(
#     iterations=500,
#     depth=8,
#     learning_rate=0.05,
#     loss_function='RMSE',
#     thread_count=N_JOBS,
#     random_seed=0,
#     verbose=100,
# )
# cat_model.fit(X_train_sc, y_train)

# print('Predicting phase 1...')
# y_pred1_cat = cat_model.predict(X_test1_sc)
# print('Predicting phase 2...')
# y_pred2_cat = cat_model.predict(X_test2_sc)

# res1_cat = evaluate_phase(y_test1, y_pred1_cat, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
# res2_cat = evaluate_phase(y_test2, y_pred2_cat, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)

# for label, res in [('Phase 1 (CatBoost pointwise)', res1_cat), ('Phase 2 (CatBoost pointwise)', res2_cat)]:
#     print(f'\n{label}')
#     _print_kl_table(res['comp_kl'])
#     kl_global    = res['kl'][~np.isnan(res['kl'])]
#     score_global = 1.0 / (1.0 + kl_global)
#     print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')
    
    
#     from sklearn.tree import DecisionTreeRegressor

# print('Training DecisionTree (pointwise)...')
# dt_model = DecisionTreeRegressor(
#     max_depth=None,
#     min_samples_leaf=20,
#     random_state=0,
# )
# dt_model.fit(X_train_sc, y_train)

# print('Predicting phase 1...')
# y_pred1_dt = dt_model.predict(X_test1_sc)
# print('Predicting phase 2...')
# y_pred2_dt = dt_model.predict(X_test2_sc)

# res1_dt = evaluate_phase(y_test1, y_pred1_dt, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
# res2_dt = evaluate_phase(y_test2, y_pred2_dt, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)

# for label, res in [('Phase 1 (DecisionTree pointwise)', res1_dt), ('Phase 2 (DecisionTree pointwise)', res2_dt)]:
#     print(f'\n{label}')
#     _print_kl_table(res['comp_kl'])
#     kl_global    = res['kl'][~np.isnan(res['kl'])]
#     score_global = 1.0 / (1.0 + kl_global)
#     print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')
    
#     import os
# from sklearn.ensemble import RandomForestRegressor

# N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

# print('Training RandomForest (pointwise)...')
# rf_model = RandomForestRegressor(
#     n_estimators=100,
#     max_depth=None,
#     min_samples_leaf=10,
#     max_features=0.5,
#     n_jobs=N_JOBS,
#     random_state=0,
# )
# rf_model.fit(X_train_sc, y_train)

# print('Predicting phase 1...')
# y_pred1_rf = rf_model.predict(X_test1_sc)
# print('Predicting phase 2...')
# y_pred2_rf = rf_model.predict(X_test2_sc)

# res1_rf = evaluate_phase(y_test1, y_pred1_rf, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
# res2_rf = evaluate_phase(y_test2, y_pred2_rf, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)

# for label, res in [('Phase 1 (RandomForest pointwise)', res1_rf), ('Phase 2 (RandomForest pointwise)', res2_rf)]:
#     print(f'\n{label}')
#     _print_kl_table(res['comp_kl'])
#     kl_global    = res['kl'][~np.isnan(res['kl'])]
#     score_global = 1.0 / (1.0 + kl_global)
#     print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')
    
import os
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor

N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

Y_train_full = y_train.reshape(n_train, nwallp)

print('Training ExtraTrees (full field)...')
et_model = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=5,
    max_features=1.0,
    n_jobs=N_JOBS,
    random_state=0,
)
et_model.fit(train_conds_sc, Y_train_full)

print('Predicting phase 1...')
y_pred1_et = et_model.predict(test1_conds_sc).reshape(-1)
print('Predicting phase 2...')
y_pred2_et = et_model.predict(test2_conds_sc).reshape(-1)

res1_et = evaluate_phase(y_test1, y_pred1_et, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
res2_et = evaluate_phase(y_test2, y_pred2_et, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)

for label, res in [('Phase 1 (ExtraTrees full field)', res1_et), ('Phase 2 (ExtraTrees full field)', res2_et)]:
    print(f'\n{label}')
    _print_kl_table(res['comp_kl'])
    kl_global = res['kl'][~np.isnan(res['kl'])]
    score_global = 1.0 / (1.0 + kl_global)
    print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')


N_POD = min(100, n_train - 1)

print('Training HistGradientBoosting (full field via POD)...')
pod = PCA(n_components=N_POD, random_state=0)
Z_train = pod.fit_transform(Y_train_full)
print(f'POD: {N_POD} modes, explained variance = {pod.explained_variance_ratio_.sum():.5f}')

hgb_model = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=None,
        max_leaf_nodes=15,
        learning_rate=0.05,
        min_samples_leaf=5,
        early_stopping=False,
        random_state=0,
    ),
    n_jobs=N_JOBS,
)
hgb_model.fit(train_conds_sc, Z_train)

print('Predicting phase 1...')
y_pred1_hgb = pod.inverse_transform(hgb_model.predict(test1_conds_sc)).reshape(-1)
print('Predicting phase 2...')
y_pred2_hgb = pod.inverse_transform(hgb_model.predict(test2_conds_sc)).reshape(-1)

res1_hgb = evaluate_phase(y_test1, y_pred1_hgb, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
res2_hgb = evaluate_phase(y_test2, y_pred2_hgb, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)

for label, res in [('Phase 1 (HistGB full field, POD)', res1_hgb), ('Phase 2 (HistGB full field, POD)', res2_hgb)]:
    print(f'\n{label}')
    _print_kl_table(res['comp_kl'])
    kl_global = res['kl'][~np.isnan(res['kl'])]
    score_global = 1.0 / (1.0 + kl_global)
    print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')
    
def weighted_pca_axes(coords, w):
    wsum = w.sum() + 1e-12
    mu = (coords * w[:, None]).sum(axis=0) / wsum
    Xc = coords - mu
    C = (Xc * w[:, None]).T @ Xc / wsum
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return mu, vecs[:, order]

def error_pca_uv(Xs, ys, yh, comp_name, gamma=2.0):
    mask = comp_masks[comp_name]
    coords = Xs[mask][:, :3]
    yt, yp = ys[mask], yh[mask]
    err = np.abs(yp - yt)
    w = err ** gamma
    mu, axes3 = weighted_pca_axes(coords, w)
    uv = (coords - mu) @ axes3[:, :2]
    return uv, yt, yp, err
    
def auto_slice_positions(uv, err, n_slices=3, min_separation=None, edge_frac=0.05):
    u = uv[:, 0]
    n_bins = 100
    edges = np.linspace(u.min(), u.max(), n_bins + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, n_bins - 1)
    err_per_bin = np.bincount(idx, weights=err, minlength=n_bins)
    cnt_per_bin = np.bincount(idx, minlength=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    lo, hi = np.quantile(u, [edge_frac, 1.0 - edge_frac])
    err_per_bin[(centers < lo) | (centers > hi)] = 0.0
    err_per_bin[cnt_per_bin < np.median(cnt_per_bin[cnt_per_bin > 0]) * 0.3] = 0.0

    if min_separation is None:
        min_separation = (u.max() - u.min()) / 10

    positions = []
    remaining = err_per_bin.copy()
    for _ in range(n_slices):
        if remaining.max() <= 0:
            break
        b = int(np.argmax(remaining))
        positions.append(centers[b])
        remaining[np.abs(centers - centers[b]) < min_separation] = 0.0
    return sorted(positions)

def plot_slice_along_v(isim, X_test, y_test, y_pred, comp_name, u0, half_width,
                       phase_label, model_name='model', gamma=2.0,
                       uv_cache=None, n_bins_v=150):
    Xs = X_test[isim * nwallp:(isim + 1) * nwallp]
    ys = y_test[isim * nwallp:(isim + 1) * nwallp]
    yh = y_pred[isim * nwallp:(isim + 1) * nwallp]
    Minf, AoA, Pi = Xs[0, COL_MINF], Xs[0, COL_AOA], Xs[0, COL_PI]

    if uv_cache is None:
        uv, yt, yp, err = error_pca_uv(Xs, ys, yh, comp_name, gamma=gamma)
    else:
        uv, yt, yp, err = uv_cache

    mask_comp = comp_masks[comp_name]
    upper = (Xs[:, 5] >= 0.0)[mask_comp]
    sel = np.abs(uv[:, 0] - u0) < half_width
    if sel.sum() < 20:
        print(f'coupe u0={u0:.2f}: seulement {sel.sum()} points, ignoree')
        return

    def _median_curve(v, vals, n_bins):
        edges = np.linspace(v.min(), v.max(), n_bins + 1)
        idx = np.clip(np.digitize(v, edges) - 1, 0, n_bins - 1)
        centers, meds = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() > 0:
                centers.append(0.5 * (edges[b] + edges[b + 1]))
                meds.append(np.median(vals[m]))
        return np.array(centers), np.array(meds)

    plt.figure(figsize=(9, 6), dpi=120)
    for side_mask, ls, side_label in [(upper, '-', 'extrados'),
                                      (~upper, '--', 'intrados')]:
        s = sel & side_mask
        if s.sum() < 10:
            continue
        vv = uv[s, 1]
        vc, rt = _median_curve(vv, yt[s], n_bins_v)
        _, rp = _median_curve(vv, yp[s], n_bins_v)
        plt.plot(vc, rt, color='red', ls=ls, lw=1.5,
                 label=f'rho vraie ({side_label})')
        plt.plot(vc, rp, color='green', ls=ls, lw=1.5,
                 label=f'rho {model_name} ({side_label})')

    plt.xlabel('v (direction traversant la structure d erreur)')
    plt.ylabel('rho / rho_inf')
    plt.title(f'Minf={Minf:.2f} AoA={AoA:.1f} Pi={Pi:.0e}  coupe u={u0:.2f}  '
              f'-- CFD et {model_name}', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.show()
    
def analyse_coupes_auto(isim, X_test, y_test, y_pred, comp_name, phase_label,
                        model_name='model', n_slices=3, gamma=2.0):
    Xs = X_test[isim * nwallp:(isim + 1) * nwallp]
    ys = y_test[isim * nwallp:(isim + 1) * nwallp]
    yh = y_pred[isim * nwallp:(isim + 1) * nwallp]
    uv, yt, yp, err = error_pca_uv(Xs, ys, yh, comp_name, gamma=gamma)

    span = uv[:, 0].max() - uv[:, 0].min()
    half_width = span / 200

    positions = auto_slice_positions(uv, err, n_slices=n_slices)
    print(f'coupes placees a u = {[f"{p:.2f}" for p in positions]}')

    for u0 in positions:
        plot_slice_along_v(isim, X_test, y_test, y_pred, comp_name, u0, half_width,
                           phase_label, model_name=model_name, gamma=gamma,
                           uv_cache=(uv, yt, yp, err))

models = {
    # 'lgb': (y_pred2_lgb, res2_lgb),
    # 'cat': (y_pred2_cat, res2_cat),
    'hgb': (y_pred2_hgb, res2_hgb),
    # 'dt':  (y_pred2_dt,  res2_dt),
    'et':  (y_pred2_et,  res2_et),
    #'rf':  (y_pred2_rf,  res2_rf),
}


_orig_show = plt.show

for name, (yp2, res) in models.items():
    isim_worst = int(np.nanargmax(np.where(test2_weights == 1.0, res['kl'], np.nan)))
    print(f'\n=== {name}: pire sim KL phase 2 = {isim_worst} ===')

    counter = {'n': 0}

    def _save_show(*args, name=name, isim=isim_worst, counter=counter, **kwargs):
        counter['n'] += 1
        fname = f'coupes_{name}_sim{isim}_coupe{counter["n"]}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  saved: {fname}')

    plt.show = _save_show
    try:
        analyse_coupes_auto(isim_worst, X_test2, y_test2, yp2, 'wing',
                            'Phase 2', model_name=name, n_slices=3)
    finally:
        plt.show = _orig_show