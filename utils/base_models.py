import json
import numpy as np
from scipy.stats import entropy, norm
import xgboost as xgb
import lightgbm as lgb

nwallp = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'
KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}


def load_data():
    X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
    y_train = np.load(DATA_DIR + 'splitv2/train_labels.npy')
    X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
    y_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_labels.npy')
    X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')
    y_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_labels.npy')
    return X_train, y_train, X_test1, y_test1, X_test2, y_test2


def load_component_masks():
    component_labels = np.load(DATA_DIR + 'component_labels_unique.npy')
    with open(DATA_DIR + 'component_map.json') as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    return {cname: component_labels == cid for cid, cname in component_map.items()}


def residual_kl_weighted(y_true, y_pred, comp_masks, comp_weights, sigma_ref, n_bins=200):
    eps = y_pred - y_true
    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = comp_weights.get(cname, 0.0)

    lim = 5.0 * sigma_ref
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx = bins[1] - bins[0]

    p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
    p = np.clip(p * dx, 1e-10, None)
    p /= p.sum()

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()

    return float(entropy(p, q))


def compute_klw(y, yhat, weights, nwallp, comp_masks, sigma_ref):
    vals = []
    for l in range(len(weights)):
        if weights[l] < 1.0:
            continue
        yc = y[l * nwallp:(l + 1) * nwallp]
        yhc = yhat[l * nwallp:(l + 1) * nwallp]
        vals.append(residual_kl_weighted(yc, yhc, comp_masks, KL_WEIGHTS, sigma_ref))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.max())


def compute_R2(y, yhat, w):
    ymean = np.mean(y)
    sse = np.sum(w * (y - yhat) ** 2)
    ssd = np.sum(w * (y - ymean) ** 2)
    return float(1.0 - sse / ssd)


def compute_worst_rMAE(y, yhat, weights, nwallp):
    vals = []
    for l in range(len(weights)):
        if weights[l] < 1.0:
            continue
        yc = y[l * nwallp:(l + 1) * nwallp]
        yhc = yhat[l * nwallp:(l + 1) * nwallp]
        vals.append(np.mean(np.abs(yc - yhc)) / np.mean(np.abs(yc)))
    vals = np.array(vals)
    return float(vals.max()), float(vals.mean())


def evaluate(y, yhat, weights, nwallp, comp_masks, sigma_ref):
    w_point = np.repeat(weights, nwallp)
    r2 = compute_R2(y, yhat, w_point)
    worst_rmae, mean_rmae = compute_worst_rMAE(y, yhat, weights, nwallp)
    mean_klw, max_klw = compute_klw(y, yhat, weights, nwallp, comp_masks, sigma_ref)
    return dict(r2=r2, worst_rmae=worst_rmae, mean_rmae=mean_rmae, mean_klw=mean_klw, max_klw=max_klw)


MODELS = {
    'xgboost': lambda: xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        tree_method='hist', subsample=0.7, n_jobs=-1, random_state=0,
    ),
    'lightgbm': lambda: lgb.LGBMRegressor(
        n_estimators=500, num_leaves=255, max_depth=8, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.8, n_jobs=-1, random_state=0,
    ),
}


def run_model(name, X_train, y_train, X_test1, y_test1, X_test2, y_test2,
              w_test1, w_test2, comp_masks, sigma_ref,
              y_train_baseline=None, y_test1_baseline=None, y_test2_baseline=None):
    model = MODELS[name]()

    target_train = y_train if y_train_baseline is None else y_train - y_train_baseline
    model.fit(X_train, target_train)

    pred1 = model.predict(X_test1)
    pred2 = model.predict(X_test2)

    if y_test1_baseline is not None:
        pred1 = pred1 + y_test1_baseline
        pred2 = pred2 + y_test2_baseline

    res1 = evaluate(y_test1, pred1, w_test1, nwallp, comp_masks, sigma_ref)
    res2 = evaluate(y_test2, pred2, w_test2, nwallp, comp_masks, sigma_ref)
    return model, pred1, pred2, res1, res2


def print_results(name, res1, res2):
    print(f'\n{name}')
    print(f'  phase1  R2={res1["r2"]:.4f}  worst_rMAE={res1["worst_rmae"]:.4f}  mean_rMAE={res1["mean_rmae"]:.4f}  mean_KLw={res1["mean_klw"]:.4f}  max_KLw={res1["max_klw"]:.4f}')
    print(f'  phase2  R2={res2["r2"]:.4f}  worst_rMAE={res2["worst_rmae"]:.4f}  mean_rMAE={res2["mean_rmae"]:.4f}  mean_KLw={res2["mean_klw"]:.4f}  max_KLw={res2["max_klw"]:.4f}')


if __name__ == '__main__':
    X_train, y_train, X_test1, y_test1, X_test2, y_test2 = load_data()

    train_conds = X_train[::nwallp, COL_MINF:COL_PI + 1]
    test1_conds = X_test1[::nwallp, COL_MINF:COL_PI + 1]
    test2_conds = X_test2[::nwallp, COL_MINF:COL_PI + 1]

    w_test1 = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
    w_test2 = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

    comp_masks = load_component_masks()
    sigma_ref = 0.01 * float(np.mean(y_train))

    for name in MODELS:
        model, pred1, pred2, res1, res2 = run_model(
            name, X_train, y_train, X_test1, y_test1, X_test2, y_test2,
            w_test1, w_test2, comp_masks, sigma_ref
        )
        print_results(name, res1, res2)