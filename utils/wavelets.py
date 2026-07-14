import json
import numpy as np
from scipy.stats import entropy, norm
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

nwallp = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'
KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}

N_GRID = 64
N_KEEP_PER_COMP = 200


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


def haar_forward_1d(x):
    coeffs = []
    a = x.astype(np.float64)
    while len(a) > 1:
        even, odd = a[0::2], a[1::2]
        approx = (even + odd) / np.sqrt(2)
        detail = (even - odd) / np.sqrt(2)
        coeffs.append(detail)
        a = approx
    coeffs.append(a)
    return coeffs


def haar_flatten(coeffs):
    return np.concatenate(coeffs)


def haar_unflatten(flat, n):
    levels = []
    size, idx = n, 0
    while size > 1:
        half = size // 2
        levels.append(flat[idx:idx + half])
        idx += half
        size = half
    levels.append(flat[idx:idx + 1])
    return levels


def haar_inverse_1d(coeffs):
    a = coeffs[-1]
    for detail in reversed(coeffs[:-1]):
        even = (a + detail) / np.sqrt(2)
        odd = (a - detail) / np.sqrt(2)
        out = np.empty(len(even) + len(odd))
        out[0::2] = even
        out[1::2] = odd
        a = out
    return a


def haar2d_forward(grid):
    n = grid.shape[0]
    rows = np.stack([haar_flatten(haar_forward_1d(grid[i])) for i in range(n)])
    return np.stack([haar_flatten(haar_forward_1d(rows[:, j])) for j in range(n)], axis=1)


def haar2d_inverse(flat, n):
    cols = np.stack([haar_inverse_1d(haar_unflatten(flat[:, j], n)) for j in range(n)], axis=1)
    return np.stack([haar_inverse_1d(haar_unflatten(cols[i], n)) for i in range(n)])


def pca_project_2d(coords):
    coords_c = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(coords_c, full_matrices=False)
    return coords_c @ vt[:2].T


def build_grid_mapping(uv, n_grid):
    umin, umax = uv[:, 0].min(), uv[:, 0].max()
    vmin, vmax = uv[:, 1].min(), uv[:, 1].max()
    ui = np.clip(((uv[:, 0] - umin) / (umax - umin) * n_grid).astype(int), 0, n_grid - 1)
    vi = np.clip(((uv[:, 1] - vmin) / (vmax - vmin) * n_grid).astype(int), 0, n_grid - 1)
    cell_idx = ui * n_grid + vi
    counts = np.bincount(cell_idx, minlength=n_grid * n_grid)
    filled = counts > 0
    fallback = np.where(filled)[0]
    nearest = np.searchsorted(fallback, np.arange(n_grid * n_grid))
    nearest = np.clip(nearest, 0, len(fallback) - 1)
    fill_for_empty = fallback[nearest]
    return cell_idx, counts, filled, fill_for_empty


def encode_grid(values, cell_idx, counts, filled, fill_for_empty, n_grid):
    grid_sum = np.bincount(cell_idx, weights=values, minlength=n_grid * n_grid)
    grid = np.zeros(n_grid * n_grid)
    grid[filled] = grid_sum[filled] / counts[filled]
    grid[~filled] = grid[fill_for_empty[~filled]]
    return grid.reshape(n_grid, n_grid)


def decode_grid(grid, cell_idx):
    return grid.reshape(-1)[cell_idx]


def fit_wavelet_component_model(X_train, y_train, comp_masks, train_conds_sc, nwallp,
                                 n_grid=N_GRID, n_keep=N_KEEP_PER_COMP):
    n_sims = y_train.shape[0] // nwallp
    Y_train_grid = y_train.reshape(n_sims, nwallp)
    X_ref = X_train[:nwallp]

    state = {}
    for cname, mask in comp_masks.items():
        coords = X_ref[mask][:, :3]
        uv = pca_project_2d(coords)
        cell_idx, counts, filled, fill_for_empty = build_grid_mapping(uv, n_grid)

        flat_list = []
        for i in range(n_sims):
            grid = encode_grid(Y_train_grid[i][mask], cell_idx, counts, filled, fill_for_empty, n_grid)
            flat_list.append(haar2d_forward(grid).reshape(-1))
        flat_coeffs = np.stack(flat_list)

        var = flat_coeffs.var(axis=0)
        top_idx = np.argsort(var)[-n_keep:]
        baseline = flat_coeffs.mean(axis=0)

        model = MultiOutputRegressor(
            xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, tree_method='hist'),
            n_jobs=-1,
        )
        model.fit(train_conds_sc, flat_coeffs[:, top_idx])

        state[cname] = dict(model=model, cell_idx=cell_idx, top_idx=top_idx,
                             baseline=baseline, n_grid=n_grid, mask=mask)
    return state


def predict_wavelet_component_model(state, conds_sc, n_points):
    n_sims = conds_sc.shape[0]
    y_pred = np.zeros((n_sims, n_points))
    for cname, s in state.items():
        n_grid = s['n_grid']
        pred_top = s['model'].predict(conds_sc)
        flat_pred = np.tile(s['baseline'], (n_sims, 1))
        flat_pred[:, s['top_idx']] = pred_top
        for i in range(n_sims):
            recon = haar2d_inverse(flat_pred[i].reshape(n_grid, n_grid), n_grid)
            y_pred[i][s['mask']] = decode_grid(recon, s['cell_idx'])
    return y_pred.reshape(-1)


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

    scaler = StandardScaler()
    train_conds_sc = scaler.fit_transform(train_conds)
    test1_conds_sc = scaler.transform(test1_conds)
    test2_conds_sc = scaler.transform(test2_conds)

    comp_masks = load_component_masks()
    sigma_ref = 0.01 * float(np.mean(y_train))

    state = fit_wavelet_component_model(X_train, y_train, comp_masks, train_conds_sc, nwallp)

    y_pred1 = predict_wavelet_component_model(state, test1_conds_sc, nwallp)
    y_pred2 = predict_wavelet_component_model(state, test2_conds_sc, nwallp)

    res1 = evaluate(y_test1, y_pred1, w_test1, nwallp, comp_masks, sigma_ref)
    res2 = evaluate(y_test2, y_pred2, w_test2, nwallp, comp_masks, sigma_ref)

    print_results('wavelet_component_2d', res1, res2)

    
# import json
# import numpy as np
# from scipy.stats import entropy, norm
# import xgboost as xgb
# import pywt
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.preprocessing import StandardScaler

# nwallp = 260774
# COL_MINF, COL_AOA, COL_PI = 6, 7, 8
# DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'
# KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}

# WAVELET = 'db4'
# LEVEL = 4
# N_KEEP = 500


# def load_data():
#     X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
#     y_train = np.load(DATA_DIR + 'splitv2/train_labels.npy')
#     X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
#     y_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_labels.npy')
#     X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')
#     y_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_labels.npy')
#     return X_train, y_train, X_test1, y_test1, X_test2, y_test2


# def load_component_masks():
#     component_labels = np.load(DATA_DIR + 'component_labels_unique.npy')
#     with open(DATA_DIR + 'component_map.json') as f:
#         component_map = {int(k): v for k, v in json.load(f).items()}
#     return {cname: component_labels == cid for cid, cname in component_map.items()}


# def residual_kl_weighted(y_true, y_pred, comp_masks, comp_weights, sigma_ref, n_bins=200):
#     eps = y_pred - y_true
#     sample_weight = np.zeros_like(eps)
#     for cname, mask in comp_masks.items():
#         sample_weight[mask] = comp_weights.get(cname, 0.0)

#     lim = 5.0 * sigma_ref
#     bins = np.linspace(-lim, lim, n_bins + 1)
#     dx = bins[1] - bins[0]

#     p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
#     p = np.clip(p * dx, 1e-10, None)
#     p /= p.sum()

#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
#     q = np.clip(q, 1e-10, None)
#     q /= q.sum()

#     return float(entropy(p, q))


# def compute_klw(y, yhat, weights, nwallp, comp_masks, sigma_ref):
#     vals = []
#     for l in range(len(weights)):
#         if weights[l] < 1.0:
#             continue
#         yc = y[l * nwallp:(l + 1) * nwallp]
#         yhc = yhat[l * nwallp:(l + 1) * nwallp]
#         vals.append(residual_kl_weighted(yc, yhc, comp_masks, KL_WEIGHTS, sigma_ref))
#     vals = np.array(vals)
#     return float(vals.mean()), float(vals.max())


# def compute_R2(y, yhat, w):
#     ymean = np.mean(y)
#     sse = np.sum(w * (y - yhat) ** 2)
#     ssd = np.sum(w * (y - ymean) ** 2)
#     return float(1.0 - sse / ssd)


# def compute_worst_rMAE(y, yhat, weights, nwallp):
#     vals = []
#     for l in range(len(weights)):
#         if weights[l] < 1.0:
#             continue
#         yc = y[l * nwallp:(l + 1) * nwallp]
#         yhc = yhat[l * nwallp:(l + 1) * nwallp]
#         vals.append(np.mean(np.abs(yc - yhc)) / np.mean(np.abs(yc)))
#     vals = np.array(vals)
#     return float(vals.max()), float(vals.mean())


# def evaluate(y, yhat, weights, nwallp, comp_masks, sigma_ref):
#     w_point = np.repeat(weights, nwallp)
#     r2 = compute_R2(y, yhat, w_point)
#     worst_rmae, mean_rmae = compute_worst_rMAE(y, yhat, weights, nwallp)
#     mean_klw, max_klw = compute_klw(y, yhat, weights, nwallp, comp_masks, sigma_ref)
#     return dict(r2=r2, worst_rmae=worst_rmae, mean_rmae=mean_rmae, mean_klw=mean_klw, max_klw=max_klw)


# def wavelet_encode(field, coord_order):
#     ordered = field[coord_order]
#     coeffs = pywt.wavedec(ordered, WAVELET, level=LEVEL)
#     flat, slices = pywt.coeffs_to_array(coeffs)
#     return flat, slices


# def wavelet_decode(flat, slices, coord_order):
#     coeffs = pywt.array_to_coeffs(flat, slices, output_format='wavedec')
#     ordered = pywt.waverec(coeffs, WAVELET)
#     field = np.empty(nwallp)
#     field[coord_order] = ordered[:nwallp]
#     return field


# def fit_wavelet_model(X_train, y_train, train_conds_sc):
#     coord_order = np.argsort(X_train[:nwallp, 0])
#     Y_train_grid = y_train.reshape(-1, nwallp)

#     encoded = [wavelet_encode(row, coord_order) for row in Y_train_grid]
#     flat_coeffs = np.stack([e[0] for e in encoded])
#     slices = encoded[0][1]

#     var = flat_coeffs.var(axis=0)
#     top_idx = np.argsort(var)[-N_KEEP:]
#     targets = flat_coeffs[:, top_idx]
#     baseline_coeffs = flat_coeffs.mean(axis=0)

#     model = MultiOutputRegressor(
#         xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, tree_method='hist'),
#         n_jobs=-1,
#     )
#     model.fit(train_conds_sc, targets)

#     return dict(model=model, coord_order=coord_order, slices=slices,
#                 top_idx=top_idx, baseline_coeffs=baseline_coeffs)


# def predict_field(state, conds_sc):
#     pred_top = state['model'].predict(conds_sc)
#     flat_pred = np.tile(state['baseline_coeffs'], (conds_sc.shape[0], 1))
#     flat_pred[:, state['top_idx']] = pred_top
#     return np.stack([wavelet_decode(row, state['slices'], state['coord_order']) for row in flat_pred])


# if __name__ == '__main__':
#     X_train, y_train, X_test1, y_test1, X_test2, y_test2 = load_data()

#     train_conds = X_train[::nwallp, COL_MINF:COL_PI + 1]
#     test1_conds = X_test1[::nwallp, COL_MINF:COL_PI + 1]
#     test2_conds = X_test2[::nwallp, COL_MINF:COL_PI + 1]

#     w_test1 = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
#     w_test2 = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

#     scaler = StandardScaler()
#     train_conds_sc = scaler.fit_transform(train_conds)
#     test1_conds_sc = scaler.transform(test1_conds)
#     test2_conds_sc = scaler.transform(test2_conds)

#     comp_masks = load_component_masks()
#     sigma_ref = 0.01 * float(np.mean(y_train))

#     state = fit_wavelet_model(X_train, y_train, train_conds_sc)

#     y_pred1_wav = predict_field(state, test1_conds_sc).reshape(-1)
#     y_pred2_wav = predict_field(state, test2_conds_sc).reshape(-1)

#     res1 = evaluate(y_test1, y_pred1_wav, w_test1, nwallp, comp_masks, sigma_ref)
#     res2 = evaluate(y_test2, y_pred2_wav, w_test2, nwallp, comp_masks, sigma_ref)

#     print(f'phase1  R2={res1["r2"]:.4f}  worst_rMAE={res1["worst_rmae"]:.4f}  mean_KLw={res1["mean_klw"]:.4f}')
#     print(f'phase2  R2={res2["r2"]:.4f}  worst_rMAE={res2["worst_rmae"]:.4f}  mean_KLw={res2["mean_klw"]:.4f}')