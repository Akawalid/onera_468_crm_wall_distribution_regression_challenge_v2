import numpy as np
import json
import torch
import torch.nn as nn
import pywt
import lightgbm as lgb
from scipy.stats import norm, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

nwallp = 260774
COL_MINF = 6
COL_AOA = 7
COL_PI = 8
DATA_DIR = '/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/'
CKPT_PATH = '/home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/FILES_RHO_ALL_POINTS_reduitfloat32/mlp_klw_globalref.pt'

WAVELET = 'db4'
ENERGY_FRAC = 0.995
GRID_SIZES = {'wing': 32, 'pylon': 32, 'fuselage': 16, 'nacelle': 16}
MAX_KEEP = 150
KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}', flush=True)

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

scaler = StandardScaler()
train_conds_sc = scaler.fit_transform(train_conds)
test1_conds_sc = scaler.transform(test1_conds)
test2_conds_sc = scaler.transform(test2_conds)

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


def _bootstrap_ci(values, stat_func=np.max, n_boot=1000, ci=95, rng=None):
    rng = rng or np.random.default_rng()
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
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


class GlobalMLP(nn.Module):
    def __init__(self, n_out, hidden, dropout=0.0, mean_field=None):
        super().__init__()
        layers, d = [], 3
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.01)]
            if dropout > 0.0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)
        mf = torch.zeros(n_out) if mean_field is None else torch.tensor(mean_field, dtype=torch.float32)
        self.register_buffer('mean_field', mf)

    def forward(self, c):
        return self.mean_field + self.net(c)


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


def wav_forward(grid):
    coeffs = pywt.wavedec2(grid, WAVELET)
    arr, slices = pywt.coeffs_to_array(coeffs)
    return arr.reshape(-1), arr.shape, slices


def wav_inverse(flat, shape, slices, n_grid):
    coeffs = pywt.array_to_coeffs(flat.reshape(shape), slices, output_format='wavedec2')
    rec = pywt.waverec2(coeffs, WAVELET)
    return rec[:n_grid, :n_grid]


X_ref = X_train[:nwallp]
upper_ref = X_ref[:, 5] >= 0.0

enc_masks = {}
for cname, mask in comp_masks.items():
    if cname not in KL_WEIGHTS:
        continue
    enc_masks[cname + '_up'] = mask & upper_ref
    enc_masks[cname + '_lo'] = mask & (~upper_ref)


def build_encoders():
    encoders = {}
    for mname, mask in enc_masks.items():
        base = mname.rsplit('_', 1)[0]
        n_grid = GRID_SIZES[base]
        uv = pca_project_2d(X_ref[mask][:, :3])
        cell_idx, counts, filled, fill_for_empty = build_grid_mapping(uv, n_grid)
        encoders[mname] = dict(mask=mask, n_grid=n_grid, cell_idx=cell_idx, counts=counts,
                               filled=filled, fill_for_empty=fill_for_empty)
    return encoders


def encode_field(field_1sim, enc):
    return encode_grid(field_1sim[enc['mask']], enc['cell_idx'], enc['counts'],
                       enc['filled'], enc['fill_for_empty'], enc['n_grid'])


def reconstruction_ceiling(y, n_sims, encoders):
    Y = y.reshape(n_sims, nwallp)
    y_rec = Y.copy()
    for mname, enc in encoders.items():
        for i in range(n_sims):
            grid = encode_field(Y[i], enc)
            flat, shape, slices = wav_forward(grid)
            rec = wav_inverse(flat, shape, slices, enc['n_grid'])
            y_rec[i][enc['mask']] = decode_grid(rec, enc['cell_idx'])
    return y_rec.reshape(-1)


def fit_wavelet_residual_model(resid_train, conds_sc, encoders):
    n_sims = resid_train.shape[0] // nwallp
    R = resid_train.reshape(n_sims, nwallp)
    state = {}
    for mname, enc in encoders.items():
        flat_list = []
        for i in range(n_sims):
            flat, shape, slices = wav_forward(encode_field(R[i], enc))
            flat_list.append(flat)
        F = np.stack(flat_list)
        var = F.var(axis=0)
        order = np.argsort(var)[::-1]
        cum = np.cumsum(var[order]) / (var.sum() + 1e-12)
        k = min(int(np.searchsorted(cum, ENERGY_FRAC)) + 1, MAX_KEEP)
        top_idx = order[:k]
        baseline = F.mean(axis=0)
        model = MultiOutputRegressor(
            lgb.LGBMRegressor(n_estimatokeprs=200, num_leaves=15, learning_rate=0.05,
                              min_child_samples=5, verbose=-1),
            n_jobs=-1,
        )
        model.fit(conds_sc, F[:, top_idx])
        state[mname] = dict(enc=enc, model=model, top_idx=top_idx, baseline=baseline,
                            shape=shape, slices=slices)
        print(f'{mname}: kept {k} of {F.shape[1]} coefficients', flush=True)
    return state


def predict_wavelet_residual(state, conds_sc):
    n_sims = conds_sc.shape[0]
    resid = np.zeros((n_sims, nwallp))
    for mname, s in state.items():
        enc = s['enc']
        pred_top = s['model'].predict(conds_sc)
        flat_pred = np.tile(s['baseline'], (n_sims, 1))
        flat_pred[:, s['top_idx']] = pred_top
        for i in range(n_sims):
            rec = wav_inverse(flat_pred[i], s['shape'], s['slices'], enc['n_grid'])
            resid[i][enc['mask']] = decode_grid(rec, enc['cell_idx'])
    return resid.reshape(-1)


def main():
    encoders = build_encoders()

    print('Reconstruction ceiling (encode true fields, decode back, no model):', flush=True)
    y_ceil1 = reconstruction_ceiling(y_test1, n_test1, encoders)
    res_ceil1 = evaluate_phase(y_test1, y_ceil1, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
    print(f'  phase1 ceiling: R2={res_ceil1["r2"]:.4f}  worst_rMAE={res_ceil1["worst_rMAE"]:.4f}', flush=True)
    y_ceil2 = reconstruction_ceiling(y_test2, n_test2, encoders)
    res_ceil2 = evaluate_phase(y_test2, y_ceil2, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)
    print(f'  phase2 ceiling: R2={res_ceil2["r2"]:.4f}  worst_rMAE={res_ceil2["worst_rMAE"]:.4f}', flush=True)

    print('Loading MLP checkpoint...', flush=True)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    mlp_base = GlobalMLP(ckpt['nwallp'], ckpt['hidden'], dropout=ckpt.get('dropout', 0.0)).to(device)
    mlp_base.load_state_dict(ckpt['state_dict'])
    mlp_base.eval()

    with torch.no_grad():
        Ctr = torch.tensor(train_conds_sc, dtype=torch.float32, device=device)
        C1 = torch.tensor(test1_conds_sc, dtype=torch.float32, device=device)
        C2 = torch.tensor(test2_conds_sc, dtype=torch.float32, device=device)
        y_base_train = mlp_base(Ctr).cpu().numpy().astype(np.float64).reshape(-1)
        y_base1 = mlp_base(C1).cpu().numpy().astype(np.float64).reshape(-1)
        y_base2 = mlp_base(C2).cpu().numpy().astype(np.float64).reshape(-1)

    print('Evaluating MLP base alone...', flush=True)
    res1_base = evaluate_phase(y_test1, y_base1, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
    res2_base = evaluate_phase(y_test2, y_base2, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)
    print_phase_summary('Phase 1 (MLP base)', res1_base)
    print_phase_summary('Phase 2 (MLP base)', res2_base)

    print('Fitting wavelet residual model (LightGBM)...', flush=True)
    resid_train = y_train - y_base_train
    state = fit_wavelet_residual_model(resid_train, train_conds_sc, encoders)

    y_pred1 = y_base1 + predict_wavelet_residual(state, test1_conds_sc)
    y_pred2 = y_base2 + predict_wavelet_residual(state, test2_conds_sc)

    np.save('y_pred1_mlpwav.npy', y_pred1)
    np.save('y_pred2_mlpwav.npy', y_pred2)

    res1 = evaluate_phase(y_test1, y_pred1, test1_weights, n_test1, nwallp, SIGMA_REF_GLOBAL)
    res2 = evaluate_phase(y_test2, y_pred2, test2_weights, n_test2, nwallp, SIGMA_REF_GLOBAL)
    print_phase_summary('Phase 1 (MLP + wavelet LightGBM residual)', res1)
    print_phase_summary('Phase 2 (MLP + wavelet LightGBM residual)', res2)


if __name__ == '__main__':
    main()