import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_pod_gp as pg

NWALLP = pg.NWALLP
COL_X, COL_Y, COL_Z, COL_NX, COL_NY, COL_NZ = 0, 1, 2, 3, 4, 5
COL_MINF, COL_AOA, COL_PI = pg.COL_MINF, pg.COL_AOA, pg.COL_PI

DATA_DIR = pg.DATA_DIR
SPLIT_DIR = pg.SPLIT_DIR

N_STATIONS = 26
N_XI_BINS = 100
MIN_STATION_NODES = 300
SHOCK_GRAD_FRAC = 0.15
STATION_DETECT_RATE_MIN = 0.10
GLOBAL_DETECT_RATE_WARN = 0.8
SHOCK_XI_CLIP = (0.03, 0.97)
SHOCK_SEARCH_XI_MIN = 0.20
SHOCK_SEARCH_XI_MAX = 0.85

VARIANCE_THRESHOLD = pg.VARIANCE_THRESHOLD
MAX_COMPONENTS = pg.MAX_COMPONENTS
KERNEL_NU = pg.KERNEL_NU
ALPHA = pg.ALPHA
N_RESTARTS = pg.N_RESTARTS
N_JOBS = pg.N_JOBS
SEED = pg.SEED

MAX_COMPONENTS_CV = 60
N_RESTARTS_CV = 2
N_RANDOM_FOLDS = 5
N_MACH_HOLDOUT = 2

OUT_NAME = 'shifted_pod_gp'


def inspect_columns(X0):
    print('Inspecting columns 0..5 of train_data.npy on the first simulation block', flush=True)
    labels = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for c, label in enumerate(labels):
        col = X0[:, c]
        print(f'  col {c} ({label}) min={col.min():.4f} max={col.max():.4f} '
              f'mean={col.mean():.4f} std={col.std():.4f}', flush=True)
    norm_sq = X0[:, COL_NX] ** 2 + X0[:, COL_NY] ** 2 + X0[:, COL_NZ] ** 2
    print(f'  cols 3,4,5 unit norm check min={norm_sq.min():.6f} max={norm_sq.max():.6f}', flush=True)
    print('  conclusion cols 0,1,2 are x,y,z surface coordinates, cols 3,4,5 are unit wall normals', flush=True)


def load_wing_mask(data_dir, split_dir):
    base = os.path.join(data_dir, split_dir)
    comp_labels = np.load(os.path.join(base, 'component_labels_unique.npy'))
    with open(os.path.join(base, 'component_map.json')) as f:
        comp_map = {int(k): v for k, v in json.load(f).items()}
    name_to_id = {v: k for k, v in comp_map.items()}
    return comp_labels == name_to_id['wing']


def build_wing_parametrization(X0, wing_mask):
    xw = X0[wing_mask, COL_X]
    yw = X0[wing_mask, COL_Y]
    nzw = X0[wing_mask, COL_NZ]
    n_wing = wing_mask.sum()

    y_edges = np.linspace(yw.min(), yw.max(), N_STATIONS + 1)
    eta_of_station = 0.5 * (y_edges[:-1] + y_edges[1:])
    station_idx = np.clip(np.digitize(yw, y_edges) - 1, 0, N_STATIONS - 1)
    upper = nzw >= 0.0

    xi = np.full(n_wing, np.nan)
    station_ok = np.zeros(N_STATIONS, dtype=bool)

    print(f'Wing has {n_wing} points, y in [{yw.min():.3f},{yw.max():.3f}], '
          f'{N_STATIONS} spanwise stations', flush=True)
    for s in range(N_STATIONS):
        total_n = int((station_idx == s).sum())
        station_ok[s] = total_n >= MIN_STATION_NODES
        for surf_name, surf_mask in (('upper', upper), ('lower', ~upper)):
            sel = (station_idx == s) & surf_mask
            n = int(sel.sum())
            if n > 0:
                xs = xw[sel]
                xmin, xmax = xs.min(), xs.max()
                if xmax > xmin:
                    xi[sel] = (xs - xmin) / (xmax - xmin)
            print(f'  station {s:2d} y=[{y_edges[s]:.2f},{y_edges[s+1]:.2f}] {surf_name} n={n}', flush=True)
        if not station_ok[s]:
            print(f'  station {s} flagged low node count {total_n} < MIN_STATION_NODES={MIN_STATION_NODES}',
                  flush=True)

    return station_idx, upper, xi, eta_of_station, station_ok


def detect_shock(xi_pts, rho_pts):
    valid = np.isfinite(xi_pts)
    xi_pts = xi_pts[valid]
    rho_pts = rho_pts[valid]
    if xi_pts.size < 10:
        return False, np.nan, 0.0, 0.0

    bin_edges = np.linspace(0.0, 1.0, N_XI_BINS + 1)
    bin_idx = np.clip(np.digitize(xi_pts, bin_edges) - 1, 0, N_XI_BINS - 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    rho_binned = np.full(N_XI_BINS, np.nan)
    for b in range(N_XI_BINS):
        sel = bin_idx == b
        if sel.any():
            rho_binned[b] = np.median(rho_pts[sel])

    valid_bins = np.flatnonzero(np.isfinite(rho_binned))
    if valid_bins.size < 5:
        return False, np.nan, 0.0, 0.0

    rho_valid = rho_binned[valid_bins]
    centers_valid = bin_centers[valid_bins]
    rho_range = float(rho_valid.max() - rho_valid.min())
    if rho_range <= 0.0:
        return False, np.nan, 0.0, rho_range

    diffs = np.abs(np.diff(rho_valid))
    mids = 0.5 * (centers_valid[:-1] + centers_valid[1:])
    search_ok = (mids >= SHOCK_SEARCH_XI_MIN) & (mids <= SHOCK_SEARCH_XI_MAX)
    if not search_ok.any():
        return False, np.nan, 0.0, rho_range

    diffs_search = np.where(search_ok, diffs, -1.0)
    peak_local = int(np.argmax(diffs_search))
    peak_jump = float(diffs[peak_local])
    shock_xi = float(mids[peak_local])

    detected = peak_jump >= SHOCK_GRAD_FRAC * rho_range
    return detected, shock_xi if detected else np.nan, peak_jump, rho_range


def detect_all_shocks(Y_train, wing_mask, station_idx, upper, xi):
    n_train = Y_train.shape[0]
    Y_wing = Y_train[:, wing_mask]

    detected = np.zeros((n_train, N_STATIONS), dtype=bool)
    shock_xi = np.full((n_train, N_STATIONS), np.nan)

    for i in range(n_train):
        rho_wing = Y_wing[i]
        for s in range(N_STATIONS):
            sel = (station_idx == s) & upper
            det, sxi, _, _ = detect_shock(xi[sel], rho_wing[sel])
            detected[i, s] = det
            shock_xi[i, s] = sxi

    return detected, shock_xi


def fit_shift_model(train_conds, eta_of_station, detected, shock_xi):
    n_train, n_stations = detected.shape
    rows = []
    targets = []
    for i in range(n_train):
        for s in range(n_stations):
            if detected[i, s]:
                rows.append([eta_of_station[s], train_conds[i, 0], train_conds[i, 1], train_conds[i, 2]])
                targets.append(shock_xi[i, s])
    rows = np.array(rows)
    targets = np.array(targets)

    feat_scaler = StandardScaler()
    rows_sc = feat_scaler.fit_transform(rows)

    linear = LinearRegression()
    linear.fit(rows_sc, targets)
    residual = targets - linear.predict(rows_sc)

    rbf = RBFInterpolator(rows_sc, residual, kernel='thin_plate_spline', smoothing=1e-4)

    print(f'Shift model fit on {len(targets)} detected (snapshot,station) pairs, '
          f'linear coef (eta,Minf,AoA,Pi) scaled = {linear.coef_}', flush=True)

    def predict_shift(eta, minf, aoa, pi):
        feats = np.column_stack([
            np.full_like(np.atleast_1d(minf), eta, dtype=np.float64),
            np.atleast_1d(minf), np.atleast_1d(aoa), np.atleast_1d(pi),
        ])
        feats_sc = feat_scaler.transform(feats)
        pred = linear.predict(feats_sc) + rbf(feats_sc)
        return np.clip(pred, SHOCK_XI_CLIP[0], SHOCK_XI_CLIP[1])

    return predict_shift


def warp_xi(xi_vals, shock_xi_used, xi_ref):
    shock_xi_used = np.clip(shock_xi_used, SHOCK_XI_CLIP[0], SHOCK_XI_CLIP[1])
    pre = xi_vals < shock_xi_used
    warped = np.empty_like(xi_vals)
    warped[pre] = xi_vals[pre] * (xi_ref / shock_xi_used)
    warped[~pre] = xi_ref + (xi_vals[~pre] - shock_xi_used) * ((1.0 - xi_ref) / (1.0 - shock_xi_used))
    return warped


class AlignedRepresentation:
    def __init__(self, wing_mask, station_idx, upper, xi, eta_of_station, warp_stations, xi_ref):
        self.wing_mask = wing_mask
        self.station_idx = station_idx
        self.upper = upper
        self.xi = xi
        self.eta_of_station = eta_of_station
        self.warp_stations = warp_stations
        self.xi_ref = xi_ref
        self.xi_grid = np.linspace(0.0, 1.0, N_XI_BINS)

        wing_idx_global = np.flatnonzero(wing_mask)
        warp_local_mask = upper & np.isin(station_idx, warp_stations)
        self.warp_global_idx = wing_idx_global[warp_local_mask]
        self.warp_local_mask = warp_local_mask
        self.rest_global_mask = np.ones(NWALLP, dtype=bool)
        self.rest_global_mask[self.warp_global_idx] = False
        self.rest_count = int(self.rest_global_mask.sum())
        self.aligned_len = self.rest_count + len(warp_stations) * N_XI_BINS

        self.station_point_local_idx = {}
        for s in warp_stations:
            sel = warp_local_mask & (station_idx == s)
            self.station_point_local_idx[s] = np.flatnonzero(sel)

    def encode(self, rho, shock_xi_per_station):
        rest_part = rho[self.rest_global_mask]
        parts = [rest_part]
        for s in self.warp_stations:
            local_idx = self.station_point_local_idx[s]
            xi_pts = self.xi[local_idx]
            rho_pts = rho[self.wing_mask][local_idx]
            valid = np.isfinite(xi_pts)
            xi_pts = xi_pts[valid]
            rho_pts = rho_pts[valid]
            shock_xi_used = shock_xi_per_station[s]
            warped = warp_xi(xi_pts, shock_xi_used, self.xi_ref[s])
            order = np.argsort(warped)
            profile = np.interp(self.xi_grid, warped[order], rho_pts[order])
            parts.append(profile)
        return np.concatenate(parts)

    def decode(self, aligned_vec, shock_xi_per_station):
        rho = np.empty(NWALLP)
        rho[self.rest_global_mask] = aligned_vec[:self.rest_count]
        offset = self.rest_count
        for s in self.warp_stations:
            profile = aligned_vec[offset:offset + N_XI_BINS]
            offset += N_XI_BINS
            local_idx = self.station_point_local_idx[s]
            xi_pts = self.xi[local_idx]
            valid = np.isfinite(xi_pts)
            shock_xi_used = shock_xi_per_station[s]
            warped = warp_xi(xi_pts[valid], shock_xi_used, self.xi_ref[s])
            rho_pts = np.interp(warped, self.xi_grid, profile)
            target_idx = np.flatnonzero(self.wing_mask)[local_idx[valid]]
            rho[target_idx] = rho_pts
        return rho


def build_encode_matrix(Y, aligned, shock_xi_table):
    n = Y.shape[0]
    out = np.empty((n, aligned.aligned_len))
    for i in range(n):
        out[i] = aligned.encode(Y[i], shock_xi_table[i])
    return out


def shock_xi_table_for_conditions(conds, aligned, predict_shift, detected=None, shock_xi_detected=None):
    n = conds.shape[0]
    table = np.empty((n, N_STATIONS))
    for s in aligned.warp_stations:
        pred = predict_shift(aligned.eta_of_station[s], conds[:, 0], conds[:, 1], conds[:, 2])
        table[:, s] = pred
        if detected is not None:
            use_detected = detected[:, s]
            table[use_detected, s] = shock_xi_detected[use_detected, s]
    return table


def fit_pod_gp(train_conds_sc, Y_aligned_train, n_components, variance_threshold, n_restarts):
    n_train = Y_aligned_train.shape[0]
    n_comp = min(n_components, n_train - 1)
    pca = PCA(n_components=n_comp, random_state=SEED, svd_solver='full')
    Z_full = pca.fit_transform(Y_aligned_train)
    if variance_threshold is None:
        k = n_comp
        cum_var = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    else:
        k, cum_var = pg.choose_n_modes(pca, variance_threshold)
    Z_train = Z_full[:, :k]
    components = pca.components_[:k]
    mean_field = pca.mean_
    gps = Parallel(n_jobs=N_JOBS)(
        delayed(pg.fit_one_gp)(train_conds_sc, Z_train[:, m], train_conds_sc.shape[1],
                                KERNEL_NU, ALPHA, n_restarts, SEED)
        for m in range(k)
    )
    return dict(pca=pca, k=k, cum_var=cum_var, components=components, mean_field=mean_field, gps=gps)


def predict_pod_gp(model, conds_sc):
    gps = model['gps']
    Z_pred = np.column_stack([gp.predict(conds_sc) for gp in gps])
    return model['mean_field'][None, :] + Z_pred @ model['components']


def mach_holdout_split(train_conds, n_holdout):
    mach = train_conds[:, 0]
    low_machs = np.unique(mach)[:n_holdout]
    high_machs = np.unique(mach)[-n_holdout:]
    holdout_mask = np.isin(mach, np.concatenate([low_machs, high_machs]))
    val_idx = np.flatnonzero(holdout_mask)
    tr_idx = np.flatnonzero(~holdout_mask)
    return tr_idx, val_idx


def flat_r2(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    sse = float(np.sum((y_true - y_pred) ** 2))
    ssd = float(np.sum((y_true - y_true.mean()) ** 2))
    if ssd < pg.EPS:
        return 0.0
    return 1.0 - sse / ssd


def cv_compare_truncation(train_conds, train_conds_sc, Y_aligned_train):
    print('CV comparing truncated vs full rank POD+GP on aligned data', flush=True)
    n_train = Y_aligned_train.shape[0]
    schemes = []

    kf = KFold(n_splits=N_RANDOM_FOLDS, shuffle=True, random_state=SEED)
    schemes.append(('random_kfold', list(kf.split(np.arange(n_train)))))

    tr_idx, val_idx = mach_holdout_split(train_conds, N_MACH_HOLDOUT)
    schemes.append(('mach_holdout', [(tr_idx, val_idx)]))

    for scheme_name, splits in schemes:
        for variant_name, variance_threshold in (('truncated', VARIANCE_THRESHOLD), ('full_rank', None)):
            r2_list = []
            for tr_i, val_i in splits:
                scaler = StandardScaler()
                conds_sc_fold = scaler.fit_transform(train_conds[tr_i])
                val_sc_fold = scaler.transform(train_conds[val_i])
                model = fit_pod_gp(conds_sc_fold, Y_aligned_train[tr_i],
                                    MAX_COMPONENTS_CV, variance_threshold, N_RESTARTS_CV)
                pred = predict_pod_gp(model, val_sc_fold)
                r2 = flat_r2(Y_aligned_train[val_i], pred)
                r2_list.append(r2)
            print(f'  scheme={scheme_name} variant={variant_name} '
                  f'mean_R2={np.mean(r2_list):.4f} folds={len(splits)}', flush=True)


def per_component_breakdown(y_true, y_pred, weights, comp_masks, n_sims, label):
    print(f'Per-component breakdown {label}', flush=True)
    Y = y_true.reshape(n_sims, NWALLP)
    Yh = y_pred.reshape(n_sims, NWALLP)
    valid = np.flatnonzero(weights >= 1.0)
    for cname, mask in comp_masks.items():
        r2_list = []
        rmae_list = []
        for i in valid:
            yt = Y[i, mask]
            yp = Yh[i, mask]
            r2 = pg.compute_R2(yt, yp, np.ones_like(yt))
            rmae = float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt)))
            r2_list.append(r2)
            rmae_list.append(rmae)
        print(f'  {cname:<10} mean_R2={np.mean(r2_list):.4f} mean_rMAE={np.mean(rmae_list):.4f} '
              f'worst_rMAE={np.max(rmae_list):.4f}', flush=True)


def main():
    out_prefix = os.path.join(DATA_DIR, OUT_NAME)

    print('Loading data', flush=True)
    data = pg.load_split(DATA_DIR, SPLIT_DIR)
    train_conds, Y_train, n_train = data['train_conds'], data['Y_train'], data['n_train']
    print(f'n_train={n_train} n_test1={data["n_test1"]} n_test2={data["n_test2"]}', flush=True)

    X0 = np.asarray(np.load(os.path.join(DATA_DIR, SPLIT_DIR, 'train_data.npy'), mmap_mode='r')[:NWALLP])
    inspect_columns(X0)

    wing_mask = load_wing_mask(DATA_DIR, SPLIT_DIR)
    station_idx, upper, xi, eta_of_station, station_ok = build_wing_parametrization(X0, wing_mask)

    print('Detecting shocks on training snapshots', flush=True)
    t0 = time.time()
    detected, shock_xi_detected = detect_all_shocks(Y_train, wing_mask, station_idx, upper, xi)
    print(f'  detection time {time.time()-t0:.1f}s', flush=True)

    n_pairs_total = int(station_ok.sum()) * n_train
    n_pairs_detected = int(detected[:, station_ok].sum())
    detect_rate_global = n_pairs_detected / max(n_pairs_total, 1)
    print(f'Global detection rate over eligible (snapshot,station) pairs = {detect_rate_global:.3f} '
          f'({n_pairs_detected}/{n_pairs_total})', flush=True)
    if detect_rate_global < GLOBAL_DETECT_RATE_WARN:
        print(f'WARNING shock detection rate {detect_rate_global:.3f} is below {GLOBAL_DETECT_RATE_WARN}, '
              f'registration may not be reliable', flush=True)

    station_detect_rate = np.zeros(N_STATIONS)
    for s in range(N_STATIONS):
        if station_ok[s]:
            station_detect_rate[s] = detected[:, s].mean()
        print(f'  station {s:2d} detect_rate={station_detect_rate[s]:.3f}', flush=True)

    warp_stations = [s for s in range(N_STATIONS)
                      if station_ok[s] and station_detect_rate[s] >= STATION_DETECT_RATE_MIN]
    print(f'Warp enabled stations: {warp_stations}', flush=True)
    if len(warp_stations) == 0:
        print('FATAL no stations qualify for warping, stopping', flush=True)
        return

    xi_ref = {}
    for s in warp_stations:
        vals = shock_xi_detected[detected[:, s], s]
        xi_ref[s] = float(np.median(vals))
    print(f'Reference shock positions per warped station: {xi_ref}', flush=True)

    predict_shift = fit_shift_model(train_conds, eta_of_station, detected, shock_xi_detected)

    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    aligned = AlignedRepresentation(wing_mask, station_idx, upper, xi, eta_of_station, warp_stations, xi_ref)
    print(f'Aligned representation length = {aligned.aligned_len} '
          f'(rest={aligned.rest_count}, warped={len(warp_stations)}*{N_XI_BINS})', flush=True)

    shock_xi_table_train = shock_xi_table_for_conditions(
        train_conds, aligned, predict_shift, detected=detected, shock_xi_detected=shock_xi_detected)

    print('Building aligned training snapshots', flush=True)
    t0 = time.time()
    Y_aligned_train = build_encode_matrix(Y_train, aligned, shock_xi_table_train)
    print(f'  build time {time.time()-t0:.1f}s', flush=True)

    print('Singular value decay raw vs aligned', flush=True)
    pca_raw = PCA(n_components=min(50, n_train - 1), random_state=SEED, svd_solver='full')
    pca_raw.fit(Y_train)
    pca_aligned = PCA(n_components=min(50, n_train - 1), random_state=SEED, svd_solver='full')
    pca_aligned.fit(Y_aligned_train)
    cum_raw = np.cumsum(pca_raw.explained_variance_ratio_)
    cum_aligned = np.cumsum(pca_aligned.explained_variance_ratio_)
    for m in (5, 10, 20, 30, 50):
        m = min(m, len(cum_raw))
        print(f'  modes={m:3d} cum_var_raw={cum_raw[m-1]:.5f} cum_var_aligned={cum_aligned[m-1]:.5f}',
              flush=True)
    decay_improved = cum_aligned[9] > cum_raw[9]
    if not decay_improved:
        print('WARNING aligned singular values do not decay faster than raw at 10 modes, '
              'registration is not concentrating variance, stopping here', flush=True)
        return
    print('Aligned spectrum decays faster than raw at 10 modes, proceeding', flush=True)

    cv_compare_truncation(train_conds, train_conds_sc, Y_aligned_train)

    print('Fitting truncated POD+GP on aligned data', flush=True)
    t0 = time.time()
    model_trunc = fit_pod_gp(train_conds_sc, Y_aligned_train, MAX_COMPONENTS, VARIANCE_THRESHOLD, N_RESTARTS)
    print(f'  k={model_trunc["k"]} cum_var={model_trunc["cum_var"]:.5f} time={time.time()-t0:.1f}s', flush=True)

    print('Fitting full rank POD+GP on aligned data', flush=True)
    t0 = time.time()
    model_full = fit_pod_gp(train_conds_sc, Y_aligned_train, n_train - 1, None, N_RESTARTS)
    print(f'  k={model_full["k"]} time={time.time()-t0:.1f}s', flush=True)

    shock_xi_table_test1 = shock_xi_table_for_conditions(data['test1_conds'], aligned, predict_shift)
    shock_xi_table_test2 = shock_xi_table_for_conditions(data['test2_conds'], aligned, predict_shift)

    comp_labels = np.load(os.path.join(DATA_DIR, SPLIT_DIR, 'component_labels_unique.npy'))
    with open(os.path.join(DATA_DIR, SPLIT_DIR, 'component_map.json')) as f:
        comp_map = {int(k): v for k, v in json.load(f).items()}
    comp_masks = {cname: comp_labels == cid for cid, cname in comp_map.items()}

    for model_name, model in (('truncated', model_trunc), ('full_rank', model_full)):
        aligned_pred1 = predict_pod_gp(model, test1_conds_sc)
        aligned_pred2 = predict_pod_gp(model, test2_conds_sc)
        y_pred1 = np.concatenate([aligned.decode(aligned_pred1[i], shock_xi_table_test1[i])
                                   for i in range(data['n_test1'])])
        y_pred2 = np.concatenate([aligned.decode(aligned_pred2[i], shock_xi_table_test2[i])
                                   for i in range(data['n_test2'])])

        res1 = pg.evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], comp_masks)
        res2 = pg.evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], comp_masks)
        pg.print_result(f'Shifted POD+GP ({model_name})', 'Phase 1 (interpolation)', res1)
        pg.print_result(f'Shifted POD+GP ({model_name})', 'Phase 2 (extrapolation)', res2)
        per_component_breakdown(data['y_test1'], y_pred1, data['w_test1'], comp_masks,
                                 data['n_test1'], f'{model_name} phase1')
        per_component_breakdown(data['y_test2'], y_pred2, data['w_test2'], comp_masks,
                                 data['n_test2'], f'{model_name} phase2')

        if model_name == 'truncated':
            try:
                np.save(f'{out_prefix}_y_pred1.npy', y_pred1)
                np.save(f'{out_prefix}_y_pred2.npy', y_pred2)
                print(f'Saved {out_prefix}_y_pred1.npy {out_prefix}_y_pred2.npy', flush=True)
            except OSError as e:
                print(f'WARN could not save predictions {e}', flush=True)

    print('Computing oracle scores using true test shock detection', flush=True)
    for phase_label, Y_test, conds_test, weights_test, n_test in (
        ('Phase 1', data['y_test1'].reshape(data['n_test1'], NWALLP), data['test1_conds'],
         data['w_test1'], data['n_test1']),
        ('Phase 2', data['y_test2'].reshape(data['n_test2'], NWALLP), data['test2_conds'],
         data['w_test2'], data['n_test2']),
    ):
        detected_test = np.zeros((n_test, N_STATIONS), dtype=bool)
        shock_xi_test = np.full((n_test, N_STATIONS), np.nan)
        for i in range(n_test):
            for s in warp_stations:
                sel = (station_idx == s) & upper
                det, sxi, _, _ = detect_shock(xi[sel], Y_test[i][wing_mask][sel])
                detected_test[i, s] = det
                shock_xi_test[i, s] = sxi
        shock_xi_table_oracle = shock_xi_table_for_conditions(
            conds_test, aligned, predict_shift, detected=detected_test, shock_xi_detected=shock_xi_test)

        Y_aligned_test_true = build_encode_matrix(Y_test, aligned, shock_xi_table_oracle)
        Z_oracle = (Y_aligned_test_true - model_trunc['mean_field']) @ model_trunc['components'].T
        aligned_oracle = model_trunc['mean_field'][None, :] + Z_oracle @ model_trunc['components']
        y_oracle = np.concatenate([aligned.decode(aligned_oracle[i], shock_xi_table_oracle[i])
                                    for i in range(n_test)])

        res_oracle = pg.evaluate_phase(Y_test.reshape(-1), y_oracle, weights_test, comp_masks)
        pg.print_result('Shifted POD+GP oracle (truncation ceiling)', phase_label, res_oracle)

    print('Ablation, same POD+GP pipeline with alignment disabled', flush=True)
    t0 = time.time()
    model_noalign = fit_pod_gp(train_conds_sc, Y_train, MAX_COMPONENTS, VARIANCE_THRESHOLD, N_RESTARTS)
    print(f'  k={model_noalign["k"]} cum_var={model_noalign["cum_var"]:.5f} time={time.time()-t0:.1f}s',
          flush=True)
    y_pred1_noalign = predict_pod_gp(model_noalign, test1_conds_sc).reshape(-1)
    y_pred2_noalign = predict_pod_gp(model_noalign, test2_conds_sc).reshape(-1)
    res1_noalign = pg.evaluate_phase(data['y_test1'], y_pred1_noalign, data['w_test1'], comp_masks)
    res2_noalign = pg.evaluate_phase(data['y_test2'], y_pred2_noalign, data['w_test2'], comp_masks)
    pg.print_result('POD+GP no alignment ablation', 'Phase 1 (interpolation)', res1_noalign)
    pg.print_result('POD+GP no alignment ablation', 'Phase 2 (extrapolation)', res2_noalign)


if __name__ == '__main__':
    main()
