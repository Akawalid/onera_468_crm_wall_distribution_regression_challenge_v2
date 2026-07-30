"""Diagnose *where* utils/simple_gnn.py's SimpleGNN is failing, not just by how much.

Three complementary angles:
  1. Per-component and per-condition breakdowns -- which body part, and which
     (Minf, AoA, Pi) regime, drives the error.
  2. Error vs. local mesh density -- a GNN-specific check: does this distance-weighted
     aggregator degrade where the kNN neighborhood is coarse/sparse relative to typical
     spacing? (own diagnostic, not from the notebook)
  3. The weighted-PCA error-slice analysis from copy.ipynb / utils/train.py
     (weighted_pca_axes / error_pca_uv / auto_slice_positions / plot_slice_along_v):
     finds the axis along which large errors are spread out (typically a shock line) and
     plots true-vs-predicted rho across it. Ported as clean standalone functions here --
     importing them from utils/train.py directly isn't possible, since that script runs a
     full multi-model training pipeline as a side effect of being imported. Run twice per
     component: once on the single worst-KL simulation (matches the notebook's original
     use), and once on the mean field averaged over every simulation in the split, which
     surfaces systematic failure locations instead of one simulation's noise.

Produces a text report (printed and saved) plus PNG figures under --out_dir.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_models import KL_WEIGHTS, residual_kl_weighted
from utils.simple_gnn import SimpleGNN
from utils.train_simple_gnn import NWALLP, load_split, load_static_graph, predict_split


# --------------------------------------------------------------------------- #
# PCA-based error-slice analysis (ported from copy.ipynb / utils/train.py)
# --------------------------------------------------------------------------- #

def weighted_pca_axes(coords, w):
    """Weighted PCA: the dominant axis is the direction along which `w` is most spread
    out. With w = |error|**gamma this is the direction an error band/shock line runs
    along, not the direction the geometry itself is longest.
    """
    wsum = w.sum() + 1e-12
    mu = (coords * w[:, None]).sum(axis=0) / wsum
    xc = coords - mu
    cov = (xc * w[:, None]).T @ xc / wsum
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return mu, vecs[:, order]


def error_weighted_uv(coords, yt, yp, gamma=2.0):
    err = np.abs(yp - yt)
    w = err ** gamma
    mu, axes3 = weighted_pca_axes(coords, w)
    uv = (coords - mu) @ axes3[:, :2]
    return uv, err


def auto_slice_positions(uv, err, n_slices=3, min_separation=None, edge_frac=0.05):
    """Place n_slices cuts along u (the dominant error axis) at the bins with the most
    concentrated weighted error, skipping the extreme edges and near-empty bins.
    """
    u = uv[:, 0]
    n_bins = 100
    edges = np.linspace(u.min(), u.max(), n_bins + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, n_bins - 1)
    err_per_bin = np.bincount(idx, weights=err, minlength=n_bins)
    cnt_per_bin = np.bincount(idx, minlength=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    lo, hi = np.quantile(u, [edge_frac, 1.0 - edge_frac])
    err_per_bin[(centers < lo) | (centers > hi)] = 0.0
    nonzero_cnt = cnt_per_bin[cnt_per_bin > 0]
    if len(nonzero_cnt):
        err_per_bin[cnt_per_bin < np.median(nonzero_cnt) * 0.3] = 0.0

    if min_separation is None:
        min_separation = (u.max() - u.min()) / 10

    positions, remaining = [], err_per_bin.copy()
    for _ in range(n_slices):
        if remaining.max() <= 0:
            break
        b = int(np.argmax(remaining))
        positions.append(float(centers[b]))
        remaining[np.abs(centers - centers[b]) < min_separation] = 0.0
    return sorted(positions)


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


def plot_error_slice(uv, yt, yp, upper, u0, half_width, title, out_path, n_bins_v=150):
    sel = np.abs(uv[:, 0] - u0) < half_width
    if sel.sum() < 20:
        return False

    plt.figure(figsize=(9, 6), dpi=120)
    for side_mask, ls, label in [(upper, '-', 'upper'), (~upper, '--', 'lower')]:
        s = sel & side_mask
        if s.sum() < 10:
            continue
        vc, rt = _median_curve(uv[s, 1], yt[s], n_bins_v)
        _, rp = _median_curve(uv[s, 1], yp[s], n_bins_v)
        plt.plot(vc, rt, color='#e34948', ls=ls, lw=1.5, label=f'true rho ({label})')
        plt.plot(vc, rp, color='#1baf7a', ls=ls, lw=1.5, label=f'pred rho ({label})')

    plt.xlabel('v (across the error band)')
    plt.ylabel('rho')
    plt.title(f'{title}  --  slice u={u0:.3f}', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def analyse_error_slices(coords, yt, yp, upper, title, out_prefix, n_slices=3, gamma=2.0):
    uv, err = error_weighted_uv(coords, yt, yp, gamma=gamma)
    span = uv[:, 0].max() - uv[:, 0].min()
    half_width = max(span / 200, 1e-9)
    positions = auto_slice_positions(uv, err, n_slices=n_slices)
    saved = []
    for i, u0 in enumerate(positions):
        path = f'{out_prefix}_slice{i + 1}.png'
        if plot_error_slice(uv, yt, yp, upper, u0, half_width, title, path):
            saved.append(path)
    return positions, saved


# --------------------------------------------------------------------------- #
# Own diagnostics: per-component, per-condition, error-vs-mesh-density
# --------------------------------------------------------------------------- #

def per_component_report(y, preds, weights, comp_masks, nwallp=NWALLP):
    n_sims = len(y) // nwallp
    Y, P = y.reshape(n_sims, nwallp), preds.reshape(n_sims, nwallp)
    keep = weights >= 1.0  # same convention as base_models' worst_rMAE/KLw: skip half-weight sims

    header = f"{'component':<10} {'mean|err|':>10} {'rMAE':>8} {'bias':>9} {'n_pts':>9}"
    lines = [header, '-' * len(header)]
    for cname, mask in comp_masks.items():
        err = P[keep][:, mask] - Y[keep][:, mask]
        mean_abs = float(np.mean(np.abs(err)))
        rmae = mean_abs / float(np.mean(np.abs(Y[keep][:, mask])))
        bias = float(np.mean(err))
        lines.append(f"{cname:<10} {mean_abs:>10.5f} {rmae:>8.4f} {bias:>+9.5f} {mask.sum():>9d}")
    return '\n'.join(lines)


def per_simulation_report(y, preds, weights, conds, comp_masks, sigma_ref, nwallp=NWALLP, n_worst=10):
    n_sims = len(y) // nwallp
    Y, P = y.reshape(n_sims, nwallp), preds.reshape(n_sims, nwallp)
    rmae = np.full(n_sims, np.nan)
    klw = np.full(n_sims, np.nan)
    for i in range(n_sims):
        if weights[i] < 1.0:
            continue
        yt, yp = Y[i], P[i]
        rmae[i] = np.mean(np.abs(yp - yt)) / np.mean(np.abs(yt))
        klw[i] = residual_kl_weighted(yt, yp, comp_masks, KL_WEIGHTS, sigma_ref)

    order = np.argsort(np.where(np.isnan(klw), -np.inf, klw))[::-1]
    header = f"{'rank':<5}{'sim':<6}{'Minf':>7}{'AoA':>7}{'Pi':>10}{'rMAE':>9}{'KLw':>9}"
    lines = [f'worst {n_worst} simulations by weighted-KL:', header]
    for rank, i in enumerate(order[:n_worst]):
        lines.append(f"{rank + 1:<5}{i:<6}{conds[i, 0]:>7.2f}{conds[i, 1]:>7.1f}"
                      f"{conds[i, 2]:>10.3g}{rmae[i]:>9.4f}{klw[i]:>9.4f}")

    valid = ~np.isnan(klw)
    corr_lines = ['condition-vs-error correlation (Pearson r, full-weight sims only):']
    for j, name in enumerate(['Minf', 'AoA', 'Pi']):
        r_rmae = np.corrcoef(conds[valid, j], rmae[valid])[0, 1]
        r_klw = np.corrcoef(conds[valid, j], klw[valid])[0, 1]
        corr_lines.append(f'  {name:<5} rMAE r={r_rmae:+.3f}   KLw r={r_klw:+.3f}')

    return '\n'.join(lines), '\n'.join(corr_lines), rmae, klw, order


def local_neighbor_distance(coords, edge_index):
    """Per-node mean distance to its kNN neighbors (in edge_index's dst-indexed sense) --
    a proxy for local mesh density. Small = fine/dense mesh, large = coarse/sparse.
    """
    src, dst = edge_index[0], edge_index[1]
    n = coords.shape[0]
    dist = (coords[dst] - coords[src]).norm(dim=-1)
    dsum = torch.zeros(n, dtype=dist.dtype)
    cnt = torch.zeros(n, dtype=dist.dtype)
    dsum.index_add_(0, dst, dist)
    cnt.index_add_(0, dst, torch.ones_like(dist))
    return (dsum / cnt.clamp_min(1)).numpy()


def density_vs_error_report(local_density, mean_abs_err, n_bins=10):
    edges = np.quantile(local_density, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(local_density, edges[1:-1]), 0, n_bins - 1)
    header = f"{'density decile':<16}{'mean nbr-dist':>14}{'mean|err|':>12}{'n_pts':>10}"
    lines = ['error vs. local mesh density (decile 1 = densest/finest mesh):', header]
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        lines.append(f"{b + 1:<16}{local_density[m].mean():>14.5f}{mean_abs_err[m].mean():>12.5f}{m.sum():>10d}")
    r = float(np.corrcoef(local_density, mean_abs_err)[0, 1])
    lines.append(f'\nPearson r(local mean neighbor distance, mean |error|) = {r:+.3f}')
    return '\n'.join(lines), r


def plot_component_bar(comp_err, out_path):
    names = list(comp_err.keys())
    vals = [comp_err[n] for n in names]
    plt.figure(figsize=(6, 4), dpi=120)
    plt.bar(names, vals, color='#2a78d6')
    plt.ylabel('mean |error|')
    plt.title('Mean absolute error by component')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_density_vs_error(local_density, mean_abs_err, out_path, n_bins=30):
    edges = np.quantile(local_density, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.digitize(local_density, edges[1:-1]), 0, n_bins - 1)
    centers, means = [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            centers.append(local_density[m].mean())
            means.append(mean_abs_err[m].mean())
    plt.figure(figsize=(7, 4.5), dpi=120)
    plt.plot(centers, means, marker='o', ms=4, lw=1.5, color='#eb6834')
    plt.xlabel('local mean neighbor distance (mesh coarseness)')
    plt.ylabel('mean |error|')
    plt.title('Error vs. local mesh density')
    plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default='data/')
    p.add_argument('--checkpoint', default='utils/simple_gnn_model.pt')
    p.add_argument('--k', type=int, default=20, help='must match the k the checkpoint was trained with.')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--split', default='test_phase1', choices=['test_phase1', 'test_phase2'])
    p.add_argument('--n_worst_sims', type=int, default=10)
    p.add_argument('--n_slice_components', type=int, default=2,
                   help='run the PCA error-slice analysis on this many worst-error components.')
    p.add_argument('--n_slices', type=int, default=3)
    p.add_argument('--gamma', type=float, default=2.0, help='error-weighting exponent for the weighted PCA.')
    p.add_argument('--out_dir', default='utils/logs/simple_gnn_error_analysis')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    report = []

    def log(s=''):
        print(s, flush=True)
        report.append(s)

    log('Loading static graph/features...')
    coords, edge_index, edge_weight, static_feats, component_labels, component_map = load_static_graph(
        args.data_dir, args.k, cache_dir
    )
    edge_index_dev = edge_index.to(device)
    edge_weight_dev = edge_weight.to(device)
    static_feats_dev = static_feats.to(device)
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}
    coords_np = coords.numpy()
    normals_z = static_feats[:, 5].numpy()  # cols: [coords_norm(3), normals(3), onehot(...)]
    upper_all = normals_z >= 0.0

    y_train, _, n_train, train_conds = load_split(args.data_dir, 'train')
    cond_mean = train_conds.mean(axis=0)
    cond_std = train_conds.std(axis=0)
    cond_std[cond_std == 0] = 1.0
    sigma_ref = 0.01 * float(np.mean(y_train))

    in_dim = static_feats.shape[1] + 3
    model = SimpleGNN(in_dim, args.hidden_dim, out_dim=1, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    log(f'Loaded checkpoint: {args.checkpoint}')

    y, weights, n_sims, conds = load_split(args.data_dir, args.split)
    weights = weights if weights is not None else np.ones(n_sims)
    y = np.asarray(y)
    preds = predict_split(model, n_sims, conds, edge_index_dev, edge_weight_dev, static_feats_dev,
                           device, cond_mean, cond_std)

    log(f'\n=== {args.split}  (n_sims={n_sims}) ===\n')

    log('--- per-component breakdown ---')
    log(per_component_report(y, preds, weights, comp_masks))

    log('\n--- per-simulation / per-condition ---')
    sim_table, corr_table, rmae, klw, order = per_simulation_report(
        y, preds, weights, conds, comp_masks, sigma_ref, n_worst=args.n_worst_sims
    )
    log(sim_table)
    log('')
    log(corr_table)

    Y, P = y.reshape(n_sims, NWALLP), preds.reshape(n_sims, NWALLP)
    mean_abs_err = np.mean(np.abs(P - Y), axis=0)  # length NWALLP, averaged over all sims

    log('\n--- error vs. local mesh density ---')
    local_density = local_neighbor_distance(coords, edge_index)
    density_table, density_corr = density_vs_error_report(local_density, mean_abs_err)
    log(density_table)
    plot_density_vs_error(local_density, mean_abs_err, os.path.join(args.out_dir, f'{args.split}_density_vs_error.png'))

    log('\n--- PCA error-slice analysis ---')
    comp_err = {cname: float(np.mean(mean_abs_err[mask])) for cname, mask in comp_masks.items()}
    plot_component_bar(comp_err, os.path.join(args.out_dir, f'{args.split}_component_bar.png'))
    worst_components = sorted(comp_err, key=comp_err.get, reverse=True)[:args.n_slice_components]
    ranked = sorted(comp_err.items(), key=lambda kv: -kv[1])
    log(f"components ranked by mean |error|: {[(c, round(e, 5)) for c, e in ranked]}")
    log(f'running slice analysis on: {worst_components}')

    isim_worst = int(order[0])
    for cname in worst_components:
        mask = comp_masks[cname]
        coords_c = coords_np[mask]
        upper_c = upper_all[mask]

        title = (f'{args.split} sim {isim_worst} ({cname}) Minf={conds[isim_worst, 0]:.2f} '
                 f'AoA={conds[isim_worst, 1]:.1f}')
        prefix = os.path.join(args.out_dir, f'{args.split}_worstsim{isim_worst}_{cname}')
        positions, saved = analyse_error_slices(
            coords_c, Y[isim_worst][mask], P[isim_worst][mask], upper_c, title, prefix,
            n_slices=args.n_slices, gamma=args.gamma
        )
        log(f'[{cname}] worst-sim (#{isim_worst}) slices at u={[round(p, 3) for p in positions]}: {saved}')

        mean_yt, mean_yp = Y[:, mask].mean(axis=0), P[:, mask].mean(axis=0)
        title2 = f'{args.split} SYSTEMATIC (mean over {n_sims} sims) ({cname})'
        prefix2 = os.path.join(args.out_dir, f'{args.split}_systematic_{cname}')
        positions2, saved2 = analyse_error_slices(
            coords_c, mean_yt, mean_yp, upper_c, title2, prefix2,
            n_slices=args.n_slices, gamma=args.gamma
        )
        log(f'[{cname}] systematic (mean-field) slices at u={[round(p, 3) for p in positions2]}: {saved2}')

    report_path = os.path.join(args.out_dir, f'{args.split}_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report) + '\n')
    log(f'\nFull report saved to {report_path}')


if __name__ == '__main__':
    main()
