"""
Visual diagnostics for one simulation:

- plot_condition: ground truth / prediction / |error| painted on the
  aircraft surface (error in red) -- the classic "where does the model
  get it wrong" view.
- plot_condition_error_pca: same |error| view, but with the two
  principal directions of the *error* (not the geometry) drawn as arrows
  on the plane body -- u follows the error structure (e.g. a shock line),
  v cuts across it.
- analyse_coupes_auto: automatically finds the u-positions where rho
  varies the most, and plots rho(true) vs rho(predicted) along v at each
  one -- i.e. slices perpendicular to the error structure, at the spots
  most likely to show it.

All of this only needs one simulation's (X, y_true, y_pred, comp_masks);
it's used after the fact on a CV fold's predictions, so it never needs
the real test set either.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .metrics import NWALLP, COL_MINF, COL_AOA, COL_PI


def _percentile_scale(field):
    """ Robust color scale: centered on the median, spread set from the
    3rd/97th percentiles so a few outlier points don't wash out the plot. """
    f03, f50, f97 = np.percentile(field, [3.0, 50.0, 97.0])
    slopem = (f50 - f03) / 0.47
    slopep = (f97 - f50) / 0.47
    return max(0.0, f50 - 0.6 * slopem), f50 + 0.6 * slopep


def plot_condition(X_sim, y_sim, yhat_sim, title_suffix='', model_name='model', elev=20):
    """ 3-panel 3-D scatter: ground truth, prediction, |error| (red). """
    X_coord, Y_coord, Z_coord = X_sim[:, 0], X_sim[:, 1], X_sim[:, 2]
    error = np.abs(y_sim - yhat_sim)

    fields = [y_sim, yhat_sim, error]
    titles = ['Ground truth rho', f'{model_name} prediction', 'Absolute error']
    cmaps  = ['jet', 'jet', 'Reds']

    fig = plt.figure(figsize=(18, 5), dpi=110)
    fig.suptitle(f'Worst condition{title_suffix}', fontsize=11)
    for col, (field, title, cmap) in enumerate(zip(fields, titles, cmaps)):
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[20, 1])
        ax = fig.add_subplot(gs[0, col], projection='3d')
        fmin, fmax = _percentile_scale(field)
        sca = ax.scatter3D(X_coord, Y_coord, Z_coord, c=field, vmin=fmin, vmax=fmax,
                            cmap=cmap, s=0.3, alpha=0.8)
        ax.view_init(elev=elev, azim=120)
        ax.set_xlim(X_coord.min() - 5.0, X_coord.max() + 5.0)
        ax.set_ylim(Y_coord.min() - 5.0, Y_coord.max() + 5.0)
        ax.set_zlim(Z_coord.min(), Z_coord.max())
        lims = np.array([getattr(ax, f'get_{a}lim')() for a in 'xyz'])
        ax.set_box_aspect(np.ptp(lims, axis=1), zoom=1.0)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        cax = fig.add_subplot(gs[1, col])
        cbar = fig.colorbar(sca, cax=cax, orientation='horizontal')
        cbar.set_label('rho' if col < 2 else '|error|', size=8)
        cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.show()


def weighted_pca_axes(coords, w):
    """ PCA axes of `coords`, weighted by `w` (e.g. by |error|^gamma so the
    principal axis follows where the error is concentrated rather than
    just the geometry). """
    wsum = w.sum() + 1e-12
    mu = (coords * w[:, None]).sum(axis=0) / wsum
    Xc = coords - mu
    C = (Xc * w[:, None]).T @ Xc / wsum
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    return mu, vecs[:, order]


def error_pca_uv(X_sim, y_sim, yhat_sim, comp_masks, comp_name, gamma=2.0):
    """ Project one component's points onto the error-weighted principal
    plane (u, v): u = direction the error structure follows, v = the
    direction that cuts across it. """
    mask = comp_masks[comp_name]
    coords = X_sim[mask][:, :3]
    yt, yp = y_sim[mask], yhat_sim[mask]
    err = np.abs(yp - yt)
    w = err ** gamma
    mu, axes3 = weighted_pca_axes(coords, w)
    uv = (coords - mu) @ axes3[:, :2]
    return uv, yt, yp, err, mu, axes3


def plot_condition_error_pca(X_sim, y_sim, yhat_sim, comp_masks, comp_name,
                              title_suffix='', gamma=2.0, vec_scale=0.6, elev=80):
    """ |error| painted on the surface, with the error-PCA (u, v) axes
    drawn as arrows anchored at the error-weighted centroid. """
    X_coord, Y_coord, Z_coord = X_sim[:, 0], X_sim[:, 1], X_sim[:, 2]
    error = np.abs(y_sim - yhat_sim)
    coords = X_sim[:, :3]

    _, _, _, _, mu, axes3 = error_pca_uv(X_sim, y_sim, yhat_sim, comp_masks, comp_name, gamma=gamma)
    u_dir, v_dir = axes3[:, 0], axes3[:, 1]
    span = coords.max(axis=0) - coords.min(axis=0)
    arrow_len = vec_scale * np.linalg.norm(span)

    fig = plt.figure(figsize=(7, 6), dpi=110)
    fig.suptitle(f'Error PCA directions{title_suffix}', fontsize=10)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[20, 1])
    ax = fig.add_subplot(gs[0, 0], projection='3d')

    fmin, fmax = _percentile_scale(error)
    sca = ax.scatter3D(X_coord, Y_coord, Z_coord, c=error, vmin=fmin, vmax=fmax,
                        cmap='Reds', s=0.3, alpha=0.8)
    ax.quiver(*mu, *(u_dir * arrow_len), color='cyan', linewidth=2,
              label='u (follows the error structure)')
    ax.quiver(*mu, *(v_dir * arrow_len * 0.5), color='lime', linewidth=2,
              label='v (cuts across it)')
    ax.legend(loc='upper left', fontsize=7)
    ax.view_init(elev=elev, azim=120)
    ax.set_xlim(X_coord.min() - 5.0, X_coord.max() + 5.0)
    ax.set_ylim(Y_coord.min() - 5.0, Y_coord.max() + 5.0)
    ax.set_zlim(Z_coord.min(), Z_coord.max())
    lims = np.array([getattr(ax, f'get_{a}lim')() for a in 'xyz'])
    ax.set_box_aspect(np.ptp(lims, axis=1), zoom=1.0)
    ax.set_axis_off()

    cax = fig.add_subplot(gs[1, 0])
    cbar = fig.colorbar(sca, cax=cax, orientation='horizontal')
    cbar.set_label('|error|', size=8)
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.show()


def auto_slice_positions_rho(uv, yt, upper, n_slices=3, n_u=100, n_v=150,
                              min_separation=None, edge_frac=0.05):
    """ Pick the `n_slices` u-positions where rho varies the most along v
    (i.e. where a slice is most informative), spaced at least
    `min_separation` apart. """
    u, v = uv[:, 0], uv[:, 1]
    ue = np.linspace(u.min(), u.max(), n_u + 1)
    ve = np.linspace(v.min(), v.max(), n_v + 1)
    ui = np.clip(np.digitize(u, ue) - 1, 0, n_u - 1)
    vi = np.clip(np.digitize(v, ve) - 1, 0, n_v - 1)

    score = np.zeros(n_u)
    for face_mask in (upper, ~upper):
        cell = ui[face_mask] * n_v + vi[face_mask]
        sums = np.bincount(cell, weights=yt[face_mask], minlength=n_u * n_v)
        cnts = np.bincount(cell, minlength=n_u * n_v)
        grid = np.full(n_u * n_v, np.nan)
        ok = cnts > 0
        grid[ok] = sums[ok] / cnts[ok]
        grid = grid.reshape(n_u, n_v)
        jump = np.abs(np.diff(grid, axis=1))
        col_score = np.where(np.all(np.isnan(jump), axis=1), 0.0, np.nanmax(jump, axis=1))
        score = np.maximum(score, col_score)

    centers = 0.5 * (ue[:-1] + ue[1:])
    cnt_u = np.bincount(ui, minlength=n_u)
    lo, hi = np.quantile(u, [edge_frac, 1.0 - edge_frac])
    score[(centers < lo) | (centers > hi)] = 0.0
    score[cnt_u < np.median(cnt_u[cnt_u > 0]) * 0.3] = 0.0

    if min_separation is None:
        min_separation = (u.max() - u.min()) / 10

    positions, remaining = [], score.copy()
    for _ in range(n_slices):
        if remaining.max() <= 0:
            break
        b = int(np.argmax(remaining))
        positions.append(centers[b])
        remaining[np.abs(centers - centers[b]) < min_separation] = 0.0
    return sorted(positions)


def _median_curve(x, vals, n_bins):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    centers, meds = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            centers.append(0.5 * (edges[b] + edges[b + 1]))
            meds.append(np.median(vals[m]))
    return np.array(centers), np.array(meds)


def plot_slice_along_v(uv, yt, yp, upper, u0, half_width, model_name='model',
                        title_suffix='', n_bins_v=150):
    """ Median rho(true) vs rho(predicted) along v, for the thin slice
    |u - u0| < half_width -- extrados/intrados plotted separately. """
    sel = np.abs(uv[:, 0] - u0) < half_width
    if sel.sum() < 20:
        print(f'  slice u0={u0:.2f}: only {sel.sum()} points, skipped')
        return

    plt.figure(figsize=(9, 6), dpi=110)
    for side_mask, ls, side_label in [(upper, '-', 'extrados'), (~upper, '--', 'intrados')]:
        s = sel & side_mask
        if s.sum() < 10:
            continue
        vv = uv[s, 1]
        vc, rt = _median_curve(vv, yt[s], n_bins_v)
        _, rp = _median_curve(vv, yp[s], n_bins_v)
        plt.plot(vc, rt, color='red', ls=ls, lw=1.5, label=f'rho true ({side_label})')
        plt.plot(vc, rp, color='green', ls=ls, lw=1.5, label=f'rho {model_name} ({side_label})')

    plt.xlabel('v (across the error structure)')
    plt.ylabel('rho / rho_inf')
    plt.title(f'slice u={u0:.2f}{title_suffix}', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.show()


def analyse_coupes_auto(X_sim, y_sim, yhat_sim, comp_masks, comp_name,
                         model_name='model', n_slices=3, gamma=2.0, title_suffix=''):
    """ Full pipeline: PCA-project the error, auto-pick the most
    informative u-positions, plot a v-slice at each. """
    uv, yt, yp, err, _, _ = error_pca_uv(X_sim, y_sim, yhat_sim, comp_masks, comp_name, gamma=gamma)
    upper = (X_sim[:, 5] >= 0.0)[comp_masks[comp_name]]

    span = uv[:, 0].max() - uv[:, 0].min()
    half_width = span / 200

    positions = auto_slice_positions_rho(uv, yt, upper, n_slices=n_slices)
    print(f'  slices placed at u = {[f"{p:.2f}" for p in positions]}')
    for u0 in positions:
        plot_slice_along_v(uv, yt, yp, upper, u0, half_width, model_name=model_name,
                            title_suffix=title_suffix)
