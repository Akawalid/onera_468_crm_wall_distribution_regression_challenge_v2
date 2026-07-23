"""
Metrics used by the starting kit -- exact same formulas as
bundle/scoring_program/scoring.py, so the numbers you see here during
cross-validation are directly comparable to what the leaderboard will show.

Primary metric: KLw (mean_KL) -- lower is better, see mlp_klw.py for an
explanation of what it measures and why we train the MLP baseline to
minimize it directly.
"""

import numpy as np

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
EPS = 1e-6

# Component-weighted residual KL-divergence weights (see scoring_program).
KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
KL_N_BINS = 200


def compute_R2(y, yhat, confidence_per_case, nwallp=NWALLP):
    """ Weighted R^2, one condition = one block of `nwallp` points. """
    y_blocks    = y.reshape(-1, nwallp)
    yhat_blocks = yhat.reshape(-1, nwallp)
    ymean       = np.mean(y)
    sq_err = ((y_blocks - yhat_blocks) ** 2).sum(axis=1)
    sq_dev = ((y_blocks - ymean) ** 2).sum(axis=1)
    SSE = float(np.dot(confidence_per_case, sq_err))
    SSD = float(np.dot(confidence_per_case, sq_dev))
    if SSD < EPS:
        return 0.0
    return 1.0 - SSE / SSD


def compute_wrMAE(y, yhat, confidence_per_case, nwallp=NWALLP):
    """ Worst-case relative MAE over high-confidence conditions. """
    y_blocks    = y.reshape(-1, nwallp)
    yhat_blocks = yhat.reshape(-1, nwallp)
    mask = confidence_per_case >= 1.0
    if not np.any(mask):
        raise ValueError('No high-confidence conditions to compute wrMAE.')
    mean_abs_diff = np.mean(np.abs(y_blocks - yhat_blocks), axis=1)
    mean_abs_y    = np.maximum(np.mean(np.abs(y_blocks), axis=1), EPS)
    relMAE = mean_abs_diff / mean_abs_y
    candidates   = np.flatnonzero(mask)
    iworst_local = int(np.argmax(relMAE[mask]))
    iworst       = int(candidates[iworst_local])
    return iworst, float(relMAE[iworst]), relMAE


def _residual_kl(y_true_case, y_pred_case, comp_masks, sigma_ref, n_bins=KL_N_BINS):
    """ KL(p_eps || N(0, sigma_ref)) of one condition's residuals, pooled
    across wall points with each point weighted by its component weight.
    See mlp_klw.py for the plain-English explanation of what this means. """
    eps = y_pred_case - y_true_case
    sigma_y = float(y_true_case.std()) + EPS

    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = KL_WEIGHTS[cname]

    lim  = 5.0 * sigma_y
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx   = bins[1] - bins[0]

    p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
    p    = np.clip(p * dx, 1e-10, None)
    p   /= p.sum()

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = np.exp(-0.5 * (bin_centers / sigma_ref) ** 2) / (sigma_ref * np.sqrt(2.0 * np.pi)) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def compute_mean_KL(y, yhat, confidence_per_case, comp_masks, sigma_ref, nwallp=NWALLP):
    """ Component-weighted residual KL, per condition, on high-confidence
    conditions only. Returns the mean (the leaderboard's primary metric),
    the worst single condition, and the full per-condition array (used for
    the bootstrap confidence interval). """
    y_blocks    = y.reshape(-1, nwallp)
    yhat_blocks = yhat.reshape(-1, nwallp)
    valid_idx = np.flatnonzero(confidence_per_case >= 1.0)
    if valid_idx.size == 0:
        raise ValueError('No high-confidence conditions to compute KLw.')

    kl_values = np.array([
        _residual_kl(y_blocks[i], yhat_blocks[i], comp_masks, sigma_ref)
        for i in valid_idx
    ])
    iworst_local = int(np.argmax(kl_values))
    iworst       = int(valid_idx[iworst_local])
    return float(kl_values.mean()), iworst, kl_values, valid_idx


def bootstrap_ci(values, stat_func=np.mean, n_boot=1000, ci=95, rng=None):
    """ Percentile bootstrap CI: resample `values` with replacement n_boot
    times, apply stat_func to each resample, take the [2.5, 97.5] (for
    ci=95) percentiles of the result. """
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = stat_func(values[idx], axis=1)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(boot_stats, [alpha, 100 - alpha])
    return float(lo), float(hi)


def evaluate(y, yhat, confidence_per_case, comp_masks, sigma_ref, nwallp=NWALLP):
    """ One-stop evaluation: pooled R2 / wrMAE / mean_KL (exactly what
    scoring.py computes) plus a bootstrap 95% CI for each, obtained by
    resampling validation conditions. """
    R2 = compute_R2(y, yhat, confidence_per_case, nwallp)
    iworst_wrmae, wrMAE, relMAE_per_case = compute_wrMAE(y, yhat, confidence_per_case, nwallp)
    mean_kl, iworst_kl, kl_values, valid_idx = compute_mean_KL(
        y, yhat, confidence_per_case, comp_masks, sigma_ref, nwallp)
    score = 5.0 * R2 + 5.0 * (1.0 - wrMAE)

    # Per-condition R2 (unweighted, single condition), for the R2 CI.
    y_blocks    = y.reshape(-1, nwallp)[valid_idx]
    yhat_blocks = yhat.reshape(-1, nwallp)[valid_idx]
    r2_per_case = np.array([
        compute_R2(y_blocks[i], yhat_blocks[i], np.array([1.0]), nwallp)
        for i in range(len(valid_idx))
    ])

    R2_ci    = bootstrap_ci(r2_per_case, np.mean)
    wrMAE_ci = bootstrap_ci(relMAE_per_case[valid_idx], np.max)
    KLw_ci   = bootstrap_ci(kl_values, np.mean)

    return dict(
        R2=R2, R2_ci=R2_ci,
        wrMAE=wrMAE, wrMAE_ci=wrMAE_ci, iworst_wrmae=iworst_wrmae,
        mean_KL=mean_kl, KLw_ci=KLw_ci, iworst_kl=iworst_kl,
        score=score,
    )


def print_leaderboard(results):
    """ results: {model_name: evaluate(...) dict}. KLw is printed first --
    it's the challenge's primary metric. """
    print(f'{"model":<12}  {"KLw (mean)":>11}  {"95% CI":>17}  {"R2":>8}  {"95% CI":>17}  '
          f'{"wrMAE":>8}  {"95% CI":>17}  {"score":>7}')
    print('-' * 108)
    for name, r in results.items():
        print(f'{name:<12}  {r["mean_KL"]:>11.4f}  [{r["KLw_ci"][0]:>6.4f}, {r["KLw_ci"][1]:>6.4f}]  '
              f'{r["R2"]:>8.4f}  [{r["R2_ci"][0]:>+6.4f}, {r["R2_ci"][1]:>+6.4f}]  '
              f'{r["wrMAE"]:>8.4f}  [{r["wrMAE_ci"][0]:>6.4f}, {r["wrMAE_ci"][1]:>6.4f}]  '
              f'{r["score"]:>7.4f}')
