import json
import os
import sys
import traceback
import numpy as np
from datetime import datetime as dt

# Codabench invokes this program as:
#   scoring.py $input $output
# Fall back to local defaults when run without arguments.
if len(sys.argv) == 1:
    input_dir  = '/app/input'
    output_dir = '/app/output/'
else:
    input_dir  = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])

reference_dir  = os.path.join(input_dir, 'ref')
prediction_dir = os.path.join(input_dir, 'res')
score_file     = os.path.join(output_dir, 'scores.json')
html_file      = os.path.join(output_dir, 'detailed_results.html')

nwallp  = 260774
epsilon = 1.e-6

# Column order in test_conditions.npy (one row per test condition).
IDX_MINF, IDX_AOA, IDX_PI = 0, 1, 2

# Component-weighted residual KL-divergence (see copy.ipynb, section 7-8).
# Each named component must be present in component_map.json / comp_masks.
KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
KL_N_BINS  = 200


def write_file(file, content, mode='a'):
    """ Write content in file ('a' to append, 'w' to (re)create). """
    with open(file, mode, encoding='utf-8') as f:
        f.write(content)


def print_bar():
    """ Display a bar ('----------') """
    print('-' * 10)


def get_data():
    """ Get ground truth (y_test) and predictions (y_pred). """
    print('[*] Reading reference data from {}'.format(reference_dir))
    try:
        y_test = np.load(os.path.join(reference_dir, 'test_labels.npy'))[:, 0]
        print('[+] Loaded {} reference values.'.format(len(y_test)))
    except Exception as e:
        print('[-] Error loading reference data: {}'.format(e))
        raise

    print('[*] Reading predictions from {}'.format(prediction_dir))
    try:
        y_pred = np.load(os.path.join(prediction_dir, 'Yhat.npy'))[:, 0]
        print('[+] Loaded {} predictions.'.format(len(y_pred)))
    except Exception as e:
        print('[-] Error loading predictions: {}'.format(e))
        raise

    return y_test, y_pred


def get_confidence():
    """ Get per-condition confidence weights and the corresponding flight
    conditions (one row per condition: [Minf, AoA, Pi], see IDX_* above).
    test_conditions.npy is a small (n_cases, 3) summary extracted once by
    the data-prep pipeline, so scoring never needs the full (multi-GB)
    test_data.npy that ingestion already receives separately as input_data
    -- avoids duplicating that large array into reference_data too. """
    print('[*] Reading confidence weights from {}'.format(reference_dir))
    try:
        confidence_per_case = np.load(os.path.join(reference_dir, 'test_weights.npy'))
        print('[+] Loaded confidence for {} test conditions.'.format(len(confidence_per_case)))
    except Exception as e:
        print('[-] Error loading confidence data: {}'.format(e))
        raise

    print('[*] Reading test conditions from {}'.format(reference_dir))
    try:
        X_conditions = np.load(os.path.join(reference_dir, 'test_conditions.npy'))
        print('[+] Loaded {} test conditions.'.format(len(X_conditions)))
    except Exception as e:
        print('[-] Error loading test conditions: {}'.format(e))
        raise

    return confidence_per_case, X_conditions


def get_components():
    """ Load per-wall-point component ids and their id->name mapping, and
    build a boolean mask (length nwallp) per named component required by
    KL_WEIGHTS (wing/pylon/fuselage/nacelle). """
    print('[*] Reading component labels from {}'.format(reference_dir))
    try:
        component_labels = np.load(os.path.join(reference_dir, 'component_labels_unique.npy'))
        with open(os.path.join(reference_dir, 'component_map.json')) as f:
            component_map = {int(k): v for k, v in json.load(f).items()}

        comp_masks = {
            cname: (component_labels == cid)
            for cid, cname in component_map.items()
            if cname in KL_WEIGHTS
        }
        missing = sorted(set(KL_WEIGHTS) - set(comp_masks))
        if missing:
            raise ValueError(f'component_map.json is missing component(s): {missing}')

        print('[+] Loaded {} wall-point component labels ({} components).'.format(
            len(component_labels), len(comp_masks)))
    except Exception as e:
        print('[-] Error loading component data: {}'.format(e))
        raise

    return comp_masks


def validate_alignment(y_test, y_pred, confidence_per_case, comp_masks):
    """ Sanity-check shapes so a data/reference mismatch fails loudly with
    a clear message instead of silently producing a wrong (or crashing)
    score. """
    if y_test.shape != y_pred.shape:
        raise ValueError(
            'Shape mismatch: prediction {} != reference {}.'.format(
                y_pred.shape, y_test.shape))
    if y_test.size % nwallp != 0:
        raise ValueError(
            'Reference length {} is not a multiple of nwallp={}.'.format(
                y_test.size, nwallp))
    n_cases = y_test.size // nwallp
    if len(confidence_per_case) != n_cases:
        raise ValueError(
            'confidence_per_case has {} entries, expected {} (= {} / {}).'.format(
                len(confidence_per_case), n_cases, y_test.size, nwallp))
    for cname, mask in comp_masks.items():
        if mask.shape[0] != nwallp:
            raise ValueError(
                'component mask "{}" has {} points, expected nwallp={}.'.format(
                    cname, mask.shape[0], nwallp))


def compute_R2(y, yhat, confidence_per_case):
    """ Weighted R^2 score, weighting each condition block by its confidence. """
    y_blocks    = y.reshape(-1, nwallp)
    yhat_blocks = yhat.reshape(-1, nwallp)
    ymean       = np.mean(y)
    sq_err      = ((y_blocks - yhat_blocks) ** 2).sum(axis=1)
    sq_dev      = ((y_blocks - ymean) ** 2).sum(axis=1)
    SSE = float(np.dot(confidence_per_case, sq_err))
    SSD = float(np.dot(confidence_per_case, sq_dev))
    if SSD < epsilon:
        # Degenerate case: reference values have (near) zero variance.
        # Avoid a division by zero that would otherwise produce a NaN
        # score (which is not even valid JSON).
        return 0.0
    return 1.0 - SSE / SSD


def compute_wrMAE(y, yhat, confidence_per_case):
    """ Worst-case relative MAE on high-confidence conditions only. """
    y_blocks    = y.reshape(-1, nwallp)
    yhat_blocks = yhat.reshape(-1, nwallp)

    mask = confidence_per_case >= 1.0
    if not np.any(mask):
        raise ValueError(
            'No test conditions with confidence >= 1.0; cannot compute wrMAE.')

    mean_abs_diff = np.mean(np.abs(y_blocks - yhat_blocks), axis=1)
    mean_abs_y    = np.maximum(np.mean(np.abs(y_blocks), axis=1), epsilon)
    relMAE        = mean_abs_diff / mean_abs_y

    candidates   = np.flatnonzero(mask)
    iworst_local = int(np.argmax(relMAE[mask]))
    iworst       = int(candidates[iworst_local])
    return iworst, float(relMAE[iworst])


def _residual_kl(y_true_case, y_pred_case, comp_masks, sigma_ref, n_bins=KL_N_BINS):
    """ KL(p_eps || N(0, sigma_ref)) of one simulation's residual
    distribution, pooled across wall points with each point weighted by
    its component weight (so wing/pylon points count more than
    fuselage/nacelle points in the same histogram). Ported from
    residual_kl_weighted() in copy.ipynb. """
    eps = y_pred_case - y_true_case
    sigma_y = float(y_true_case.std()) + epsilon

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


def compute_mean_KL(y, yhat, confidence_per_case, comp_masks, sigma_ref):
    """ Component-weighted residual KL-divergence, computed per simulation
    on high-confidence conditions only, then averaged across simulations.
    Lower is better; 0 means the residual distribution exactly matches the
    reference N(0, sigma_ref). Also returns the index/value of the single
    worst (highest-KL) simulation. """
    y_blocks    = y.reshape(-1, nwallp)
    yhat_blocks = yhat.reshape(-1, nwallp)

    valid_idx = np.flatnonzero(confidence_per_case >= 1.0)
    if valid_idx.size == 0:
        raise ValueError(
            'No test conditions with confidence >= 1.0; cannot compute mean_KL.')

    kl_values = np.array([
        _residual_kl(y_blocks[i], yhat_blocks[i], comp_masks, sigma_ref)
        for i in valid_idx
    ])
    iworst_local = int(np.argmax(kl_values))
    iworst       = int(valid_idx[iworst_local])
    return float(kl_values.mean()), iworst, float(kl_values[iworst_local])


def main():
    """ The scoring program. """
    print_bar()
    print('Scoring program - ONERA 468 CRM challenge rho.')

    scores = {'mean_KL': None, 'R2': 0.0, 'wrMAE': 1.0, 'score': 0.0}
    start_time = dt.now()

    try:
        os.makedirs(output_dir, exist_ok=True)
        write_file(html_file, '<h1>ONERA 468 CRM, Challenge &rho;</h1>\n', mode='w')

        print_bar()
        y_test, y_pred = get_data()
        confidence_per_case, X_conditions = get_confidence()
        comp_masks = get_components()
        validate_alignment(y_test, y_pred, confidence_per_case, comp_masks)

        # Reference-derived width for the target residual distribution
        # N(0, sigma_ref), used by the mean_KL metric.
        sigma_ref = max(0.01 * float(np.mean(y_test)), epsilon)

        print_bar()
        print('Computing R2.')
        R2 = compute_R2(y_test, y_pred, confidence_per_case)

        print('Computing wrMAE.')
        iworst_wrmae, wrMAE = compute_wrMAE(y_test, y_pred, confidence_per_case)

        print('Computing mean_KL.')
        mean_kl, iworst_kl, worst_kl = compute_mean_KL(
            y_test, y_pred, confidence_per_case, comp_masks, sigma_ref)

        score = 5.0 * R2 + 5.0 * (1.0 - wrMAE)

        scores['mean_KL'] = mean_kl
        scores['R2']      = R2
        scores['wrMAE']   = wrMAE
        scores['score']   = score

        worst_row_wrmae = X_conditions[iworst_wrmae]
        worst_row_kl    = X_conditions[iworst_kl]

        print_bar()
        print('mean_KL : {:.6f}'.format(mean_kl))
        print('R2      : {:.6f}'.format(R2))
        print('wrMAE   : {:.6f}'.format(wrMAE))
        print('score   : {:.6f}'.format(score))

        print('Worst wrMAE case: Pi={:.1f}e5 Pa  Minf={:.2f}  AoA={:.1f}°  (rMAE={:.6f})'.format(
            worst_row_wrmae[IDX_PI], worst_row_wrmae[IDX_MINF], worst_row_wrmae[IDX_AOA], wrMAE))
        print('Worst KL case   : Pi={:.1f}e5 Pa  Minf={:.2f}  AoA={:.1f}°  (KL={:.6f})'.format(
            worst_row_kl[IDX_PI], worst_row_kl[IDX_MINF], worst_row_kl[IDX_AOA], worst_kl))

        write_file(html_file, '<h2>Scores</h2>\n<ul>\n')
        write_file(html_file, '  <li>mean_KL : {:.6f}</li>\n'.format(mean_kl))
        write_file(html_file, '  <li>R2 : {:.6f}</li>\n'.format(R2))
        write_file(html_file, '  <li>wrMAE : {:.6f}</li>\n'.format(wrMAE))
        write_file(html_file, '  <li>score : {:.6f}</li>\n'.format(score))
        write_file(html_file, '</ul>\n')

        write_file(html_file, '<h2>Worst predicted condition (wrMAE={:.6f})</h2>\n<ul>\n'.format(wrMAE))
        write_file(html_file, '  <li>Pi = {:.1f}&times;10<sup>5</sup> Pa</li>\n'.format(worst_row_wrmae[IDX_PI]))
        write_file(html_file, '  <li>M<sub>&infin;</sub> = {:.2f}</li>\n'.format(worst_row_wrmae[IDX_MINF]))
        write_file(html_file, '  <li>AoA = {:.1f}&deg;</li>\n'.format(worst_row_wrmae[IDX_AOA]))
        write_file(html_file, '</ul>\n')

        write_file(html_file, '<h2>Worst predicted condition (KL={:.6f})</h2>\n<ul>\n'.format(worst_kl))
        write_file(html_file, '  <li>Pi = {:.1f}&times;10<sup>5</sup> Pa</li>\n'.format(worst_row_kl[IDX_PI]))
        write_file(html_file, '  <li>M<sub>&infin;</sub> = {:.2f}</li>\n'.format(worst_row_kl[IDX_MINF]))
        write_file(html_file, '  <li>AoA = {:.1f}&deg;</li>\n'.format(worst_row_kl[IDX_AOA]))
        write_file(html_file, '</ul>\n')

    except Exception as inst:
        # Catch-all: a broken submission or corrupt reference data must
        # never crash the scorer with a raw traceback. Log full details
        # for organizers, report cleanly, and fall back to worst-case
        # scores so the leaderboard/JSON stay well-formed.
        print('[ERR] Scoring failed: {}'.format(inst))
        traceback.print_exc()
        try:
            write_file(html_file, '<p style="color:red;">ERREUR : {}</p>\n'.format(inst))
        except Exception as e:
            print('[ERR] Could not write error to HTML report: {}'.format(e))
        scores['mean_KL'] = None
        scores['R2']      = 0.0
        scores['wrMAE']   = 1.0
        scores['score']   = 0.0

    scores['scoring_duration'] = (dt.now() - start_time).total_seconds()

    try:
        with open(os.path.join(prediction_dir, 'metadata.json')) as f:
            scores['duration'] = json.load(f).get('duration', -1)
    except Exception:
        scores['duration'] = -1

    print_bar()
    print('Scoring program finished. Writing scores.')
    print(scores)
    try:
        with open(score_file, 'w') as f:
            json.dump(scores, f, indent=4)
    except Exception as e:
        print('[ERR] Could not write score file: {}'.format(e))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Last-resort safety net: main() already catches its own errors,
        # this only guards against a bug in the error-handling path itself.
        print('[FATAL] Scoring program crashed: {}'.format(e))
        traceback.print_exc()
        sys.exit(1)
