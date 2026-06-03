import json
import os
# import sys
import numpy as np
import pandas as pd
from datetime import datetime as dt

input_dir      = '/app/input'
output_dir     = '/app/output/'

# input_dir      = '/tmp/test_bundle/input_data/'
# output_dir     = '/tmp/test_bundle/output/'

reference_dir  = os.path.join(input_dir, 'ref')
prediction_dir = os.path.join(input_dir, 'res')
score_file     = os.path.join(output_dir, 'scores.json')
html_file      = os.path.join(output_dir, 'detailed_results.html')

nwallp  = 260774
ntest   = 156 * nwallp
epsilon = 1.e-6


def write_file(file, content):
    """ Write content in file. """
    with open(file, 'a', encoding='utf-8') as f:
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
    print('[*] Reading confidence weights from {}'.format(reference_dir))
    try:
        confidence_per_case  = np.load(os.path.join(reference_dir, 'test_weights.npy'))
        confidence_pointwise = np.repeat(confidence_per_case, nwallp)
        print('[+] Loaded confidence for {} test conditions.'.format(len(confidence_per_case)))
    except Exception as e:
        print('[-] Error loading confidence data: {}'.format(e))
        raise

    print('[*] Reading test data from {}'.format(reference_dir))
    try:
        X_test = np.load(os.path.join(reference_dir, 'test_data.npy'))
        X_conditions = X_test[::nwallp]
        print('[+] Loaded {} test conditions.'.format(len(X_conditions)))
    except Exception as e:
        print('[-] Error loading test data: {}'.format(e))
        raise

    return confidence_per_case, confidence_pointwise, X_conditions


def compute_R2(y, yhat, confidence_pointwise):
    """ Weighted R^2 score. """
    ymean = np.mean(y)
    SSE   = np.sum(confidence_pointwise * (y - yhat) ** 2)
    SSD   = np.sum(confidence_pointwise * (y - ymean) ** 2)
    return float(1.0 - SSE / SSD)


def compute_wrMAE(y, yhat, confidence_per_case):
    """ Worst-case relative MAE on high-confidence conditions only. """
    ncasetest   = len(confidence_per_case)
    relMAE_list = []
    idx_list    = []

    for l in range(ncasetest):
        if confidence_per_case[l] < 1.0:
            continue
        ycase    = y   [l * nwallp:(l + 1) * nwallp]
        yhatcase = yhat[l * nwallp:(l + 1) * nwallp]
        diff     = np.abs(ycase - yhatcase)
        relMAE_list.append(np.mean(diff) / np.mean(np.abs(ycase)))
        idx_list.append(l)

    relMAE_arr   = np.array(relMAE_list)
    iworst_local = int(np.argmax(relMAE_arr))
    return idx_list[iworst_local], float(relMAE_arr[iworst_local])


def main():
    """ The scoring program. """
    print_bar()
    print('Scoring program - ONERA 468 CRM challenge rho.')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    write_file(html_file, '<h1>ONERA 468 CRM, Challenge &rho;</h1>\n')

    scores = {}
    start_time = dt.now()

    try:
        print_bar()
        y_test, y_pred = get_data()

        # assert that y_test.shape==confidence_pointwise

        if y_test.shape != y_pred.shape:
            raise ValueError(
                'Shape mismatch: prediction {} != reference {}.'.format(
                    y_pred.shape, y_test.shape))

        confidence_per_case, confidence_pointwise, X_conditions = get_confidence()
        
        print_bar()
        print('Computing R2.')
        R2 = compute_R2(y_test, y_pred, confidence_pointwise)

        print('Computing wrMAE.')
        iworst, wrMAE = compute_wrMAE(y_test, y_pred, confidence_per_case)

        score = 5.0 * R2 + 5.0 * (1.0 - wrMAE)

        scores['R2']    = R2
        scores['wrMAE'] = wrMAE
        scores['score'] = score

        worst_row = X_conditions[iworst]

        print_bar()
        print('R2    : {:.6f}'.format(R2))
        print('wrMAE : {:.6f}'.format(wrMAE))
        print('score : {:.6f}'.format(score))

        print('Worst case: Pi={:.1f}e5 Pa  Minf={:.2f}  AoA={:.1f}°'.format(
            worst_row[8], worst_row[6], worst_row[7]))

        write_file(html_file, '<h2>Scores</h2>\n<ul>\n')
        write_file(html_file, '  <li>R2 : {:.6f}</li>\n'.format(R2))
        write_file(html_file, '  <li>wrMAE : {:.6f}</li>\n'.format(wrMAE))
        write_file(html_file, '  <li>score : {:.6f}</li>\n'.format(score))
        write_file(html_file, '</ul>\n')
        write_file(html_file, '<h2>Worst predicted condition (wrMAE)</h2>\n<ul>\n')
        write_file(html_file, '  <li>Pi = {:.1f}&times;10<sup>5</sup> Pa</li>\n'.format(worst_row[8]))
        write_file(html_file, '  <li>M<sub>&infin;</sub> = {:.2f}</li>\n'.format(worst_row[6]))
        write_file(html_file, '  <li>AoA = {:.1f}&deg;</li>\n'.format(worst_row[7]))
        write_file(html_file, '</ul>\n')

    except Exception as inst:
        print('ERREUR scoring : {}'.format(inst))
        write_file(html_file, '<p style="color:red;">ERREUR : {}</p>\n'.format(inst))
        scores['R2']    = 0.0
        scores['wrMAE'] = 1.0
        scores['score'] = 0.0

    end_time  = dt.now()
    duration  = (end_time - start_time).total_seconds()
    scores['scoring_duration'] = duration

    try:
        with open(os.path.join(prediction_dir, 'metadata.json')) as f:
            scores['duration'] = json.load(f).get('duration', -1)
    except Exception:
        scores['duration'] = -1

    print_bar()
    print('Scoring program finished. Writing scores.')
    print(scores)
    with open(score_file, 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == '__main__':
    main()