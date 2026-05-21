import json
import os
import numpy as np
import pandas as pd

input_dir      = '/app/input'
output_dir     = '/app/output/'
# input_dir  = '/tmp/test_bundle/input'
# output_dir = '/tmp/test_bundle/output/'

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
    y_test = np.load(os.path.join(reference_dir, 'rho_test_fl32.npy'))[:, 0]
    y_pred = np.load(os.path.join(prediction_dir, 'Yhat.npy'))[:, 0]
    return y_test, y_pred


def get_confidence():
    """ Get confidence score per test condition from the CSV. """
    df = pd.read_csv(
        os.path.join(reference_dir, 'fullfiles_PiMinfAoA_with_scores.csv')
    )
    df_test = df.loc[~df['Train']]
    confidence_per_case = df_test['confidence_score_simple_4'].values
    confidence_pointwise   = np.repeat(confidence_per_case, nwallp)
    return confidence_per_case, confidence_pointwise, df_test.reset_index(drop=True)


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

    write_file(html_file, '<h1>ONERA 468 CRM, Challenge &rho;</h1>\n')

    scores = {}

    try:
        print_bar()
        print('Reading data.')
        y_test, y_pred = get_data()

        if y_test.shape != y_pred.shape:
            raise ValueError(
                'Shape mismatch: prediction {} != reference {}.'.format(
                    y_pred.shape, y_test.shape))

        print('Reading confidence weights.')
        confidence_per_case, confidence_pointwise, df_test = get_confidence()

        print('Computing R2.')
        R2 = compute_R2(y_test, y_pred, confidence_pointwise)

        print('Computing wrMAE.')
        iworst, wrMAE = compute_wrMAE(y_test, y_pred, confidence_per_case)

        score = 5.0 * R2 + 5.0 * (1.0 - wrMAE)

        scores['R2']    = R2
        scores['wrMAE'] = wrMAE
        scores['score'] = score

        worst_row = df_test.iloc[iworst]

        print_bar()
        print('R2    : {:.6f}'.format(R2))
        print('wrMAE : {:.6f}'.format(wrMAE))
        print('score : {:.6f}'.format(score))
        print('Worst case: Pi={:.1f}e5 Pa  Minf={:.2f}  AoA={:.1f}°'.format(
            worst_row['Pi'], worst_row['Mach'], worst_row['AoA']))

        write_file(html_file, '<h2>Scores</h2>\n<ul>\n')
        write_file(html_file, '  <li>R2 : {:.6f}</li>\n'.format(R2))
        write_file(html_file, '  <li>wrMAE : {:.6f}</li>\n'.format(wrMAE))
        write_file(html_file, '  <li>score : {:.6f}</li>\n'.format(score))
        write_file(html_file, '</ul>\n')
        write_file(html_file, '<h2>Worst predicted condition (wrMAE)</h2>\n<ul>\n')
        write_file(html_file, '  <li>Pi = {:.1f}&times;10<sup>5</sup> Pa</li>\n'.format(worst_row['Pi']))
        write_file(html_file, '  <li>M<sub>&infin;</sub> = {:.2f}</li>\n'.format(worst_row['Mach']))
        write_file(html_file, '  <li>AoA = {:.1f}&deg;</li>\n'.format(worst_row['AoA']))
        write_file(html_file, '</ul>\n')

    except Exception as inst:
        print('ERREUR scoring : {}'.format(inst))
        write_file(html_file, '<p style="color:red;">ERREUR : {}</p>\n'.format(inst))
        scores['R2']    = 0.0
        scores['wrMAE'] = 1.0
        scores['score'] = 0.0

    try:
        with open(os.path.join(prediction_dir, 'metadata.json')) as f:
            scores['duration'] = json.load(f).get('duration', -1)
    except Exception:
        scores['duration'] = -1

    print_bar()
    print('Scoring program finished. Writing scores.')
    print(scores)
    write_file(score_file, json.dumps(scores))


if __name__ == '__main__':
    main()