"""
IsoMap embedding + gradient-boosting decoder, replacing the kNN backmap in paper_isomap_rbf.py.
One of three backmap replacements tested to isolate whether IsoMap's embedding or its kNN
backmapping is the bigger factor behind that model's Phase 2 (extrapolation) weakness -- see
paper_isomap_rbf.py's module docstring for the kNN-can't-extrapolate reasoning this is testing.
The other two (ridge, GP) were cheap enough to run ad hoc locally:
  - Ridge decoder: worse than kNN on every metric, both phases (linear decoder mismatched to
    IsoMap's deliberately nonlinear embedding).
  - GP decoder: narrow win on Phase 1, clearly worse than kNN on Phase 2.
This script is the third: MultiOutputRegressor(HistGradientBoostingRegressor). Unlike Ridge/GP,
boosting can't share a single factorization across outputs (each of the NWALLP wall points needs
its own independently-fit tree ensemble) -- measured at ~28ms/output locally, i.e. ~124 minutes
for the real full field even on 12 cores, hence running this on the cluster instead of ad hoc.

Same IsoMap+RBF setup as paper_isomap_rbf.py (n_neighbors=15, n_components=3,
kernel='thin_plate_spline'), same official metrics (ported from scoring.py, via import).
"""

import os

# Must happen before numpy/scipy/sklearn are imported anywhere (including transitively via
# paper_isomap_rbf below) -- see tune_pod_gp.py's identical block for the full oversubscription
# reasoning: MultiOutputRegressor(n_jobs=-1) spawns one joblib worker process per output-batch,
# and each worker's own BLAS calls would ALSO try to multithread across every core on the node.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import sys
import time

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.manifold import Isomap
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_isomap_rbf as pir

MAX_ITER = 200       # matches the local timing probe (28ms/output at this setting)
N_JOBS = -1
SEED = 0

OUT_NAME = 'isomap_boosting_decoder'


def main():
    out_prefix = os.path.join(pir.DATA_DIR, OUT_NAME)   # iceberg/shared drive, not utils/

    print('Loading data...', flush=True)
    data = pir.load_split(pir.DATA_DIR, pir.SPLIT_DIR)
    train_conds, Y_train, n_train = data['train_conds'], data['Y_train'], data['n_train']
    print(f'n_train={n_train}  n_test1={data["n_test1"]}  n_test2={data["n_test2"]}', flush=True)

    cond_scaler = StandardScaler()
    train_conds_sc = cond_scaler.fit_transform(train_conds)
    test1_conds_sc = cond_scaler.transform(data['test1_conds'])
    test2_conds_sc = cond_scaler.transform(data['test2_conds'])

    print(f'Fitting IsoMap (n_neighbors={pir.ISOMAP_K}, n_components={pir.N_COMPONENTS})...', flush=True)
    isomap = Isomap(n_neighbors=pir.ISOMAP_K, n_components=pir.N_COMPONENTS)
    Z_train = isomap.fit_transform(Y_train)

    print(f'Fitting RBFInterpolator (kernel={pir.RBF_KERNEL}) for p -> z...', flush=True)
    rbf = RBFInterpolator(train_conds_sc, Z_train, kernel=pir.RBF_KERNEL)
    Z_pred1 = rbf(test1_conds_sc)
    Z_pred2 = rbf(test2_conds_sc)

    print(f'Fitting MultiOutputRegressor(HistGradientBoostingRegressor(max_iter={MAX_ITER})) '
          f'decoder z -> y ({Y_train.shape[1]} independent models, n_jobs={N_JOBS})...', flush=True)
    t0 = time.time()
    decoder = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=MAX_ITER, random_state=SEED),
        n_jobs=N_JOBS,
    )
    decoder.fit(Z_train, Y_train)
    print(f'  fit time: {time.time() - t0:.1f}s', flush=True)

    print('Predicting phase 1...', flush=True)
    y_pred1 = decoder.predict(Z_pred1).reshape(-1)
    print('Predicting phase 2...', flush=True)
    y_pred2 = decoder.predict(Z_pred2).reshape(-1)

    # Metrics first, saving last -- same fix as the other paper_*.py scripts: a save failure must
    # never cost results that are already computed.
    res1 = pir.evaluate_phase(data['y_test1'], y_pred1, data['w_test1'], data['comp_masks'])
    res2 = pir.evaluate_phase(data['y_test2'], y_pred2, data['w_test2'], data['comp_masks'])
    pir.print_result('IsoMap+Boosting (kNN backmap replaced)', 'Phase 1 (interpolation)', res1)
    pir.print_result('IsoMap+Boosting (kNN backmap replaced)', 'Phase 2 (extrapolation)', res2)

    try:
        np.save(f'{out_prefix}_y_pred1.npy', y_pred1)
        np.save(f'{out_prefix}_y_pred2.npy', y_pred2)
        print(f'\nSaved: {out_prefix}_y_pred1.npy, {out_prefix}_y_pred2.npy', flush=True)
    except OSError as e:
        print(f'\n[WARN] Could not save predictions: {e}', flush=True)


if __name__ == '__main__':
    main()
