"""
Field baseline: POD (PCA on the wall rho fields) + one independent Gaussian
Process per retained mode, mapping (Minf, AoA, Pi) -> mode coefficient.
Reconstruction is the exact linear POD decoder (mean_field + Z @ components).
Hyperparameters tuned via Optuna over this project's dev split -- see
utils/gp_pod_model.py (the plain pipeline they were re-verified against) in
the parent repo. Slow: N_MODES independent GPs per fold, so cv_predict here
is not meant to be run casually -- see section 2.3 of the notebook.
"""

import os

# Must happen before numpy/scipy/sklearn/joblib are imported anywhere -- BLAS thread counts are
# fixed at library load time. Without this, joblib.Parallel(n_jobs=-1)'s worker processes (one per
# POD mode) would each also try to multithread their own BLAS calls, oversubscribing the machine.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from .metrics import NWALLP, COL_MINF, COL_PI
from .data import mach_fold_splits, select_sims

N_MODES            = 55
KERNEL_NU          = 1.5
ALPHA              = 7.485874594219525e-05
LENGTH_SCALE_UPPER = 728.6705690631443
NOISE_LEVEL_UPPER  = 4.043220411002399
N_RESTARTS         = 5
SEED               = 0


def _make_kernel(n_dims):
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, LENGTH_SCALE_UPPER), nu=KERNEL_NU)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-12, NOISE_LEVEL_UPPER)))


def _fit_one_gp(conds_sc, z_col, n_dims):
    gp = GaussianProcessRegressor(
        kernel=_make_kernel(n_dims), alpha=ALPHA, n_restarts_optimizer=N_RESTARTS,
        normalize_y=True, random_state=SEED,
    )
    gp.fit(conds_sc, z_col)
    return gp


class Model:
    """ Matches the Codabench Model contract: fit(X, y) / predict(X). Field
    regressor -- receives the full pointwise arrays and reshapes internally
    (one row per simulation, condition columns 6:9), unlike the pointwise
    baselines above. """

    def __init__(self):
        self.scaler = StandardScaler()
        self.gps = None
        self.components = None
        self.mean_field = None
        self.k = None

    def fit(self, X, y):
        n_train = X.shape[0] // NWALLP
        train_conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1])
        Y_train = np.asarray(y).reshape(n_train, NWALLP)

        conds_sc = self.scaler.fit_transform(train_conds)

        n_components = min(N_MODES, n_train - 1)
        pca = PCA(n_components=n_components, random_state=SEED)
        Z_train = pca.fit_transform(Y_train)   # mean-subtracted internally by PCA
        self.components = pca.components_
        self.mean_field = pca.mean_
        self.k = n_components

        n_dims = conds_sc.shape[1]
        self.gps = Parallel(n_jobs=-1)(
            delayed(_fit_one_gp)(conds_sc, Z_train[:, m], n_dims)
            for m in range(self.k)
        )
        return self

    def predict(self, X):
        n_test = X.shape[0] // NWALLP
        test_conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1])
        conds_sc = self.scaler.transform(test_conds)

        Z_pred = np.column_stack([gp.predict(conds_sc) for gp in self.gps])
        y_pred = self.mean_field[None, :] + Z_pred @ self.components   # exact linear POD decoder
        return y_pred.reshape(-1)


def cv_predict(X_train, y_train, conds, nwallp=NWALLP):
    """ Out-of-fold predictions for every training simulation, via
    leave-two-consecutive-Machs-out CV. Pass an already-subsampled
    (X_train, y_train, conds) -- and expect it to still be slow, this is a
    heavy baseline (see module docstring). """
    y_cv_pred = np.zeros_like(y_train)
    for train_idx, val_idx, label in mach_fold_splits(conds):
        X_fit, y_fit = select_sims(X_train, y_train, train_idx, nwallp)
        X_val, _     = select_sims(X_train, y_train, val_idx, nwallp)

        model = Model()
        model.fit(X_fit, y_fit)
        y_val_pred = model.predict(X_val)

        for local_i, sim_i in enumerate(val_idx):
            start = sim_i * nwallp
            y_cv_pred[start:start + nwallp] = y_val_pred[local_i * nwallp:(local_i + 1) * nwallp]
        print(f'  fold {label}: {len(val_idx)} val sim(s) done')
    return y_cv_pred
