import os

# Must happen before numpy/scipy/sklearn/joblib are imported anywhere -- BLAS thread counts are
# fixed at library load time. Without this, joblib.Parallel(n_jobs=-1)'s worker PROCESSES (one per
# POD mode, see _fit_one_gp below) would ALSO each try to multithread their own BLAS calls, which
# oversubscribes whatever CPU allocation the Codabench worker has. Pin each worker to 1 BLAS
# thread and let joblib alone own the parallelism (same fix as this project's
# utils/tune_pod_gp.py).
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8

# Hyperparameters found via Optuna (leave-2-consecutive-Machs-out CV, 181 trials, CV
# mean_KL=2.28673) over this project's own dev split, then independently re-verified by refitting
# through the plain (non-Optuna) POD+GP pipeline and confirming both agree.
N_MODES = 55
KERNEL_NU = 1.5
ALPHA = 7.485874594219525e-05
LENGTH_SCALE_UPPER = 728.6705690631443
NOISE_LEVEL_UPPER = 4.043220411002399
N_RESTARTS = 5
SEED = 0


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
    """ POD + Gaussian Process: reduces each simulation's wall rho field to N_MODES POD
    coefficients (mean-subtracted PCA), then fits one independent ARD Matern GP per mode mapping
    (Minf, AoA, Pi) -> coefficient. Reconstruction is the exact linear POD decoder
    (mean_field + Z @ components), no neighbor-averaging/interpolation involved.

    fit()/predict() receive the full pointwise [n_sims*NWALLP, 9] arrays Codabench's ingestion
    program passes (see bundle/pages/extra_packages.md) -- conditions are extracted internally
    (one row per simulation, columns 6:9) since this is a field regressor, not a pointwise one.
    """

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
