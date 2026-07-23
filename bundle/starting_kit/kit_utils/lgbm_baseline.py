"""
Simple baseline: a pointwise LightGBM regressor.

Unlike the full-field models (KNN, MLP -- see mlp_klw.py), this one treats
every (wall point, condition) pair as an independent training row: the 9
input columns are (x, y, z, nx, ny, nz, Minf, AoA, Pi), the target is the
local rho at that point. No need to know nwallp or reshape by simulation
to *train* it -- that's what makes it the simplest model to understand
here. `nwallp` is only inferred at fit-time to (a) subsample points per
simulation for speed and (b) let predict() work on any array laid out the
same way (Codabench convention: rows grouped by simulation).
"""

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from .metrics import NWALLP, COL_MINF, COL_PI
from .data import mach_fold_splits, select_sims


def _infer_nwallp(X):
    cond0 = X[0, COL_MINF:COL_PI + 1]
    return int(np.argmax(np.any(X[:, COL_MINF:COL_PI + 1] != cond0, axis=1)))


def _subsample_idx(n_rows, nwallp, stride):
    """ Keep every `stride`-th point *within each simulation block* --
    training on a sparser version of the surface is enough to fit the
    trend, and cuts LightGBM's training time roughly by `stride`. """
    n_sims = n_rows // nwallp
    block_idx = np.arange(0, nwallp, stride)
    return (np.arange(n_sims)[:, None] * nwallp + block_idx[None, :]).reshape(-1)


class Model:
    """ Matches the Codabench Model contract: fit(X, y) / predict(X). """

    def __init__(self, point_stride=8, n_estimators=200):
        self.point_stride = point_stride
        self.scaler = StandardScaler()
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=0,
            verbose=-1,
        )
        self.nwallp = None

    def fit(self, X, y):
        self.nwallp = _infer_nwallp(X)
        idx = _subsample_idx(X.shape[0], self.nwallp, self.point_stride)
        X_sc = self.scaler.fit_transform(X[idx])
        self.model.fit(X_sc, y[idx])
        return self

    def predict(self, X):
        X_sc = self.scaler.transform(X)
        return self.model.predict(X_sc)


def cv_predict(X_train, y_train, conds, nwallp=NWALLP, point_stride=8, n_estimators=200):
    """ Out-of-fold predictions for every training simulation, via
    leave-two-consecutive-Machs-out CV (see data.mach_fold_splits). """
    y_cv_pred = np.zeros_like(y_train)
    for train_idx, val_idx, label in mach_fold_splits(conds):
        X_fit, y_fit = select_sims(X_train, y_train, train_idx, nwallp)
        X_val, _     = select_sims(X_train, y_train, val_idx, nwallp)

        model = Model(point_stride=point_stride, n_estimators=n_estimators)
        model.fit(X_fit, y_fit)
        y_val_pred = model.predict(X_val)

        for local_i, sim_i in enumerate(val_idx):
            start = sim_i * nwallp
            y_cv_pred[start:start + nwallp] = y_val_pred[local_i * nwallp:(local_i + 1) * nwallp]
        print(f'  fold {label}: {len(val_idx)} val sim(s) done')
    return y_cv_pred
