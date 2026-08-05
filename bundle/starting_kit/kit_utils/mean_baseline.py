"""
Pointwise baseline: ignores all 9 input columns (geometry + flow conditions)
entirely and predicts the global training mean of rho for every row. The
simplest possible baseline -- it exists as a sanity floor, not something to
tune. For a quick local cross-validation run, subsample simulations *before*
calling cv_predict (see kit_utils/data.py's subsample_sims_per_mach and
section 2 of the notebook), rather than shrinking the model itself.
"""

import numpy as np

from .metrics import NWALLP
from .data import mach_fold_splits, select_sims


class Model:
    """ Matches the Codabench Model contract: fit(X, y) / predict(X). """

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def cv_predict(X_train, y_train, conds, nwallp=NWALLP):
    """ Out-of-fold predictions for every training simulation, via
    leave-two-consecutive-Machs-out CV. Pass an already-subsampled
    (X_train, y_train, conds) to keep this fast. """
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
