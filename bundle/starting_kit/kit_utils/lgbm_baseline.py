"""
Pointwise baseline: a single LightGBM regressor fit on all 9 input columns
(geometry + flow conditions), predicting the density at each row directly.
Same model and hyperparameters as utils/base_models.py's "lightgbm" baseline
-- no subsampling here, so this is exactly what gets trained on the full
data when Codabench calls fit() on a real submission. For a quick local
cross-validation run, subsample simulations *before* calling cv_predict
(see kit_utils/data.py's subsample_sims_per_mach and section 2 of the
notebook), rather than shrinking the model itself.
"""

import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from .metrics import NWALLP
from .data import mach_fold_splits, select_sims

N_ESTIMATORS = 500
NUM_LEAVES   = 255
MAX_DEPTH    = 8
LR           = 0.05
SUBSAMPLE    = 0.7
COLSAMPLE    = 0.8


class Model:
    """ Matches the Codabench Model contract: fit(X, y) / predict(X). """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES, max_depth=MAX_DEPTH,
            learning_rate=LR, subsample=SUBSAMPLE, colsample_bytree=COLSAMPLE,
            n_jobs=-1, random_state=0, verbose=-1,
        )

    def fit(self, X, y):
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, y)
        return self

    def predict(self, X):
        X_sc = self.scaler.transform(X)
        return self.model.predict(X_sc)


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
