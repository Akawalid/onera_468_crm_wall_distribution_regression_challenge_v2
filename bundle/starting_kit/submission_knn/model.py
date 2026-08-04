import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8

N_NEIGHBORS = 5


class Model:
    """ Input-space kNN baseline, same as the one in copy.ipynb: a single KNeighborsRegressor
    fit on scaled (Minf, AoA, Pi) condition vectors (one row per simulation), predicting the
    entire wall rho field directly -- sklearn's KNeighborsRegressor supports multi-output y
    natively, so no per-point looping is needed.

    fit()/predict() receive the full pointwise [n_sims*NWALLP, 9] arrays Codabench's ingestion
    program passes -- conditions are extracted internally (one row per simulation, columns 6:9)
    since this is a field regressor, not a pointwise one.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS, algorithm='auto', n_jobs=-1)

    def fit(self, X, y):
        n_train = X.shape[0] // NWALLP
        train_conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1])
        Y_train = np.asarray(y).reshape(n_train, NWALLP)

        conds_sc = self.scaler.fit_transform(train_conds)
        self.knn.fit(conds_sc, Y_train)
        return self

    def predict(self, X):
        test_conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1])
        conds_sc = self.scaler.transform(test_conds)
        y_pred = self.knn.predict(conds_sc)
        return y_pred.reshape(-1)
