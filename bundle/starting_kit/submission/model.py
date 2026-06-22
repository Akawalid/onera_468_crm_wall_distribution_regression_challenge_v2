import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

COL_MINF, COL_AOA, COL_PI = 6, 7, 8

class Model:

    def __init__(self):
        self.knn    = KNeighborsRegressor(n_neighbors=5, algorithm='auto', n_jobs=-1)
        self.scaler = StandardScaler()
        self.nwallp = None

    def fit(self, X, y):
        cond0 = X[0, COL_MINF:COL_PI+1]
        self.nwallp = int(np.argmax(np.any(X[:, COL_MINF:COL_PI+1] != cond0, axis=1)))
        n_train = X.shape[0] // self.nwallp

        train_conds = X[::self.nwallp, COL_MINF:COL_PI+1]
        train_conds_scaled = self.scaler.fit_transform(train_conds)

        y_per_sim = y.reshape(n_train, self.nwallp)
        self.knn.fit(train_conds_scaled, y_per_sim)

    def predict(self, X):
        n_test = X.shape[0] // self.nwallp
        test_conds = X[::self.nwallp, COL_MINF:COL_PI+1]
        test_conds_scaled = self.scaler.transform(test_conds)
        y_pred = self.knn.predict(test_conds_scaled)
        return y_pred.reshape(-1)
