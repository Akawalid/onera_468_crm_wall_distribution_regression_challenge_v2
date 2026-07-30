import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

N_ESTIMATORS, NUM_LEAVES, MAX_DEPTH = 500, 255, 8
LR, SUBSAMPLE, COLSAMPLE = 0.05, 0.7, 0.8


class Model:

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
