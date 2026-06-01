import numpy as np
from lightgbm import LGBMRegressor

class Model:
    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).astype(np.float32)