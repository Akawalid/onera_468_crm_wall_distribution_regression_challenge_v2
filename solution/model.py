import numpy as np
from sklearn.linear_model import Ridge

class Model:
    def __init__(self):
        self.regressor = Ridge(alpha=1.0)

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)