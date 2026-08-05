import numpy as np


class Model:

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)
