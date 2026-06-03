import numpy as np
from sklearn.linear_model import Ridge

class Model:
    def __init__(self):
        """
        Ridge regression for volumetric density (rho) prediction.
        Predicts rho at each point from (x, y, z, nx, ny, nz, Minf, AoA, pi).
        """
        self.regressor = Ridge(alpha=1.0)

    def fit(self, X, y):
        """
        Train the model.
        Args:
            X: Training data matrix of shape (num_samples, 9), type np.ndarray.
               Columns: x, y, z, nx, ny, nz, Minf, AoA, pi
            y: Training target vector of shape (num_samples,), type np.ndarray.
               Values: adimensional volumetric density rho
        """
        self.regressor.fit(X, y)

    def predict(self, X):
        """
        Predict rho values.
        Args:
            X: Data matrix of shape (num_samples, 9), type np.ndarray.
        Returns:
            y: Predicted rho vector of shape (num_samples,), type np.ndarray.
        """
        return self.regressor.predict(X)