# This is a sample code submission.
# It is a simple machine learning regressor.

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class Model:
    def __init__(self):
        """ <ADD DOCUMENTATION HERE>
        """
        self.regressor = KNeighborsRegressor()
    def fit(self, X, y):
        """ Train the model.

        Args:
            X: Training data matrix of shape (num-samples, num-features), type np.ndarray.
            y: Training label vector of shape (num-samples), type np.ndarray.
        """
        self.regressor.fit(X, y)
    def predict(self, X):
        """ Predict values.
        
        Args:
          X: Data matrix of shape (num-samples, num-features) to pass to the model for inference, type np.ndarray.
        """
        y = self.regressor.predict(X)
        return y