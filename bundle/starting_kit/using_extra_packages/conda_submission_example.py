# --------------------------------------------------------------------------
# This submission file shows how to include extra Python packages in your submission.
# Here we use LightGBM, which is not available by default in the Codabench environment.
#
# To prepare your submission:
# 1) Create a fresh conda environment from the environment.yml file placed at the root of the starting_kit:
#    conda env create -f environment.yml
# 2) Activate the environment:
#    conda activate codabench-env
# 3) Create a folder named `python_packages` in the same directory as this file
# 4) Install LightGBM into that folder:
#    pip install --target python_packages lightgbm
# 5) Edit this file with your model
# 6) Compress your submission into a zip file (the python_packages folder will be included):
#    cd PATH_TO_YOUR_FOLDER && zip -r ../submission.zip .
# 7) Submit the zip file in the "My Submissions" tab on Codabench
# --------------------------------------------------------------------------

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent / "python_packages"))
import lightgbm as lgb


class Model:
    def __init__(self):
        """
        LightGBM regressor for volumetric density (rho) prediction.
        Predicts rho at each point from (x, y, z, nx, ny, nz, Minf, AoA, pi).
        """
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            n_jobs=-1
        )

    def fit(self, X, y):
        """
        Train the model.
        Args:
            X: Training data matrix of shape (num_samples, 9), type np.ndarray.
               Columns: x, y, z, nx, ny, nz, Minf, AoA, pi
            y: Training target vector of shape (num_samples,), type np.ndarray.
               Values: adimensional volumetric density rho
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict rho values.
        Args:
            X: Data matrix of shape (num_samples, 9), type np.ndarray.
        Returns:
            y: Predicted rho vector of shape (num_samples,), type np.ndarray.
        """
        return self.model.predict(X)