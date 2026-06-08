import sys
from pathlib import Path

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
            n_jobs=-1, 
            random_state=42
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
    
import numpy as np
nwallp = 260774

def compute_R2(y, yhat, confidence_pointwise):
    """ Weighted R^2 score. """
    ymean = np.mean(y)
    SSE   = np.sum(confidence_pointwise * (y - yhat) ** 2)
    SSD   = np.sum(confidence_pointwise * (y - ymean) ** 2)
    return float(1.0 - SSE / SSD)


def compute_wrMAE(y, yhat, confidence_per_case):
    """ Worst-case relative MAE on high-confidence conditions only. """
    ncasetest   = len(confidence_per_case)
    relMAE_list = []
    idx_list    = []

    for l in range(ncasetest):
        if confidence_per_case[l] < 1.0:
            continue
        ycase    = y   [l * nwallp:(l + 1) * nwallp]
        yhatcase = yhat[l * nwallp:(l + 1) * nwallp]
        diff     = np.abs(ycase - yhatcase)
        relMAE_list.append(np.mean(diff) / np.mean(np.abs(ycase)))
        idx_list.append(l)

    relMAE_arr   = np.array(relMAE_list)
    iworst_local = int(np.argmax(relMAE_arr))
    return idx_list[iworst_local], float(relMAE_arr[iworst_local])


x = np.load("final_phase/input_data/train_data.npy")
y  = np.load("final_phase/input_data/train_labels.npy")[:, 0]
# tw = np.load("final/phase/input_data/train_weights.npy")
sx = np.load("final_phase/input_data/test_data.npy")
sy = np.load("final_phase/ref/test_labels.npy")[:, 0]  # flatten comme dans scoring.py
sw = np.load("final_phase/ref/test_weights.npy")

m = Model()
m.fit(x, y)
yhat = m.predict(sx)

r2 = compute_R2(sy, yhat, np.repeat(sw, nwallp))
wmae = compute_wrMAE(sy, yhat, sw)

print(f"R2: {r2}, wmae: {wmae}")