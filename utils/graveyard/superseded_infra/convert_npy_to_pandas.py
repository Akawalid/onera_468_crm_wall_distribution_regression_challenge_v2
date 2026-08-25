import numpy as np
import pandas as pd


path="/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/"
nwallp = 260774

X = np.load(path + "train_data.npy")
y = np.load(path + "train_labels.npy")
y = y.reshape(-1)
w = np.load(path + "train_weights.npy")
w = np.repeat(w, nwallp)

# X = np.load(path + "test_phase1_data.npy")
# y = np.load(path + "test_phase1_labels.npy")
# y = y.reshape(-1)
# w = np.load(path + "test_phase1_weights.npy")
# w = np.repeat(w, nwallp)

# X = np.load(path + "test_phase2_data.npy")
# y = np.load(path + "test_phase2_labels.npy")
# y = y.reshape(-1)
# w = np.load(path + "test_phase2_weights.npy")
# w = np.repeat(w, nwallp)

# Feature names for X
feature_names = ["x", "y", "z", "nx", "ny", "nz", "minf", "aoa", "pi"]

# Build the DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["weight"] = w
df["rho"] = y

print(df.shape)
print(df.head())

df.to_parquet(path + "X_train.parquet", index=False)