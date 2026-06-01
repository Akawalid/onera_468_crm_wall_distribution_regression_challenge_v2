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
    
nwallp  = 260774
ntest   = 156 * nwallp
epsilon = 1.e-6


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

path="/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/"
model = Model()

trainx = np.load(path + 'train_data.npy')
trainy = np.load(path + 'train_labels.npy')
testx = np.load(path + 'test_phase1_data.npy')
testy = np.load(path +  'test_phase1_labels.npy')
testw = np.load(path +  'test_phase1_weights.npy')

print(testw.shape, testw[:10], np.unique(testw))

#========= we need this for R2 !!!!!!!!!!!!!!!!!!!!!!!!!!!! but is it really the repeat function that should be used?
# testw = np.repeat(testw, nwallp)

trainy = trainy.reshape(-1)
testy = testy.reshape(-1)

print("aaaaaaaaaaaaaaaaaaaaaaaaaa", type(trainx), type(testx))

print("=============== begin training ===============")

model.fit(trainx, trainy)
ypred = model.predict(testx)

print(f"trainx.shape: {trainx.shape}, testx.shape: {testx.shape}, testw.shape: {testw.shape},"\
       f"ypred.shape: {ypred.shape}, testy.shape: {testy.shape}")

print("================ begin scoring ================")

wmae = compute_wrMAE(testy, ypred, testw)
#r2 = compute_R2(testy, ypred, testw)

print(f"r2: {0}, wmae: {wmae}")

# #===========phase 2
# testx = np.load(path + 'test_phase2_data.npy')
# testy = np.load(path +  'test_phase2_labels.npy')
# testw = np.load(path +  'test_phase2_weights.npy')

# model.fit(trainx, trainy)
# ypred = model.predict(testx)

# r2 = compute_R2(testy, ypred, testw)
# wmae = compute_wrMAE(testy, ypred, testw)

# print(f"r2: {r2}, wmae: {wmae}")