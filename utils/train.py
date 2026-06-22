import numpy as np
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor


# ============================================================
#  Models
# ============================================================

class LGBMModel:
    name = "LightGBM"

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


class KNNModel:
    name = "KNN (k=10)"

    def __init__(self):
        self.model = KNeighborsRegressor(
            n_neighbors=10,
            weights="distance",
            n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).astype(np.float32)


class RidgeModel:
    name = "Ridge Regression"

    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).astype(np.float32)


class MeanBaseline:
    name = "Mean Baseline"

    def __init__(self):
        self.model = DummyRegressor(strategy="mean")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).astype(np.float32)


# ============================================================
#  Metrics
# ============================================================

def compute_R2(y, yhat, confidence_pointwise):
    ymean = np.mean(y)
    SSE   = np.sum(confidence_pointwise * (y - yhat) ** 2)
    SSD   = np.sum(confidence_pointwise * (y - ymean) ** 2)
    return float(1.0 - SSE / SSD)


def compute_wrMAE(y, yhat, confidence_per_case):
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


def print_scores(model_name, r2, wmae):
    worst_case_idx, worst_case_val = wmae
    print(f"  {'Model':<22}: {model_name}")
    print(f"  {'R2':<22}: {r2:.6f}")
    print(f"  {'wrMAE':<22}: {worst_case_val:.6f}  (worst case idx: {worst_case_idx})")
    print()


def evaluate_all(models, trainx, trainy, testx, testy, testw):
    testw_pointwise = np.repeat(testw, nwallp)
    trainy_flat     = trainy.reshape(-1)
    testy_flat      = testy.reshape(-1)

    results = []
    for model in models:
        print(f"  [Training]  {model.name} ...")
        model.fit(trainx, trainy_flat)

        print(f"  [Scoring]   {model.name} ...")
        ypred = model.predict(testx)
        r2    = compute_R2(testy_flat, ypred, testw_pointwise)
        wmae  = compute_wrMAE(testy_flat, ypred, testw)
        results.append((model.name, r2, wmae))
        print()

    return results


def print_summary(results, phase_label):
    w = 60
    print("=" * w)
    print(f"  SUMMARY {phase_label}")
    print("=" * w)
    print(f"  {'Model':<24} {'R2':>10}  {'wrMAE':>12}  {'Worst idx':>10}")
    print("-" * w)
    for name, r2, (worst_idx, worst_val) in results:
        print(f"  {name:<24} {r2:>10.6f}  {worst_val:>12.6f}  {worst_idx:>10}")
    print("=" * w)
    print()


# ============================================================
#  Main
# ============================================================

nwallp  = 260774
epsilon = 1.e-6

path = "/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/"

models = [
    LGBMModel(),
    # KNNModel(),
    # RidgeModel(),
    # MeanBaseline(),
]

print("Loading training data ...")
trainx = np.load(path + "train_data.npy")
trainy = np.load(path + "train_labels.npy")

# ── Phase 1 ──────────────────────────────────────────────────
print()
print("=" * 60)
print("  PHASE 1")
print("=" * 60)

testx = np.load(path + "test_phase1_data.npy")
testy = np.load(path + "test_phase1_labels.npy")
testw = np.load(path + "test_phase1_weights.npy")

results_p1 = evaluate_all(models, trainx, trainy, testx, testy, testw)
print_summary(results_p1, "Phase 1")

# ── Phase 2 ──────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 2")
print("=" * 60)

testx = np.load(path + "test_phase2_data.npy")
testy = np.load(path + "test_phase2_labels.npy")
testw = np.load(path + "test_phase2_weights.npy")

results_p2 = evaluate_all(models, trainx, trainy, testx, testy, testw)
print_summary(results_p2, "Phase 2")