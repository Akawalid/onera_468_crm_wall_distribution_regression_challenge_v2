import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
EPS = 1e-6

N_COMPONENTS = 3          # r: paper finds this matches the 3 physical flow parameters
ISOMAP_K = 15              # IsoMap's own neighbor-graph size (not stated in the paper); smallest
                            # value found to keep the neighbor graph fully connected on this
                            # project's training set
RBF_KERNEL = 'thin_plate_spline'
K_CANDIDATES = range(2, 13)   # paper's exact candidate range for the backmapping k
N_REPEATS = 20
N_HOLDOUT = 10             # paper: "repeatedly removing only ten randomly selected wall distributions"
SEED = 0


def _knn_backmap(Z_query, Z_pool, Y_pool, k):
    """ Inverse-distance-weighted kNN in latent space -- paper's Eq. 5 (input-space kNN, section
    5.2), applied to z-space neighbors instead of p-space neighbors since IsoMap has no natural
    backmapping (section 5.4). """
    k = min(k, len(Z_pool))
    nn = NearestNeighbors(n_neighbors=k).fit(Z_pool)
    dist, idx = nn.kneighbors(Z_query)
    dist = np.maximum(dist, 1e-12)
    w = 1.0 / dist
    w /= w.sum(axis=1, keepdims=True)
    y_pred = np.empty((Z_query.shape[0], Y_pool.shape[1]), dtype=np.float64)
    for i in range(Z_query.shape[0]):
        y_pred[i] = w[i] @ Y_pool[idx[i]]
    return y_pred


def _compute_R2(y, yhat, confidence_per_case):
    y_blocks = y.reshape(-1, NWALLP)
    yhat_blocks = yhat.reshape(-1, NWALLP)
    ymean = np.mean(y)
    sq_err = ((y_blocks - yhat_blocks) ** 2).sum(axis=1)
    sq_dev = ((y_blocks - ymean) ** 2).sum(axis=1)
    SSE = float(np.dot(confidence_per_case, sq_err))
    SSD = float(np.dot(confidence_per_case, sq_dev))
    if SSD < EPS:
        return 0.0
    return 1.0 - SSE / SSD


def _select_backmap_k(Z_train, Y_train, n_repeats=N_REPEATS, n_holdout=N_HOLDOUT,
                       k_candidates=K_CANDIDATES, seed=SEED):
    """ Paper's procedure (section 5.2/5.4): repeatedly hold out ~10 random training snapshots and
    measure the accuracy of the kNN-backmapped prediction for each candidate k, then keep the best
    on average. """
    rng = np.random.default_rng(seed)
    n_train = Z_train.shape[0]
    scores = {k: [] for k in k_candidates}

    for _ in range(n_repeats):
        holdout_idx = rng.choice(n_train, size=min(n_holdout, n_train - 1), replace=False)
        keep_mask = np.ones(n_train, dtype=bool)
        keep_mask[holdout_idx] = False
        keep_idx = np.flatnonzero(keep_mask)

        Z_holdout = Z_train[holdout_idx]
        Y_holdout = Y_train[holdout_idx]

        for k in k_candidates:
            y_pred = _knn_backmap(Z_holdout, Z_train[keep_idx], Y_train[keep_idx], k)
            r2 = _compute_R2(Y_holdout.reshape(-1), y_pred.reshape(-1), np.ones(len(holdout_idx)))
            scores[k].append(r2)

    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    return max(mean_scores, key=mean_scores.get)


class Model:
    """ IsoMap+RBF, reproduced from Peter et al. 2025 ("ONERA's CRM WBPN database for machine
    learning activities..."), section 5.4 -- same model as utils/paper_isomap_rbf.py in the
    project repo:
      1. IsoMap embeds the training wall-field snapshots onto an r=3 latent manifold (matching the
         paper's finding that r=3 matches the 3 physical flow parameters).
      2. RBFInterpolator regresses the r latent coordinates as functions of (Minf, AoA, Pi) (the
         paper cites the SMT toolbox for this; scipy.interpolate.RBFInterpolator is the standard
         substitute -- SMT isn't in Codabench's base image).
      3. IsoMap has no natural backmapping, so kNN interpolation (inverse-distance weighted)
         reconstructs the full field from the predicted latent coordinates; k is chosen via the
         paper's own leave-10-out CV procedure, run once during fit().

    fit()/predict() receive the full pointwise [n_sims*NWALLP, 9] arrays Codabench's ingestion
    program passes -- conditions are extracted internally (one row per simulation, columns 6:9)
    since this is a field regressor, not a pointwise one.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.isomap = None
        self.rbf = None
        self.Z_train = None
        self.Y_train = None
        self.best_k = None

    def fit(self, X, y):
        n_train = X.shape[0] // NWALLP
        train_conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1])
        Y_train = np.asarray(y).reshape(n_train, NWALLP)

        conds_sc = self.scaler.fit_transform(train_conds)

        self.isomap = Isomap(n_neighbors=ISOMAP_K, n_components=N_COMPONENTS)
        Z_train = self.isomap.fit_transform(Y_train)

        self.rbf = RBFInterpolator(conds_sc, Z_train, kernel=RBF_KERNEL)

        self.best_k = _select_backmap_k(Z_train, Y_train)

        self.Z_train = Z_train
        self.Y_train = Y_train
        return self

    def predict(self, X):
        test_conds = np.asarray(X[::NWALLP, COL_MINF:COL_PI + 1])
        conds_sc = self.scaler.transform(test_conds)

        Z_pred = self.rbf(conds_sc)
        y_pred = _knn_backmap(Z_pred, self.Z_train, self.Y_train, self.best_k)
        return y_pred.reshape(-1)
