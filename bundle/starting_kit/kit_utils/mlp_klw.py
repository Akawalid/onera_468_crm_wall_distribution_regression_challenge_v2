"""
Main baseline: a small "full-field" MLP, trained to directly minimize an
approximation of the KLw metric (replaces the KNN baseline from earlier
versions of the starting kit).

Full-field, like KNN: the model never sees geometry. It only takes the 3
scalar flow conditions (Minf, AoA, Pi) for one simulation and outputs the
entire rho field (nwallp values) in one shot -- the wall-point order is
memorized implicitly by the output layer, since it's identical across all
simulations, train and test alike.

--------------------------------------------------------------------------
Loss function, explained simply
--------------------------------------------------------------------------
The evaluation metric (mean_KL / KLw, see metrics.py) measures how close
the *distribution* of prediction residuals (eps = y_pred - y_true) is to a
narrow reference distribution N(0, sigma_ref) -- i.e. it rewards residuals
that are both centered at 0 (unbiased) and tightly concentrated (accurate)
around it. The real metric estimates this with a histogram, which isn't
differentiable, so we can't use it directly as a training loss.

Instead we assume the residuals are approximately Gaussian and use the
closed-form KL divergence between two Gaussians:

    KL( N(bias, spread^2) || N(0, sigma_ref^2) )
      = log(sigma_ref / spread) + (spread^2 + bias^2) / (2 * sigma_ref^2) - 1/2

`bias` and `spread` are just the (component-weighted) mean and standard
deviation of the residuals for one simulation. This is a smooth function
of the model's output, trivial to differentiate, and pushes training in
the same direction as the real metric: shrink both the bias and the
spread of the errors. We still *evaluate* with the exact histogram-based
KLw from metrics.py, this Gaussian version is only used as the training
loss.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .metrics import NWALLP, COL_MINF, COL_PI, KL_WEIGHTS

# Small on purpose: a starting-kit baseline should train in well under a
# minute, not need a GPU, and leave plenty of headroom under Codabench's
# execution time limit.
HIDDEN     = (64,)
DROPOUT    = 0.1
N_EPOCHS   = 40
BATCH      = 16
LR         = 3e-3
VAL_FRAC   = 0.15
PATIENCE   = 8
SEED       = 0


class GlobalMLP(nn.Module):
    """ (Minf, AoA, Pi) -> full rho field of length n_out (= nwallp). """

    def __init__(self, n_out, hidden=HIDDEN, dropout=DROPOUT, mean_field=None):
        super().__init__()
        layers, d = [], 3
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.01)]
            if dropout > 0.0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)
        mf = torch.zeros(n_out) if mean_field is None else torch.as_tensor(mean_field, dtype=torch.float32)
        self.register_buffer('mean_field', mf)  # start from the training mean, only learn a delta

    def forward(self, c):
        return self.mean_field + self.net(c)


def _component_weights(comp_masks, nwallp=NWALLP):
    """ Per-point weight (sums to 1), used to weight the residual mean/std
    by component the same way the evaluation metric does. """
    w = np.zeros(nwallp, dtype=np.float32)
    for cname, mask in comp_masks.items():
        w[mask] = KL_WEIGHTS.get(cname, 0.0)
    return w / w.sum()


def gaussian_kl_loss(y_pred, y_true, w_pts, sigma_ref):
    """ Closed-form KL(N(bias, spread^2) || N(0, sigma_ref^2)) per row of
    a (batch, nwallp) residual matrix -- see module docstring. """
    eps    = y_pred - y_true                        # (batch, nwallp)
    bias   = eps @ w_pts                             # (batch,) weighted mean
    var    = ((eps - bias.unsqueeze(1)) ** 2) @ w_pts  # (batch,) weighted variance
    spread = torch.sqrt(var + 1e-12)
    kl = torch.log(sigma_ref / spread) + (var + bias ** 2) / (2.0 * sigma_ref ** 2) - 0.5
    return kl  # (batch,)


def train_mlp(conds, Y, comp_masks, sigma_ref, hidden=HIDDEN, n_epochs=N_EPOCHS,
              batch=BATCH, lr=LR, val_frac=VAL_FRAC, patience=PATIENCE, verbose=True):
    """ Fit a GlobalMLP on (conds, Y) -- conds: (n_sims, 3) raw flow
    conditions, Y: (n_sims, nwallp) rho fields. Returns the fitted model
    and the StandardScaler used on the conditions. Carves its own small
    validation split out of `conds`/`Y` for early stopping -- this never
    touches the real test set. """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler   = StandardScaler()
    conds_sc = scaler.fit_transform(conds)
    nwallp   = Y.shape[1]
    w_pts    = torch.tensor(_component_weights(comp_masks, nwallp), device=device)

    n = len(conds)
    n_val = max(1, int(round(val_frac * n)))
    perm  = np.random.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    C = torch.tensor(conds_sc, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    sigma_ref_t = torch.tensor(float(sigma_ref), device=device)

    mean_field = Yt[tr_idx].mean(dim=0).cpu().numpy()
    model = GlobalMLP(nwallp, hidden=hidden, mean_field=mean_field).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_val, best_state, bad_epochs = float('inf'), None, 0
    tr_idx_t, val_idx_t = torch.tensor(tr_idx), torch.tensor(val_idx)

    for epoch in range(n_epochs):
        model.train()
        perm_t = tr_idx_t[torch.randperm(len(tr_idx_t))]
        for i in range(0, len(perm_t), batch):
            idx = perm_t[i:i + batch]
            pred = model(C[idx])
            loss = gaussian_kl_loss(pred, Yt[idx], w_pts, sigma_ref_t).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = gaussian_kl_loss(model(C[val_idx_t]), Yt[val_idx_t], w_pts, sigma_ref_t).mean().item()

        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f'    epoch {epoch:3d}  val Gaussian-KL = {val_loss:.4f}  (best {best_val:.4f})')
        if bad_epochs >= patience:
            if verbose:
                print(f'    early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def predict_mlp(model, scaler, conds):
    device = next(model.parameters()).device
    with torch.no_grad():
        C = torch.tensor(scaler.transform(conds), dtype=torch.float32, device=device)
        return model(C).cpu().numpy().astype(np.float64)


class Model:
    """ Matches the Codabench Model contract: fit(X, y) / predict(X).
    This is the class written out to submission/model.py. """

    def __init__(self, hidden=HIDDEN, n_epochs=N_EPOCHS):
        self.hidden = hidden
        self.n_epochs = n_epochs
        self.model = None
        self.scaler = None
        self.nwallp = None

    def fit(self, X, y):
        cond0 = X[0, COL_MINF:COL_PI + 1]
        self.nwallp = int(np.argmax(np.any(X[:, COL_MINF:COL_PI + 1] != cond0, axis=1)))
        n_sims = X.shape[0] // self.nwallp
        conds  = X[::self.nwallp, COL_MINF:COL_PI + 1]
        Y      = y.reshape(n_sims, self.nwallp)

        comp_masks = {'all': np.ones(self.nwallp, dtype=bool)}  # no component split available at submission time
        sigma_ref  = max(0.01 * float(np.mean(y)), 1e-6)

        self.model, self.scaler = train_mlp(
            conds, Y, comp_masks, sigma_ref, hidden=self.hidden,
            n_epochs=self.n_epochs, verbose=False)
        return self

    def predict(self, X):
        n_test = X.shape[0] // self.nwallp
        conds  = X[::self.nwallp, COL_MINF:COL_PI + 1]
        y_pred = predict_mlp(self.model, self.scaler, conds)
        return y_pred.reshape(-1)


def cv_predict(conds, Y, comp_masks, sigma_ref, mach_fold_splits_fn, hidden=HIDDEN, n_epochs=N_EPOCHS):
    """ Out-of-fold predictions for every training simulation, via
    leave-two-consecutive-Machs-out CV. """
    n_sims, nwallp = Y.shape
    y_cv_pred = np.zeros(n_sims * nwallp, dtype=np.float64)
    for train_idx, val_idx, label in mach_fold_splits_fn(conds):
        model, scaler = train_mlp(conds[train_idx], Y[train_idx], comp_masks, sigma_ref,
                                   hidden=hidden, n_epochs=n_epochs, verbose=False)
        y_val_pred = predict_mlp(model, scaler, conds[val_idx])
        for local_i, sim_i in enumerate(val_idx):
            y_cv_pred[sim_i * nwallp:(sim_i + 1) * nwallp] = y_val_pred[local_i]
        print(f'  fold {label}: {len(val_idx)} val sim(s) done')
    return y_cv_pred
