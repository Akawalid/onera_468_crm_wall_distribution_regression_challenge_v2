import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

COL_MINF, COL_AOA, COL_PI = 6, 7, 8
HIDDEN, DROPOUT = (64,), 0.1
N_EPOCHS, BATCH, LR = 40, 16, 3e-3
VAL_FRAC, PATIENCE, SEED = 0.15, 8, 0


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
        self.register_buffer('mean_field', mf)

    def forward(self, c):
        return self.mean_field + self.net(c)


def gaussian_kl_loss(y_pred, y_true, w_pts, sigma_ref):
    """ Closed-form KL(N(bias, spread^2) || N(0, sigma_ref^2)) per row --
    see the starting kit notebook (section 2.2) for the explanation. """
    eps    = y_pred - y_true
    bias   = eps @ w_pts
    var    = ((eps - bias.unsqueeze(1)) ** 2) @ w_pts
    spread = torch.sqrt(var + 1e-12)
    return torch.log(sigma_ref / spread) + (var + bias ** 2) / (2.0 * sigma_ref ** 2) - 0.5


class Model:

    def __init__(self):
        self.model  = None
        self.scaler = StandardScaler()
        self.nwallp = None

    def fit(self, X, y):
        cond0 = X[0, COL_MINF:COL_PI + 1]
        self.nwallp = int(np.argmax(np.any(X[:, COL_MINF:COL_PI + 1] != cond0, axis=1)))
        n_sims = X.shape[0] // self.nwallp
        conds  = X[::self.nwallp, COL_MINF:COL_PI + 1]
        Y      = y.reshape(n_sims, self.nwallp)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        conds_sc  = self.scaler.fit_transform(conds)
        sigma_ref = torch.tensor(max(0.01 * float(np.mean(y)), 1e-6), device=device)
        # No component_map.json available here, so weight every point equally
        # (see notebook section 5 for how to change this).
        w_pts = torch.full((self.nwallp,), 1.0 / self.nwallp, device=device)

        n_val = max(1, int(round(VAL_FRAC * n_sims)))
        perm  = np.random.permutation(n_sims)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        C  = torch.tensor(conds_sc, dtype=torch.float32, device=device)
        Yt = torch.tensor(Y, dtype=torch.float32, device=device)
        mean_field = Yt[tr_idx].mean(dim=0).cpu().numpy()

        self.model = GlobalMLP(self.nwallp, mean_field=mean_field).to(device)
        opt   = torch.optim.Adam(self.model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

        best_val, best_state, bad_epochs = float('inf'), None, 0
        tr_idx_t  = torch.tensor(tr_idx)
        val_idx_t = torch.tensor(val_idx)

        for epoch in range(N_EPOCHS):
            self.model.train()
            perm_t = tr_idx_t[torch.randperm(len(tr_idx_t))]
            for i in range(0, len(perm_t), BATCH):
                idx  = perm_t[i:i + BATCH]
                pred = self.model(C[idx])
                loss = gaussian_kl_loss(pred, Yt[idx], w_pts, sigma_ref).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = gaussian_kl_loss(
                    self.model(C[val_idx_t]), Yt[val_idx_t], w_pts, sigma_ref).mean().item()
            if val_loss < best_val - 1e-4:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X):
        device = next(self.model.parameters()).device
        conds  = X[::self.nwallp, COL_MINF:COL_PI + 1]
        with torch.no_grad():
            C = torch.tensor(self.scaler.transform(conds), dtype=torch.float32, device=device)
            y_pred = self.model(C).cpu().numpy().astype(np.float64)
        return y_pred.reshape(-1)
