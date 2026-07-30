"""
Reference full-field MLP baseline: production-scale network trained with
the same differentiable KLw loss as utils/train_mlp_with_kl.py.

This is heavier than the starting kit's LightGBM baseline (bigger network,
many more epochs) and is meant to be run on its own, not from the
notebook. It only uses the training simulations, holding a slice of them
out for early stopping and evaluation, since that's all a participant has
access to.

Usage:
    python mlp_klw.py --data-dir /path/to/input_data
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}

HIDDEN, DROPOUT = (128, 256, 512), 0.2
N_EPOCHS, BATCH, LR = 400, 2, 1e-3
VAL_FRAC, PATIENCE, SEED = 0.1, 30, 0
N_BINS = 200
TAU = 10.0 / N_BINS


def load_data(data_dir):
    X_train = np.load(f'{data_dir}/train_data.npy')
    y_train = np.load(f'{data_dir}/train_labels.npy')[:, 0]
    component_labels = np.load(f'{data_dir}/component_labels_unique.npy')
    with open(f'{data_dir}/component_map.json') as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}
    return X_train, y_train, comp_masks


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


def make_klw_loss(sigma_s, device, n_bins=N_BINS, tau=TAU):
    """ Differentiable KLw loss: each residual softly votes into n_bins
    bins through a Gaussian-shaped softmax kernel, giving a smoothed
    histogram p compared via discrete KL to the reference N(0, 0.1*sigma_s). """
    edges   = torch.linspace(-5.0, 5.0, n_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])

    centers_np = centers.cpu().numpy()
    q = np.exp(-0.5 * (centers_np / 0.1) ** 2)
    q = np.clip(q / q.sum(), 1e-10, None)
    log_q = torch.tensor(np.log(q), dtype=torch.float32, device=device)

    def loss_fn(y_pred, y_true, w_pts):
        eps  = (y_pred - y_true) / sigma_s
        d    = (eps.unsqueeze(2) - centers.view(1, 1, -1)) / tau
        soft = torch.softmax(-0.5 * d * d, dim=2)
        p    = torch.einsum('p,spb->sb', w_pts, soft)
        p    = torch.clamp(p, min=1e-10)
        p    = p / p.sum(dim=1, keepdim=True)
        return (p * (p.log() - log_q)).sum(dim=1)

    return loss_fn


def component_weights(comp_masks, nwallp=NWALLP):
    w = np.zeros(nwallp, dtype=np.float32)
    for cname, mask in comp_masks.items():
        w[mask] = KL_WEIGHTS.get(cname, 0.0)
    return w / w.sum()


def compute_R2(y, yhat):
    ymean = np.mean(y)
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - ymean) ** 2))


def compute_rMAE(y, yhat):
    return float(np.mean(np.abs(y - yhat)) / np.mean(np.abs(y)))


def evaluate_hard_klw(y_blocks, yhat_blocks, comp_masks, sigma_ref, n_bins=N_BINS):
    """ Exact histogram-based KLw (same formula as kit_utils/metrics.py),
    for reporting on the held-out slice with the real leaderboard metric. """
    sample_weight = np.zeros(y_blocks.shape[1], dtype=np.float32)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = KL_WEIGHTS.get(cname, 0.0)

    kls = []
    for y, yhat in zip(y_blocks, yhat_blocks):
        eps = yhat - y
        sigma_y = float(y.std()) + 1e-6
        lim = 5.0 * sigma_y
        bins = np.linspace(-lim, lim, n_bins + 1)
        dx = bins[1] - bins[0]
        p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
        p = np.clip(p * dx, 1e-10, None)
        p /= p.sum()
        centers = 0.5 * (bins[:-1] + bins[1:])
        q = np.exp(-0.5 * (centers / sigma_ref) ** 2) / (sigma_ref * np.sqrt(2 * np.pi)) * dx
        q = np.clip(q, 1e-10, None)
        q /= q.sum()
        kls.append(float(np.sum(p * np.log(p / q))))
    return float(np.mean(kls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='../../../feedback_phase/input_data',
                     help='path to the downloaded input_data/ folder')
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('Loading data...')
    X_train, y_train, comp_masks = load_data(args.data_dir)
    conds  = X_train[::NWALLP, COL_MINF:COL_PI + 1]
    n_sims = X_train.shape[0] // NWALLP
    Y      = y_train.reshape(n_sims, NWALLP)
    print(f'n_sims={n_sims}')

    sigma_ref = max(0.01 * float(np.mean(y_train)), 1e-6)
    sigma_s   = 10.0 * sigma_ref
    klw_loss  = make_klw_loss(sigma_s, device)
    w_pts     = torch.tensor(component_weights(comp_masks), device=device)

    n_val = max(1, int(round(VAL_FRAC * n_sims)))
    perm  = np.random.permutation(n_sims)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f'train/val split: {len(tr_idx)} train  {len(val_idx)} val (held out from the training simulations)')

    C = torch.tensor(conds, dtype=torch.float32, device=device)
    C_mean, C_std = C.mean(dim=0, keepdim=True), C.std(dim=0, keepdim=True)
    C = (C - C_mean) / C_std
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)

    mean_field = Yt[tr_idx].mean(dim=0).cpu().numpy()
    model = GlobalMLP(NWALLP, mean_field=mean_field).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    tr_idx_t, val_idx_t = torch.tensor(tr_idx), torch.tensor(val_idx)
    best_val, best_state, bad_epochs = float('inf'), None, 0

    print('Training (loss = differentiable KLw, production-scale network)...')
    for epoch in range(N_EPOCHS):
        model.train()
        perm_t = tr_idx_t[torch.randperm(len(tr_idx_t))]
        for i in range(0, len(perm_t), BATCH):
            idx  = perm_t[i:i + BATCH]
            pred = model(C[idx])
            loss = klw_loss(pred, Yt[idx], w_pts).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_losses = [
                klw_loss(model(C[val_idx_t[i:i + BATCH]]), Yt[val_idx_t[i:i + BATCH]], w_pts)
                for i in range(0, len(val_idx_t), BATCH)
            ]
            val_loss = torch.cat(val_losses).mean().item()

        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f'  epoch {epoch:3d}  val KLw = {val_loss:.4f}  (best {best_val:.4f})')
        if bad_epochs >= PATIENCE:
            print(f'  early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    print('Evaluating on the held-out validation simulations...')
    with torch.no_grad():
        yhat_val = model(C[val_idx_t]).cpu().numpy()
    y_val = Y[val_idx]

    r2   = compute_R2(y_val.reshape(-1), yhat_val.reshape(-1))
    rmae = compute_rMAE(y_val.reshape(-1), yhat_val.reshape(-1))
    klw  = evaluate_hard_klw(y_val, yhat_val, comp_masks, sigma_ref)
    print(f'\nR2={r2:.4f}  rMAE={rmae:.4f}  KLw={klw:.4f}')

    torch.save(model.state_dict(), 'mlp_klw.pt')
    print('Saved weights to mlp_klw.pt')


if __name__ == '__main__':
    main()
