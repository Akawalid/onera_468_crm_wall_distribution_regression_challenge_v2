import json
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm, entropy
from sklearn.preprocessing import StandardScaler

nwallp   = 260774
COL_MINF = 6
COL_AOA  = 7
COL_PI   = 8
DATA_DIR = '/home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/FILES_RHO_ALL_POINTS_reduitfloat32/'

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}

N_BINS     = 200
SIGMA_REF  = 0.1
MEAN_FLOOR = 5e-3
TAU        = 10.0 / N_BINS
N_EPOCHS   = 400
BATCH      = 2
LR         = 1e-3
# HIDDEN     = (128, 256, 512)
HIDDEN = (75, 120, 1226, 16490)
DROPOUT    = 0.2
VAL_FRAC   = 0.1
PATIENCE   = 30
SEED       = 0

BETA = 1.178


def sigma_scale(y_true):
    return BETA * max(1.0 - float(np.mean(y_true)), MEAN_FLOOR)


def residual_kl_normal(y_true, y_pred, sigma_ref_frac=0.1, n_bins=200):
    eps       = y_pred - y_true
    sigma_s   = sigma_scale(y_true)
    sigma_ref = sigma_ref_frac * sigma_s
    lim  = 5.0 * sigma_s
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx   = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, density=True)
    p    = np.clip(p * dx, 1e-10, None)
    p   /= p.sum()
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q    = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q    = np.clip(q, 1e-10, None)
    q   /= q.sum()
    kl   = float(entropy(p, q))
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)),
            'bias': float(np.mean(eps) / sigma_s), 'spread': float(np.std(eps) / sigma_s)}


def residual_kl_weighted(y_true, y_pred, comp_masks, comp_weights, sigma_ref_frac=0.1, n_bins=200):
    eps       = y_pred - y_true
    sigma_s   = sigma_scale(y_true)
    sigma_ref = sigma_ref_frac * sigma_s
    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = comp_weights.get(cname, 0.0)
    lim  = 5.0 * sigma_s
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx   = bins[1] - bins[0]
    p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
    p    = np.clip(p * dx, 1e-10, None)
    p   /= p.sum()
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()
    kl = float(entropy(p, q))
    bias   = float(np.average(eps, weights=sample_weight)) / sigma_s
    spread = float(np.sqrt(np.average((eps - eps.mean()) ** 2, weights=sample_weight))) / sigma_s
    return {'kl': kl, 'score': float(1.0 / (1.0 + kl)), 'bias': bias, 'spread': spread}


def compute_R2(y, yhat, confidence_pointwise):
    ymean = np.mean(y)
    SSE   = np.sum(confidence_pointwise * (y - yhat) ** 2)
    SSD   = np.sum(confidence_pointwise * (y - ymean) ** 2)
    return float(1.0 - SSE / SSD)


def compute_worst_rMAE(y, yhat, confidence_per_case):
    rMAE_list, idx_list = [], []
    for l in range(len(confidence_per_case)):
        if confidence_per_case[l] < 1.0:
            continue
        ycase    = y[l * nwallp:(l + 1) * nwallp]
        yhatcase = yhat[l * nwallp:(l + 1) * nwallp]
        rMAE_list.append(np.mean(np.abs(ycase - yhatcase)) / np.mean(np.abs(ycase)))
        idx_list.append(l)
    rMAE_arr     = np.array(rMAE_list)
    iworst_local = int(np.argmax(rMAE_arr))
    return idx_list[iworst_local], float(rMAE_arr[iworst_local])


def evaluate_phase(y, y_pred, weights, n_sims, comp_masks):
    Y, Yh = y.reshape(n_sims, nwallp), y_pred.reshape(n_sims, nwallp)
    confidence_pointwise = np.repeat(weights, nwallp)
    iworst, worst_rMAE = compute_worst_rMAE(y, y_pred, weights)
    r2_global = compute_R2(y, y_pred, confidence_pointwise)
    comp_kl   = {c: [None] * n_sims for c in KL_WEIGHTS}
    kl_w      = np.full(n_sims, np.nan)
    valid_idx = np.where(weights == 1.0)[0]
    for i in valid_idx:
        yc, yhatc = Y[i], Yh[i]
        for cname, mask in comp_masks.items():
            if cname not in KL_WEIGHTS:
                continue
            comp_kl[cname][i] = residual_kl_normal(yc[mask], yhatc[mask])
        kl_w[i] = residual_kl_weighted(yc, yhatc, comp_masks, KL_WEIGHTS)['kl']
    return dict(Y=Y, Yh=Yh, iworst=iworst, worst_rMAE=worst_rMAE, r2=r2_global,
                kl=kl_w, comp_kl=comp_kl)


def print_phase_summary(label, res):
    i_mae = res['iworst']
    i_kl  = int(np.nanargmax(res['kl']))
    rMAE_at_worst_kl = np.mean(np.abs(res['Y'][i_kl] - res['Yh'][i_kl])) / np.mean(np.abs(res['Y'][i_kl]))
    print(f'\n{label}')
    print(f'  R2 global: {res["r2"]:.4f}')
    print(f'  worst rMAE: sim {i_mae}  rMAE={res["worst_rMAE"]:.4f}  KL={res["kl"][i_mae]:.4f}  score={1.0 / (1.0 + res["kl"][i_mae]):.4f}')
    print(f'  worst KL  : sim {i_kl}  KL={res["kl"][i_kl]:.4f}  score={1.0 / (1.0 + res["kl"][i_kl]):.4f}  rMAE={rMAE_at_worst_kl:.4f}')
    print(f'\n  KL by component')
    print(f'  {"component":<10}  {"mean KL":>9}  {"max KL":>9}  {"mean score":>10}  {"mean bias":>10}  {"mean spread":>11}')
    for cname in KL_WEIGHTS:
        vals = [v for v in res['comp_kl'][cname] if v is not None]
        kl_c = np.array([v['kl'] for v in vals])
        sc_c = np.array([v['score'] for v in vals])
        bi_c = np.array([v['bias'] for v in vals])
        sp_c = np.array([v['spread'] for v in vals])
        print(f'  {cname:<10}  {kl_c.mean():>9.4f}  {kl_c.max():>9.4f}  {sc_c.mean():>10.4f}  {bi_c.mean():>10.4f}  {sp_c.mean():>11.4f}')
    kl_global    = res['kl'][~np.isnan(res['kl'])]
    score_global = 1.0 / (1.0 + kl_global)
    print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')


class GlobalMLP(nn.Module):
    def __init__(self, n_out, hidden, dropout=0.0, mean_field=None):
        super().__init__()
        layers, d = [], 3
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.01)]
            if dropout > 0.0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)
        mf = torch.zeros(n_out) if mean_field is None else torch.tensor(mean_field, dtype=torch.float32)
        self.register_buffer('mean_field', mf)

    def forward(self, c):
        return self.mean_field + self.net(c)


def main():
    global BETA
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    print('Loading data...')
    X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
    y_train = np.load(DATA_DIR + 'splitv2/train_labels.npy')
    X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
    y_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_labels.npy')
    X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')
    y_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_labels.npy')

    component_labels = np.load(DATA_DIR + 'component_labels_unique.npy')
    with open(DATA_DIR + 'component_map.json') as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    train_conds = X_train[::nwallp, COL_MINF:COL_PI + 1]
    test1_conds = X_test1[::nwallp, COL_MINF:COL_PI + 1]
    test2_conds = X_test2[::nwallp, COL_MINF:COL_PI + 1]

    n_train = X_train.shape[0] // nwallp
    n_test1 = X_test1.shape[0] // nwallp
    n_test2 = X_test2.shape[0] // nwallp
    print(f'n_train={n_train}  n_test1={n_test1}  n_test2={n_test2}')

    test1_weights = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
    test2_weights = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

    scaler         = StandardScaler()
    train_conds_sc = scaler.fit_transform(train_conds)
    test1_conds_sc = scaler.transform(test1_conds)
    test2_conds_sc = scaler.transform(test2_conds)

    Y_all_train = y_train.reshape(n_train, nwallp)
    means_tr    = Y_all_train.mean(axis=1)
    sigmas_tr   = Y_all_train.std(axis=1)
    x_fit       = 1.0 - means_tr
    BETA        = float(np.sum(sigmas_tr * x_fit) / np.sum(x_fit * x_fit))
    print(f'BETA (fitted on train) = {BETA:.4f}')

    edges   = torch.linspace(-5.0, 5.0, N_BINS + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])

    sw = np.zeros(nwallp, dtype=np.float32)
    for cname, mask in comp_masks.items():
        sw[mask] = KL_WEIGHTS.get(cname, 0.0)
    w_pts = torch.tensor(sw / sw.sum(), device=device)

    q = norm.pdf(centers.cpu().numpy(), loc=0.0, scale=SIGMA_REF)
    q = np.clip(q / q.sum(), 1e-10, None)
    log_q = torch.tensor(np.log(q), dtype=torch.float32, device=device)

    def klw_loss(y_pred, y_true, sigma_s):
        eps  = (y_pred - y_true) / sigma_s.unsqueeze(1)
        d    = (eps.unsqueeze(2) - centers.view(1, 1, -1)) / TAU
        soft = torch.softmax(-0.5 * d * d, dim=2)
        p    = torch.einsum('p,spb->sb', w_pts, soft)
        p    = torch.clamp(p, min=1e-10)
        p    = p / p.sum(dim=1, keepdim=True)
        return (p * (p.log() - log_q)).sum(dim=1)

    n_val   = max(1, int(round(VAL_FRAC * n_train)))
    perm0   = np.random.permutation(n_train)
    val_idx = perm0[:n_val]
    tr_idx  = perm0[n_val:]
    n_tr    = len(tr_idx)
    print(f'train/val split: {n_tr} train  {n_val} val')

    Y_full_t   = torch.tensor(Y_all_train, dtype=torch.float32, device=device)
    C_full_t   = torch.tensor(train_conds_sc, dtype=torch.float32, device=device)
    sigma_full = torch.tensor(BETA * np.clip(1.0 - means_tr, MEAN_FLOOR, None),
                              dtype=torch.float32, device=device)

    tr_idx_t  = torch.tensor(tr_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    mean_field = Y_all_train[tr_idx].mean(axis=0)

    model = GlobalMLP(nwallp, HIDDEN, dropout=DROPOUT, mean_field=mean_field).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    best_val   = float('inf')
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    print('Training MLP (loss = sum of per simulation weighted KL, new sigma formula)...')
    for epoch in range(N_EPOCHS):
        perm, tot = torch.randperm(n_tr, device=device), 0.0
        model.train()
        for i in range(0, n_tr, BATCH):
            idx  = tr_idx_t[perm[i:i + BATCH]]
            pred = model(C_full_t[idx])
            loss = klw_loss(pred, Y_full_t[idx], sigma_full[idx]).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_tot = 0.0
            for i in range(0, n_val, BATCH):
                idx = val_idx_t[i:i + BATCH]
                pred = model(C_full_t[idx])
                val_tot += klw_loss(pred, Y_full_t[idx], sigma_full[idx]).sum().item()
        val_mean = val_tot / n_val

        if val_mean < best_val - 1e-4:
            best_val   = val_mean
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f'epoch {epoch:4d}  train KLw mean = {tot / n_tr:.4f}  val KLw mean = {val_mean:.4f}  best = {best_val:.4f} (epoch {best_epoch})')

        if bad_epochs >= PATIENCE:
            print(f'early stopping at epoch {epoch} (best epoch {best_epoch}, val KLw mean {best_val:.4f})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        C1 = torch.tensor(test1_conds_sc, dtype=torch.float32, device=device)
        C2 = torch.tensor(test2_conds_sc, dtype=torch.float32, device=device)
        y_pred1 = model(C1).cpu().numpy().astype(np.float64).reshape(-1)
        y_pred2 = model(C2).cpu().numpy().astype(np.float64).reshape(-1)

    torch.save({'state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
                'hidden': HIDDEN, 'dropout': DROPOUT, 'beta': BETA,
                'nwallp': nwallp, 'best_epoch': best_epoch},
               'mlp_klw_v2.pt')
    print(f'Saved: mlp_klw_v2.pt (best epoch {best_epoch})')

    res1 = evaluate_phase(y_test1, y_pred1, test1_weights, n_test1, comp_masks)
    res2 = evaluate_phase(y_test2, y_pred2, test2_weights, n_test2, comp_masks)
    for label, res in [('Phase 1 (global MLP v2)', res1), ('Phase 2 (global MLP v2)', res2)]:
        print_phase_summary(label, res)


if __name__ == '__main__':
    main()

# import json
# import numpy as np
# import torch
# import torch.nn as nn
# from scipy.stats import norm, entropy
# from sklearn.preprocessing import StandardScaler

# nwallp   = 260774
# COL_MINF = 6
# COL_AOA  = 7
# COL_PI   = 8
# DATA_DIR = '/home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/FILES_RHO_ALL_POINTS_reduitfloat32/'

# KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}

# N_BINS    = 200
# SIGMA_REF = 0.1
# TAU       = 10.0 / N_BINS
# N_EPOCHS  = 400
# BATCH     = 2
# LR        = 1e-3
# HIDDEN    = (128, 256, 512)
# SEED      = 0


# def residual_kl_normal(y_true, y_pred, sigma_ref_frac=0.1, n_bins=200):
#     eps       = y_pred - y_true
#     sigma_y   = y_true.std() + 1e-12
#     sigma_ref = sigma_ref_frac * sigma_y
#     lim  = 5.0 * sigma_y
#     bins = np.linspace(-lim, lim, n_bins + 1)
#     dx   = bins[1] - bins[0]
#     p, _ = np.histogram(eps, bins=bins, density=True)
#     p    = np.clip(p * dx, 1e-10, None)
#     p   /= p.sum()
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     q    = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
#     q    = np.clip(q, 1e-10, None)
#     q   /= q.sum()
#     kl   = float(entropy(p, q))
#     return {'kl': kl, 'score': float(1.0 / (1.0 + kl)),
#             'bias': float(np.mean(eps) / sigma_y), 'spread': float(np.std(eps) / sigma_y)}


# def residual_kl_weighted(y_true, y_pred, comp_masks, comp_weights, sigma_ref_frac=0.1, n_bins=200):
#     eps       = y_pred - y_true
#     sigma_y   = y_true.std() + 1e-12
#     sigma_ref = sigma_ref_frac * sigma_y
#     sample_weight = np.zeros_like(eps)
#     for cname, mask in comp_masks.items():
#         sample_weight[mask] = comp_weights.get(cname, 0.0)
#     lim  = 5.0 * sigma_y
#     bins = np.linspace(-lim, lim, n_bins + 1)
#     dx   = bins[1] - bins[0]
#     p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
#     p    = np.clip(p * dx, 1e-10, None)
#     p   /= p.sum()
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
#     q = np.clip(q, 1e-10, None)
#     q /= q.sum()
#     kl = float(entropy(p, q))
#     bias   = float(np.average(eps, weights=sample_weight)) / sigma_y
#     spread = float(np.sqrt(np.average((eps - eps.mean()) ** 2, weights=sample_weight))) / sigma_y
#     return {'kl': kl, 'score': float(1.0 / (1.0 + kl)), 'bias': bias, 'spread': spread}


# def compute_R2(y, yhat, confidence_pointwise):
#     ymean = np.mean(y)
#     SSE   = np.sum(confidence_pointwise * (y - yhat) ** 2)
#     SSD   = np.sum(confidence_pointwise * (y - ymean) ** 2)
#     return float(1.0 - SSE / SSD)


# def compute_worst_rMAE(y, yhat, confidence_per_case):
#     rMAE_list, idx_list = [], []
#     for l in range(len(confidence_per_case)):
#         if confidence_per_case[l] < 1.0:
#             continue
#         ycase    = y[l * nwallp:(l + 1) * nwallp]
#         yhatcase = yhat[l * nwallp:(l + 1) * nwallp]
#         rMAE_list.append(np.mean(np.abs(ycase - yhatcase)) / np.mean(np.abs(ycase)))
#         idx_list.append(l)
#     rMAE_arr     = np.array(rMAE_list)
#     iworst_local = int(np.argmax(rMAE_arr))
#     return idx_list[iworst_local], float(rMAE_arr[iworst_local])


# def evaluate_phase(y, y_pred, weights, n_sims, comp_masks):
#     Y, Yh = y.reshape(n_sims, nwallp), y_pred.reshape(n_sims, nwallp)
#     confidence_pointwise = np.repeat(weights, nwallp)
#     iworst, worst_rMAE = compute_worst_rMAE(y, y_pred, weights)
#     r2_global = compute_R2(y, y_pred, confidence_pointwise)
#     comp_kl   = {c: [None] * n_sims for c in KL_WEIGHTS}
#     kl_w      = np.full(n_sims, np.nan)
#     valid_idx = np.where(weights == 1.0)[0]
#     for i in valid_idx:
#         yc, yhatc = Y[i], Yh[i]
#         for cname, mask in comp_masks.items():
#             if cname not in KL_WEIGHTS:
#                 continue
#             comp_kl[cname][i] = residual_kl_normal(yc[mask], yhatc[mask])
#         kl_w[i] = residual_kl_weighted(yc, yhatc, comp_masks, KL_WEIGHTS)['kl']
#     return dict(Y=Y, Yh=Yh, iworst=iworst, worst_rMAE=worst_rMAE, r2=r2_global,
#                 kl=kl_w, comp_kl=comp_kl)


# def print_phase_summary(label, res):
#     i_mae = res['iworst']
#     i_kl  = int(np.nanargmax(res['kl']))
#     rMAE_at_worst_kl = np.mean(np.abs(res['Y'][i_kl] - res['Yh'][i_kl])) / np.mean(np.abs(res['Y'][i_kl]))
#     print(f'\n{label}')
#     print(f'  R2 global: {res["r2"]:.4f}')
#     print(f'  worst rMAE: sim {i_mae}  rMAE={res["worst_rMAE"]:.4f}  KL={res["kl"][i_mae]:.4f}  score={1.0 / (1.0 + res["kl"][i_mae]):.4f}')
#     print(f'  worst KL  : sim {i_kl}  KL={res["kl"][i_kl]:.4f}  score={1.0 / (1.0 + res["kl"][i_kl]):.4f}  rMAE={rMAE_at_worst_kl:.4f}')
#     print(f'\n  KL by component')
#     print(f'  {"component":<10}  {"mean KL":>9}  {"max KL":>9}  {"mean score":>10}  {"mean bias":>10}  {"mean spread":>11}')
#     for cname in KL_WEIGHTS:
#         vals = [v for v in res['comp_kl'][cname] if v is not None]
#         kl_c = np.array([v['kl'] for v in vals])
#         sc_c = np.array([v['score'] for v in vals])
#         bi_c = np.array([v['bias'] for v in vals])
#         sp_c = np.array([v['spread'] for v in vals])
#         print(f'  {cname:<10}  {kl_c.mean():>9.4f}  {kl_c.max():>9.4f}  {sc_c.mean():>10.4f}  {bi_c.mean():>10.4f}  {sp_c.mean():>11.4f}')
#     kl_global    = res['kl'][~np.isnan(res['kl'])]
#     score_global = 1.0 / (1.0 + kl_global)
#     print(f'\n  Global weighted KL: mean={kl_global.mean():.4f}  max={kl_global.max():.4f}  mean score={score_global.mean():.4f}')


# class GlobalMLP(nn.Module):
#     def __init__(self, n_out, hidden, mean_field=None):
#         super().__init__()
#         layers, d = [], 3
#         for h in hidden:
#             layers += [nn.Linear(d, h), nn.LeakyReLU(0.01)]
#             d = h
#         layers += [nn.Linear(d, n_out)]
#         self.net = nn.Sequential(*layers)
#         mf = torch.zeros(n_out) if mean_field is None else torch.tensor(mean_field, dtype=torch.float32)
#         self.register_buffer('mean_field', mf)

#     def forward(self, c):
#         return self.mean_field + self.net(c)


# def main():
#     torch.manual_seed(SEED)
#     np.random.seed(SEED)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'device: {device}')

#     print('Loading data...')
#     X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
#     y_train = np.load(DATA_DIR + 'splitv2/train_labels.npy')
#     X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
#     y_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_labels.npy')
#     X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')
#     y_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_labels.npy')

#     component_labels = np.load(DATA_DIR + 'component_labels_unique.npy')
#     with open(DATA_DIR + 'component_map.json') as f:
#         component_map = {int(k): v for k, v in json.load(f).items()}
#     comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

#     train_conds = X_train[::nwallp, COL_MINF:COL_PI + 1]
#     test1_conds = X_test1[::nwallp, COL_MINF:COL_PI + 1]
#     test2_conds = X_test2[::nwallp, COL_MINF:COL_PI + 1]

#     n_train = X_train.shape[0] // nwallp
#     n_test1 = X_test1.shape[0] // nwallp
#     n_test2 = X_test2.shape[0] // nwallp
#     print(f'n_train={n_train}  n_test1={n_test1}  n_test2={n_test2}')

#     test1_weights = np.where(np.abs(test1_conds[:, 1]) < 10.0, 1.0, 0.5)
#     test2_weights = np.where(np.abs(test2_conds[:, 1]) < 10.0, 1.0, 0.5)

#     scaler         = StandardScaler()
#     train_conds_sc = scaler.fit_transform(train_conds)
#     test1_conds_sc = scaler.transform(test1_conds)
#     test2_conds_sc = scaler.transform(test2_conds)

#     edges   = torch.linspace(-5.0, 5.0, N_BINS + 1, device=device)
#     centers = 0.5 * (edges[:-1] + edges[1:])

#     sw = np.zeros(nwallp, dtype=np.float32)
#     for cname, mask in comp_masks.items():
#         sw[mask] = KL_WEIGHTS.get(cname, 0.0)
#     w_pts = torch.tensor(sw / sw.sum(), device=device)

#     q = norm.pdf(centers.cpu().numpy(), loc=0.0, scale=SIGMA_REF)
#     q = np.clip(q / q.sum(), 1e-10, None)
#     log_q = torch.tensor(np.log(q), dtype=torch.float32, device=device)

#     def klw_loss(y_pred, y_true, sigma_y):
#         eps  = (y_pred - y_true) / sigma_y.unsqueeze(1)
#         d    = (eps.unsqueeze(2) - centers.view(1, 1, -1)) / TAU
#         soft = torch.softmax(-0.5 * d * d, dim=2)
#         p    = torch.einsum('p,spb->sb', w_pts, soft)
#         p    = torch.clamp(p, min=1e-10)
#         p    = p / p.sum(dim=1, keepdim=True)
#         return (p * (p.log() - log_q)).sum(dim=1)

#     Y_train_t   = torch.tensor(y_train.reshape(n_train, nwallp), dtype=torch.float32, device=device)
#     C_train_t   = torch.tensor(train_conds_sc, dtype=torch.float32, device=device)
#     sigma_train = Y_train_t.std(dim=1) + 1e-12
#     mean_field  = Y_train_t.mean(dim=0).cpu().numpy()

#     model = GlobalMLP(nwallp, HIDDEN, mean_field=mean_field).to(device)
#     opt   = torch.optim.Adam(model.parameters(), lr=LR)
#     sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

#     print('Training MLP (loss = sum of per simulation weighted KL)...')
#     for epoch in range(N_EPOCHS):
#         perm, tot = torch.randperm(n_train, device=device), 0.0
#         model.train()
#         for i in range(0, n_train, BATCH):
#             idx  = perm[i:i + BATCH]
#             pred = model(C_train_t[idx])
#             loss = klw_loss(pred, Y_train_t[idx], sigma_train[idx]).sum()
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             tot += loss.item()
#         sched.step()
#         if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
#             print(f'epoch {epoch:4d}  sum KLw train = {tot:.4f}  mean = {tot / n_train:.4f}')

#     model.eval()
#     with torch.no_grad():
#         C1 = torch.tensor(test1_conds_sc, dtype=torch.float32, device=device)
#         C2 = torch.tensor(test2_conds_sc, dtype=torch.float32, device=device)
#         y_pred1 = model(C1).cpu().numpy().astype(np.float64).reshape(-1)
#         y_pred2 = model(C2).cpu().numpy().astype(np.float64).reshape(-1)

#     torch.save({'state_dict': model.state_dict(),
#                 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
#                 'hidden': HIDDEN, 'nwallp': nwallp},
#                'mlp_klw.pt')
#     print('Saved: mlp_klw.pt')

#     res1 = evaluate_phase(y_test1, y_pred1, test1_weights, n_test1, comp_masks)
#     res2 = evaluate_phase(y_test2, y_pred2, test2_weights, n_test2, comp_masks)
#     for label, res in [('Phase 1 (global MLP, KLw loss)', res1), ('Phase 2 (global MLP, KLw loss)', res2)]:
#         print_phase_summary(label, res)


# if __name__ == '__main__':
#     main()