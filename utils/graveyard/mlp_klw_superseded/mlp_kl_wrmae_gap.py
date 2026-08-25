"""
Adversarial "gap finder": trains a GlobalMLP (same architecture as
train_mlp_with_kl.py) with a flipped objective -- instead of minimizing
KLw, it MAXIMIZES KLw while keeping relMAE (the per-simulation quantity
behind the challenge's wrMAE metric) capped under a "looks fine" budget.
The point is to construct a concrete counter-example showing that KLw and
wrMAE measure different things and can strongly disagree:

  - wrMAE only looks at the mean *magnitude* of the residual
    (mean(|y_pred - y_true|) / mean(|y_true|)).
  - KLw looks at the whole *shape* of the (component-weighted) residual
    distribution against a narrow reference N(0, sigma_ref).

Two predictions can have the same mean absolute error while looking
totally different histogram-wise: errors spread thinly and uniformly vs.
errors concentrated (heavy tails / off-center bulk) on a subset of points.
The first stays close to the reference and scores well on both metrics,
the second keeps relMAE low but blows up KLw. Gradient ascent on KLw
(under the relMAE cap) is exactly the search procedure to find that
second kind of prediction automatically, for real (Minf, AoA, Pi)
conditions in the dataset.

--------------------------------------------------------------------------
Training objective
--------------------------------------------------------------------------
For each simulation, per training step:

  relMAE   = mean(|eps|) / max(mean(|y_true|), EPS)          differentiable,
             identical formula to compute_wrMAE in
             bundle/starting_kit/kit_utils/metrics.py (just not maximized
             over conditions yet -- that comes later, at ranking time).
  KLw_soft = soft-histogram KL divergence of the component-weighted
             residuals against N(0, sigma_ref), same softmax relaxation as
             utils/train_mlp_with_kl.py (the real histogram KL isn't
             differentiable).

  loss = -KLw_soft + HINGE_WEIGHT * relu(relMAE - RELMAE_CAP) ** 2

i.e. unconstrained ascent on KLw, with a one-sided penalty that only
kicks in once relMAE crosses RELMAE_CAP. Below the cap there is no
pressure to shrink relMAE further, so the optimizer spends its whole
budget reshaping the residual distribution instead. Set RELMAE_CAP close
to what an actual decent baseline achieves, so the resulting example is
"as good as a normal model" by wrMAE.

--------------------------------------------------------------------------
Reporting
--------------------------------------------------------------------------
The soft KL used for training is only a training-time relaxation. Once
training is done (or periodically during it), every simulation's field is
re-scored with the *exact* histogram-based KLw and relMAE formulas from
metrics.py (reimplemented locally here, same convention as the other
scripts in utils/), and simulations are ranked by
(real_KLw - real_relMAE) so the printed / saved examples are genuine,
leaderboard-comparable numbers -- not artifacts of the soft relaxation.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

NWALLP   = 260774
COL_MINF = 6
COL_AOA  = 7
COL_PI   = 8
DATA_DIR = '/home/tau/ochabane/onera_468_crm_wall_distribution_regression_challenge_v2/FILES_RHO_ALL_POINTS_reduitfloat32/'

KL_WEIGHTS = {'wing': 0.3, 'pylon': 0.3, 'fuselage': 0.2, 'nacelle': 0.2}
KL_N_BINS  = 200
EPS        = 1e-6


# ---------------------------------------------------------------------------
# Model -- identical shape to the "real" baseline in train_mlp_with_kl.py, so
# the adversarial example is achievable by the same model class used for the
# actual submissions, not just an unconstrained per-point array.
# ---------------------------------------------------------------------------

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
        mf = torch.zeros(n_out) if mean_field is None else torch.as_tensor(mean_field, dtype=torch.float32)
        self.register_buffer('mean_field', mf)

    def forward(self, c):
        return self.mean_field + self.net(c)


# ---------------------------------------------------------------------------
# Differentiable training losses (soft KL histogram + plain relMAE)
# ---------------------------------------------------------------------------

def make_soft_kl_fn(w_pts, centers, log_q, tau):
    """ w_pts, centers, log_q: torch tensors already on `device`. Returns a
    function (y_pred, y_true, sigma_scale) -> per-row soft KLw, batched. """
    def klw_soft(y_pred, y_true, sigma_scale):
        eps  = (y_pred - y_true) / sigma_scale
        d    = (eps.unsqueeze(2) - centers.view(1, 1, -1)) / tau
        soft = torch.softmax(-0.5 * d * d, dim=2)
        p    = torch.einsum('p,spb->sb', w_pts, soft)
        p    = torch.clamp(p, min=1e-10)
        p    = p / p.sum(dim=1, keepdim=True)
        return (p * (p.log() - log_q)).sum(dim=1)
    return klw_soft


def relmae_soft(y_pred, y_true):
    """ Differentiable per-row relMAE, same formula as compute_wrMAE. """
    eps = y_pred - y_true
    mean_abs_diff = eps.abs().mean(dim=1)
    mean_abs_y    = torch.clamp(y_true.abs().mean(dim=1), min=EPS)
    return mean_abs_diff / mean_abs_y


# ---------------------------------------------------------------------------
# Exact (non-differentiable) metrics for reporting -- same formulas as
# bundle/starting_kit/kit_utils/metrics.py, reimplemented locally.
# ---------------------------------------------------------------------------

def real_residual_kl(y_true_case, y_pred_case, comp_masks, sigma_ref, n_bins=KL_N_BINS):
    eps = y_pred_case - y_true_case
    sigma_y = float(y_true_case.std()) + EPS

    sample_weight = np.zeros_like(eps)
    for cname, mask in comp_masks.items():
        sample_weight[mask] = KL_WEIGHTS.get(cname, 0.0)

    lim  = 5.0 * sigma_y
    bins = np.linspace(-lim, lim, n_bins + 1)
    dx   = bins[1] - bins[0]

    p, _ = np.histogram(eps, bins=bins, weights=sample_weight, density=True)
    p    = np.clip(p * dx, 1e-10, None)
    p   /= p.sum()

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    q = norm.pdf(bin_centers, loc=0.0, scale=sigma_ref) * dx
    q = np.clip(q, 1e-10, None)
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def real_relmae(y_true_case, y_pred_case):
    return float(np.mean(np.abs(y_pred_case - y_true_case)) / max(np.mean(np.abs(y_true_case)), EPS))


# ---------------------------------------------------------------------------
# Data loading -- train + both labeled test phases pooled into one set, since
# this is a search over already-labeled conditions, not a generalization
# exercise: we want the worst gap example among everything we can score.
# ---------------------------------------------------------------------------

def load_all(data_dir, max_sims_per_split=None):
    """ max_sims_per_split: keep only the first k simulations of each split.
    None (default, used for the real cluster run) keeps everything. Only
    meant as a knob for quick local sanity checks on a laptop/CPU, where
    the full dataset + soft-histogram tensors would be far too slow/heavy. """
    print('Loading data...', flush=True)
    # mmap: X_* arrays are only used for their (Minf, AoA, Pi) columns, one
    # row per simulation -- no need to pull the full point cloud into RAM.
    X_train = np.load(data_dir + 'splitv2/train_data.npy', mmap_mode='r')
    y_train = np.load(data_dir + 'splitv2/train_labels.npy')
    X_test1 = np.load(data_dir + 'splitv2/test_phase1_data.npy', mmap_mode='r')
    y_test1 = np.load(data_dir + 'splitv2/test_phase1_labels.npy')
    X_test2 = np.load(data_dir + 'splitv2/test_phase2_data.npy', mmap_mode='r')
    y_test2 = np.load(data_dir + 'splitv2/test_phase2_labels.npy')

    component_labels = np.load(data_dir + 'component_labels_unique.npy')
    with open(data_dir + 'component_map.json') as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    n_train = X_train.shape[0] // NWALLP
    n_test1 = X_test1.shape[0] // NWALLP
    n_test2 = X_test2.shape[0] // NWALLP
    print(f'n_train={n_train}  n_test1={n_test1}  n_test2={n_test2}', flush=True)

    conds_train = np.asarray(X_train[::NWALLP, COL_MINF:COL_PI + 1])
    conds_test1 = np.asarray(X_test1[::NWALLP, COL_MINF:COL_PI + 1])
    conds_test2 = np.asarray(X_test2[::NWALLP, COL_MINF:COL_PI + 1])

    conf_train = np.ones(n_train)
    conf_test1 = np.where(np.abs(conds_test1[:, 1]) < 10.0, 1.0, 0.5)
    conf_test2 = np.where(np.abs(conds_test2[:, 1]) < 10.0, 1.0, 0.5)

    Y_train = y_train.reshape(n_train, NWALLP)
    Y_test1 = y_test1.reshape(n_test1, NWALLP)
    Y_test2 = y_test2.reshape(n_test2, NWALLP)

    if max_sims_per_split is not None:
        k = max_sims_per_split
        conds_train, conds_test1, conds_test2 = conds_train[:k], conds_test1[:k], conds_test2[:k]
        conf_train, conf_test1, conf_test2 = conf_train[:k], conf_test1[:k], conf_test2[:k]
        Y_train, Y_test1, Y_test2 = Y_train[:k], Y_test1[:k], Y_test2[:k]
        n_train, n_test1, n_test2 = len(conds_train), len(conds_test1), len(conds_test2)
        print(f'subsampled to n_train={n_train}  n_test1={n_test1}  n_test2={n_test2}', flush=True)

    conds = np.concatenate([conds_train, conds_test1, conds_test2], axis=0)
    Y = np.concatenate([Y_train, Y_test1, Y_test2], axis=0)
    confidence = np.concatenate([conf_train, conf_test1, conf_test2])
    split = np.array(['train'] * n_train + ['test1'] * n_test1 + ['test2'] * n_test2)
    sim_in_split = np.concatenate([np.arange(n_train), np.arange(n_test1), np.arange(n_test2)])

    return conds, Y, confidence, split, sim_in_split, comp_masks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gap_mlp(conds, Y, comp_masks, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scaler   = StandardScaler()
    conds_sc = scaler.fit_transform(conds)

    mean_global = float(np.mean(Y))
    sigma_scale = args.sigma_scale_frac * mean_global   # bin range + reference scale for the soft KL
    sigma_ref   = args.sigma_ref_frac * mean_global      # reference used both in training and real eval
    print(f'global mean = {mean_global:.5f}  sigma_scale = {sigma_scale:.5f}  sigma_ref = {sigma_ref:.5f}', flush=True)

    n = conds.shape[0]
    edges   = torch.linspace(-5.0, 5.0, KL_N_BINS + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tau     = 10.0 / KL_N_BINS

    sw = np.zeros(NWALLP, dtype=np.float32)
    for cname, mask in comp_masks.items():
        sw[mask] = KL_WEIGHTS.get(cname, 0.0)
    w_pts = torch.tensor(sw / sw.sum(), device=device)

    q = norm.pdf(centers.cpu().numpy(), loc=0.0, scale=sigma_ref / sigma_scale)
    q = np.clip(q / q.sum(), 1e-10, None)
    log_q = torch.tensor(np.log(q), dtype=torch.float32, device=device)

    klw_soft = make_soft_kl_fn(w_pts, centers, log_q, tau)

    C = torch.tensor(conds_sc, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    mean_field = Yt.mean(dim=0).cpu().numpy()

    model = GlobalMLP(NWALLP, args.hidden, dropout=args.dropout, mean_field=mean_field).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)

    # Constrained checkpoint selection, feasibility first: once an epoch's
    # relMAE has satisfied the cap for the first time, only feasible epochs
    # are ever accepted again (and among those, the one with the highest
    # KL wins). Before that point, we just keep whichever epoch had the
    # lowest relMAE so far, since satisfying the constraint comes first.
    best_state, best_epoch, bad_epochs = None, -1, 0
    been_feasible = False
    best_feasible_gap = -float('inf')
    best_infeasible_relmae = float('inf')
    hinge_weight = args.hinge_weight

    print('Training MLP to MAXIMIZE KLw while forcing relMAE under the cap '
          f'(cap={args.relmae_cap}, hinge_weight={hinge_weight} -> escalates x{args.hinge_growth} '
          f'per epoch while the cap is violated, max {args.hinge_weight_max})...', flush=True)
    for epoch in range(args.n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_kl, tot_relmae, n_batches = 0.0, 0.0, 0
        for i in range(0, n, args.batch):
            idx  = perm[i:i + args.batch]
            pred = model(C[idx])
            kl_batch     = klw_soft(pred, Yt[idx], sigma_scale)
            relmae_batch = relmae_soft(pred, Yt[idx])
            # quadratic penalty above the cap, plus a small always-on linear
            # pull so relMAE keeps drifting down even once under the cap,
            # instead of the hinge going fully slack there.
            hinge = F.relu(relmae_batch - args.relmae_cap) ** 2
            loss = (-kl_batch + hinge_weight * hinge + args.relmae_pull * relmae_batch).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_kl     += kl_batch.mean().item()
            tot_relmae += relmae_batch.mean().item()
            n_batches  += 1
        sched.step()

        mean_kl, mean_relmae = tot_kl / n_batches, tot_relmae / n_batches
        gap = mean_kl - mean_relmae
        feasible = mean_relmae <= args.relmae_cap

        improved = False
        if feasible:
            if not been_feasible or gap > best_feasible_gap + 1e-4:
                best_feasible_gap = gap
                improved = True
            been_feasible = True
        elif not been_feasible and mean_relmae < best_infeasible_relmae - 1e-4:
            best_infeasible_relmae = mean_relmae
            improved = True

        if improved:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch, bad_epochs = epoch, 0
        else:
            bad_epochs += 1

        # escalate the penalty while the constraint is still violated, so
        # it eventually dominates regardless of how --hinge_weight was set
        if not feasible:
            hinge_weight = min(hinge_weight * args.hinge_growth, args.hinge_weight_max)

        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            flag = 'OK' if feasible else 'violated'
            print(f'epoch {epoch:4d}  soft KLw={mean_kl:.4f}  relMAE={mean_relmae:.4f} ({flag})  '
                  f'gap={gap:.4f}  hinge_w={hinge_weight:.1f}  best_epoch={best_epoch}', flush=True)
        if bad_epochs >= args.patience:
            print(f'plateau at epoch {epoch} (best epoch {best_epoch})', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler, sigma_ref


# ---------------------------------------------------------------------------
# Ranking with the exact (hard-histogram) metrics
# ---------------------------------------------------------------------------

def rank_by_real_gap(model, scaler, conds, Y, confidence, split, sim_in_split, comp_masks, sigma_ref, device):
    with torch.no_grad():
        C = torch.tensor(scaler.transform(conds), dtype=torch.float32, device=device)
        Yhat = model(C).cpu().numpy().astype(np.float64)

    valid = np.flatnonzero(confidence >= 1.0)
    rows = []
    for i in valid:
        kl = real_residual_kl(Y[i], Yhat[i], comp_masks, sigma_ref)
        rmae = real_relmae(Y[i], Yhat[i])
        rows.append(dict(idx=int(i), split=str(split[i]), sim_in_split=int(sim_in_split[i]),
                          real_KLw=kl, real_relMAE=rmae, gap=kl - rmae))
    rows.sort(key=lambda r: r['gap'], reverse=True)
    return rows, Yhat


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--max_sims_per_split', type=int, default=None,
                   help='keep only the first k sims of each split, for a quick local sanity check '
                        '(leave unset for the real cluster run).')
    p.add_argument('--hidden', type=int, nargs='+', default=[128, 256, 512])
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--n_epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--relmae_cap', type=float, default=0.05,
                   help='relMAE ceiling: quadratic penalty above it, small constant pull below it '
                        '(see --relmae_pull). Keep comfortably under wrMAE values a normal model gets.')
    p.add_argument('--hinge_weight', type=float, default=50.0,
                   help='starting weight of the over-cap penalty; escalates automatically (see --hinge_growth) '
                        'so the cap gets enforced regardless of this initial value.')
    p.add_argument('--hinge_growth', type=float, default=1.15,
                   help='multiply hinge_weight by this factor every epoch relMAE is still above the cap.')
    p.add_argument('--hinge_weight_max', type=float, default=1e6)
    p.add_argument('--relmae_pull', type=float, default=1.0,
                   help='small constant linear penalty on relMAE, always on (even under the cap), so it keeps '
                        'drifting toward 0 instead of the hinge going fully slack once under the cap.')
    p.add_argument('--sigma_scale_frac', type=float, default=0.1,
                   help='soft-KL bin half-range / scale, as a fraction of the global mean field.')
    p.add_argument('--sigma_ref_frac', type=float, default=0.01,
                   help='KLw reference std, as a fraction of the global mean field '
                        '(matches sigma_ref = 1%% of global mean used at scoring time).')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--out_prefix', default='gap_mlp')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    conds, Y, confidence, split, sim_in_split, comp_masks = load_all(args.data_dir, args.max_sims_per_split)

    model, scaler, sigma_ref = train_gap_mlp(conds, Y, comp_masks, args, device)

    torch.save({'state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
                'hidden': args.hidden, 'dropout': args.dropout,
                'sigma_ref': sigma_ref, 'nwallp': NWALLP},
               f'{args.out_prefix}_checkpoint.pt')
    print(f'Saved: {args.out_prefix}_checkpoint.pt', flush=True)

    print('\nScoring every (labeled) simulation with the exact histogram KLw / relMAE...', flush=True)
    rows, Yhat = rank_by_real_gap(model, scaler, conds, Y, confidence, split, sim_in_split, comp_masks, sigma_ref, device)

    print(f'\nTop {args.top_k} examples where KLw is bad but relMAE (wrMAE-like) looks fine:')
    print(f'{"rank":>4}  {"split":<6}  {"sim":>5}  {"Minf":>7}  {"AoA":>7}  {"Pi":>10}  '
          f'{"real_KLw":>10}  {"real_relMAE":>12}  {"gap":>10}')
    for rank, r in enumerate(rows[:args.top_k]):
        Minf, AoA, Pi = conds[r['idx']]
        print(f'{rank:>4}  {r["split"]:<6}  {r["sim_in_split"]:>5}  {Minf:>7.4f}  {AoA:>7.2f}  {Pi:>10.4g}  '
              f'{r["real_KLw"]:>10.4f}  {r["real_relMAE"]:>12.4f}  {r["gap"]:>10.4f}')

    with open(f'{args.out_prefix}_report.csv', 'w') as f:
        f.write('rank,idx,split,sim_in_split,Minf,AoA,Pi,real_KLw,real_relMAE,gap\n')
        for rank, r in enumerate(rows):
            Minf, AoA, Pi = conds[r['idx']]
            f.write(f'{rank},{r["idx"]},{r["split"]},{r["sim_in_split"]},'
                    f'{Minf:.6f},{AoA:.4f},{Pi:.6g},{r["real_KLw"]:.6f},{r["real_relMAE"]:.6f},{r["gap"]:.6f}\n')
    print(f'\nSaved: {args.out_prefix}_report.csv ({len(rows)} rows)', flush=True)

    top_idx = [r['idx'] for r in rows[:args.top_k]]
    np.savez(f'{args.out_prefix}_top_examples.npz',
             idx=np.array(top_idx),
             y_true=Y[top_idx],
             y_pred=Yhat[top_idx],
             conds=conds[top_idx])
    print(f'Saved: {args.out_prefix}_top_examples.npz (y_true/y_pred fields for the top {len(top_idx)} examples)', flush=True)


if __name__ == '__main__':
    main()
