"""
Plots ground truth / adversarial prediction / absolute error for one or more
simulations found by mlp_kl_wrmae_gap.py -- same 3-panel 3-D scatter layout
as plot_condition() in copy.ipynb, except the title reports KLw and the
per-simulation relMAE (the gap the adversarial MLP was trained to open)
instead of wrMAE/R2.

Takes the *_top_examples.npz and *_report.csv produced by
mlp_kl_wrmae_gap.py, re-loads the wall-point (x, y, z) coordinates for the
requested simulation(s) from the raw split file (mmap'd, so only that one
simulation's block is actually read), and saves one PNG per requested rank.
"""

import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NWALLP = 260774

SPLIT_FILES = {
    'train': 'splitv2/train_data.npy',
    'test1': 'splitv2/test_phase1_data.npy',
    'test2': 'splitv2/test_phase2_data.npy',
}


def _scale(field):
    """ Same robust percentile-based color scaling as copy.ipynb's plot_condition. """
    f03, f50, f97 = np.percentile(field, [3., 50., 97.])
    slopem = (f50 - f03) / 0.47
    slopep = (f97 - f50) / 0.47
    return max(0., f50 - 0.6 * slopem), f50 + 0.6 * slopep


def plot_gap_example(coords, y_true, y_pred, kl, relmae, gap, Minf, AoA, Pi, label, out_path, elev=80):
    error = np.abs(y_true - y_pred)
    X_coord, Y_coord, Z_coord = coords[:, 0], coords[:, 1], coords[:, 2]

    fields = [y_true, y_pred, error]
    titles = ['Ground truth rho', 'Adversarial prediction rho', 'Absolute error']
    cmaps  = ['jet', 'jet', 'Reds']

    fig = plt.figure(figsize=(18, 6), dpi=120)
    fig.suptitle(
        f'{label}  --  Minf={Minf:.2f}  AoA={AoA:.1f} deg  Pi={Pi:.0e} Pa\n'
        f'real KLw = {kl:.4f}  (bad)      sim relMAE = {relmae:.4f}  (looks fine)      '
        f'gap (KLw - relMAE) = {gap:.4f}',
        fontsize=12
    )
    for col, (field, title, cmap) in enumerate(zip(fields, titles, cmaps)):
        gs  = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[20, 1])
        ax  = fig.add_subplot(gs[0, col], projection='3d')
        fmin, fmax = _scale(field)
        sca = ax.scatter3D(X_coord, Y_coord, Z_coord,
                            c=field, vmin=fmin, vmax=fmax,
                            cmap=cmap, s=0.3, alpha=0.8)
        ax.view_init(elev=elev, azim=120)
        ax.set_xlim(X_coord.min() - 5.0, X_coord.max() + 5.0)
        ax.set_ylim(Y_coord.min() - 5.0, Y_coord.max() + 5.0)
        ax.set_zlim(Z_coord.min(), Z_coord.max())
        lims = np.array([getattr(ax, f'get_{a}lim')() for a in 'xyz'])
        ax.set_box_aspect(np.ptp(lims, axis=1), zoom=1.)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
        cax  = fig.add_subplot(gs[1, col])
        cbar = fig.colorbar(sca, cax=cax, orientation='horizontal')
        cbar.set_label('rho' if col < 2 else '|error|', size=8)
        cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def load_report_rows(report_path):
    with open(report_path) as f:
        return list(csv.DictReader(f))


def load_coords(data_dir, split, sim_in_split):
    path = data_dir + SPLIT_FILES[split]
    X = np.load(path, mmap_mode='r')
    block = X[sim_in_split * NWALLP:(sim_in_split + 1) * NWALLP, :3]
    return np.asarray(block)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--npz', required=True, help='*_top_examples.npz from mlp_kl_wrmae_gap.py')
    p.add_argument('--report', required=True, help='*_report.csv from mlp_kl_wrmae_gap.py')
    p.add_argument('--data_dir', required=True, help='directory containing splitv2/ (same as --data_dir used to train)')
    p.add_argument('--ranks', type=int, nargs='+', default=[0], help='which ranks (0 = biggest gap) to plot')
    p.add_argument('--out_prefix', default='gap_example')
    p.add_argument('--elev', type=float, default=80.0)
    args = p.parse_args()

    npz = np.load(args.npz)
    rows = load_report_rows(args.report)  # rows[i] corresponds to npz arrays at position i

    for rank in args.ranks:
        row = rows[rank]
        coords = load_coords(args.data_dir, row['split'], int(row['sim_in_split']))
        y_true = npz['y_true'][rank]
        y_pred = npz['y_pred'][rank]
        Minf, AoA, Pi = float(row['Minf']), float(row['AoA']), float(row['Pi'])
        kl, relmae, gap = float(row['real_KLw']), float(row['real_relMAE']), float(row['gap'])
        label = f'rank {rank}  [{row["split"]} sim {row["sim_in_split"]}]'
        out_path = f'{args.out_prefix}_rank{rank}.png'
        plot_gap_example(coords, y_true, y_pred, kl, relmae, gap, Minf, AoA, Pi, label, out_path, elev=args.elev)


if __name__ == '__main__':
    main()
