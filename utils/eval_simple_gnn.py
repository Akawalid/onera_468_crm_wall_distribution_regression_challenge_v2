"""Evaluate a saved SimpleGNN checkpoint (utils/simple_gnn.py) on held-out splits.

train_simple_gnn.py only evaluates test_phase1 during training, for monitoring. This script
re-evaluates a saved checkpoint on any split -- in particular test_phase2, which the training
script never touches, so it's the closer-to-independent check of how well the model
generalizes rather than just fits the split it was watched against while training.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.base_models import evaluate
from utils.simple_gnn import SimpleGNN
from utils.train_simple_gnn import NWALLP, load_split, load_static_graph, predict_split


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--data_dir', default='data/')
    p.add_argument('--split_dir', default='splitv2', help='must match the split the checkpoint was trained on.')
    p.add_argument('--checkpoint', default='utils/simple_gnn_model.pt')
    p.add_argument('--k', type=int, default=20, help='must match the k the checkpoint was trained with.')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--splits', nargs='+', default=['test_phase1', 'test_phase2'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    cache_dir = os.path.join(args.data_dir, 'cache')

    print('Loading static graph/features...', flush=True)
    coords, edge_index, edge_weight, static_feats, component_labels, component_map = load_static_graph(
        args.data_dir, args.k, cache_dir, split_dir=args.split_dir
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    static_feats = static_feats.to(device)
    comp_masks = {cname: component_labels == cid for cid, cname in component_map.items()}

    y_train, _, n_train, train_conds = load_split(args.data_dir, 'train', split_dir=args.split_dir)
    cond_mean = train_conds.mean(axis=0)
    cond_std = train_conds.std(axis=0)
    cond_std[cond_std == 0] = 1.0
    sigma_ref = 0.01 * float(np.mean(y_train))

    in_dim = static_feats.shape[1] + 3  # + (Minf, AoA, Pi)
    model = SimpleGNN(in_dim, args.hidden_dim, out_dim=1, n_layers=args.n_layers).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}', flush=True)

    for split in args.splits:
        y, weights, n_sims, conds = load_split(args.data_dir, split, split_dir=args.split_dir)
        preds = predict_split(model, n_sims, conds, edge_index, edge_weight, static_feats,
                               device, cond_mean, cond_std)
        w = weights if weights is not None else np.ones(n_sims)
        res = evaluate(np.asarray(y), preds, w, NWALLP, comp_masks, sigma_ref)
        print(f'{split}  (n_sims={n_sims})  R2={res["r2"]:.4f}  worst_rMAE={res["worst_rmae"]:.4f}  '
              f'mean_rMAE={res["mean_rmae"]:.4f}  mean_KLw={res["mean_klw"]:.4f}  '
              f'max_KLw={res["max_klw"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
