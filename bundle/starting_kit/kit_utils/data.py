"""
Data loading + cross-validation splitting, shared by every baseline in the
starting kit.

We only ever load `input_data` (train_data.npy / train_labels.npy) plus the
small component files -- never test_data.npy or anything under
reference_data/. Participants never have access to the test set, so all
model selection here has to happen through cross-validation on the
training simulations alone.
"""

import json
import os

import numpy as np

from .metrics import NWALLP, COL_MINF, COL_AOA, COL_PI, EPS


def load_train(input_dir):
    """ Load the training features/labels only (no test data). """
    X_train = np.load(os.path.join(input_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(input_dir, 'train_labels.npy'))[:, 0]
    return X_train, y_train


def load_component_masks(input_dir):
    """ {component_name: boolean mask of shape (NWALLP,)} """
    component_labels = np.load(os.path.join(input_dir, 'component_labels_unique.npy'))
    with open(os.path.join(input_dir, 'component_map.json')) as f:
        component_map = {int(k): v for k, v in json.load(f).items()}
    return {cname: (component_labels == cid) for cid, cname in component_map.items()}


def train_conditions(X_train, nwallp=NWALLP):
    """ One (Minf, AoA, Pi) row per simulation, and the simulation count. """
    n_sims = X_train.shape[0] // nwallp
    conds  = X_train[::nwallp, COL_MINF:COL_PI + 1]
    return conds, n_sims


def confidence_weights(conds):
    """ Same rule as the official reference data: conditions with
    |AoA| >= 10 deg count half as much (they're the noisier, separated-flow
    cases). """
    return np.where(np.abs(conds[:, 1]) < 10.0, 1.0, 0.5)


def select_sims(X, y, sim_idx, nwallp=NWALLP):
    """ Concatenate the (X, y) rows for just the given simulation indices,
    preserving per-simulation block structure. """
    X_sel = np.concatenate([X[i * nwallp:(i + 1) * nwallp] for i in sim_idx], axis=0)
    y_sel = np.concatenate([y[i * nwallp:(i + 1) * nwallp] for i in sim_idx], axis=0)
    return X_sel, y_sel


def mach_fold_splits(conds, epsilon=EPS):
    """ Leave-two-consecutive-Machs-out CV folds: yields
    (train_sim_idx, val_sim_idx, label) so that every simulation gets
    validated on exactly once, while training folds still cover a wide
    Mach range (a single Mach held out alone would be too little
    validation data per fold). """
    mach_values = np.unique(conds[:, 0])
    for i in range(len(mach_values) - 1):
        m0, m1 = mach_values[i], mach_values[i + 1]
        val_mask = (np.abs(conds[:, 0] - m0) < epsilon) | (np.abs(conds[:, 0] - m1) < epsilon)
        yield np.flatnonzero(~val_mask), np.flatnonzero(val_mask), f'{m0:.2f}+{m1:.2f}'
