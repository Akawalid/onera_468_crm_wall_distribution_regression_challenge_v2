"""
Split v3: rebalances the Mach-block split so train >= 2/3 of the 468
simulations (supervisor's floor, see the ONERA feedback on splitting), while
keeping distinct interpolation (Phase 1) / extrapolation (Phase 2) test sets.

Unlike train_test_splitting.py (v2), this derives (Minf, AoA, Pi) directly
from the block-averaged columns of X9_ALL_POINT_fl32.npy itself instead of
joining on the row order of traintest_splitting1_MinfAoAPi_with_scores.csv --
that CSV's row order does NOT match the npy's block order (verified: it
diverges from row 3 onward), so indexing by its 'idx' column would have
silently pulled the wrong simulations.

    Train         : 0.50, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93  (9 Mach,  324 sims, 69.2%)
    Test Phase 1  : 0.84, 0.86                                           (2 Mach,   72 sims, 15.4%) -- interpolation
    Test Phase 2  : 0.30, 0.96                                           (2 Mach,   72 sims, 15.4%) -- extrapolation
"""

import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, '.FILES_RHO_ALL_POINTS_reduitfloat32')
X9_FILE = os.path.join(RAW_DIR, 'X9_ALL_POINT_fl32.npy')
RHO_FILE = os.path.join(RAW_DIR, 'RHO_ALL_POINT_fl32.npy')

OUT_DIR = os.path.join(ROOT, 'data', 'splitv3')
COMP_SRC_DIR = os.path.join(ROOT, 'data')
COMP_FILES = ['component_labels_unique.npy', 'component_map.json', 'component_nn_distances.npy']

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8
EPS = 1e-6

MACH_TRAIN = [0.50, 0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93]
MACH_TEST_PHASE1 = [0.84, 0.86]
MACH_TEST_PHASE2 = [0.30, 0.96]
AOA_EXTREME_THRESHOLD = 10.0


def assign_split(mach):
    if any(abs(mach - m) < EPS for m in MACH_TRAIN):
        return 'train'
    if any(abs(mach - m) < EPS for m in MACH_TEST_PHASE1):
        return 'test_phase1'
    if any(abs(mach - m) < EPS for m in MACH_TEST_PHASE2):
        return 'test_phase2'
    return 'unknown'


def extract_blocks(arr, block_idx, nwallp=NWALLP):
    return np.concatenate([arr[i * nwallp:(i + 1) * nwallp] for i in block_idx], axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    x9 = np.load(X9_FILE, mmap_mode='r')
    n_sims = x9.shape[0] // NWALLP
    conds = np.array(x9[::NWALLP, COL_MINF:COL_PI + 1])  # (n_sims, 3): Minf, AoA, Pi

    splits = np.array([assign_split(m) for m in conds[:, 0]])
    unknown = np.flatnonzero(splits == 'unknown')
    if unknown.size:
        raise ValueError(f'Mach values with no split assignment: {conds[unknown, 0]}')

    weights = np.where(np.abs(conds[:, 1]) < AOA_EXTREME_THRESHOLD, 1.0, 0.5)

    print('Simulation counts per split:')
    for name in ('train', 'test_phase1', 'test_phase2'):
        idx = np.flatnonzero(splits == name)
        print(f'  {name:12} : {len(idx):3} sims  ({len(idx) / n_sims:.1%})'
              f'  weight=1.0: {(weights[idx] >= 1.0).sum():3}'
              f'  weight=0.5: {(weights[idx] < 1.0).sum():3}')

    rho = np.load(RHO_FILE, mmap_mode='r')[:, 0]

    for name in ('train', 'test_phase1', 'test_phase2'):
        idx = np.flatnonzero(splits == name)
        print(f'\nExtracting {name} ({len(idx)} sims)...')
        X = extract_blocks(x9, idx)
        y = extract_blocks(rho, idx)
        np.save(os.path.join(OUT_DIR, f'{name}_data.npy'), X)
        np.save(os.path.join(OUT_DIR, f'{name}_labels.npy'), y)
        np.save(os.path.join(OUT_DIR, f'{name}_weights.npy'), weights[idx])
        print(f'  {name}_data.npy   {X.shape}')
        print(f'  {name}_labels.npy {y.shape}')
        print(f'  {name}_weights.npy {weights[idx].shape}')

    print('\nCopying component files...')
    for fname in COMP_FILES:
        src = os.path.join(COMP_SRC_DIR, fname)
        dst = os.path.join(OUT_DIR, fname)
        shutil.copy2(src, dst)
        print(f'  {fname} -> {dst}')

    print('\nDone.')


if __name__ == '__main__':
    main()
