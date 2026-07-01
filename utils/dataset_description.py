#to check after....

import numpy as np
from collections import Counter

nwallp = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8

print('Loading data...')
X_train = np.load(DATA_DIR + 'splitv2/train_data.npy')
X_test1 = np.load(DATA_DIR + 'splitv2/test_phase1_data.npy')
X_test2 = np.load(DATA_DIR + 'splitv2/test_phase2_data.npy')

train_conds = X_train[::nwallp, COL_MINF:COL_PI+1]
test1_conds = X_test1[::nwallp, COL_MINF:COL_PI+1]
test2_conds = X_test2[::nwallp, COL_MINF:COL_PI+1]

all_conds = np.vstack([train_conds, test1_conds, test2_conds])
labels = (
    ['train'] * len(train_conds) +
    ['phase1'] * len(test1_conds) +
    ['phase2'] * len(test2_conds)
)

print(f'\nTotal simulations: {len(all_conds)}')
print(f'  Train  : {len(train_conds)}')
print(f'  Phase 1: {len(test1_conds)}')
print(f'  Phase 2: {len(test2_conds)}')

for col, name, fmt in [
    (COL_MINF - COL_MINF, 'Mach (M_inf)',  '{:.2f}'),
    (COL_AOA  - COL_MINF, 'AoA (alpha)',   '{:.1f}'),
    (COL_PI   - COL_MINF, 'Pi (Pa)',        '{:.0f}'),
]:
    vals = np.round(all_conds[:, col], 6)
    unique = sorted(np.unique(vals))
    counts = Counter(vals)
    print(f'\n── {name} ── {len(unique)} unique values')
    print(f'  {"Value":>12}  {"Total":>6}  {"Train":>6}  {"Phase1":>7}  {"Phase2":>7}')
    print(f'  {"-"*12}  {"-"*6}  {"-"*6}  {"-"*7}  {"-"*7}')
    for v in unique:
        mask = np.isclose(all_conds[:, col], v)
        n_all = mask.sum()
        n_tr  = np.isclose(train_conds[:, col], v).sum()
        n_t1  = np.isclose(test1_conds[:, col], v).sum()
        n_t2  = np.isclose(test2_conds[:, col], v).sum()
        print(f'  {fmt.format(v):>12}  {n_all:>6}  {n_tr:>6}  {n_t1:>7}  {n_t2:>7}')