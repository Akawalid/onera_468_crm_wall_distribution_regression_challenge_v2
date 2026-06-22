#srun --account=tau --partition=gpu-best --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:2 --mem=64G --time=01:00:00 --pty bash
import numpy as np
import pandas as pd

path="/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/"

# Config
MAIN_FILE = path + 'X9_ALL_POINT_fl32.npy'
COEF_FILE = path + 'traintest_splitting1_MinfAoAPi_with_scores.csv'

MACH_TEST_PHASE1 = [0.82, 0.84, 0.85, 0.86]
MACH_TEST_PHASE2 = [0.30, 0.96]
MACH_TRAIN       = [0.50, 0.70, 0.75, 0.80, 0.88,0.90, 0.93]

AOA_EXTREME_THRESHOLD = 10.0

df = pd.read_csv(COEF_FILE)
df = df.rename(columns={'Unnamed: 0': 'idx'})

# Assigner les weights selon AoA
df['weight'] = np.where(np.abs(df['AoA']) < AOA_EXTREME_THRESHOLD, 1.0, 0.5)

# Assigner les splits selon Mach
def assign_split(mach):
    if mach in MACH_TRAIN:
        return 'train'
    elif mach in MACH_TEST_PHASE1:
        return 'test_phase1'
    elif mach in MACH_TEST_PHASE2:
        return 'test_phase2'
    else:
        return 'unknown'

df['split'] = df['Mach'].apply(assign_split)

# Vérification
print("Répartition des simulations :")
print(df.groupby(['split', 'Mach']).size().reset_index(name='count').to_string())
print()
print("Weights par split :")
for split in ['train', 'test_phase1', 'test_phase2']:
    sub = df[df['split'] == split]
    n_full = (sub['weight'] == 1.0).sum()
    n_half = (sub['weight'] == 0.5).sum()
    print(f"  {split:15} : {len(sub):4} sims | weight=1.0: {n_full:4} | weight=0.5: {n_half:4}")

# Extraire les indices par split
train_idx      = df[df['split'] == 'train']['idx'].values
test_phase1_idx = df[df['split'] == 'test_phase1']['idx'].values
test_phase2_idx = df[df['split'] == 'test_phase2']['idx'].values

train_weights      = df[df['split'] == 'train']['weight'].values
test_phase1_weights = df[df['split'] == 'test_phase1']['weight'].values
test_phase2_weights = df[df['split'] == 'test_phase2']['weight'].values

# Charger les données principales et extraire les splits
data     = np.load(MAIN_FILE)
n_points = data.shape[0] // 468

def extract_split(data, indices, n_points):
    rows = []
    for idx in indices:
        start = idx * n_points
        end   = start + n_points
        rows.append(data[start:end])
    return np.vstack(rows)

print("\nExtraction des données...")
train_data      = extract_split(data, train_idx,       n_points)
test_phase1_data = extract_split(data, test_phase1_idx, n_points)
test_phase2_data = extract_split(data, test_phase2_idx, n_points)

LABEL_FILE = path + 'RHO_ALL_POINT_fl32.npy'
labels = np.load(LABEL_FILE)[:, 0]

train_labels       = extract_split(labels, train_idx,       n_points).reshape(-1)
test_phase1_labels = extract_split(labels, test_phase1_idx, n_points).reshape(-1)
test_phase2_labels = extract_split(labels, test_phase2_idx, n_points).reshape(-1)

print(f"Train      : {train_data.shape}")
print(f"Test Phase1: {test_phase1_data.shape}")
print(f"Test Phase2: {test_phase2_data.shape}")

# Sauvegarder
np.save(path + 'splitv2/train_data.npy',       train_data)
np.save(path + 'splitv2/test_phase1_data.npy', test_phase1_data)
np.save(path + 'splitv2/test_phase2_data.npy', test_phase2_data)

np.save(path + 'splitv2/train_weights.npy',       train_weights)
np.save(path + 'splitv2/test_phase1_weights.npy', test_phase1_weights)
np.save(path + 'splitv2/test_phase2_weights.npy', test_phase2_weights)

np.save(path + 'splitv2/train_labels.npy',       train_labels)
np.save(path + 'splitv2/test_phase1_labels.npy', test_phase1_labels)
np.save(path + 'splitv2/test_phase2_labels.npy', test_phase2_labels)

# df[df['split'] == 'train'].to_csv('train_coefs.csv',       index=False)
# df[df['split'] == 'test_phase1'].to_csv('test_phase1_coefs.csv', index=False)
# df[df['split'] == 'test_phase2'].to_csv('test_phase2_coefs.csv', index=False)

print("\nFichiers sauvegardés.")