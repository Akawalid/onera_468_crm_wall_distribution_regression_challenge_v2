#srun --account=tau --partition=gpu-best --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:2 --mem=64G --time=01:00:00 --pty bash
# KNN from charbonnier to the current geometry, with taking in consideration units and normalization 
import numpy as np
import json
from pathlib import Path
from scipy.spatial import cKDTree

CHALLENGE_DIR   = Path(".FILES_RHO_ALL_POINTS_reduitfloat32/COORDINATES")
CHARBONNIER_DIR = Path(".")
OUTPUT_DIR      = Path("./output")

INCHES_TO_METERS = 1.0 / 39.3701

COMPONENTS = ["fuselage", "nacelle", "pylon", "wing"]

print("Chargement des points challenge (mètres)...")
X = np.load(CHALLENGE_DIR / "XX.npy")
Y = np.load(CHALLENGE_DIR / "YY.npy")
Z = np.load(CHALLENGE_DIR / "ZZ.npy")
challenge_xyz = np.column_stack([X, Y, Z])
print(f"  {len(challenge_xyz):,} pts  X=[{X.min():.3f}, {X.max():.3f}]  Y=[{Y.min():.3f}, {Y.max():.3f}]  Z=[{Z.min():.3f}, {Z.max():.3f}]")

print("\nChargement des points Charbonnier (pouces → mètres)...")
charb_xyz_list   = []
charb_label_list = []
for i, comp in enumerate(COMPONENTS):
    fpath = CHARBONNIER_DIR / f"charbonnier_{comp}.npy"
    if not fpath.exists():
        print(f"  [WARN] manquant : {fpath}")
        continue
    pts = np.load(fpath) * INCHES_TO_METERS
    charb_xyz_list.append(pts)
    charb_label_list.append(np.full(len(pts), i, dtype=np.int32))
    print(f"  {comp:12s} : {len(pts):>7,} pts  X=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")

charb_xyz    = np.vstack(charb_xyz_list)
charb_labels = np.concatenate(charb_label_list)
print(f"  TOTAL : {len(charb_xyz):,} pts")

print("\nConstruction du KD-tree...")
tree = cKDTree(charb_xyz)

print("KNN (k=1)...")
distances, indices = tree.query(challenge_xyz, k=1, workers=-1)
labels = charb_labels[indices]

print(f"\nDistances au plus proche voisin (mètres) :")
print(f"  mean : {distances.mean():.5f}")
print(f"  p50  : {np.percentile(distances, 50):.5f}")
print(f"  p95  : {np.percentile(distances, 95):.5f}")
print(f"  p99  : {np.percentile(distances, 99):.5f}")
print(f"  max  : {distances.max():.5f}")

print(f"\nRépartition des labels :")
for i, name in enumerate(COMPONENTS):
    cnt = (labels == i).sum()
    print(f"  {name:12s} : {cnt:>8,} pts  ({100*cnt/len(labels):.1f}%)")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_DIR / "component_labels_unique.npy", labels)
np.save(OUTPUT_DIR / "component_nn_distances.npy",  distances)
code_map = {str(i): name for i, name in enumerate(COMPONENTS)}
with open(OUTPUT_DIR / "component_map.json", "w") as f:
    json.dump(code_map, f, indent=2)

print(f"\nSauvegardé dans {OUTPUT_DIR}/")
print(f"  component_labels_unique.npy  {labels.shape}")
print(f"  component_nn_distances.npy   {distances.shape}")
print(f"  component_map.json")