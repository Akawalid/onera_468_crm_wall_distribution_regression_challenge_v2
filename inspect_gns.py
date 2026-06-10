"""
assign_component_labels.py
==========================
Classifie chaque point de la peau du challenge ONERA 468 CRM
par composant géométrique : fuselage, wing, pylon, nacelle.

Approche : règles spatiales pures sur (x, y, z) — pas besoin de CGNS.
Les seuils ont été déterminés par analyse des histogrammes des 260 774
points du fichier X9_ALL_POINT_fl32.npy.

Géométrie CRM (unités du challenge) :
  x : [2.35, 65.09]  axe longitudinal nez→queue
  y : [0.00, 29.45]  envergure  (0 = plan de symétrie)
  z : [-0.07, 8.72]  hauteur

Composants identifiés :
  fuselage : y < 1.5  (corps, plan de symétrie)
  nacelle  : y=[5,12]  z < 3.0  x=[24, 31]  (sous l'aile)
  pylon    : y=[8,12]  z=[3, 5]  x=[24, 39]  (entre nacelle et aile)
  wing     : tout le reste

Outputs
-------
  component_labels_unique.npy  shape (260774,)  int32
  component_labels_full.npy    shape (N_sims x 260774,)  int32
  component_labels.csv         x, y, z, component, component_code
  component_map.json           code -> nom

Usage
-----
    python assign_component_labels.py \\
        --points X9_ALL_POINT_fl32.npy \\
        --out_dir ./output

    # Ajuster les seuils si besoin :
    python assign_component_labels.py \\
        --points X9_ALL_POINT_fl32.npy \\
        --y_fus 1.5 --z_nac 3.0 --y_nac_lo 5.0 --y_nac_hi 12.0 \\
        --x_nac_lo 24.0 --x_nac_hi 31.0 \\
        --y_pyl_lo 8.0 --y_pyl_hi 12.0 --z_pyl_lo 3.0 --z_pyl_hi 5.2 \\
        --x_pyl_lo 24.0 --x_pyl_hi 39.0 \\
        --out_dir ./output
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_WALL = 260_774

COMP_TO_CODE = {"fuselage": 0, "nacelle": 1, "pylon": 2, "wing": 3}
CODE_TO_COMP = {v: k for k, v in COMP_TO_CODE.items()}


def classify_points(xyz, args):
    """
    Retourne un array int32 (N,) avec le code composant de chaque point.
    Ordre de priorité : nacelle > pylon > fuselage > wing
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    labels = np.full(len(xyz), COMP_TO_CODE["wing"], dtype=np.int32)

    mask_fus = y < args.y_fus
    labels[mask_fus] = COMP_TO_CODE["fuselage"]

    mask_pyl = (
        (y >= args.y_pyl_lo) & (y <= args.y_pyl_hi) &
        (z >= args.z_pyl_lo) & (z <= args.z_pyl_hi) &
        (x >= args.x_pyl_lo) & (x <= args.x_pyl_hi)
    )
    labels[mask_pyl] = COMP_TO_CODE["pylon"]

    mask_nac = (
        (y >= args.y_nac_lo) & (y <= args.y_nac_hi) &
        (z < args.z_nac) &
        (x >= args.x_nac_lo) & (x <= args.x_nac_hi)
    )
    labels[mask_nac] = COMP_TO_CODE["nacelle"]

    return labels


def print_distribution(labels, xyz, title="Distribution par composant"):
    print(f"\n{title} :")
    n = len(labels)
    for code, name in CODE_TO_COMP.items():
        m = labels == code
        cnt = m.sum()
        if cnt == 0:
            continue
        x, y, z = xyz[m, 0], xyz[m, 1], xyz[m, 2]
        print(f"  {name:10s} : {cnt:7d} pts ({100*cnt/n:.1f}%)"
              f"  x=[{x.min():.2f},{x.max():.2f}]"
              f"  y=[{y.min():.2f},{y.max():.2f}]"
              f"  z=[{z.min():.2f},{z.max():.2f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", required=True,
                        help="Fichier .npy des points (shape [N_sims*260774, 9])")
    parser.add_argument("--out_dir", default="./output")

    g = parser.add_argument_group("Seuils géométriques (ajustables)")
    g.add_argument("--y_fus",    type=float, default=1.5,
                   help="y < seuil → fuselage (défaut: 1.5)")
    g.add_argument("--y_nac_lo", type=float, default=5.0)
    g.add_argument("--y_nac_hi", type=float, default=12.0)
    g.add_argument("--z_nac",    type=float, default=3.0,
                   help="z < seuil → nacelle (dans la fenêtre y_nac)")
    g.add_argument("--x_nac_lo", type=float, default=24.0)
    g.add_argument("--x_nac_hi", type=float, default=31.0)
    g.add_argument("--y_pyl_lo", type=float, default=8.0)
    g.add_argument("--y_pyl_hi", type=float, default=12.0)
    g.add_argument("--z_pyl_lo", type=float, default=3.0)
    g.add_argument("--z_pyl_hi", type=float, default=5.2)
    g.add_argument("--x_pyl_lo", type=float, default=24.0)
    g.add_argument("--x_pyl_hi", type=float, default=39.0)

    args = parser.parse_args()

    points_path = Path(args.points)
    out_dir = Path(args.out_dir)

    if not points_path.exists():
        print(f"[ERREUR] Fichier introuvable : {points_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  ONERA CRM — classification géométrique par composant")
    print("=" * 60)

    print("\n[1] Chargement des points …")
    arr = np.load(points_path, mmap_mode="r")
    print(f"    Shape brut : {arr.shape}  dtype : {arr.dtype}")

    total = len(arr)
    n_sims = total // N_WALL
    remainder = total % N_WALL
    if remainder != 0:
        print(f"    [WARNING] {total} % {N_WALL} = {remainder} — vérifier N_WALL")

    print(f"    Détecté : {n_sims} simulations x {N_WALL} points")

    xyz_unique = arr[:N_WALL, :3].astype(np.float64)
    print(f"    Coordonnées uniques : {xyz_unique.shape}")
    print(f"    x=[{xyz_unique[:,0].min():.3f}, {xyz_unique[:,0].max():.3f}]  "
          f"y=[{xyz_unique[:,1].min():.3f}, {xyz_unique[:,1].max():.3f}]  "
          f"z=[{xyz_unique[:,2].min():.3f}, {xyz_unique[:,2].max():.3f}]")

    print("\n[2] Seuils utilisés :")
    print(f"    fuselage : y < {args.y_fus}")
    print(f"    nacelle  : y=[{args.y_nac_lo},{args.y_nac_hi}]  z<{args.z_nac}  x=[{args.x_nac_lo},{args.x_nac_hi}]")
    print(f"    pylon    : y=[{args.y_pyl_lo},{args.y_pyl_hi}]  z=[{args.z_pyl_lo},{args.z_pyl_hi}]  x=[{args.x_pyl_lo},{args.x_pyl_hi}]")
    print(f"    wing     : tout le reste")

    print("\n[3] Classification …")
    labels_unique = classify_points(xyz_unique, args)
    print_distribution(labels_unique, xyz_unique, "Distribution sur les 260 774 points uniques")

    print("\n[4] Réplication pour toutes les simulations …")
    labels_full = np.tile(labels_unique, n_sims)
    print(f"    Labels complets : {labels_full.shape}  (= {n_sims} x {N_WALL})")

    print("\n[5] Sauvegarde …")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "component_labels_unique.npy", labels_unique)
    print(f"    component_labels_unique.npy  shape={labels_unique.shape}")

    np.save(out_dir / "component_labels_full.npy", labels_full)
    print(f"    component_labels_full.npy    shape={labels_full.shape}")

    comp_names = np.array([CODE_TO_COMP[c] for c in labels_unique])
    df = pd.DataFrame({
        "x": xyz_unique[:, 0],
        "y": xyz_unique[:, 1],
        "z": xyz_unique[:, 2],
        "component": comp_names,
        "component_code": labels_unique,
    })
    df.to_csv(out_dir / "component_labels.csv", index=False)
    print(f"    component_labels.csv")

    with open(out_dir / "component_map.json", "w") as f:
        json.dump({str(v): k for k, v in COMP_TO_CODE.items()}, f, indent=2)
    print(f"    component_map.json")

    print("\n[OK] Terminé.")
    print(f"\nRésumé final :")
    for code, name in CODE_TO_COMP.items():
        cnt = (labels_unique == code).sum()
        print(f"  {name:10s} : {cnt:7d} pts ({100*cnt/N_WALL:.1f}%)")


if __name__ == "__main__":
    main()