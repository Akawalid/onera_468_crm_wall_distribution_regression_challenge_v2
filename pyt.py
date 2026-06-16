#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import json

cameraSettings = {
    'elevation': 20.,
    'azimuth': 120.,
    'zoom': 1.,
    'xlim': [None, None],
    'ylim': [None, None],
    'zlim': [None, None],
    'xoffsets': [0., 0.],
    'yoffsets': [0., 0.],
    'zoffsets': [0., 0.]
}
plotSettings = {
    'dpi': 400,
    'fileName': 'fig_components_CRM.png',
    'title': 'CRM WBPN — Classification par composant géométrique',
    'title_fontsize': 12,
}
COMP_COLORS = {
    'fuselage': '#4878CF',
    'wing':     '#6ACC65',
    'pylon':    '#D65F5F',
    'nacelle':  '#F5C518',
    'unknown':  '#999999',
}

X = np.load('.FILES_RHO_ALL_POINTS_reduitfloat32/COORDINATES/XX.npy')
Y = np.load('.FILES_RHO_ALL_POINTS_reduitfloat32/COORDINATES/YY.npy')
Z = np.load('.FILES_RHO_ALL_POINTS_reduitfloat32/COORDINATES/ZZ.npy')

labels_int = np.load('./output/component_labels_unique.npy')
with open('./output/component_map.json') as f:
    code_map = json.load(f)

int_to_name = {int(k): v for k, v in code_map.items()}

point_colors = np.array([COMP_COLORS.get(int_to_name.get(c, 'unknown'), '#999999')
                         for c in labels_int])

xmin, xmax = np.min(X), np.max(X)
ymin, ymax = np.min(Y), np.max(Y)
zmin, zmax = np.min(Z), np.max(Z)

figa = plt.figure()
gs = gridspec.GridSpec(nrows=2, ncols=3,
                       width_ratios=[1., 1., 1.],
                       height_ratios=[20, 1])
ax = figa.add_subplot(gs[0, :], projection=Axes3D.name)

ax.scatter3D(X, Y, Z, c=point_colors, s=0.3, alpha=0.7,
             clip_on=False, rasterized=True)

if cameraSettings['xlim'][0] is None: cameraSettings['xlim'][0] = xmin
if cameraSettings['xlim'][1] is None: cameraSettings['xlim'][1] = xmax
if cameraSettings['ylim'][0] is None: cameraSettings['ylim'][0] = ymin
if cameraSettings['ylim'][1] is None: cameraSettings['ylim'][1] = ymax
if cameraSettings['zlim'][0] is None: cameraSettings['zlim'][0] = zmin
if cameraSettings['zlim'][1] is None: cameraSettings['zlim'][1] = zmax

ax.view_init(elev=cameraSettings['elevation'], azim=cameraSettings['azimuth'])
ax.set(
    xlim=(cameraSettings['xlim'][0] + cameraSettings['xoffsets'][0],
          cameraSettings['xlim'][1] + cameraSettings['xoffsets'][1]),
    ylim=(cameraSettings['ylim'][0] + cameraSettings['yoffsets'][0],
          cameraSettings['ylim'][1] + cameraSettings['yoffsets'][1]),
    zlim=(cameraSettings['zlim'][0] + cameraSettings['zoffsets'][0],
          cameraSettings['zlim'][1] + cameraSettings['zoffsets'][1]),
)
ax.set_axis_off()

limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
ax.set_box_aspect(np.ptp(limits, axis=1), zoom=cameraSettings['zoom'])

figa.subplots_adjust(left=0, right=1, bottom=0, top=1)

legend_patches = []
for code, name in int_to_name.items():
    cnt = (labels_int == code).sum()
    color = COMP_COLORS.get(name, '#999999')
    legend_patches.append(
        mpatches.Patch(color=color, label=f'{name} ({cnt:,} pts)')
    )
ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
          framealpha=0.8, markerscale=2)

figa.suptitle(plotSettings['title'], fontsize=plotSettings['title_fontsize'])
figa.tight_layout(pad=0.)
figa.savefig(plotSettings['fileName'], dpi=plotSettings['dpi'])
print(f"Figure sauvegardée : {plotSettings['fileName']}")
# KNN from charbonnier to the current geometry, with taking in consideration units and normalization 
# import numpy as np
# import json
# from pathlib import Path
# from scipy.spatial import cKDTree

# CHALLENGE_DIR   = Path(".FILES_RHO_ALL_POINTS_reduitfloat32/COORDINATES")
# CHARBONNIER_DIR = Path(".")
# OUTPUT_DIR      = Path("./output")

# INCHES_TO_METERS = 1.0 / 39.3701

# COMPONENTS = ["fuselage", "nacelle", "pylon", "wing"]

# print("Chargement des points challenge (mètres)...")
# X = np.load(CHALLENGE_DIR / "XX.npy")
# Y = np.load(CHALLENGE_DIR / "YY.npy")
# Z = np.load(CHALLENGE_DIR / "ZZ.npy")
# challenge_xyz = np.column_stack([X, Y, Z])
# print(f"  {len(challenge_xyz):,} pts  X=[{X.min():.3f}, {X.max():.3f}]  Y=[{Y.min():.3f}, {Y.max():.3f}]  Z=[{Z.min():.3f}, {Z.max():.3f}]")

# print("\nChargement des points Charbonnier (pouces → mètres)...")
# charb_xyz_list   = []
# charb_label_list = []
# for i, comp in enumerate(COMPONENTS):
#     fpath = CHARBONNIER_DIR / f"charbonnier_{comp}.npy"
#     if not fpath.exists():
#         print(f"  [WARN] manquant : {fpath}")
#         continue
#     pts = np.load(fpath) * INCHES_TO_METERS
#     charb_xyz_list.append(pts)
#     charb_label_list.append(np.full(len(pts), i, dtype=np.int32))
#     print(f"  {comp:12s} : {len(pts):>7,} pts  X=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")

# charb_xyz    = np.vstack(charb_xyz_list)
# charb_labels = np.concatenate(charb_label_list)
# print(f"  TOTAL : {len(charb_xyz):,} pts")

# print("\nConstruction du KD-tree...")
# tree = cKDTree(charb_xyz)

# print("KNN (k=1)...")
# distances, indices = tree.query(challenge_xyz, k=1, workers=-1)
# labels = charb_labels[indices]

# print(f"\nDistances au plus proche voisin (mètres) :")
# print(f"  mean : {distances.mean():.5f}")
# print(f"  p50  : {np.percentile(distances, 50):.5f}")
# print(f"  p95  : {np.percentile(distances, 95):.5f}")
# print(f"  p99  : {np.percentile(distances, 99):.5f}")
# print(f"  max  : {distances.max():.5f}")

# print(f"\nRépartition des labels :")
# for i, name in enumerate(COMPONENTS):
#     cnt = (labels == i).sum()
#     print(f"  {name:12s} : {cnt:>8,} pts  ({100*cnt/len(labels):.1f}%)")

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# np.save(OUTPUT_DIR / "component_labels_unique.npy", labels)
# np.save(OUTPUT_DIR / "component_nn_distances.npy",  distances)
# code_map = {str(i): name for i, name in enumerate(COMPONENTS)}
# with open(OUTPUT_DIR / "component_map.json", "w") as f:
#     json.dump(code_map, f, indent=2)

# print(f"\nSauvegardé dans {OUTPUT_DIR}/")
# print(f"  component_labels_unique.npy  {labels.shape}")
# print(f"  component_nn_distances.npy   {distances.shape}")
# print(f"  component_map.json")