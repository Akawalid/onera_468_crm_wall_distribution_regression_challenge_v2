#srun --account=tau --partition=gpu-best --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:2 --mem=64G --time=01:00:00 --pty bash
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

#==================== load data
path="/data/tau/iceberg_1/shared/ochabane/FILES_RHO_ALL_POINTS_reduitfloat32/"

# x = np.load(path + "X9_ALL_POINT_fl32.npy")
# y = np.load(path + "RHO_ALL_POINT_fl32.npy")


#==================== 
# df_full = pd.read_csv(path+'fullfiles_PiMinfAoA_with_scores.csv')
# df_split = pd.read_csv(path+'traintest_splitting1_MinfAoAPi_with_scores.csv')

# Load geometry
X = np.load(path+'COORDINATES/XX.npy')
Y = np.load(path+'COORDINATES/YY.npy')
Z = np.load(path+'COORDINATES/ZZ.npy')

xyz = np.stack([X, Y, Z], axis=1) 

# # KMeans
# scaler = StandardScaler()
# xyz_scaled = scaler.fit_transform(xyz)

# k = 32  # wing, fuselage, pylon, nacelle
# km = KMeans(n_clusters=k, random_state=42, n_init=10)
# labels = km.fit_predict(xyz_scaled)  # (200k,)

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=50, n_jobs=-1)
labels = db.fit_predict(xyz)
k="DBSCAN"

# Plotting — adapted from your script
cameraSettings = {
    'elevation': 20.,
    'azimuth': 120.,
    'zoom': 1.,
    'xlim': [None, None],
    'ylim': [None, 5.],
    'zlim': [None, None],
    'xoffsets': [10., -14.],
    'yoffsets': [0., 0.],
    'zoffsets': [0., 0.]
}

xmin, xmax = np.min(X), np.max(X)
ymin, ymax = np.min(Y), np.max(Y)
zmin, zmax = np.min(Z), np.max(Z)

if cameraSettings['xlim'][0] is None: cameraSettings['xlim'][0] = xmin
if cameraSettings['xlim'][1] is None: cameraSettings['xlim'][1] = xmax
if cameraSettings['ylim'][0] is None: cameraSettings['ylim'][0] = ymin
if cameraSettings['ylim'][1] is None: cameraSettings['ylim'][1] = ymax
if cameraSettings['zlim'][0] is None: cameraSettings['zlim'][0] = zmin
if cameraSettings['zlim'][1] is None: cameraSettings['zlim'][1] = zmax

figa = plt.figure(dpi=400)
gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[1., 1, 1.], height_ratios=[20, 1])
ax = figa.add_subplot(gs[0, :], projection=Axes3D.name)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
cmap = plt.cm.colors.ListedColormap(colors) if hasattr(plt.cm, 'colors') else 'tab10'

sca = ax.scatter3D(X, Y, Z, c=labels, cmap='tab10', clip_on=False, s=0.1)

ax.view_init(elev=cameraSettings['elevation'], azim=cameraSettings['azimuth'])
ax.set(
    xlim=(cameraSettings['xlim'][0] + cameraSettings['xoffsets'][0],
          cameraSettings['xlim'][1] + cameraSettings['xoffsets'][1]),
    ylim=(cameraSettings['ylim'][0] + cameraSettings['yoffsets'][0],
          cameraSettings['ylim'][1] + cameraSettings['yoffsets'][1]),
    zlim=(cameraSettings['zlim'][0] + cameraSettings['zoffsets'][0],
          cameraSettings['zlim'][1] + cameraSettings['zoffsets'][1])
)
ax.set_axis_off()
limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
ax.set_box_aspect(np.ptp(limits, axis=1), zoom=cameraSettings['zoom'])
figa.subplots_adjust(left=0, right=1, bottom=0, top=1)

cax = figa.add_subplot(gs[1, 1])
cbar = figa.colorbar(sca, cax=cax, orientation='horizontal')
cbar.set_label('Component', size=8)
cbar.ax.tick_params(labelsize=6)
cbar.ax.xaxis.set_label_position('top')

figa.suptitle(f'KMeans segmentation k={k}', fontsize=12)
figa.tight_layout(pad=0.)
figa.savefig(f'fig_kmeans_k{k}.png', dpi=400)