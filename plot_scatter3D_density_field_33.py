#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# User inputs
# -----------

sMinf='0.7'
sAoA='8.0'
sPi='1.0'
snumFlow='33'

# -----------

# Input datasets
dataFiles = {
  'X': './COORDINATES/XX.npy',
  'Y': './COORDINATES/YY.npy',
  'Z': './COORDINATES/ZZ.npy',
    'density': './DENSITY_FIELD_'+snumFlow+'/mla_density.npy',
}

# Adjusting camera position & margins
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

plotSettings = {
  'figsize': None, # (12,4)
  'dpi': 400,
  'fileName': 'fig_density_Pi_'+sPi+'_Minf_'+sMinf+'_AoA_'+sAoA+'.png',
  'title': 'Computed density/density_inf : Wind Tunnel Pi ='+sPi+', Minf = '+sMinf+', AoA = '+sAoA,
  'title_fontsize': 12,
  'cmap': 'jet', # others can be: 'hsv', 'RdBu' 'PiYG'
  'cbar': {
      'title': 'Density ',
      'title_fontsize': 8,
      'ticks_fontsize': 6,
  }
}

########################################################################
# this avoids the pertubation of scales by marginal values
#================================================

def defminmaxscale(field):
    #
    f03 = np.percentile(field,3.)
    f50 = np.percentile(field,50.)
    f97 = np.percentile(field,97.)
    slopem = (f50-f03)/0.47
    slopep  = (f97-f50)/0.47
    scalemin = f50 - 0.6*slopem   # 20% margin wrt linearly spread date
    scalemax = f50 + 0.6*slopep
    #
    return scalemin,scalemax

#########################################################################
# Creating datasets
#==========================

X = np.load(dataFiles['X'])
Y = np.load(dataFiles['Y'])
Z = np.load(dataFiles['Z'])
dens = np.load(dataFiles['density'])

xmin, xmax = np.min(X), np.max(X)
ymin, ymax = np.min(Y), np.max(Y)
zmin, zmax = np.min(Z), np.max(Z)

densmin,densmax = defminmaxscale(dens) 
densmin = max(0.,densmin)

print("Xmin: {}, Xmax: {}".format(xmin, xmax))
print("Ymin: {}, Ymax: {}".format(ymin, ymax))
print("Zmin: {}, Zmax: {}".format(zmin, zmax))

#######################################################################
# Creating figures
#=============================

if plotSettings['figsize'] is None: figa = plt.figure()
else: figa = plt.figure(figsize=plotSettings['figsize'])

gs = gridspec.GridSpec(
    nrows=2, ncols=3,
    width_ratios=[1.,1,1.],
    height_ratios=[20,1]
)

ax = figa.add_subplot(gs[0,:], projection=Axes3D.name)
sca = ax.scatter3D(X, Y, Z, vmin=densmin, vmax=densmax, c=dens,
                   cmap=plt.get_cmap(plotSettings['cmap']),
                   clip_on=False)

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
        cameraSettings['zlim'][1] + cameraSettings['zoffsets'][1])
)

ax.set_axis_off()
limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
ax.set_box_aspect(np.ptp(limits, axis=1), zoom=cameraSettings['zoom'])
figa.subplots_adjust(left=0, right=1, bottom=0, top=1)

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('EXA')    

cax = figa.add_subplot(gs[1,1])
cbar = figa.colorbar(sca, cax=cax, orientation='horizontal') #shrink=0.5) 
cbar.set_label(plotSettings['cbar']['title'],
               size=plotSettings['cbar']['title_fontsize'])
cbar.ax.tick_params(labelsize=plotSettings['cbar']['ticks_fontsize'])
cbar.ax.xaxis.set_label_position('top')

title = plotSettings.get('title', None)
if title is not None:
    figa.suptitle(plotSettings['title'],
                  fontsize=plotSettings['title_fontsize'])
figa.tight_layout(pad=0.)
figa.savefig(plotSettings['fileName'], dpi=plotSettings['dpi'])
#plt.show()

#############################################################################
