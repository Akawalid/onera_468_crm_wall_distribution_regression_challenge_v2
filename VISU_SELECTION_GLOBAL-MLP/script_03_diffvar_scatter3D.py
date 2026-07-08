#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

#####################################################################################################

def plot_diffvar_scatter3D(sMinf,sAoA,sPi,sVar1,svar2,sDiff,varfile1,varfile2,varmin,varmax):

    dataFiles = {
      'X': 'INPUT-FILES_SURFACE/xc_once.npy',
      'Y': 'INPUT-FILES_SURFACE/yc_once.npy',
      'Z': 'INPUT-FILES_SURFACE/zc_once.npy'
       }

    # Adjusting camera position & margins
    cameraSettings = {
        'elevation': 50, # 40, # 20.,
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
        'fileName': 'PLOTS_SCATTER_ERR/figerror_scatter_Minf_'+sMinf+'_AoA_'+sAoA+'_Pi_'+sPi+'_'+sVar1+'.png',
        'title': sDiff+' - Wind Tunnel, Minf = '+sMinf+', AoA = '+sAoA+' Pi= '+sPi+' 10^6',
        'title_fontsize': 12,
        'cmap': 'jet', # others can be: 'hsv', 'RdBu' 'PiYG'
        'cbar': { 'title': 'cf', 'title_fontsize': 8, 'ticks_fontsize': 6,}
    }

    #########################################################################
    # Creating datasets
    
    X = np.fromfile(dataFiles['X'], dtype='float')
    Y = np.fromfile(dataFiles['Y'], dtype='float')
    Z = np.fromfile(dataFiles['Z'], dtype='float')
    myvar1 = np.load(varfile1)
    myvar2 = np.load(varfile2)
    myvar = np.abs(myvar1-myvar2)
    
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    zmin, zmax = np.min(Z), np.max(Z)

    #######################################################################
    # Creating figure
    
    if plotSettings['figsize'] is None: figa = plt.figure()
    else: figa = plt.figure(figsize=plotSettings['figsize'])

    gs = gridspec.GridSpec(nrows=2, ncols=3,
                           width_ratios=[1.,1,1.],
                           height_ratios=[20,1]
    )
    
    ax = figa.add_subplot(gs[0,:], projection=Axes3D.name)
    sca = ax.scatter3D(X, Y, Z, vmin=varmin, vmax=varmax, c=myvar,
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

#############################################################################


#=========================================================

plot_diffvar_scatter3D('0.5','0.0','4.0','cf','cfhat',' abs(cf-cfhat) ',     \
                       'INPUT-FILES_SURFACE/cftesthat_Minf_0.5_AoA_0.0_Pi_4.0.npy',  \
                       'INPUT-FILES_SURFACE/cftest_Minf_0.5_AoA_0.0_Pi_4.0.npy',  \
                       0.0,0.001)

plot_diffvar_scatter3D('0.5','0.0','4.0','cp','cphat',' abs(cp-cphat) ',     \
                       'INPUT-FILES_SURFACE/cptesthat_Minf_0.5_AoA_0.0_Pi_4.0.npy',  \
                       'INPUT-FILES_SURFACE/cptest_Minf_0.5_AoA_0.0_Pi_4.0.npy',  \
                       0.0,0.2)

#=========================================================

plot_diffvar_scatter3D('0.82','3.0','2.0','cf','cfhat',' abs(cf-cfhat) ',     \
                       'INPUT-FILES_SURFACE/cftesthat_Minf_0.82_AoA_3.0_Pi_2.0.npy',  \
                       'INPUT-FILES_SURFACE/cftest_Minf_0.82_AoA_3.0_Pi_2.0.npy',  \
                       0.0,0.001)

plot_diffvar_scatter3D('0.82','3.0','2.0','cp','cphat',' abs(cp-cphat) ',     \
                       'INPUT-FILES_SURFACE/cptesthat_Minf_0.82_AoA_3.0_Pi_2.0.npy',  \
                       'INPUT-FILES_SURFACE/cptest_Minf_0.82_AoA_3.0_Pi_2.0.npy',  \
                       0.0,0.2)

#=========================================================

plot_diffvar_scatter3D('0.85','1.5','1.0','cf','cfhat',' abs(cf-cfhat) ',     \
                       'INPUT-FILES_SURFACE/cftesthat_Minf_0.85_AoA_1.5_Pi_1.0.npy',  \
                       'INPUT-FILES_SURFACE/cftest_Minf_0.85_AoA_1.5_Pi_1.0.npy',  \
                       0.0,0.001)

plot_diffvar_scatter3D('0.85','1.5','1.0','cp','cphat',' abs(cp-cphat) ',     \
                       'INPUT-FILES_SURFACE/cptesthat_Minf_0.85_AoA_1.5_Pi_1.0.npy',  \
                       'INPUT-FILES_SURFACE/cptest_Minf_0.85_AoA_1.5_Pi_1.0.npy',  \
                       0.0,0.2)

#=========================================================

plot_diffvar_scatter3D('0.96','-2.0','2.0','cf','cfhat',' abs(cf-cfhat) ',     \
                       'INPUT-FILES_SURFACE/cftesthat_Minf_0.96_AoA_-2.0_Pi_2.0.npy',  \
                       'INPUT-FILES_SURFACE/cftest_Minf_0.96_AoA_-2.0_Pi_2.0.npy',  \
                       0.0,0.001)

plot_diffvar_scatter3D('0.96','-2.0','2.0','cp','cphat',' abs(cp-cphat) ',     \
                       'INPUT-FILES_SURFACE/cptesthat_Minf_0.96_AoA_-2.0_Pi_2.0.npy',  \
                       'INPUT-FILES_SURFACE/cptest_Minf_0.96_AoA_-2.0_Pi_2.0.npy',  \
                       0.0,0.2)

#############################################################################
