
import csv
import numpy as np
import time as time
from scipy import stats

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

############################################################################

PLOT_DIR = "PLOTS_SLICES"
SLICES_DIR = "FILES_SLICES"

cpmin = -1.1 ; cpmax = 0.6
cfmin =  0.  ; cfmax = 0.007

#############################################################################

def make_sliceplots_145678(flowst):
    
    flow_sl8 = np.load(SLICES_DIR + "/coord_"+flowst+"_YYhatcpcf_sly28.5.npy")
    yy = 28.5
    x        = flow_sl8[:,0]
    mcp      = -flow_sl8[:,3]
    mcphat   = -flow_sl8[:,4]
    cf       = flow_sl8[:,5]
    cfhat  = flow_sl8[:,6]
    
    plt.plot(x, mcp, color='r', label='-CP')
    plt.plot(x, mcphat, color='g', label='-CPHAT')
    plt.xlabel(" x ")
    plt.ylabel(" -CP ")
    plt.ylim(-cpmax, -cpmin)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cpcphat.png") 
    plt.close()
    
    plt.plot(x, cf, color='r', label='CF')
    plt.plot(x, cfhat, color='g', label='CFHAT')
    plt.xlabel(" x ")
    plt.ylabel(" CF ")
    plt.ylim(cfmin, cfmax)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cfcfhat.png")
    plt.close()
    
    #=============================================================================
    
    flow_sl7 = np.load(SLICES_DIR + "/coord_"+flowst+"_YYhatcpcf_sly26.npy")
    yy = 26.0
    
    x        = flow_sl7[:,0]
    mcp      = -flow_sl7[:,3]
    mcphat   = -flow_sl7[:,4]
    cf       = flow_sl7[:,5]
    cfhat  = flow_sl7[:,6]
    
    plt.plot(x, mcp, color='r', label='-CP')
    plt.plot(x, mcphat, color='g', label='-CPHAT')
    plt.xlabel(" x ")
    plt.ylabel(" -CP ")
    plt.ylim(-cpmax, -cpmin)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cpcphat.png")
    plt.close()
    
    plt.plot(x, cf, color='r', label='CF')
    plt.plot(x, cfhat, color='g', label='CFHAT')
    plt.xlabel(" x ")
    plt.ylabel(" CF ")
    plt.ylim(cfmin, cfmax)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cfcfhat.png")
    plt.close()
    
    #===========================================================================
    
    flow_sl6 = np.load(SLICES_DIR + "/coord_"+flowst+"_YYhatcpcf_sly21.9.npy")
    yy = 21.9
    
    x        = flow_sl6[:,0]
    mcp      = -flow_sl6[:,3]
    mcphat   = -flow_sl6[:,4]
    cf       = flow_sl6[:,5]
    cfhat  = flow_sl6[:,6]
    
    plt.plot(x, mcp, color='r', label='-CP')
    plt.plot(x, mcphat, color='g', label='-CPHAT')
    plt.xlabel(" x ")
    plt.ylabel(" -CP ")
    plt.ylim(-cpmax, -cpmin)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cpcphat.png")
    plt.close()
    
    plt.plot(x, cf, color='r', label='CF')
    plt.plot(x, cfhat, color='g', label='CFHAT')
    plt.xlabel(" x ")
    plt.ylabel(" CF ")
    plt.ylim(cfmin, cfmax)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cfcfhat.png")
    plt.close()
    
    #===========================================================================
    
    flow_sl5 = np.load(SLICES_DIR + "/coord_"+flowst+"_YYhatcpcf_sly17.6.npy")
    yy = 17.6
    
    x        = flow_sl5[:,0]
    mcp      = -flow_sl5[:,3]
    mcphat   = -flow_sl5[:,4]
    cf       = flow_sl5[:,5]
    cfhat  = flow_sl5[:,6]
    
    plt.plot(x, mcp, color='r', label='-CP')
    plt.plot(x, mcphat, color='g', label='-CPHAT')
    plt.xlabel(" x ")
    plt.ylabel(" -CP ")
    plt.ylim(-cpmax, -cpmin)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cpcphat.png")
    plt.close()
    
    plt.plot(x, cf, color='r', label='CF')
    plt.plot(x, cfhat, color='g', label='CFHAT')
    plt.xlabel(" x ")
    plt.ylabel(" CF ")
    plt.ylim(cfmin, cfmax)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cfcfhat.png")
    plt.close()
    
    #============================================================================
    
    flow_sl4 = np.load(SLICES_DIR + "/coord_"+flowst+"_YYhatcpcf_sly13.4.npy")
    yy = 13.4
    
    x        = flow_sl4[:,0]
    mcp      = -flow_sl4[:,3]
    mcphat   = -flow_sl4[:,4]
    cf       = flow_sl4[:,5]
    cfhat  = flow_sl4[:,6]
    
    plt.plot(x, mcp, color='r', label='-CP')
    plt.plot(x, mcphat, color='g', label='-CPHAT')
    plt.xlabel(" x ")
    plt.ylabel(" -CP ")
    plt.ylim(-cpmax, -cpmin)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cpcphat.png")
    plt.close()
    
    plt.plot(x, cf, color='r', label='CF')
    plt.plot(x, cfhat, color='g', label='CFHAT')
    plt.xlabel(" x ")
    plt.ylabel(" CF ")
    plt.ylim(cfmin, cfmax)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cfcfhat.png")
    plt.close()

    #============================================================================

    flow_sl1 = np.load(SLICES_DIR + "/coord_"+flowst+"_YYhatcpcf_sly5.6.npy")
    yy = 5.6
    
    x        = flow_sl1[:,0]
    mcp      = -flow_sl1[:,3]
    mcphat   = -flow_sl1[:,4]
    cf       = flow_sl1[:,5]
    cfhat  = flow_sl1[:,6]
    
    plt.plot(x, mcp, color='r', label='-CP')
    plt.plot(x, mcphat, color='g', label='-CPHAT')
    plt.xlabel(" x ")
    plt.ylabel(" -CP ")
    plt.ylim(-cpmax, -cpmin)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cpcphat.png")
    plt.close()
    
    plt.plot(x, cf, color='r', label='CF')
    plt.plot(x, cfhat, color='g', label='CFHAT')
    plt.xlabel(" x ")
    plt.ylabel(" CF ")
    plt.ylim(cfmin, cfmax)
    plt.title(flowst+" slice Y="+str(yy)+"  -- CFD and isoMap+ML")
    plt.legend()
    plt.savefig(PLOT_DIR + "/fig_"+flowst+"_slicey"+str(yy)+"_cfcfhat.png")
    plt.close()

#############################################################################
# in case subprocess is available  directory can be used
#############################################################################

make_sliceplots_145678("Minf_0.5_AoA_0.0_Pi_4.0")
make_sliceplots_145678("Minf_0.82_AoA_3.0_Pi_2.0")
make_sliceplots_145678("Minf_0.85_AoA_1.5_Pi_1.0")
make_sliceplots_145678("Minf_0.96_AoA_-2.0_Pi_2.0")

#############################################################################

