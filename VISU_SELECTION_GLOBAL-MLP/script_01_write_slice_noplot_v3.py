######################################################################################

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

import sys

sizesc = 260774

#################################################################################################
#################################################################################################

def compinfo_monofield(f,bsize,textdef):

    fmin =  333333333.
    fmax = -777777777.
    fsum = 0.
    fstd = 0.
    for l in range (bsize):
        fsum = fsum + f[l]
        if  f[l]>fmax:
            fmax = f[l]
        if f[l]<fmin:
            fmin = f[l]
    fmean = fsum / float(bsize)
    for l in range(bsize):
        fstd = fstd + (fmean-f[l])**2
    fstd = np.sqrt(fstd/float(bsize-1))    

    print(" ")
    print(" ************************************************************************")
    print("  **** champ "+textdef )
    print(" ************************************************************")
    print("  mean      ",fmean)
    print("  stdv        ",fstd)
    print("  min         ",fmin)
    print("  max        ",fmax)
    print("   " )               
    #
    #
    return

def printshape(array,name):
    print("--- array "+ name+" --- shape  ",array.shape)

#################################################################################################
################################################################################################

SLICES_DIR = "FILES_SLICES"

XC = np.fromfile("./INPUT-FILES_SURFACE/xc_once.npy",dtype="float")
YC = np.fromfile("./INPUT-FILES_SURFACE/yc_once.npy",dtype="float")
ZC = np.fromfile("./INPUT-FILES_SURFACE/zc_once.npy",dtype="float")

printshape(XC," XC ")
printshape(YC," YC ")
printshape(ZC," XC ")

compinfo_monofield(XC, sizesc,  " **** XC coordinate *** ")
compinfo_monofield(YC, sizesc,  " **** YC coordinate ***  ")
compinfo_monofield(ZC, sizesc,  " **** ZC coordinate ***  ")

FLOWA="Minf_0.5_AoA_0.0_Pi_4.0"
FLOWB="Minf_0.82_AoA_3.0_Pi_2.0"
FLOWC="Minf_0.85_AoA_1.5_Pi_1.0"
FLOWD="Minf_0.96_AoA_-2.0_Pi_2.0"

CPa = np.load("INPUT-FILES_SURFACE/cptest_Minf_0.5_AoA_0.0_Pi_4.0.npy")
CPb = np.load("INPUT-FILES_SURFACE/cptest_Minf_0.82_AoA_3.0_Pi_2.0.npy")
CPc = np.load("INPUT-FILES_SURFACE/cptest_Minf_0.85_AoA_1.5_Pi_1.0.npy")
CPd = np.load("INPUT-FILES_SURFACE/cptest_Minf_0.96_AoA_-2.0_Pi_2.0.npy")

CPHATa = np.load("INPUT-FILES_SURFACE/cptesthat_Minf_0.5_AoA_0.0_Pi_4.0.npy")
CPHATb = np.load("INPUT-FILES_SURFACE/cptesthat_Minf_0.82_AoA_3.0_Pi_2.0.npy")
CPHATc = np.load("INPUT-FILES_SURFACE/cptesthat_Minf_0.85_AoA_1.5_Pi_1.0.npy")
CPHATd = np.load("INPUT-FILES_SURFACE/cptesthat_Minf_0.96_AoA_-2.0_Pi_2.0.npy")

CFa = np.load("INPUT-FILES_SURFACE/cftest_Minf_0.5_AoA_0.0_Pi_4.0.npy")
CFb = np.load("INPUT-FILES_SURFACE/cftest_Minf_0.82_AoA_3.0_Pi_2.0.npy")
CFc = np.load("INPUT-FILES_SURFACE/cftest_Minf_0.85_AoA_1.5_Pi_1.0.npy")
CFd = np.load("INPUT-FILES_SURFACE/cftest_Minf_0.96_AoA_-2.0_Pi_2.0.npy")

CFHATa = np.load("INPUT-FILES_SURFACE/cftesthat_Minf_0.5_AoA_0.0_Pi_4.0.npy")
CFHATb = np.load("INPUT-FILES_SURFACE/cftesthat_Minf_0.82_AoA_3.0_Pi_2.0.npy")
CFHATc = np.load("INPUT-FILES_SURFACE/cftesthat_Minf_0.85_AoA_1.5_Pi_1.0.npy")
CFHATd = np.load("INPUT-FILES_SURFACE/cftesthat_Minf_0.96_AoA_-2.0_Pi_2.0.npy")

printshape(CPa," CPa ")
printshape(CPHATa," CPHATa ")

#####################################################################################
######## try to write the output for the first column###############################################

yy_list = [ {"yy" : 5.57, "SL" : "sly5.6"  , "dd":0.05 },
                 {"yy" : 13.4, "SL" : "sly13.4", "dd":0.1   },
                 {"yy" : 17.6, "SL" : "sly17.6", "dd":0.15  }, 
                 {"yy" : 21.9, "SL" : "sly21.9", "dd":0.10 }, 
                 {"yy" : 26.0, "SL" : "sly26"   , "dd":0.05 },
                 {"yy" : 28.5, "SL" : "sly28.5", "dd":0.05 },
]

for slicetmp in yy_list:
    yy = slicetmp["yy"]
    SL = slicetmp["SL"]
    dd = slicetmp["dd"]
    countp = 0
    for l in range(sizesc):
        if (YC[l]<yy+dd) and (YC[l]>yy-dd) : countp=countp+1 

    print(" \n=======  basket  Y in [ ",(yy-dd) ," , ",(yy+dd)," ] == ",countp," points ===== ") 

    #########################################################################################

    coord_flowa_YYhatcpcf_sl =np.zeros((countp,7))

    ind= 0
    for l in range(sizesc):
        if (YC[l]<yy+dd) and (YC[l]>yy-dd):
            coord_flowa_YYhatcpcf_sl[ind,0] = XC[l]
            coord_flowa_YYhatcpcf_sl[ind,1] = YC[l]
            coord_flowa_YYhatcpcf_sl[ind,2] = ZC[l]
            coord_flowa_YYhatcpcf_sl[ind,3] = CPa[l]
            coord_flowa_YYhatcpcf_sl[ind,4] = CPHATa[l]
            coord_flowa_YYhatcpcf_sl[ind,5] = CFa[l]
            coord_flowa_YYhatcpcf_sl[ind,6] = CFHATa[l]
            ind = ind+1

    np.save(SLICES_DIR + "/coord_"+FLOWA+"_YYhatcpcf_"+SL+".npy",coord_flowa_YYhatcpcf_sl)        

    #===========================================

    coord_flowb_YYhatcpcf_sl =np.zeros((countp,7))

    ind= 0
    for l in range(sizesc):
        if (YC[l]<yy+dd) and (YC[l]>yy-dd):
            coord_flowb_YYhatcpcf_sl[ind,0] = XC[l]
            coord_flowb_YYhatcpcf_sl[ind,1] = YC[l]
            coord_flowb_YYhatcpcf_sl[ind,2] = ZC[l]
            coord_flowb_YYhatcpcf_sl[ind,3] = CPb[l]
            coord_flowb_YYhatcpcf_sl[ind,4] = CPHATb[l]
            coord_flowb_YYhatcpcf_sl[ind,5] = CFb[l]
            coord_flowb_YYhatcpcf_sl[ind,6] = CFHATb[l]
            ind = ind+1

    np.save(SLICES_DIR + "/coord_"+FLOWB+"_YYhatcpcf_"+SL+".npy",coord_flowb_YYhatcpcf_sl)    

    #===========================================
    
    coord_flowc_YYhatcpcf_sl =np.zeros((countp,7))

    ind= 0
    for l in range(sizesc):
        if (YC[l]<yy+dd) and (YC[l]>yy-dd):
            coord_flowc_YYhatcpcf_sl[ind,0] = XC[l]
            coord_flowc_YYhatcpcf_sl[ind,1] = YC[l]
            coord_flowc_YYhatcpcf_sl[ind,2] = ZC[l]
            coord_flowc_YYhatcpcf_sl[ind,3] = CPc[l]
            coord_flowc_YYhatcpcf_sl[ind,4] = CPHATc[l]
            coord_flowc_YYhatcpcf_sl[ind,5] = CFc[l]
            coord_flowc_YYhatcpcf_sl[ind,6] = CFHATc[l]
            ind = ind+1

    np.save(SLICES_DIR + "/coord_"+FLOWC+"_YYhatcpcf_"+SL+".npy",coord_flowc_YYhatcpcf_sl)        

    #===========================================

    coord_flowd_YYhatcpcf_sl =np.zeros((countp,7))

    ind= 0
    for l in range(sizesc):
        if (YC[l]<yy+dd) and (YC[l]>yy-dd):
            coord_flowd_YYhatcpcf_sl[ind,0] = XC[l]
            coord_flowd_YYhatcpcf_sl[ind,1] = YC[l]
            coord_flowd_YYhatcpcf_sl[ind,2] = ZC[l]
            coord_flowd_YYhatcpcf_sl[ind,3] = CPd[l]
            coord_flowd_YYhatcpcf_sl[ind,4] = CPHATd[l]
            coord_flowd_YYhatcpcf_sl[ind,5] = CFd[l]
            coord_flowd_YYhatcpcf_sl[ind,6] = CFHATd[l]
            ind = ind+1

    np.save(SLICES_DIR + "/coord_"+FLOWD+"_YYhatcpcf_"+SL+".npy",coord_flowd_YYhatcpcf_sl)


     

#########################################################################
#########################################################################

