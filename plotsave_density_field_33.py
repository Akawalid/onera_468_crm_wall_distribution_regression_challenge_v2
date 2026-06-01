import numpy as np

##########################################################################"

npwall = 260774
ncase = 33                                                           # from 0 up to 467
#
#--------------------------------------
#  if no memory problem 
#-------------------------------------------
#XI = np.load('X9_ALL_POINT_fl64.npy')
#print(" XI.shape ", XI.shape)
#X = XI[ncase*npwall:(ncase+1)*npwall,0]      # actually the scattered mesh is the same for all cases
#Y = XI[ncase*npwall:(ncase+1)*npwall,1]      # XI[0:npwall,..]   would do       
#Z = XI[ncase*npwall:(ncase+1)*npwall,2]  
#
Y = np.load('RHO_ALL_POINT_fl64.npy')
#
print(" Y.shape ", Y.shape)
#
#
RHO33 = Y[ncase*npwall:(ncase+1)*npwall]    
print(" RHO33 ",RHO33)
print(" RHO33.shape ", RHO33.shape)

print(" ======================================================= ")
print("   aerodynamic conditions of computation ", ncase,"  of the series ")
print("   get info (from line  ",(ncase+2)," ) starting with ",ncase," of the csv file ")
print("   temporarly hard coded = 1.,0.7,8.0  " )
print(" ======================================================== ")
#
#-------------------------------
#  if no memory problem 
#------------------------------------
#print("   Minf            ")   #,XI[ncase*npwall,6] )  
#print("   AoA             ")  #,XI[ncase*npwall,7] )  
#print("   Pi (*10-6)    ")  #,XI[ncase*npwall,8] )  
print(" =================================================== ")

#=========================================================================

DIRWRITE = "./DENSITY_FIELD_33/"
#
#np.save("COORDINATES/XX.npy",X)
#np.save("COORDINATES/YY.npy",Y)
#np.save("COORDINATES/ZZ.npy",Z)
#
np.save(DIRWRITE+"mla_density.npy",RHO33)

#############################################################################
