# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:58:13 2017
PLOT LIBRARY
@author: ManhDuy
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import Solve_Equation_SVM_RBF as SESVM_RBF

def Plot_Class(X, Y, shape = 'circle'):
    gin = np.where(Y == -1)
    go = np.where(Y == 1)
    XGIN = X[gin,:][0]
    XGO = X[go,:][0]
    
    if (shape == 'circle'):
        plt.plot(XGO[:,0], XGO[:,1], 'bo')
        plt.plot(XGIN[:,0], XGIN[:,1], 'ro')
    elif (shape == 'square'):
        plt.plot(XGO[:,0], XGO[:,1], 'bs')
        plt.plot(XGIN[:,0], XGIN[:,1], 'rs')
    elif (shape == 'triangle'):
        plt.plot(XGO[:,0], XGO[:,1], 'b^')
        plt.plot(XGIN[:,0], XGIN[:,1], 'r^')
        
    


# ---------- PLOT 3D KERNEL RBF SVM ----------    
def Plot_3D_KSVM_RBF(XG, YG, bK, BetaK, YK, XK, GammaK):
    gin = np.where(YG == -1)
    go = np.where(YG == 1)
    
    XGIN = XG[gin,:][0]
    XGO = XG[go,:][0]
    
    ZGIN = np.array([SESVM_RBF.SVM_Func(XGIN[i], bK, BetaK, YK, XK, GammaK) for i in range(XGIN.shape[0])])
    ZGO = np.array([SESVM_RBF.SVM_Func(XGO[i], bK, BetaK, YK, XK, GammaK) for i in range(XGO.shape[0])])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(XGIN[:,0],XGIN[:,1],ZGIN[:], color='red')
    ax.scatter(XGO[:,0],XGO[:,1],ZGO[:], color='blue') 

        
# ---------- PLOT FINAL COMPARING ----------
def Mesh_2D_Coordinates(X_Mesh, Y_Mesh):
    XX, YY = np.meshgrid(X_Mesh, Y_Mesh, sparse=False)
    C_Mesh = np.append(XX[:,0][np.newaxis].T, YY[:,0][np.newaxis].T,axis = 1)
    for i in range(1,XX.shape[1]):
        tmp = np.append(XX[:,i][np.newaxis].T, YY[:,i][np.newaxis].T,axis = 1)
        C_Mesh = np.append(C_Mesh, tmp, axis = 0)
    return C_Mesh

def Plot_2D_Mesh(C_Mesh, L_Mesh):    
    co = np.where(L_Mesh == 1)
    ci = np.where(L_Mesh ==- 1)
    XO = C_Mesh[co]
    XI = C_Mesh[ci]
    
    plt.figure()
    plt.plot(XO[:,0], XO[:,1], 'bo')
    plt.plot(XI[:,0], XI[:,1], 'ro')

def Mesh_3D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh):
    X, Y, Z = np.meshgrid(X_Mesh, Y_Mesh, Z_Mesh, sparse=False)
    XR = X.ravel()
    YR = Y.ravel()
    ZR = Z.ravel()
    C_Mesh = np.hstack([XR[:,np.newaxis],YR[:,np.newaxis],ZR[:,np.newaxis]])    
    return C_Mesh

def Plot_3D_Mesh(C_Mesh, L_Mesh):    
    #co = np.where(L_Mesh == 1)
    ci = np.where(L_Mesh <=- 1)
    #XO = C_Mesh[co]
    XI = C_Mesh[ci]
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(XI[:,0], XI[:,1], XI[:,2], s = 5, c = 'r', marker = '*')
    #ax.scatter(XO[:,0], XO[:,1], XO[:,2], c = 'b', marker = 'o')
    
def Mesh_4D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh):
    X, Y, Z, T = np.meshgrid(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, sparse=False)
    XR = X.ravel()
    YR = Y.ravel()
    ZR = Z.ravel()
    TR = T.ravel()
    C_Mesh = np.hstack([XR[:,np.newaxis], YR[:,np.newaxis], ZR[:,np.newaxis], TR[:,np.newaxis]])    
    return C_Mesh
