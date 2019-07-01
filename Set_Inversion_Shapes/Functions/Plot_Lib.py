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

# ---------- PLOT SAMPLE 2D CIRCLE ----------
def Plot_2D_Random_Circle(X_samp, R = 2):
    K_near = int(X_samp.shape[0]/4)
    plt.figure()
    plt.plot(X_samp[0:K_near,0], X_samp[0:K_near,1], 'ro')
    plt.plot(X_samp[K_near:2*K_near,0], X_samp[K_near:2*K_near,1], 'r*')
    plt.plot(X_samp[2*K_near:3*K_near,0], X_samp[2*K_near:3*K_near,1], 'bo')
    plt.plot(X_samp[3*K_near:4*K_near,0], X_samp[3*K_near:4*K_near,1], 'b*')
    ax = plt.gca()
    circle = plt.Circle((0, 0), np.sqrt(R), color='g', fill=False)
    ax.add_artist(circle)
    
# ---------- PLOT SAMPLE 3D Sphere ----------
def Plot_3D_Random_Sphere(X_samp):
    K_near = int(X_samp.shape[0]/4)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_samp[0:K_near,0], X_samp[0:K_near,1], X_samp[0:K_near,2], c = 'r', marker = 'o')
    ax.scatter(X_samp[K_near:2*K_near,0], X_samp[K_near:2*K_near,1], X_samp[K_near:2*K_near,2], c = 'r', marker = '*')
    ax.scatter(X_samp[2*K_near:3*K_near,0], X_samp[2*K_near:3*K_near,1], X_samp[2*K_near:3*K_near,2], c = 'b', marker = 'o')
    ax.scatter(X_samp[3*K_near:4*K_near,0], X_samp[3*K_near:4*K_near,1], X_samp[3*K_near:4*K_near,2], c = 'b', marker = '*')

# ---------- PLOT SAMPLE 3D Sphere ----------
def Plot_3D_Surface_Sphere(R = 2):
    Npoints = 100
    u = np.linspace(0, 2 * np.pi, Npoints)
    v = np.linspace(0, np.pi, Npoints)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax = plt.gca()
    # ax.plot_surface(x, y, z, rstride=8, cstride=8, alpha=0.3)
    ax.plot_surface(x, y, z, color = 'g', alpha=0.15)

# ---------- PLOT SAMPLE 2D SQUARE ----------
def Plot_2D_Random_Square(X_samp, R = 2):
    K_near = int(X_samp.shape[0]/4)
    plt.figure()
    plt.plot(X_samp[0:K_near,0], X_samp[0:K_near,1], 'ro')
    plt.plot(X_samp[K_near:2*K_near,0], X_samp[K_near:2*K_near,1], 'ro')
    plt.plot(X_samp[2*K_near:3*K_near,0], X_samp[2*K_near:3*K_near,1], 'bo')
    plt.plot(X_samp[3*K_near:4*K_near,0], X_samp[3*K_near:4*K_near,1], 'bo')
    ax = plt.gca()
    ax.add_patch(
        patches.Rectangle(
            (-R, -R), R*2, R*2, fill=False,      # remove background
            edgecolor="green"
        )
    )
    
# ---------- PLOT SAMPLE 2D DONUT ----------
def Plot_2D_Random_Ring(X_samp, R1 = 1, R2 = 2):
    K_near = int(X_samp.shape[0]/4)
    plt.figure()
    plt.plot(X_samp[0:K_near,0], X_samp[0:K_near,1], 'ro')
    plt.plot(X_samp[K_near:2*K_near,0], X_samp[K_near:2*K_near,1], 'r*')
    plt.plot(X_samp[2*K_near:3*K_near,0], X_samp[2*K_near:3*K_near,1], 'bo')
    plt.plot(X_samp[3*K_near:4*K_near,0], X_samp[3*K_near:4*K_near,1], 'b*')
    ax = plt.gca()
    circle1 = plt.Circle((0, 0), np.sqrt(R1), color='g', fill=False)
    ax.add_artist(circle1)
    circle2 = plt.Circle((0, 0), np.sqrt(R2), color='g', fill=False)
    ax.add_artist(circle2) 

# ---------- PLOT 2D KERNEL SVM SQUARE ----------    
def Plot_2D_Circle(XG, YG, R = 2):
    gin = np.where(YG == -1)
    go = np.where(YG == 1)
    
    XGIN = XG[gin,:][0]
    XGO = XG[go,:][0]
    
    plt.figure()
    plt.plot(XGIN[:,0], XGIN[:,1], 'ro', zorder = 1)
    plt.plot(XGO[:,0], XGO[:,1], 'bo', zorder = 1)
    ax = plt.gca()
    circle = plt.Circle((0, 0), np.sqrt(R), color='g', fill=False)
    ax.add_artist(circle)
    
# ---------- PLOT 2D KERNEL SVM SQUARE ----------    
def Plot_2D_Square(XG, YG, R = 2):
    gin = np.where(YG == -1)
    go = np.where(YG == 1)
    
    XGIN = XG[gin,:][0]
    XGO = XG[go,:][0]
    
    plt.figure()
    plt.plot(XGIN[:,0], XGIN[:,1], 'ro')
    plt.plot(XGO[:,0], XGO[:,1], 'bo')
    ax = plt.gca()
    ax.add_patch(
        patches.Rectangle(
            (-R, -R), R*2, R*2, fill=False,      # remove background
            edgecolor="green"
        )
    )    

# ---------- PLOT 2D KERNEL SVM RING ----------    
def Plot_2D_Ring(XG, YG, R1 = 1, R2 = 2):
    gin = np.where(YG == -1)
    go = np.where(YG == 1)
    
    XGIN = XG[gin,:][0]
    XGO = XG[go,:][0]
    
    plt.figure()
    plt.plot(XGIN[:,0], XGIN[:,1], 'ro')
    plt.plot(XGO[:,0], XGO[:,1], 'bo')
    ax = plt.gca()
    circle1 = plt.Circle((0, 0), np.sqrt(R1), color='g', fill=False)
    ax.add_artist(circle1)
    circle2 = plt.Circle((0, 0), np.sqrt(R2), color='g', fill=False)
    ax.add_artist(circle2)

# ---------- PLOT 2D DOUGHNUT ----------
def Plot_2D_Doughnut(XG, YG, a = 1, b = 2):
    gin = np.where(YG == -1)
    go = np.where(YG == 1)
    
    XGIN = XG[gin,:][0]
    XGO = XG[go,:][0]
    
    plt.figure()
    plt.plot(XGIN[:,0], XGIN[:,1], 'ro')
    plt.plot(XGO[:,0], XGO[:,1], 'bo')
    
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = X**2 + Y**2 + X*Y - 1
    Z2 = X**2 + Y**2 + X*Y - 2
    plt.contour(X, Y, Z1, [0], colors = 'green')
    plt.contour(X, Y, Z2, [0], colors = 'green')
    plt.axis('equal')

    
        
# ---------- PLOT FINAL COMPARING ----------
def Mesh_Coordinates(X_Mesh, Y_Mesh):
    XX, YY = np.meshgrid(X_Mesh, Y_Mesh, sparse=False)
    C_Mesh = np.append(XX[:,0][np.newaxis].T, YY[:,0][np.newaxis].T,axis = 1)
    for i in range(1,XX.shape[1]):
        tmp = np.append(XX[:,i][np.newaxis].T, YY[:,i][np.newaxis].T,axis = 1)
        C_Mesh = np.append(C_Mesh, tmp, axis = 0)
    return C_Mesh

def Plot_Mesh(C_Mesh, L_Mesh):    
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

def Mesh_5D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh):
    X, Y, Z, T, P = np.meshgrid(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, sparse=False)
    XR = X.ravel()
    YR = Y.ravel()
    ZR = Z.ravel()
    TR = T.ravel()
    PR = P.ravel()
    C_Mesh = np.hstack([XR[:,np.newaxis], YR[:,np.newaxis], ZR[:,np.newaxis], TR[:,np.newaxis], PR[:,np.newaxis]])    
    return C_Mesh

def Mesh_6D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh):
    X, Y, Z, T, P, K = np.meshgrid(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh, sparse=False)
    XR = X.ravel()
    YR = Y.ravel()
    ZR = Z.ravel()
    TR = T.ravel()
    PR = P.ravel()
    KR = K.ravel()
    C_Mesh = np.hstack([XR[:,np.newaxis], YR[:,np.newaxis], ZR[:,np.newaxis], TR[:,np.newaxis], PR[:,np.newaxis], KR[:,np.newaxis]])    
    return C_Mesh

def Mesh_7D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh, L_Mesh):
    X, Y, Z, T, P, K, L = np.meshgrid(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh, L_Mesh, sparse=False)
    XR = X.ravel()
    YR = Y.ravel()
    ZR = Z.ravel()
    TR = T.ravel()
    PR = P.ravel()
    KR = K.ravel()
    LR = L.ravel()
    C_Mesh = np.hstack([XR[:,np.newaxis], YR[:,np.newaxis], ZR[:,np.newaxis], TR[:,np.newaxis], PR[:,np.newaxis], KR[:,np.newaxis], LR[:,np.newaxis]])    
    return C_Mesh

def Mesh_8D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh, L_Mesh, H_Mesh):
    X, Y, Z, T, P, K, L, H = np.meshgrid(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh, L_Mesh, H_Mesh, sparse=False)
    XR = X.ravel()
    YR = Y.ravel()
    ZR = Z.ravel()
    TR = T.ravel()
    PR = P.ravel()
    KR = K.ravel()
    LR = L.ravel()
    HR = H.ravel()
    C_Mesh = np.hstack([XR[:,np.newaxis], YR[:,np.newaxis], ZR[:,np.newaxis], TR[:,np.newaxis], PR[:,np.newaxis], KR[:,np.newaxis], LR[:,np.newaxis], HR[:,np.newaxis]])    
    return C_Mesh