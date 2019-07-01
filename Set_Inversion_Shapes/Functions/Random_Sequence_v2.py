#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:30:56 2019
Random library for Set Inversion Project
(include Normal Random, LHS, MCMC)
@author: duynguyen
"""

from __future__ import division
from collections import Sequence
import numpy as np
import random
import math
import pyDOE as pD
from scipy.stats.distributions import norm
import sobol_seq as sbs

random.seed(5)
class FRange(Sequence):
    """ Lazily evaluated floating point range of evenly spaced floats
        (inclusive at both ends)

        >>> list(FRange(low=10, high=20, num_points=5))
        [10.0, 12.5, 15.0, 17.5, 20.0]

    """
    def __init__(self, low, high, num_points):
        self.low = low
        self.high = high
        self.num_points = num_points

    def __len__(self):
        return self.num_points

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('Out of range')
        p = index / (self.num_points - 1)
        return self.low * (1.0 - p) + self.high * p
    
def Convert_Range(x, OR, NR):
    y = NR[0] + (x - OR[0])*(NR[1] - NR[0])/(OR[1] - OR[0])
    return y

# ============================================================================
# ----- Random for 2D Circle -----    
def Random_2D_Circle(RLim, OLim, step = 1000, num_points = 1, type_random = 'Normal'):
    # random num_points inside the shape of circle (within RLim and OLim limit)
    if (type_random == 'LHS'):
        t = pD.lhs(2, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        theta = t[:,1]
        theta = Convert_Range(theta, [0, 1], OLim)
        theta = theta.ravel()
    elif (type_random == 'Sobol'):
        t = sbs.i4_sobol_generate(2, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        theta = t[:,1]
        theta = Convert_Range(theta, [0, 1], OLim)
        theta = theta.ravel()
    else: # Normal
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        theta = np.array(random.sample(FRange(low = OLim[0], high = OLim[1], num_points = step), k = num_points))
        
    xval = np.array([r[i]*math.cos(theta[i]) for i in range(num_points)])
    yval = np.array([r[i]*math.sin(theta[i]) for i in range(num_points)])
    Co = np.append([xval],[yval],axis = 0)
    Co = Co.T
    return Co

def Random_2D_Circle_Full(R = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # R is the radius of circle --> warning --> x^2 + y^2 = R^2 (while our boundary function is x^2 + y^2 = R)
    # need to be careful about the square root
    # random num_points points in full (inside and outside the circle) --> half for each one
    Rin = [0.02*R, 0.98*R]
    Rout = [1.02*R, 1.98*R]
    OLim = [0, 2*math.pi]
    IN = Random_2D_Circle(Rin, OLim, step, int(num_points/2), type_random = type_random) # CLASS -1
    OUT = Random_2D_Circle(Rout, OLim, step, int(num_points/2), type_random = type_random) # CLASS 1
    X_samp = np.append(IN, OUT, axis = 0)
    return X_samp

# ============================================================================
# ----- Random for 2D Square -----  
def Random_2D_Square(XLim, YLim, step = 1000, num_points = 1, type_random = 'Normal'):
    if (type_random == 'LHS'):
        t = pD.lhs(2, num_points)
        xval = t[:,0]
        xval = Convert_Range(xval, [0, 1], XLim)
        xval = xval.ravel()
        yval = t[:,1]
        yval = Convert_Range(yval, [0, 1], YLim)
        yval = yval.ravel()
    elif (type_random == 'Sobol'):
        t = sbs.i4_sobol_generate(2, num_points)
        xval = t[:,0]
        xval = Convert_Range(xval, [0, 1], XLim)
        xval = xval.ravel()
        yval = t[:,1]
        yval = Convert_Range(yval, [0, 1], YLim)
        yval = yval.ravel()
    else:
        xval = np.array(random.sample(FRange(low = XLim[0], high = XLim[1], num_points = step), k = num_points))
        yval = np.array(random.sample(FRange(low = YLim[0], high = YLim[1], num_points = step), k = num_points))
    
    Co = np.append([xval],[yval],axis = 0)
    Co = Co.T    
    return Co

def Random_2D_Square_Full(R1 = 2, R2 = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # R1 = 2 --> x_range of square will be [-2, 2]
    # R2 = 3 --> y_range of square will be [-3, 3]
    Rin_1 = [-0.98*R1, 0.98*R1]
    Rin_2 = [-0.98*R2, 0.98*R2]
    IN = Random_2D_Square(Rin_1, Rin_2, step = step, num_points = int(num_points/2), type_random = type_random)    
    
    # Random outside (4 direction)
    alpha = 1.02
    beta = 1.98
    Rup_1 = [-beta*R1, beta*R1]
    Rup_2 = [alpha*R2, beta*R2]
    Rright_1 = [alpha*R1, beta*R1]
    Rright_2 = [-alpha*R2, alpha*R2]
    Rdown_1 = [-beta*R1, beta*R1]
    Rdown_2 = [-beta*R2, -alpha*R2]
    Rleft_1 = [-beta*R1, -alpha*R1]
    Rleft_2 = [-alpha*R2, alpha*R2]
    
    FO1 = Random_2D_Square(Rup_1, Rup_2, step = step, num_points = int(num_points/8), type_random = type_random)
    FO2 = Random_2D_Square(Rright_1, Rright_2, step = step, num_points = int(num_points/8), type_random = type_random)
    FO3 = Random_2D_Square(Rdown_1, Rdown_2, step = step, num_points = int(num_points/8), type_random = type_random)
    FO4 = Random_2D_Square(Rleft_1, Rleft_2, step = step, num_points = int(num_points/2) - 3*int(num_points/8), type_random = type_random)
    FO = np.append(FO1, FO2, axis = 0)
    FO = np.append(FO, FO3, axis = 0)
    FO = np.append(FO, FO4, axis = 0)
    
    X_samp = np.append(IN, FO, axis = 0)    

    return X_samp

# ============================================================================ing
# ----- Random for 2D Ring -----  
def Random_2D_Ring_Full(R1 = 1, R2 = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Fx in this function: R1^2 < x^2 + y^2 < R2^2 --> warning --> Boundary function R1 < x^2 + y^2 < R2
    Rin = [1.02*R1, 0.98*R2]
    Rout_1 = [0.02*R1, 0.98*R1]
    Rout_2 = [1.02*R2, 1.98*R2]
    OLim = [0, 2*math.pi]
    IN = Random_2D_Circle(Rin, OLim, step, num_points = int(num_points/2), type_random = type_random)
    FO1 = Random_2D_Circle(Rout_1, OLim, step, num_points = int(num_points/4), type_random = type_random)
    FO2 = Random_2D_Circle(Rout_2, OLim, step, num_points = int(num_points/4), type_random = type_random)
    FO = np.append(FO1, FO2, axis = 0)
    X_samp = np.append(IN, FO, axis = 0)
    
    return X_samp
    

# ============================================================================
# ----- Random for 2D Doughnut  -----
def Random_2D_Doughnut(RLim, OLim, step = 1000, num_points = 1, type_random = 'Normal'):
    # random num_points inside the shape of circle (within RLim and OLim limit)
    if (type_random == 'LHS'):
        t = pD.lhs(2, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        theta = t[:,1]
        theta = Convert_Range(theta, [0, 1], OLim)
        theta = theta.ravel()
    elif (type_random == 'Sobol'):
        t = sbs.i4_sobol_generate(2, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        theta = t[:,1]
        theta = Convert_Range(theta, [0, 1], OLim)
        theta = theta.ravel()        
    else: # Normal
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        theta = np.array(random.sample(FRange(low = OLim[0], high = OLim[1], num_points = step), k = num_points))
        
    xval = np.array([r[i]*math.cos(theta[i])-r[i]*math.sin(theta[i])/np.sqrt(3) for i in range(num_points)])
    yval = np.array([2*r[i]*math.sin(theta[i])/np.sqrt(3) for i in range(num_points)])
    Co = np.append([xval],[yval],axis = 0)
    Co = Co.T
    return Co

def Random_2D_Doughnut_Full(R1 = 1, R2 = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Fx in this function: R1^2 < x^2 + y^2 < R2^2 --> warning --> Boundary function R1 < x^2 + y^2 < R2
    Rin = [1.02*R1, 0.98*R2]
    Rout_1 = [0.02*R1, 0.98*R1]
    Rout_2 = [1.02*R2, 1.98*R2]
    OLim = [0, 2*math.pi]
    IN = Random_2D_Doughnut(Rin, OLim, step, num_points = int(num_points/2), type_random = type_random)
    FO1 = Random_2D_Doughnut(Rout_1, OLim, step, num_points = int(num_points/4), type_random = type_random)
    FO2 = Random_2D_Doughnut(Rout_2, OLim, step, num_points = int(num_points/4), type_random = type_random)
    FO = np.append(FO1, FO2, axis = 0)
    X_samp = np.append(IN, FO, axis = 0)
    
    return X_samp

# ============================================================================
# ----- Random for 3D Sphere  -----
def Random_3D_Sphere(RLim, T1Lim, T2Lim, step = 1000, num_points = 1, type_random = 'Normal'):
    # 0 < T1 < pi, 0 < T2 < 2pi
    # [0, math.pi], [0, 2*math.pi]
    if (type_random == 'LHS' or type_random == 'Sobol'):
        if (type_random == 'LHS'):
            t = pD.lhs(3, num_points)
        else:
            t = sbs.i4_sobol_generate(3, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        t1 = t[:,1]
        t1 = Convert_Range(t1, [0, 1], T1Lim)
        t1 = t1.ravel()
        t2 = t[:,2]
        t2 = Convert_Range(t2, [0, 1], T2Lim)
        t2 = t2.ravel()       
    else:
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        t1 = np.array(random.sample(FRange(low = T1Lim[0], high = T1Lim[1], num_points = step), k = num_points))
        t2 = np.array(random.sample(FRange(low = T2Lim[0], high = T2Lim[1], num_points = step), k = num_points))
    
    xval = np.array([r[i]*math.cos(t1[i]) for i in range(num_points)])
    yval = np.array([r[i]*math.sin(t1[i])*math.cos(t2[i]) for i in range(num_points)])
    zval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i]) for i in range(num_points)])
    Co = np.hstack([xval[:,np.newaxis],yval[:,np.newaxis],zval[:,np.newaxis]])
    return Co

def Random_3D_Sphere_Full(R = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Same with 2D_Circle_Full
    Rin = [0.02*R, 0.98*R]
    Rout = [1.02*R, 1.98*R]
    T1Lim = [0, math.pi]
    T2Lim = [0, 2*math.pi]
    IN = Random_3D_Sphere(Rin, T1Lim, T2Lim, step, int(num_points/2), type_random = type_random) # CLASS -1
    OUT = Random_3D_Sphere(Rout, T1Lim, T2Lim, step, int(num_points/2), type_random = type_random) # CLASS 1
    X_samp = np.append(IN, OUT, axis = 0)
    return X_samp

# ============================================================================
# ----- Random for 4D Sphere  -----
def Random_4D_Sphere(RLim, T1Lim, T2Lim, T3Lim, step = 1000, num_points = 1, type_random = 'Normal'):
    # 0 < t1 < pi, 0 < t2, t3 < 2pi
    if (type_random == 'LHS' or type_random == 'Sobol'):
        if (type_random == 'LHS'):
            t = pD.lhs(4, num_points)
        else:
            t = sbs.i4_sobol_generate(4, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        t1 = t[:,1]
        t1 = Convert_Range(t1, [0, 1], T1Lim)
        t1 = t1.ravel()
        t2 = t[:,2]
        t2 = Convert_Range(t2, [0, 1], T2Lim)
        t2 = t2.ravel()       
        t3 = t[:,3]
        t3 = Convert_Range(t3, [0, 1], T3Lim)
        t3 = t3.ravel()  
    else:
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        t1 = np.array(random.sample(FRange(low = T1Lim[0], high = T1Lim[1], num_points = step), k = num_points))
        t2 = np.array(random.sample(FRange(low = T2Lim[0], high = T2Lim[1], num_points = step), k = num_points))
        t3 = np.array(random.sample(FRange(low = T3Lim[0], high = T3Lim[1], num_points = step), k = num_points))
    
    xval = np.array([r[i]*math.cos(t1[i]) for i in range(num_points)])
    yval = np.array([r[i]*math.sin(t1[i])*math.cos(t2[i]) for i in range(num_points)])
    zval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.cos(t3[i]) for i in range(num_points)])
    tval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i]) for i in range(num_points)])
    Co = np.hstack([xval[:,np.newaxis],yval[:,np.newaxis],zval[:,np.newaxis],tval[:,np.newaxis]])
    return Co    

def Random_4D_Sphere_Full(R = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Same with 2D_Circle_Full
    Rin = [0.02*R, 0.98*R]
    Rout = [1.02*R, 1.98*R]
    T1Lim = [0, math.pi]
    T2Lim = [0, 2*math.pi]
    T3Lim = [0, 2*math.pi]
    IN = Random_4D_Sphere(Rin, T1Lim, T2Lim, T3Lim, step, int(num_points/2), type_random = type_random) # CLASS -1
    OUT = Random_4D_Sphere(Rout, T1Lim, T2Lim, T3Lim, step, int(num_points/2), type_random = type_random) # CLASS 1
    X_samp = np.append(IN, OUT, axis = 0)
    return X_samp

# ============================================================================
# ----- Random for 5D Sphere  -----
def Random_5D_Sphere(RLim, T1Lim, T2Lim, T3Lim, T4Lim, step = 1000, num_points = 1, type_random = 'Normal'):
    # 0 < t1 < pi, 0 < t2, t3 < 2pi
    if (type_random == 'LHS' or type_random == 'Sobol'):
        if (type_random == 'LHS'):
            t = pD.lhs(5, num_points)
        else:
            t = sbs.i4_sobol_generate(5, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        t1 = t[:,1]
        t1 = Convert_Range(t1, [0, 1], T1Lim)
        t1 = t1.ravel()
        t2 = t[:,2]
        t2 = Convert_Range(t2, [0, 1], T2Lim)
        t2 = t2.ravel()       
        t3 = t[:,3]
        t3 = Convert_Range(t3, [0, 1], T3Lim)
        t3 = t3.ravel()  
        t4 = t[:,4]
        t4 = Convert_Range(t4, [0, 1], T4Lim)
        t4 = t4.ravel()  
    else:
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        t1 = np.array(random.sample(FRange(low = T1Lim[0], high = T1Lim[1], num_points = step), k = num_points))
        t2 = np.array(random.sample(FRange(low = T2Lim[0], high = T2Lim[1], num_points = step), k = num_points))
        t3 = np.array(random.sample(FRange(low = T3Lim[0], high = T3Lim[1], num_points = step), k = num_points))
        t4 = np.array(random.sample(FRange(low = T4Lim[0], high = T4Lim[1], num_points = step), k = num_points))
        
    xval = np.array([r[i]*math.cos(t1[i]) for i in range(num_points)])
    yval = np.array([r[i]*math.sin(t1[i])*math.cos(t2[i]) for i in range(num_points)])
    zval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.cos(t3[i]) for i in range(num_points)])
    tval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.cos(t4[i]) for i in range(num_points)])
    pval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i]) for i in range(num_points)])
    Co = np.hstack([xval[:,np.newaxis],yval[:,np.newaxis],zval[:,np.newaxis],tval[:,np.newaxis],pval[:,np.newaxis]])
    return Co   

def Random_5D_Sphere_Full(R = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Same with 2D_Circle_Full
    Rin = [0.02*R, 0.98*R]
    Rout = [1.02*R, 1.98*R]
    T1Lim = [0, math.pi]
    T2Lim = [0, 2*math.pi]
    T3Lim = [0, 2*math.pi]
    T4Lim = [0, 2*math.pi]
    IN = Random_5D_Sphere(Rin, T1Lim, T2Lim, T3Lim, T4Lim, step, int(num_points/2), type_random = type_random) # CLASS -1
    OUT = Random_5D_Sphere(Rout, T1Lim, T2Lim, T3Lim, T4Lim, step, int(num_points/2), type_random = type_random) # CLASS 1
    X_samp = np.append(IN, OUT, axis = 0)
    return X_samp

# ============================================================================
# ----- Random for 6D Sphere  -----
def Random_6D_Sphere(RLim, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, step = 1000, num_points = 1, type_random = 'Normal'):
    # 0 < t1 < pi, 0 < t2, t3 < 2pi
    if (type_random == 'LHS' or type_random == 'Sobol'):
        if (type_random == 'LHS'):
            t = pD.lhs(6, num_points)
        else:
            t = sbs.i4_sobol_generate(6, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        t1 = t[:,1]
        t1 = Convert_Range(t1, [0, 1], T1Lim)
        t1 = t1.ravel()
        t2 = t[:,2]
        t2 = Convert_Range(t2, [0, 1], T2Lim)
        t2 = t2.ravel()       
        t3 = t[:,3]
        t3 = Convert_Range(t3, [0, 1], T3Lim)
        t3 = t3.ravel()  
        t4 = t[:,4]
        t4 = Convert_Range(t4, [0, 1], T4Lim)
        t4 = t4.ravel()  
        t5 = t[:,5]
        t5 = Convert_Range(t5, [0, 1], T5Lim)
        t5 = t5.ravel()  
    else:
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        t1 = np.array(random.sample(FRange(low = T1Lim[0], high = T1Lim[1], num_points = step), k = num_points))
        t2 = np.array(random.sample(FRange(low = T2Lim[0], high = T2Lim[1], num_points = step), k = num_points))
        t3 = np.array(random.sample(FRange(low = T3Lim[0], high = T3Lim[1], num_points = step), k = num_points))
        t4 = np.array(random.sample(FRange(low = T4Lim[0], high = T4Lim[1], num_points = step), k = num_points))
        t5 = np.array(random.sample(FRange(low = T5Lim[0], high = T5Lim[1], num_points = step), k = num_points))
        
    xval = np.array([r[i]*math.cos(t1[i]) for i in range(num_points)])
    yval = np.array([r[i]*math.sin(t1[i])*math.cos(t2[i]) for i in range(num_points)])
    zval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.cos(t3[i]) for i in range(num_points)])
    tval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.cos(t4[i]) for i in range(num_points)])
    pval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.cos(t5[i]) for i in range(num_points)])
    kval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.sin(t5[i]) for i in range(num_points)])
    Co = np.hstack([xval[:,np.newaxis],yval[:,np.newaxis],zval[:,np.newaxis],tval[:,np.newaxis],pval[:,np.newaxis],kval[:,np.newaxis]])
    return Co 

def Random_6D_Sphere_Full(R = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Same with 2D_Circle_Full
    Rin = [0.02*R, 0.98*R]
    Rout = [1.02*R, 1.98*R]
    T1Lim = [0, math.pi]
    T2Lim = [0, 2*math.pi]
    T3Lim = [0, 2*math.pi]
    T4Lim = [0, 2*math.pi]
    T5Lim = [0, 2*math.pi]
    IN = Random_6D_Sphere(Rin, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, step, int(num_points/2), type_random = type_random) # CLASS -1
    OUT = Random_6D_Sphere(Rout, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, step, int(num_points/2), type_random = type_random) # CLASS 1
    X_samp = np.append(IN, OUT, axis = 0)
    return X_samp

# ============================================================================
# ----- Random for 7D Sphere  -----
def Random_7D_Sphere(RLim, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, T6Lim, step = 1000, num_points = 1, type_random = 'Normal'):
    # 0 < t1 < pi, 0 < t2, t3 < 2pi
    if (type_random == 'LHS' or type_random == 'Sobol'):
        if (type_random == 'LHS'):
            t = pD.lhs(7, num_points)
        else:
            t = sbs.i4_sobol_generate(7, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        t1 = t[:,1]
        t1 = Convert_Range(t1, [0, 1], T1Lim)
        t1 = t1.ravel()
        t2 = t[:,2]
        t2 = Convert_Range(t2, [0, 1], T2Lim)
        t2 = t2.ravel()       
        t3 = t[:,3]
        t3 = Convert_Range(t3, [0, 1], T3Lim)
        t3 = t3.ravel()  
        t4 = t[:,4]
        t4 = Convert_Range(t4, [0, 1], T4Lim)
        t4 = t4.ravel()  
        t5 = t[:,5]
        t5 = Convert_Range(t5, [0, 1], T5Lim)
        t5 = t5.ravel()
        t6 = t[:,6]
        t6 = Convert_Range(t6, [0, 1], T6Lim)
        t6 = t6.ravel()  
    else:
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        t1 = np.array(random.sample(FRange(low = T1Lim[0], high = T1Lim[1], num_points = step), k = num_points))
        t2 = np.array(random.sample(FRange(low = T2Lim[0], high = T2Lim[1], num_points = step), k = num_points))
        t3 = np.array(random.sample(FRange(low = T3Lim[0], high = T3Lim[1], num_points = step), k = num_points))
        t4 = np.array(random.sample(FRange(low = T4Lim[0], high = T4Lim[1], num_points = step), k = num_points))
        t5 = np.array(random.sample(FRange(low = T5Lim[0], high = T5Lim[1], num_points = step), k = num_points))
        t6 = np.array(random.sample(FRange(low = T6Lim[0], high = T6Lim[1], num_points = step), k = num_points))
        
    xval = np.array([r[i]*math.cos(t1[i]) for i in range(num_points)])
    yval = np.array([r[i]*math.sin(t1[i])*math.cos(t2[i]) for i in range(num_points)])
    zval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.cos(t3[i]) for i in range(num_points)])
    tval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.cos(t4[i]) for i in range(num_points)])
    pval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.cos(t5[i]) for i in range(num_points)])
    kval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.sin(t5[i])*math.cos(t6[i]) for i in range(num_points)])
    lval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.sin(t5[i])*math.sin(t6[i]) for i in range(num_points)])
    Co = np.hstack([xval[:,np.newaxis],yval[:,np.newaxis],zval[:,np.newaxis],tval[:,np.newaxis],pval[:,np.newaxis],kval[:,np.newaxis], lval[:,np.newaxis]])
    return Co

def Random_7D_Sphere_Full(R = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Same with 2D_Circle_Full
    Rin = [0.02*R, 0.98*R]
    Rout = [1.02*R, 1.98*R]
    T1Lim = [0, math.pi]
    T2Lim = [0, 2*math.pi]
    T3Lim = [0, 2*math.pi]
    T4Lim = [0, 2*math.pi]
    T5Lim = [0, 2*math.pi]
    T6Lim = [0, 2*math.pi]
    IN = Random_7D_Sphere(Rin, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, T6Lim, step, int(num_points/2), type_random = type_random) # CLASS -1
    OUT = Random_7D_Sphere(Rout, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, T6Lim, step, int(num_points/2), type_random = type_random) # CLASS 1
    X_samp = np.append(IN, OUT, axis = 0)
    return X_samp

# ============================================================================
# ----- Random for 8D Sphere  -----
def Random_8D_Sphere(RLim, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, T6Lim, T7Lim, step = 1000, num_points = 1, type_random = 'Normal'):
    # 0 < t1 < pi, 0 < t2, t3 < 2pi
    if (type_random == 'LHS' or type_random == 'Sobol'):
        if (type_random == 'LHS'):
            t = pD.lhs(8, num_points)
        else:
            t = sbs.i4_sobol_generate(8, num_points)
        r = t[:,0]
        r = Convert_Range(r, [0, 1], RLim)
        r = r.ravel()
        t1 = t[:,1]
        t1 = Convert_Range(t1, [0, 1], T1Lim)
        t1 = t1.ravel()
        t2 = t[:,2]
        t2 = Convert_Range(t2, [0, 1], T2Lim)
        t2 = t2.ravel()       
        t3 = t[:,3]
        t3 = Convert_Range(t3, [0, 1], T3Lim)
        t3 = t3.ravel()  
        t4 = t[:,4]
        t4 = Convert_Range(t4, [0, 1], T4Lim)
        t4 = t4.ravel()  
        t5 = t[:,5]
        t5 = Convert_Range(t5, [0, 1], T5Lim)
        t5 = t5.ravel()
        t6 = t[:,6]
        t6 = Convert_Range(t6, [0, 1], T6Lim)
        t6 = t6.ravel()  
        t7 = t[:,7]
        t7 = Convert_Range(t7, [0, 1], T7Lim)
        t7 = t7.ravel() 
    else:
        r = np.array(random.sample(FRange(low = RLim[0], high = RLim[1], num_points = step), k = num_points))
        t1 = np.array(random.sample(FRange(low = T1Lim[0], high = T1Lim[1], num_points = step), k = num_points))
        t2 = np.array(random.sample(FRange(low = T2Lim[0], high = T2Lim[1], num_points = step), k = num_points))
        t3 = np.array(random.sample(FRange(low = T3Lim[0], high = T3Lim[1], num_points = step), k = num_points))
        t4 = np.array(random.sample(FRange(low = T4Lim[0], high = T4Lim[1], num_points = step), k = num_points))
        t5 = np.array(random.sample(FRange(low = T5Lim[0], high = T5Lim[1], num_points = step), k = num_points))
        t6 = np.array(random.sample(FRange(low = T6Lim[0], high = T6Lim[1], num_points = step), k = num_points))
        t7 = np.array(random.sample(FRange(low = T7Lim[0], high = T7Lim[1], num_points = step), k = num_points))
        
    xval = np.array([r[i]*math.cos(t1[i]) for i in range(num_points)])
    yval = np.array([r[i]*math.sin(t1[i])*math.cos(t2[i]) for i in range(num_points)])
    zval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.cos(t3[i]) for i in range(num_points)])
    tval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.cos(t4[i]) for i in range(num_points)])
    pval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.cos(t5[i]) for i in range(num_points)])
    kval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.sin(t5[i])*math.cos(t6[i]) for i in range(num_points)])
    lval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.sin(t5[i])*math.sin(t6[i])*math.cos(t7[i]) for i in range(num_points)])
    hval = np.array([r[i]*math.sin(t1[i])*math.sin(t2[i])*math.sin(t3[i])*math.sin(t4[i])*math.sin(t5[i])*math.sin(t6[i])*math.sin(t7[i]) for i in range(num_points)])
    Co = np.hstack([xval[:,np.newaxis],yval[:,np.newaxis],zval[:,np.newaxis],tval[:,np.newaxis],pval[:,np.newaxis],kval[:,np.newaxis], lval[:,np.newaxis], hval[:,np.newaxis]])
    return Co

def Random_8D_Sphere_Full(R = 2, step = 1000, num_points = 2, type_random = 'Normal'):
    # Same with 2D_Circle_Full
    Rin = [0.02*R, 0.98*R]
    Rout = [1.02*R, 1.98*R]
    T1Lim = [0, math.pi]
    T2Lim = [0, 2*math.pi]
    T3Lim = [0, 2*math.pi]
    T4Lim = [0, 2*math.pi]
    T5Lim = [0, 2*math.pi]
    T6Lim = [0, 2*math.pi]
    T7Lim = [0, 2*math.pi]
    IN = Random_8D_Sphere(Rin, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, T6Lim, T7Lim, step, int(num_points/2), type_random = type_random) # CLASS -1
    OUT = Random_8D_Sphere(Rout, T1Lim, T2Lim, T3Lim, T4Lim, T5Lim, T6Lim, T7Lim, step, int(num_points/2), type_random = type_random) # CLASS 1
    X_samp = np.append(IN, OUT, axis = 0)
    return X_samp

#import pymc3 as pm
#import numpy as np
#
#data = 0.25 * np.random.randn(20) + 0.5 # (mean = 0.5, sigma = 0.5)
#
#with pm.Model():
#    mu = pm.Normal('mu', 0, 1)
#    sigma = 1.
#    returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)
#    
#    step = pm.Metropolis()
#    trace = pm.sample(15000, step)