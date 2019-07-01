# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:54:13 2017
Function Fx for Setinversion
@author: ManhDuy
"""

import numpy as np

def Fx_Circle(x, R = 2): # define class of function F(x)
    a = x[0]**2 + x[1]**2 - R
    if (a <= 0):
        return -1
    else:
        return 1
    
def Fx_Square(x, R = 2): # define class of function F(x)
    if (x[0] <= R and x[0] >= -R and x[1] <= R and x[1] >= -R):
        return -1
    else:
        return 1
    
def Fx_Ring(x, R1 = 1, R2 = 2): # define class of function F(x)
    a = x[0]**2 + x[1]**2
    low = R1
    high = R2
    if (a >= low and a <= high):
        return -1
    else:
        return 1

def Fx_Doughnut(x, a = 1, b = 2):
    t1 = x[0]**2 + x[1]**2 + x[0]*x[1] - a
    t2 = x[0]**2 + x[1]**2 + x[0]*x[1] - b
    if (t1 >= 0 and t2 <= 0):
        return -1
    else:
        return 1
    
def Fx_ElipCurve(x, a = -2, b = -1):
    v = x[1]**2 - x[0]**3 - a*x[0] - b
    if (v <= 0):
        return -1
    else:
        return 1
    
def Fx_Sphere(x, R = 2): # define class of function F(x)
    a = np.sum(x**2) - R
    if (a <= 0):
        return -1
    else:
        return 1 
    
