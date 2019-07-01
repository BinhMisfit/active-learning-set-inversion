#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:30:32 2019
Solve the SIR ODE functions base on alpha and beta parameters
Note that in the paper we estimate alpha and R0 (not beta) --> need to convert R0 to beta then apply this function
The formular converting R0 to beta is applied in the file Find_GT.py
@author: duynguyen
"""

'''
Function ODE
dSdt = -beta * S * I
dIdt = beta*S*I - alpha*I

solve function of alpha and beta --> need to convert from R0 to beta (in Find_GT.py) before apply this function
The condition of the SIR problem (stated in the paper): |R_{predicted} - R_{true}|<= max(R_{true}/10,5)
'''

import numpy as np
import pylab as pyl
import scipy as scp
import matplotlib.pyplot as plt
import pandas as pd

def dX_dt(X, t, alpha, beta):
    return np.array([ -beta * X[0] * X[1],  
                      beta * X[0] * X[1] - alpha * X[1]])
    
def Solve_SIR(X0, Time, Step, alpha, beta):
    t = np.linspace(0, Time,  Step) 
    X, infodict = scp.integrate.odeint(dX_dt, X0, t, args=(alpha, beta), full_output = 1)
    
    return X
    
def Decision_SIR(X0, Time, Step, alpha, beta, data):
    # data is pandas data frame read from csv file
    time_vec = np.linspace(0, Time,  Step) 
    R = Solve_SIR(X0, Time, Step, alpha, beta)
    idx = [np.where(time_vec == x)[0][0] for x in data.iloc[:,0]]
    S = R[idx, 0:1] # Take S values at the row that matches time in data
    I = R[idx, 1:2] # Take I values at the row that matches time in data
    R1= 261-S-I # recover people (R_predicted) = total pop (261) - S - I --> need to change the constant number 261 to sum of population in the csv file (update in later version)
    D = data.iloc[:, 3:4 ] # take R (R_true) from data

    Dif = abs(R1 - D)-np.maximum(0.1*D,5) # |R_{predicted} - R_{true}|<= max(R_{true}/10,5) --> condition stated in the paper
    
    tl = np.all(Dif.max(axis = 1) <=0) # check whether R_predicted at all timepoints matches the above condition or not
    if (tl):
        result = -1
    else:
        result = 1
    return result
       

    