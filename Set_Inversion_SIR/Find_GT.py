#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 03:29:10 2019
Find Groundtruth of SIR inversion
- parameter to estimate is alpha and R0
- Formular converting R0 to beta: beta = R0 * alpha / Totalpop (Totalpop = S + I + R = 261, as recorded in Data.csv)
@author: duynguyen
"""

import numpy as np
import pylab as pyl
import Functions.Plot_Lib as pll
from scipy import integrate
import scipy as scp
import Functions.solve_SIR as solsir
import time
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd

###### SET UP PARAMS (BECAREFUL WITH YOUR CHANGES) ######
# Read Data.csv recorded
data = pd.read_csv('Data.csv')
X0 = np.asarray(data.iloc[0, 1 : 3]) # initial population [pop of Susceptible, Infected, Recovered]
# params with the SIR problem
Time = 4 # Max time to run the model
N = 9 # Time step from 0 to Time
###### END SET UP PARAMS ######





###### DO NOT CHANGE THIS ######
CurDir = os.getcwd()
Link_GT = CurDir + '/Points_GT/'
if not os.path.exists(Link_GT):
    os.makedirs(Link_GT)
###### END DO NOT CHANGE THIS ######




    
###### CREATE MESHGRID ######
print('Creating Meshgrid ... ... ')
lnal_Mesh = np.linspace(0.6, 1.3, 1000) # ln alpha
lnR0_Mesh = np.linspace(np.log(1.5), np.log(2.0), 1000) # ln R0
Point = pll.Mesh_2D_Coordinates(lnal_Mesh, lnR0_Mesh)
print('Total points: ' + str(Point.shape[0]))





###### FIND THEIR LABELS ######
print('Solving ... ... ')
tic = time.clock()

# convert R0 to beta by the formular: np.exp(Point[i,1])*np.exp(Point[i,0])/261
GT = np.array([solsir.Decision_SIR(X0, Time, N, np.exp(Point[i,0]), np.exp(Point[i,1])*np.exp(Point[i,0])/261,data) for i in range(Point.shape[0])])

toc = time.clock()
print('Total Solving Time: ' + str(toc - tic))





###### PLOT and SAVE FIGURES ######
pll.Plot_2D_Mesh(Point, GT)
plt.xlabel(r'$log(\alpha)$', fontsize = 14)
plt.ylabel(r'$log(R_0)$', fontsize = 14)    
plt.savefig('GT_SIR.png')





###### SAVE FILE ######
# If change saved name --> change in Random_Lib.py too
f = open(Link_GT + 'Points_GT.pckl', 'wb') # DO NOT CHANGE THE NAME
pickle.dump(Point, f)
f.close()

g = open(Link_GT + 'GT.pckl', 'wb') # DO NOT CHANGE THE NAME
pickle.dump(GT, g)
g.close()
