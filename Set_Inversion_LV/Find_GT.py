# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:50:42 2018
Find groundtruth of equation (6) Marvel version 2
- parameter to estimate is p2, p4 (death rate of preys, grow rate of predators)
@author: ManhDuy
"""

import os
import numpy as np
import Functions.Plot_Lib as pll
import Functions.solve_lotka as sollot
import time
import matplotlib.pyplot as plt
import pickle
from scipy import integrate


###### SET UP PARAMS (BECAREFUL WITH YOUR CHANGES) ######
# params with the Lotka voltera problem
N = 200 # time step from 0 to Time
X0 = [50, 50] # initial population [pop of prey, pop of predator]
Time = 20 # Max time to run the model
M0 = 10 # condition that the population of prey is always above this threshold
p1 = 1 # grow rate of preys
p3 = 1 # death rate of predators
###### END SET UP PARAMS ######





###### DO NOT CHANGE THIS ######
CurDir = os.getcwd()
Link_GT = CurDir + '/Points_GT/'
if not os.path.exists(Link_GT):
    os.makedirs(Link_GT)
###### END DO NOT CHANGE THIS ######





###### CREATE MESHGRID ######
print('Creating Meshgrid ... ... ')
p2_Mesh = np.linspace(0.01, 0.1, 1000)
p4_Mesh = np.linspace(0.01, 0.1, 1000)
Point = pll.Mesh_2D_Coordinates(p2_Mesh, p4_Mesh) # Coord
print('Total point: ' + str(Point.shape[0]))





###### FIND THEIR LABELS ######
print('Solving ... ... ')
tic = time.clock()

GT = np.array([sollot.Decision_Lotka(X0, Time, N, M0, p1, Point[i,0], p3, Point[i,1]) for i in range(Point.shape[0])]) # true labels

toc = time.clock()
print('Total Solving Time: ' + str(toc - tic))





###### PLOT and SAVE FIGURES ######
pll.Plot_2D_Mesh(Point, GT)
plt.xlabel('p2')
plt.ylabel('p4')    
plt.savefig('GT_LV.png')





###### SAVE FILE ######
# If change saved name --> change in Random_Lib.py too
f = open(Link_GT + 'Points_GT.pckl', 'wb') # DO NOT CHANGE THE NAME
pickle.dump(Point, f)
f.close()

g = open(Link_GT + 'GT.pckl', 'wb') # DO NOT CHANGE THE NAME
pickle.dump(GT, g)
g.close()