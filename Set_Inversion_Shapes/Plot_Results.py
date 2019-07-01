#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 01:40:11 2019
Plot the result (Points_Figure) with Overlay Form
Make sure that these following are matched with models you have trained
- Shape_vec
- Random_Type
- K_samp
- Active_Points
@author: duynguyen
"""

import warnings
warnings.filterwarnings('ignore')
import os
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import ConvexHull
import numpy as np


###### PARAMETERS (BECAREFUL WITH YOUR CHANGES) ######
# List all shapes that you want to run (only support specific shapes --> See Appendix)
Shape_vec = ['2D_Circle', '2D_Ring']
Random_Type = 'LHS' # Change this for sampling initial points
K_samp = 100
Active_Points = 400
idx_method_vec = [0, 1, 2, 3] # remove a number corresponding to the method that do not want to run, Exp: remove 2, to not run MLP 
###### END PARAMETERS ######





###### SET UP ######
def Plot_Mesh(C_Mesh, L_Mesh):    
    co = np.where(L_Mesh == 1)
    ci = np.where(L_Mesh ==- 1)
    XO = C_Mesh[co]
    XI = C_Mesh[ci]
    plt.plot(XO[:,0], XO[:,1], 'bo', markersize = 2, zorder = 1)
    plt.plot(XI[:,0], XI[:,1], 'ro', markersize = 2, zorder = 1)
###### END SET UP ######
    
    
    
    
    
###### DO NOT CHANGE THIS ######
R = 2 # For others shape
R1 = 1 # For Ring
R2 = 2 # For Ring
a_dn = 1 # Doughnut
b_dn = 2 # Doughnut
R_3D = 0.5
R_4D = 0.25
R_5D = 0.25
R_High = 0.25

xmin = -3.05
xmax = 3.05
ymin = -3.05
ymax = 3.05
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

Random_Type_Generate = 'Normal'

method = ['SVM', 'KNN', 'MLP', 'RF']

CurDir = os.getcwd()
Link_Points = CurDir + '/Points/' # Directory Path to Points (to plot)
Link_Figures = CurDir + '/Figures/' # Directory Path to save Figures
if not os.path.exists(Link_Figures):
    os.makedirs(Link_Figures)
###### END DO NOT CHANGE THIS ######







###### START RUNNING ######
for idx_shape in range(len(Shape_vec)):
    Shape = Shape_vec[idx_shape]
    print('===== PLOT ' + Shape + ' =====')
    # ----- List models -----
    
    tail_string = '_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.pkl'
    model_name = ['Points_Figure_' + x + tail_string for x in method]
    
    # ----- Run prediction for each method
    for idx_method in idx_method_vec:
        # Load Trained model
        print('[Loading File] ' + model_name[idx_method])
        result_dict = pickle.load(open(Link_Points + model_name[idx_method], 'rb'))
        
        GT = result_dict['CT']
        Points = result_dict['grid']
        Pred = result_dict['CP']
        
        idx_in_GT = np.where(GT == -1)[0]
        idx_out_GT = np.where(GT == 1)[0]
        
        idx_in_Pred = np.where(Pred == -1)[0]
        idx_out_Pred = np.where(Pred == 1)[0]
        
        Points_in_GT = Points[idx_in_GT, ]
        Points_out_GT = Points[idx_out_GT, ]
        
        Points_in_Pred = Points[idx_in_Pred, ]
        Points_out_Pred = Points[idx_out_Pred, ]
        
        Hull_GT = ConvexHull(Points_in_GT)
        
        backgroud_hex = '#efefef'
        gt_hex = '#FF3F3F'
        pred_hex = '#eae55d'
        
        if (Shape == '2D_Doughnut'): 
            Z1 = X**2 + Y**2 + X*Y - a_dn
            Z2 = X**2 + Y**2 + X*Y - b_dn
            plt.contour(X, Y, Z1, [0], colors = gt_hex, linestyles = 'dashed', linewidths = 1.6, zorder = 2)
            plt.contour(X, Y, Z2, [0], colors = gt_hex, linestyles = 'dashed', linewidths = 1.6, zorder = 2)
        elif (Shape == '2D_Ring'):
            Z1 = X**2 + Y**2 - R1
            Z2 = X**2 + Y**2 - R2
            plt.contour(X, Y, Z1, [0], colors = gt_hex, linestyles = 'dashed', linewidths = 1.6, zorder = 2)
            plt.contour(X, Y, Z2, [0], colors = gt_hex, linestyles = 'dashed', linewidths = 1.6, zorder = 2)
        else:
            plt.plot(Points_in_GT[Hull_GT.vertices,0], Points_in_GT[Hull_GT.vertices,1], '--', color = gt_hex, lw=1.6)
        plt.plot(Points_out_Pred[:,0], Points_out_Pred[:,1], 'o', color = backgroud_hex, markersize = 2, zorder = 0, alpha = 0.3)
        plt.plot(Points_in_Pred[:,0], Points_in_Pred[:,1], 'o', color = pred_hex, markersize = 2, zorder = 0)
        
        axes = plt.gca()
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
        axes.set_xlabel(r'$x$', fontsize = 14)
        axes.set_ylabel(r'$y$', fontsize = 14)
        
        # SAVE FIGURES
        plt.savefig(Link_Figures + method[idx_method] + '_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.png',
                            bbox_inches = 'tight', pad_inches = 0.1,dpi = 600)
        plt.close()
