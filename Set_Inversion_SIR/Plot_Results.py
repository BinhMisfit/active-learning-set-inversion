#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 23:42:55 2019
New plot SIR
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
K_samp = 400 # Initial random points
Active_Points = 400 # number of activating points (solve boundary function)
idx_method = [0, 1, 2, 3]
CurDir = os.getcwd()
Link_Models = CurDir + '/Points/' # Folder containing Points_Figures to plot
Link_Figures = CurDir + '/Figures/'
if not os.path.exists(Link_Figures):
    os.makedirs(Link_Figures)
###### END PARAMETERS ######






###### DO NOT CHANGE THIS ######
Random_Type = 'Normal'
Random_Type_Generate = 'Normal'
method = ['SVM', 'KNN', 'MLP', 'RF']
tail_string = '_SIR_alpha_R0_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.pkl'
model_name = ['Points_Figure_' + method[x] + tail_string for x in idx_method]
# Range limit of plot
xmin = 0.6
xmax = 1.3
ymin = 0.4
ymax = 0.7
###### END DO NOT CHANGE THIS ######






###### START RUNNING ######
for idx_model in range(len(model_name)):
    # Load Trained model
    print('[Loading File] ' + model_name[idx_model])
    result_dict = pickle.load(open(Link_Models + model_name[idx_model], 'rb'))    
    
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
    
    plt.plot(Points_out_Pred[:,0], Points_out_Pred[:,1], 'o', color = backgroud_hex, markersize = 2, zorder = 0, alpha = 0.3)
    plt.plot(Points_in_GT[Hull_GT.vertices,0], Points_in_GT[Hull_GT.vertices,1], '--', color = gt_hex, lw=1.6)
    plt.plot(Points_in_Pred[:,0], Points_in_Pred[:,1], 'o', color = pred_hex, markersize = 2, zorder = 0)
    
    axes = plt.gca()
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    axes.set_xlabel(r'$log(\alpha)$', fontsize = 14)
    axes.set_ylabel(r'$log(R_0)$', fontsize = 14)
    
    plt.savefig(Link_Figures + 'SIR_alpha_R0_GT_' + method[idx_method[idx_model]] + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.png',
                        bbox_inches = 'tight', pad_inches = 0.1,dpi = 600)
    plt.close()

        