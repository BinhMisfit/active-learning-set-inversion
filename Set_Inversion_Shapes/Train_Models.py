#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:56:51 2019
- run set inversion all shapes for all method but solve SVM as our own function, 
- other method is solve by scipy.optimize.newton function of prob
- (HOWEVER) not optimize parameter --> current is default parameter
- random function is now from Random_Sequence_v2
***** Update 2019/01/25*****
Add optimal run selection. In this run, we will find optimal hyperparameters in each models.
There are some initial values in the set. The code will run all of them and find out the one have the best performance
If there are some parameters obtaining same accuracy --> smallest parameters
***** Update 2019/02/13*****
- Add sobol_seq sampling
- Gamma SVM now have 2 addition options for Gamma (1/n_features and 1/(n_features * X.std())) --> check sklearn
***** Update 2019/03/12*****
- APT: Active Point Time --> Average time to find an active point (by solving boundary)
- Add APT calculation (only for 5D --> 8D)
- Not perfectly work for 4D below
***** Update 2019/05/31*****
- Clean the code
- Save result at specific Folders
- APT might work well now

@author: duynguyen
"""

import os
import numpy as np
import scipy.optimize as so
import math
import random
import time
import Functions.Plot_Lib as pll
import Functions.Function_Fx as FF
import pandas as pd
import Functions.Random_Sequence_v2 as RS
import pickle
import Functions.Solve_Equation_SVM_RBF as SESVM_RBF
import warnings
warnings.filterwarnings('ignore')

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

'''
APPENDIX PARAMETERS
* Shape
    + 2D_Circle
    + 2D_Ring
    + 2D_Doughnut
    + 3D_Sphere (Up to 8D_Sphere)

* Sampling
    + Normal
    + LHS
    + Sobol
'''


###### PARAMETERS (BECAREFUL WITH YOUR CHANGES) ######
Support_Shape = ['2D_Circle', '2D_Ring', '2D_Doughnut', '3D_Sphere', '4D_Sphere',
                 '5D_Sphere', '6D_Sphere', '7D_Sphere', '8D_Sphere'] # DO NOT CHANGE THIS
# List all shapes that you want to run (only support specific shapes --> See Appendix)
Shape_vec = ['2D_Circle', '2D_Ring']
#Shape_vec = Support_Shape # Run all Shapes

Random_Type = 'LHS' # Change this for sampling initial points (can be Normal, LHS, Sobol)

method_numb_idx = [0, 1, 2, 3] # remove a number corresponding to the method that do not want to run, Exp: remove 2, to not run MLP --> see method variable below 

K_samp = 100 # Number of initial points
Active_Points = 400 # Number of activating points
tol_threshold = 0.55 # threshold of tolerance for solving boundary function

Optimal_Run = True # set True if run optimal, False to use default parameters

# Optimal Run --> Algorithm will search which values in the list will be the optimal paras
Gamma_SVM_Optimal_Run = np.array([0, 0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 200]) # gamma in SVM
K_KNN_Optimal_Run = np.array([0, 5, 7, 10, 12, 15, 18, 20]) # K in KNN
HLS_MLP_Optimal_Run = np.array([2, 5, 10, 15, 20]) # Nodes in a hidden layer (only 2 hidden layers in MLP)
Trees_RF_Optimal_Run = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # Tree in Random Forest

# Default Run --> Algorithm will run with the given paras
Gamma_SVM_Default = 1/2
K_KNN_Default = 10
HLS_MLP_Default = 10
Trees_RF_Default = 20
###### END PARAMETERS ######





###### SET UP FUNCTION ######
random.seed(5)
def MyFunc(x, clf):
    x = np.array(x)
    x = x.reshape(1, -1)
    val = ((clf.predict_proba(x))[0, 0] - 1/2)**2
    return(val)
###### END SET UP FUNCTION ######



    
###### DO NOT CHANGE THIS ######
Random_Type_Generate = 'Normal' 
method = ['SVM', 'KNN', 'MLP', 'RF'] # order of method has to be the same with classifiers

R = 2 # For Circle
R1 = 1 # For Ring
R2 = 2 # For Ring
a_dn = 1 # Doughnut
b_dn = 2 # Doughnut
R_3D = 0.5
R_4D = 0.25
R_5D = 0.25
R_High = 0.25
    
res_2d = 601 # resolution for 2D
res_3d = 151 # resolution for 3D
res_4d = 41 # resolution for 4D Sphere
res_5d = 31 # resolution for 5D Sphere
res_6d = 18 # resolution for 6D Sphere
res_7d = 12 # resolution for 7D Sphere
res_8d = 9 # resolution for 8D Sphere

# Create Folder to save result
CurDir = os.getcwd()
Link_Models = CurDir + '/Models/' # Directory Path to save Model
Link_Points = CurDir + '/Points/' # Directory Path to save Points (to plot)
Link_CSV = CurDir + '/CSV/' # Directory Path to save CSV files
if not os.path.exists(Link_Models):
    os.makedirs(Link_Models)
if not os.path.exists(Link_Points):
    os.makedirs(Link_Points)
if not os.path.exists(Link_CSV):
    os.makedirs(Link_CSV)
###### END DO NOT CHANGE THIS ######





###### START RUNNING ######
# Check if all shapes are supported --> if not, end programme
check_shape = True
for shape in Shape_vec:
    try:
        valid_shape_boolean = Support_Shape.index(shape)
    except ValueError:
        print('[WARNING] ' + shape + ' is not in the supported shape list --> END PROGRAM')
        check_shape = False

if check_shape:
    print('[CHECKING] All shapes are supported! --> GOOD!')
    
    for idx_shape in range(len(Shape_vec)):
        Shape = Shape_vec[idx_shape]
        print('***** START *****\nSHAPE: ' + Shape + '\nRANDOM_TYPE: ' + Random_Type + ' (' + Random_Type_Generate + ')\nOPTIMAL_RUN: ' + str(Optimal_Run) + 
              '\nTOL_THRESHOLD: ' + str(tol_threshold) + '\nK_SAMP: ' + str(K_samp) + '\nACTIVATE_POINTS: ' + str(Active_Points) + 
              '\n*****************')
        
        classifiers = [
            svm.SVC(kernel="rbf", gamma = Gamma_SVM_Default, random_state = 0),
            KNeighborsClassifier(n_neighbors = K_KNN_Default),
            MLPClassifier(solver='adam', activation = 'relu', hidden_layer_sizes=(HLS_MLP_Default, HLS_MLP_Default), random_state = 0, max_iter = 2000),
            RandomForestClassifier(n_estimators = Trees_RF_Default, random_state = 0, n_jobs = -1)#,
        ]
        
        if (Shape[:2] == '2D'): # resolution 601   
            xx, yy = np.mgrid[-3:3.01:.01, -3:3.01:.01]
            grid = np.c_[xx.ravel(), yy.ravel()]
        if (Shape[:2] == '3D'): # resolution 151
            xx, yy, zz = np.mgrid[-1.5:1.51:.02, -1.5:1.51:.02, -1.5:1.51:.02]
            grid_3d = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        if (Shape[:2] == '4D'): # resolution 41
            xx, yy, zz, tt = np.mgrid[-1:1.01:.05, -1:1.01:.05, -1:1.01:.05, -1:1.01:.05]
            grid_4d = np.c_[xx.ravel(), yy.ravel(), zz.ravel(), tt.ravel()]
        if (Shape[:2] == '5D'): # resolution res_5d
            X_Mesh = np.linspace(-0.6, 0.6, res_5d)
            Y_Mesh = np.linspace(-0.6, 0.6, res_5d)
            Z_Mesh = np.linspace(-0.6, 0.6, res_5d)
            T_Mesh = np.linspace(-0.6, 0.6, res_5d)
            P_Mesh = np.linspace(-0.6, 0.6, res_5d)
            grid_5d = pll.Mesh_5D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh)
        if (Shape[:2] == '6D'): # resolution res_6d
            X_Mesh = np.linspace(-0.6, 0.6, res_6d)
            Y_Mesh = np.linspace(-0.6, 0.6, res_6d)
            Z_Mesh = np.linspace(-0.6, 0.6, res_6d)
            T_Mesh = np.linspace(-0.6, 0.6, res_6d)
            P_Mesh = np.linspace(-0.6, 0.6, res_6d)
            K_Mesh = np.linspace(-0.6, 0.6, res_6d)
            grid_6d = pll.Mesh_6D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh)
        if (Shape[:2] == '7D'): # resolution res_7d
            X_Mesh = np.linspace(-0.6, 0.6, res_7d)
            Y_Mesh = np.linspace(-0.6, 0.6, res_7d)
            Z_Mesh = np.linspace(-0.6, 0.6, res_7d)
            T_Mesh = np.linspace(-0.6, 0.6, res_7d)
            P_Mesh = np.linspace(-0.6, 0.6, res_7d)
            K_Mesh = np.linspace(-0.6, 0.6, res_7d)
            L_Mesh = np.linspace(-0.6, 0.6, res_7d)
            grid_7d = pll.Mesh_7D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh, L_Mesh)
        if (Shape[:2] == '8D'): # resolution res_8d
            X_Mesh = np.linspace(-0.6, 0.6, res_8d)
            Y_Mesh = np.linspace(-0.6, 0.6, res_8d)
            Z_Mesh = np.linspace(-0.6, 0.6, res_8d)
            T_Mesh = np.linspace(-0.6, 0.6, res_8d)
            P_Mesh = np.linspace(-0.6, 0.6, res_8d)
            K_Mesh = np.linspace(-0.6, 0.6, res_8d)
            L_Mesh = np.linspace(-0.6, 0.6, res_8d)
            H_Mesh = np.linspace(-0.6, 0.6, res_8d)
            grid_8d = pll.Mesh_8D_Coordinates(X_Mesh, Y_Mesh, Z_Mesh, T_Mesh, P_Mesh, K_Mesh, L_Mesh, H_Mesh)
        
        N_Points = K_samp
        N_Thres = K_samp + Active_Points
        K_samp_each = int(K_samp/2)
        Y_samp = np.append(-1*np.ones(K_samp_each),np.ones(K_samp_each))
        
        # ---------- SAMPLING TRAINING DATA ----------
        if (Shape == '2D_Circle'): # Normal Random
            X_samp = RS.Random_2D_Circle_Full(R = np.sqrt(R), step = 1500, num_points = K_samp, type_random = Random_Type)
            
        if (Shape == '2D_Ring'):
            X_samp = RS.Random_2D_Ring_Full(R1 = np.sqrt(R1), R2 = np.sqrt(R2), step = 1500, num_points = K_samp, type_random = Random_Type)
            
        if (Shape == '2D_Doughnut'):
            X_samp = RS.Random_2D_Doughnut_Full(R1 = np.sqrt(a_dn), R2 = np.sqrt(b_dn), step = 1500, num_points = K_samp, type_random = Random_Type)
            
        if (Shape == '3D_Sphere'):
            X_samp = RS.Random_3D_Sphere_Full(R = np.sqrt(R_3D), step = 1500, num_points = K_samp, type_random = Random_Type)
            
        if (Shape == '4D_Sphere'):
            X_samp = RS.Random_4D_Sphere_Full(R = np.sqrt(R_4D), step = 1500, num_points = K_samp, type_random = Random_Type)
        
        if (Shape == '5D_Sphere'):
            X_samp = RS.Random_5D_Sphere_Full(R = np.sqrt(R_5D), step = 1500, num_points = K_samp, type_random = Random_Type)
        
        if (Shape == '6D_Sphere'):
            X_samp = RS.Random_6D_Sphere_Full(R = np.sqrt(R_High), step = 1500, num_points = K_samp, type_random = Random_Type)
        
        if (Shape == '7D_Sphere'):
            X_samp = RS.Random_7D_Sphere_Full(R = np.sqrt(R_High), step = 1500, num_points = K_samp, type_random = Random_Type)
        
        if (Shape == '8D_Sphere'):
            X_samp = RS.Random_8D_Sphere_Full(R = np.sqrt(R_High), step = 1500, num_points = K_samp, type_random = Random_Type)
            
        # ---------- TRAINING ---------- 
        Train_Acc_Vec = np.zeros(len(method))
        Predict_Acc_Vec = np.zeros(len(method))
        Train_Time_Vec = np.zeros(len(method))
        Predict_Time_Vec = np.zeros(len(method))
        APT_Vec = np.zeros(len(method))
        
        if (Optimal_Run):
            Gamma_SVM = Gamma_SVM_Optimal_Run
            Accuracy_SVM = np.zeros(Gamma_SVM.shape[0])
            
            K_KNN = K_KNN_Optimal_Run
            Accuracy_KNN = np.zeros(K_KNN.shape[0])
            
            HLS_MLP = HLS_MLP_Optimal_Run
            Accuracy_MLP = np.zeros(HLS_MLP.shape[0])
            
            Trees_RF = Trees_RF_Optimal_Run
            Accuracy_RF = np.zeros(Trees_RF.shape[0])
            
            Hyper_String = ['Gamma_SVM' for i in range(len(Gamma_SVM))]
            Hyper_String = Hyper_String + ['K_KNN' for i in range(len(K_KNN))]
            Hyper_String = Hyper_String + ['HLS_MLP' for i in range(len(HLS_MLP))]
            Hyper_String = Hyper_String + ['Trees_RF' for i in range(len(Trees_RF))]
            len_paras = len(Hyper_String)
            Hyper_String = Hyper_String * (Active_Points + 1)
            values_paras = np.zeros(len(Hyper_String))
            accuracy_paras = np.zeros(len(Hyper_String))
            loop_paras = np.linspace(0, Active_Points, num = Active_Points + 1)
            loop_paras = np.repeat(loop_paras, len_paras)
                
            values_optimal = np.zeros(len(method))
            accuracy_optimal = np.zeros(len(method))
        
        for idx_method in method_numb_idx:
            print('========== Method: ' + method[idx_method] + '[' + str(idx_method + 1) + ' / ' + str(len(method)) + '] ==========')
        
            X_train = X_samp
            Y_train = Y_samp 
            N_Points = K_samp 
            G_Points = np.zeros((N_Thres-N_Points, int(Shape[0])))
            G_Points_Fail = np.zeros((0,int(Shape[0])))
            
            tic = time.clock()
            for g in range(G_Points.shape[0] + 1):
                
                if (Optimal_Run): ##### AUTOMATICALLY CHOOSE OPTIMAL PARAS #####
                    ##### SVM #####
                    if (method[idx_method] == 'SVM'):
                        Gamma_SVM[0] = 1/(X_train.shape[1])
                        Gamma_SVM[1] = 1/(X_train.shape[1] * X_train.std())
                        X_dict = {}
                        Y_dict = {}
                        BetaK_dict = {}
                        bK_list = np.zeros(Gamma_SVM.shape[0])
                        
                        # train SVM to find suitable value of gamma
                        for i in range(Gamma_SVM.shape[0]):
                            clf = svm.SVC(kernel='rbf', gamma=Gamma_SVM[i])
                            clf.fit(X_train, Y_train)
                            Accuracy_SVM[i] = clf.score(X_train,Y_train)
                            
                            bK_list[i] = clf.intercept_[0]
                            X_dict[i] = clf.support_vectors_
                            Y_dict[i] = Y_train[clf.support_]
                            BetaK_dict[i] = clf.dual_coef_ / Y_dict[i]
                        
                        # find min value of gamma having highest accuracy
                        pos_max_acc = np.where(Accuracy_SVM==Accuracy_SVM.max()) # find position of highest accuracy
                        Gamma_max = Gamma_SVM[pos_max_acc] # values of gamma having highest accuracy
                        pos_min_gamma = pos_max_acc[0][np.argmin(Gamma_max)] # index of min gamma
                        
                        GammaK = Gamma_SVM[pos_min_gamma]
                        bK = bK_list[pos_min_gamma]
                        BetaK = BetaK_dict[pos_min_gamma][0]
                        XK = X_dict[pos_min_gamma]
                        YK = Y_dict[pos_min_gamma]
                        
                        Gamma_SVM_optimal = GammaK
                        clf = svm.SVC(kernel='rbf', gamma=Gamma_SVM_optimal)
                        clf.fit(X_train, Y_train)
                        
                        values_paras[(g*len_paras + 0) : (g*len_paras + len(Gamma_SVM))] = Gamma_SVM
                        accuracy_paras[(g*len_paras + 0) : (g*len_paras + len(Gamma_SVM))] = Accuracy_SVM
                        values_optimal[0] = Gamma_SVM_optimal
                        accuracy_optimal[0] = Accuracy_SVM.max()
                    
                    ##### KNN #####
                    if (method[idx_method] == 'KNN'):
                        K_KNN[0] = int(np.sqrt(N_Points))
                        # train KNN to find optimal value of K_KNN
                        for i in range(K_KNN.shape[0]):
                            clf = KNeighborsClassifier(n_neighbors = K_KNN[i])
                            clf.fit(X_train, Y_train)
                            Accuracy_KNN[i] = clf.score(X_train,Y_train)
                        
                        # find min value of gamma having highest accuracy
                        pos_max_acc = np.where(Accuracy_KNN==Accuracy_KNN.max()) # find position of highest accuracy
                        K_max = K_KNN[pos_max_acc] # values of K_KNN having highest accuracy
                        pos_min_K = pos_max_acc[0][np.argmin(K_max)] # index of min K_KNN
                        
                        K_KNN_optimal = K_KNN[pos_min_K] # smallest optimal K_KNN
                        clf = KNeighborsClassifier(n_neighbors = K_KNN_optimal) # run for the optimal parameters
                        clf.fit(X_train, Y_train)
                        
                        start_idx = g*len_paras + len(Gamma_SVM)
                        values_paras[start_idx : (start_idx + len(K_KNN))] = K_KNN
                        accuracy_paras[start_idx : (start_idx + len(K_KNN))] = Accuracy_KNN
                        values_optimal[1] = K_KNN_optimal
                        accuracy_optimal[1] = Accuracy_KNN.max()
                        
                    ##### MLP #####
                    if (method[idx_method] == 'MLP'):
                        for i in range(HLS_MLP.shape[0]):
                            hls = HLS_MLP[i]
                            clf = MLPClassifier(solver='adam', activation = 'relu', hidden_layer_sizes=(hls, hls), random_state = 0, max_iter = 2000)
                            clf.fit(X_train, Y_train)
                            Accuracy_MLP[i] = clf.score(X_train,Y_train)
                        
                        # find min value of gamma having highest accuracy
                        pos_max_acc = np.where(Accuracy_MLP==Accuracy_MLP.max())
                        HLS_max = HLS_MLP[pos_max_acc] 
                        pos_min_HLS = pos_max_acc[0][np.argmin(HLS_max)]
                        
                        HLS_MLP_optimal = HLS_MLP[pos_min_HLS] # smallest optimal HLS_MLP
                        clf = MLPClassifier(solver='adam', activation = 'relu', hidden_layer_sizes=(HLS_MLP_optimal, HLS_MLP_optimal), random_state = 0, max_iter = 2000)
                        clf.fit(X_train, Y_train)
                        
                        start_idx = g*len_paras + len(Gamma_SVM) + len(K_KNN)
                        values_paras[start_idx : (start_idx + len(HLS_MLP))] = HLS_MLP
                        accuracy_paras[start_idx : (start_idx + len(HLS_MLP))] = Accuracy_MLP
                        values_optimal[2] = HLS_MLP_optimal
                        accuracy_optimal[2] = Accuracy_MLP.max()
                    
                    ##### RF #####
                    if(method[idx_method] == 'RF'):
                        for i in range(Trees_RF.shape[0]):
                            clf = RandomForestClassifier(n_estimators = Trees_RF[i], random_state = 0, n_jobs = -1)
                            clf.fit(X_train, Y_train)
                            Accuracy_RF[i] = clf.score(X_train,Y_train)
                        
                        # find min value of gamma having highest accuracy
                        pos_max_acc = np.where(Accuracy_RF==Accuracy_RF.max())
                        Trees_max = Trees_RF[pos_max_acc] 
                        pos_min_Trees = pos_max_acc[0][np.argmin(Trees_max)]
                        
                        Trees_RF_optimal = Trees_RF[pos_min_Trees] # smallest optimal Trees_RF
                        clf = RandomForestClassifier(n_estimators = Trees_RF_optimal, random_state = 0, n_jobs = -1)
                        clf.fit(X_train, Y_train)
                        
                        start_idx = g*len_paras + len(Gamma_SVM) + len(K_KNN) + len(HLS_MLP)
                        values_paras[start_idx : (start_idx + len(Trees_RF))] = Trees_RF
                        accuracy_paras[start_idx : (start_idx + len(Trees_RF))] = Accuracy_RF
                        values_optimal[3] = Trees_RF_optimal
                        accuracy_optimal[3] = Accuracy_RF.max()
                    
                else: ##### USE DEFAULT PARAS --> NO OPTIMAL #####
                    clf = classifiers[idx_method]
                    clf.fit(X_train, Y_train)
                    if (method[idx_method] == 'SVM'):
                        bK = clf.intercept_[0]
                        XK = clf.support_vectors_
                        YK = Y_train[clf.support_]
                        BetaK = clf.dual_coef_ / YK
                        BetaK = BetaK[0]
                        GammaK = clf.get_params(deep = False)['gamma']
                
                if (g < G_Points.shape[0]):
                    
                    if (np.mod(g + 1, 50) == 0 or g == 0):
                        print('Generating Point: ' + str(g))
                        
                    break_flag = 1 # flag for solving boundary function (1 -- solve fail --> 0 solve successful)
        
                    # Find active points
                    while break_flag:
                        # ---------- RANDOM POINT ----------
                        if (Shape == '2D_Circle'):
                            initial_guess = RS.Random_2D_Circle([np.sqrt(0.1*R), np.sqrt(1.9*R)], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]            
                        if (Shape == '2D_Ring'):
                            initial_guess = RS.Random_2D_Circle([np.sqrt(1.2*R1), np.sqrt(0.9*R2)], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]
                        if (Shape == '2D_Doughnut'):
                            initial_guess = RS.Random_2D_Doughnut([np.sqrt(0.1*a_dn), np.sqrt(1.9*b_dn)], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]
                        if (Shape == '3D_Sphere'):
                            initial_guess = RS.Random_3D_Sphere([np.sqrt(0.1*R_3D), np.sqrt(1.9*R_3D)], [0, math.pi], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]
                        if (Shape == '4D_Sphere'):
                            initial_guess = RS.Random_4D_Sphere([np.sqrt(0.1*R_4D), np.sqrt(1.9*R_4D)], [0, math.pi], [0, 2*math.pi], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]
                        if (Shape == '5D_Sphere'):
                            initial_guess = RS.Random_5D_Sphere([np.sqrt(0.1*R_5D), np.sqrt(1.9*R_5D)], [0, math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]
                        if (Shape == '6D_Sphere'):
                            initial_guess = RS.Random_6D_Sphere([np.sqrt(0.1*R_High), np.sqrt(1.9*R_High)], [0, math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]
                        if (Shape == '7D_Sphere'):
                            initial_guess = RS.Random_7D_Sphere([np.sqrt(0.1*R_High), np.sqrt(1.9*R_High)], [0, math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]
                        if (Shape == '8D_Sphere'):
                            initial_guess = RS.Random_8D_Sphere([np.sqrt(0.1*R_High), np.sqrt(1.9*R_High)], [0, math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], [0, 2*math.pi], 1000, 1, type_random = Random_Type_Generate)[0]    
                        
                        if (method[idx_method] == 'SVM'):
                            random_point = initial_guess + np.random.uniform(-0.5,0.5,int(Shape[0]))
                            Solution = SESVM_RBF.Solve_Equation_SVM(XK, YK, BetaK, GammaK, bK, initial_guess, random_point, 1 , 5000)
                            if (Solution.success):
                                #print('Solve Equation Successfully')
                                g_points = Solution.x
                                break_flag = 0
                        
                        else: # other classifiers
                            Solution = so.minimize(MyFunc, initial_guess, args = (clf,), method = 'COBYLA', tol = 1e-10)
                            
                            if (Solution.fun < tol_threshold):    
                                #print('Solve Equation Successfully')
                                g_points = Solution.x
                                break_flag = 0
                    
                    if (Shape == '2D_Circle'):
                        tl = FF.Fx_Circle(g_points, R)
                    if (Shape == '2D_Ring'):
                        tl = FF.Fx_Ring(g_points, R1, R2)
                    if (Shape == '2D_Doughnut'):
                        tl = FF.Fx_Doughnut(g_points, a_dn, b_dn)
                    if (Shape == '3D_Sphere'):
                        tl = FF.Fx_Sphere(g_points, R_3D)
                    if (Shape == '4D_Sphere'):
                        tl = FF.Fx_Sphere(g_points, R_4D)
                    if (Shape == '5D_Sphere'):
                        tl = FF.Fx_Sphere(g_points, R_5D)
                    if (int(Shape[:1]) > 5): # Higher than 5D
                        tl = FF.Fx_Sphere(g_points, R_High)
                    
                    N_Points += 1 
                    
                    G_Points[g] = g_points
                    X_train = np.append(X_train,[G_Points[g]],axis=0)
                    Y_train = np.append(Y_train, np.array([tl])) # true label
            
            toc = time.clock()
            Accuracy = clf.score(X_train,Y_train)
            print('Training Accuracy: ' + str(Accuracy))
            training_time = toc - tic
            print('Total Training Time: ' + str(training_time))
            
            Train_Acc_Vec[idx_method] = Accuracy
            Train_Time_Vec[idx_method] = training_time
            
            filename = 'Model_' + method[idx_method] + '_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.sav'
            pickle.dump(clf, open(Link_Models + filename, 'wb'))
        
            
            # ---------- PLOT FINAL MESHGRID ----------
            
            if (Shape[:2] == '2D'):
                grid_predict = grid
                res = res_2d
                dimen = 2
            if (Shape[:2] == '3D'):
                grid_predict = grid_3d
                res = res_3d
                dimen = 3
            if (Shape[:2] == '4D'):
                grid_predict = grid_4d
                res = res_4d
                dimen = 4
            if (Shape[:2] == '5D'):
                grid_predict = grid_5d
                res = res_5d
                dimen = 5
            if (Shape[:2] == '6D'):
                grid_predict = grid_6d
                res = res_6d
                dimen = 6
            if (Shape[:2] == '7D'):
                grid_predict = grid_7d
                res = res_7d
                dimen = 7
            if (Shape[:2] == '8D'):
                grid_predict = grid_8d
                res = res_8d
                dimen = 8
            
            print('Total points for prediction: ' + str(grid_predict.shape[0]))
            
            tic = time.clock()
            CP = clf.predict(grid_predict) # psi label
            print('Predict: Done!!!')
            toc = time.clock()
            predicting_time = toc - tic
            print('Total Predicting Time: ' + str(predicting_time))
        
            if (Shape == '2D_Circle'):
                CT = np.array([FF.Fx_Circle(grid[i], R) for i in range(grid.shape[0])])
            if (Shape == '2D_Ring'):
                CT = np.array([FF.Fx_Ring(grid[i], R1, R2) for i in range(grid.shape[0])])
            if (Shape == '2D_Doughnut'):
                CT = np.array([FF.Fx_Doughnut(grid[i], a_dn, b_dn) for i in range(grid.shape[0])])
            if (Shape == '3D_Sphere'):
                CT = np.array([FF.Fx_Sphere(grid_3d[i], R_3D) for i in range(grid_3d.shape[0])])
            if (Shape == '4D_Sphere'):
                CT = np.array([FF.Fx_Sphere(grid_4d[i], R_4D) for i in range(grid_4d.shape[0])])
            if (Shape == '5D_Sphere'):
                CT = np.array([FF.Fx_Sphere(grid_5d[i], R_5D) for i in range(grid_5d.shape[0])])    
            if (Shape == '6D_Sphere'):
                CT = np.array([FF.Fx_Sphere(grid_6d[i], R_High) for i in range(grid_6d.shape[0])])    
            if (Shape == '7D_Sphere'):
                CT = np.array([FF.Fx_Sphere(grid_7d[i], R_High) for i in range(grid_7d.shape[0])])    
            if (Shape == '8D_Sphere'):
                CT = np.array([FF.Fx_Sphere(grid_8d[i], R_High) for i in range(grid_8d.shape[0])])   
                
            Acc = np.where((CP - CT) == 0)
            Acc = Acc[0]
            Acc = Acc.shape[0] / CP.shape[0]
            print('Overall Accuracy: ' + str(Acc))
            
            Predict_Acc_Vec[idx_method] = Acc
            Predict_Time_Vec[idx_method] = predicting_time
            APT_Vec[idx_method] = predicting_time * 1000 / (res ** dimen)
            
            if (Shape[:2] == '2D'):
                figure_dict = {'grid': grid_predict, 'CP': CP, 'CT': CT, 'GP': G_Points, 'X_samp': X_samp}
                filename_fig = 'Points_Figure_' + method[idx_method] + '_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.pkl'
                pickle.dump(figure_dict, open(Link_Points + filename_fig, 'wb'))
        
            Result_pd = pd.DataFrame(method, columns=['Method'])   
            Result_pd = Result_pd.assign(Training_Time = pd.Series(Train_Time_Vec).values, 
                                         Predict_Time = pd.Series(Predict_Time_Vec).values,
                                         Train_Acc = pd.Series(Train_Acc_Vec).values,
                                         Predict_Acc = pd.Series(Predict_Acc_Vec).values,
                                         APT = pd.Series(APT_Vec).values)
            
            if (Optimal_Run):
                filename_csv = 'Result_Accuracy_Optimal_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.csv'    
            else:
                filename_csv = 'Result_Accuracy_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.csv'
            Result_pd.to_csv(Link_CSV + filename_csv, sep='\t', encoding='utf-8')
        
        optimal_pd = pd.DataFrame(loop_paras, columns=['Loop']) # loop based on number of activated points
        optimal_pd = optimal_pd.assign(Hyper_Paras = pd.Series(Hyper_String).values, 
                                         Values_Paras = pd.Series(values_paras).values,
                                         Accuracy_Paras = pd.Series(accuracy_paras).values)
        
        list_optimal = ['Gamma_SVM_optimal', 'K_KNN_optimal', 'HLS_MLP_optimal', 'Trees_RF_optimal']
        optimal_paras = pd.DataFrame(list_optimal, columns=['Paras_Optimal'])
        optimal_paras = optimal_paras.assign(Values_Optimal = pd.Series(values_optimal),
                                             Accuracy_Optimal = pd.Series(accuracy_optimal)) 
        
        filename_csv = 'Result_Optimal_Full_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.csv'
        optimal_pd.to_csv(Link_CSV + filename_csv, sep='\t', encoding='utf-8')
        filename_csv = 'Result_Optimal_' + Shape + '_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.csv'
        optimal_paras.to_csv(Link_CSV + filename_csv, sep='\t', encoding='utf-8')
        
        print('***** FINISHED *****\nSHAPE: ' + Shape + '\nRANDOM_TYPE: ' + Random_Type + ' (' + Random_Type_Generate + ')\nOPTIMAL_RUN: ' + str(Optimal_Run) + 
              '\nTOL_THRESHOLD: ' + str(tol_threshold) + '\nK_SAMP: ' + str(K_samp) + '\nACTIVATE_POINTS: ' + str(Active_Points) + 
              '\n********************')