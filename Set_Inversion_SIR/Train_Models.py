#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:23:46 2019
Apply OASIS to solve SIR problem using SVM, KNN, MLP, RF
Estimate alpha and R0
@author: duynguyen
"""

import numpy as np
import random
import time
import Functions.Solve_Equation_SVM_RBF as SESVM_RBF
import pickle
import Functions.Random_Lib as RL
import Functions.solve_SIR as solsir
import os

import scipy.optimize as so
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



###### PARAMETERS (BECAREFUL WITH YOUR CHANGES) ######
K_samp = 400 # Random K_samp points initially
Active_Points = 400 # Quantity of points would be generated in active learning
tol_threshold = 0.55 # threshold to solve boundary function

method_numb_idx = [0, 1, 2, 3] # remove a number corresponding to the method that do not want to run, Exp: remove 2, to not run MLP --> see method variable below

# --- Set parameters for ODE function
# IF CHANGE THIS ONE --> RUN FIND_GT AGAIN TO HAVE TRUE LABELS
# Read Data.csv recorded
data = pd.read_csv('Data.csv')

X0 = np.asarray(data.iloc[0, 1 : 3]) # Initial population of [Susceptible, Infected, Recovered]
Time = 4 # Max time to run the model
N = 9 # Time step from 0 to Time

# ----- Set up parameters for classifiers
# Optimal Run --> Algorithm will search which values in the list will be the optimal paras
Gamma_SVM_Optimal_Run = np.array([0, 0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 6, 7, 8, 9, 10, 15, 20, 50, 100, 200]) # gamma in SVM
K_KNN_Optimal_Run = np.array([0, 5, 7, 10, 12, 15, 18, 20]) # K in KNN
HLS_MLP_Optimal_Run = np.array([2, 5, 10, 15, 20]) # Nodes in a hidden layer (only 2 hidden layers in MLP)
Trees_RF_Optimal_Run = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) # Tree in Random Forest
###### END PARAMETERS ######





# ----- SET UP -----
random.seed(15)
def MyFunc(x, clf):
    x = np.array(x)
    x = x.reshape(1, -1)
    val = ((clf.predict_proba(x))[0, 0] - 1/2)**2
    return(val)


    


###### DO NOT CHANGE THIS ######
K_bound = 0 # Random more boundary points
K_samp_each = int(K_samp/2) # number of point randomed for each class
K_bound_each = int(K_bound/2)
N_Thres = K_samp + K_bound + Active_Points
# Directory to Points groundtruth folder
CurDir = os.getcwd()
Link_GT = CurDir + '/Points_GT/' # Directory Path where saved Ground Truth data Points
Link_Models = CurDir + '/Models/' # Directory Path to save Model
Link_Points = CurDir + '/Points/' # Directory Path to save Points (to plot)
Link_CSV = CurDir + '/CSV/' # Directory Path to save CSV files
if not os.path.exists(Link_Models):
    os.makedirs(Link_Models)
if not os.path.exists(Link_Points):
    os.makedirs(Link_Points)
if not os.path.exists(Link_CSV):
    os.makedirs(Link_CSV)
    
Random_Type = 'Normal' # initial sampling
Random_Type_Generate = 'Normal' # sampling in solving boundary function

Shape = '2D'
Type_Random = 'True_Random'
method = ['SVM', 'KNN', 'MLP', 'RF'] # order of method has to be the same with classifiers
###### END DO NOT CHANGE THIS ######




    
###### START RUNNING ######
# ---------- RANDOM ----------
print('Sampling initial points ... ...')
X_samp, Y_samp = RL.True_Random_SIR(K_samp, Link_GT)

print('Training ... ...')
# ---------- Load data of points ----------
f = open(Link_GT + 'Points_GT.pckl', 'rb')
C_Mesh = pickle.load(f)
f.close()
print('Total points: ' + str(C_Mesh.shape[0]))

# load groundtruth
f = open(Link_GT + 'GT.pckl', 'rb')
CT = pickle.load(f)
f.close()


# ---------- TRAIN CLASSIFIERS ----------
# if class -1 --> random inside, class 1 --> random outside
Train_Acc_Vec = np.zeros(len(method))
Predict_Acc_Vec = np.zeros(len(method))
Train_Time_Vec = np.zeros(len(method))
Predict_Time_Vec = np.zeros(len(method))

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

    #Active_times = np.zeros(Active_Points)
    X_train = X_samp
    Y_train = Y_samp 
    N_Points = K_samp + K_bound
    G_Points = np.zeros((N_Thres-N_Points, int(Shape[0])))
    G_Points_Fail = np.zeros((0,int(Shape[0])))
    
    tic = time.clock()
    for g in range(G_Points.shape[0] + 1):
        
        
        if (method[idx_method] == 'SVM'): #SVM
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
            values_optimal[1] =K_KNN_optimal
            accuracy_optimal[1] = Accuracy_KNN.max()  
            
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
            
            HLS_MLP_optimal = HLS_MLP[pos_min_HLS] # smallest optimal K_KNN
            clf = MLPClassifier(solver='adam', activation = 'relu', hidden_layer_sizes=(HLS_MLP_optimal, HLS_MLP_optimal), random_state = 0, max_iter = 2000)
            clf.fit(X_train, Y_train)
            
            start_idx = g*len_paras + len(Gamma_SVM) + len(K_KNN)
            values_paras[start_idx : (start_idx + len(HLS_MLP))] = HLS_MLP
            accuracy_paras[start_idx : (start_idx + len(HLS_MLP))] = Accuracy_MLP
            values_optimal[2] = HLS_MLP_optimal
            accuracy_optimal[2] = Accuracy_MLP.max()
            
        if(method[idx_method] == 'RF'):
            for i in range(Trees_RF.shape[0]):
                clf = RandomForestClassifier(n_estimators = Trees_RF[i], random_state = 0, n_jobs = -1)
                clf.fit(X_train, Y_train)
                Accuracy_RF[i] = clf.score(X_train,Y_train)
            
            # find min value of gamma having highest accuracy
            pos_max_acc = np.where(Accuracy_RF==Accuracy_RF.max())
            Trees_max = Trees_RF[pos_max_acc] 
            pos_min_Trees = pos_max_acc[0][np.argmin(Trees_max)]
            
            Trees_RF_optimal = Trees_RF[pos_min_Trees] # smallest optimal K_KNN
            clf = RandomForestClassifier(n_estimators = Trees_RF_optimal, random_state = 0, n_jobs = -1)
            clf.fit(X_train, Y_train)
            
            start_idx = g*len_paras + len(Gamma_SVM) + len(K_KNN) + len(HLS_MLP)
            values_paras[start_idx : (start_idx + len(Trees_RF))] = Trees_RF
            accuracy_paras[start_idx : (start_idx + len(Trees_RF))] = Accuracy_RF
            values_optimal[3] = Trees_RF_optimal
            accuracy_optimal[3] = Accuracy_RF.max()
            
        if (g < G_Points.shape[0]):
                
            if (np.mod(g + 1, 50) == 0 or g == 0):
                print('Generating Point: ' + str(g))
                
            break_flag = 1 # flag for solving boundary function (1 -- solve fail --> 0 solve successful)

            # Find active points
            while break_flag:
                # ---------- RANDOM POINT ----------                
                initial_guess = np.asarray([np.random.uniform(0.6, 1.3), np.random.uniform(np.log(1.5),np.log(2))])                    
                                   
                if (method[idx_method] == 'SVM'):
                    random_point = initial_guess + np.random.uniform(0, 0.005,int(Shape[0]))
                    Solution = SESVM_RBF.Solve_Equation_SVM(XK, YK, BetaK, GammaK, bK, initial_guess, random_point, 1, 5000)
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
                        
            # find true lable
            tl = solsir.Decision_SIR(X0, Time, N, np.exp(g_points[0]), np.exp(g_points[1])*np.exp(g_points[0])/261, data)
            
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
    
    filename = 'Model_' + method[idx_method] + '_SIR_alpha_R0_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp + K_bound) + '_' + str(Active_Points) + '.sav'
    pickle.dump(clf, open(Link_Models + filename, 'wb'))
    
    grid_predict = C_Mesh
    print('Total points for prediction: ' + str(grid_predict.shape[0]))
    
    tic = time.clock()
    CP = clf.predict(grid_predict) # psi label
    print('Predict: Done!!!')
    toc = time.clock()
    predicting_time = toc - tic
    print('Total New Predicting Time: ' + str(predicting_time))
    
    Acc = np.where((CP - CT) == 0)
    Acc = Acc[0]
    Acc = Acc.shape[0] / CP.shape[0]
    print('Overall Accuracy: ' + str(Acc))
    
    Predict_Acc_Vec[idx_method] = Acc
    Predict_Time_Vec[idx_method] = predicting_time
    
    figure_dict = {'grid': grid_predict, 'CP': CP, 'CT': CT, 'GP': G_Points, 'X_samp': X_samp, 'Y_samp': Y_samp}
    filename_fig = 'Points_Figure_' + method[idx_method] + '_SIR_alpha_R0_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp + K_bound) + '_' + str(Active_Points) + '.pkl'
    pickle.dump(figure_dict, open(Link_Points + filename_fig, 'wb'))
    
    Result_pd = pd.DataFrame(method, columns=['Method'])   
    Result_pd = Result_pd.assign(Training_Time = pd.Series(Train_Time_Vec).values, 
                                 Predict_Time = pd.Series(Predict_Time_Vec).values,
                                 Train_Acc = pd.Series(Train_Acc_Vec).values,
                                 Predict_Acc = pd.Series(Predict_Acc_Vec).values)

    filename_csv = 'Result_Accuracy_Optimal_SIR_alpha_R0_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp + K_bound) + '_' + str(Active_Points) + '.csv'
    Result_pd.to_csv(Link_CSV + filename_csv, sep='\t', encoding='utf-8')
    
optimal_pd = pd.DataFrame(loop_paras, columns=['Loop']) # loop based on number of activated points
optimal_pd = optimal_pd.assign(Hyper_Paras = pd.Series(Hyper_String).values, 
                                 Values_Paras = pd.Series(values_paras).values,
                                 Accuracy_Paras = pd.Series(accuracy_paras).values)

list_optimal = ['Gamma_SVM_optimal', 'K_KNN_optimal', 'HLS_MLP_optimal', 'Trees_RF_optimal']
optimal_paras = pd.DataFrame(list_optimal, columns=['Paras_Optimal'])
optimal_paras = optimal_paras.assign(Values_Optimal = pd.Series(values_optimal),
                                     Accuracy_Optimal = pd.Series(accuracy_optimal)) 

filename_csv = 'Result_Optimal_Full_SIR_alpha_R0_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.csv'
optimal_pd.to_csv(Link_CSV + filename_csv, sep='\t', encoding='utf-8')
filename_csv = 'Result_Optimal_SIR_alpha_R0_' + Random_Type + '_' + Random_Type_Generate + '_' + str(K_samp) + '_' + str(Active_Points) + '.csv'
optimal_paras.to_csv(Link_CSV + filename_csv, sep='\t', encoding='utf-8')