# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 00:57:56 2018
Sampling initial points for training for solving ODE
Return Sample Coord and their labels
@author: ManhDuy
"""

import numpy as np
import random
import pickle

def Random_Lotka(Number, path = 'Points_GT/'):    
    # load file
    f = open(path + 'GT.pckl', 'rb')
    GT = pickle.load(f)
    f.close()
    f = open(path + 'Points_GT.pckl', 'rb')
    Point_GT = pickle.load(f)
    f.close()
        
    index = np.linspace(0, GT.shape[0] - 1, GT.shape[0])
    index = index.astype(int)
    rand_index = np.array(random.sample(list(index),Number))
    rand_Point = Point_GT[rand_index]
    label_Point = GT[rand_index]
    g = label_Point.argsort()
    rand_Point = rand_Point[g]
    label_Point = label_Point[g]
    g = np.where(label_Point == 1)[0]
    n_o = g.shape[0]
    n_i = Number - n_o
    print('Total Out: ' + str(n_o) + ' - Total In: ' + str(n_i))
    return(rand_Point, label_Point)