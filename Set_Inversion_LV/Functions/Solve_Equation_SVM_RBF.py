# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:26:36 2017
Solve equation of Kernel SVM RBF
@author: ManhDuy
"""

import numpy as np
import scipy as scp
import math

'''
Betak = np.array([1, 1])
Yk = np.array([1, -1])
Xk = np.array([[0, 0], [1, 1]])
N_point = Xk.shape[0] 
Gamma = 1
B = -0.25
initial_guess = np.array([3, -5]) # real x solution start from initial guess
random_point = np.array([5, 5]) # random point to find min distance
'''

# Kernel function
def Kernel_func(x, y, gamma =1):
        return math.exp(-(np.linalg.norm(x-y)**2)*gamma)
    
def Solve_Equation_SVM(Xk, Yk, Betak, Gamma, B, initial_guess, random_point, choice = 2, iters = 5000):
    '''
        Xk, Yk: support vectors and true labels
        BetaK: Lambda in SVM equation
        Gamma: gamma parameter in kernel RBF function
        B: free parameter of A*x + B in SVM equation
        initial_guess, random point: 2 points contributing to finding the root of SVM equation
        choice: 2 options of solve equation
        iters: number of iterations to solve equation
    '''
    N_point = Yk.shape[0]
            
    # min of distance function from a random point A = [0, 0,...] to point x needed to find
    def objective(x,a=np.zeros(Xk.shape[1])):
        return np.linalg.norm(x-a)**2
    
    # derivative of the above function
    def objective_deriv(x,a=np.zeros(Xk.shape[1])):
        obj_der = np.array([2*x[0]-2*a[0]])
        for i in range(1, a.shape[0]):
            obj_der = np.append(obj_der,[2*x[i]-2*a[i]])
        return obj_der
        
    # constrain function which is \tsi(x) = 0
    def constraint(x, N = N_point, b = B, beta = Betak, Y = Yk, X = Xk, gamma = Gamma):
        cons_func = b
        for i in range(N):
            cons_func = cons_func + beta[i]*Y[i]*Kernel_func(X[i,:],x,gamma)
        return cons_func
    
    def mconstraint(x, N = N_point, b = B, beta = Betak, Y = Yk, X = Xk, gamma = Gamma):
        cons_func = b
        for i in range(N):
            cons_func = cons_func + beta[i]*Y[i]*Kernel_func(X[i,:],x,gamma)
        return (0 - cons_func)
    
    # derivative of the above function    
    def constraint_deriv(x, N = N_point, b = B, beta = Betak, Y = Yk, X = Xk, gamma = Gamma):
        cons = 0
        for i in range(N):
            cons += beta[i]*Y[i]*Kernel_func(X[i,:],x,gamma)*2*(X[i,0]-x[0])*gamma
        cons_der = np.array([cons])
        for j in range(1, X.shape[1]):
            cons = 0
            for i in range(N):
                cons += beta[i]*Y[i]*Kernel_func(X[i,:],x,gamma)*2*(X[i,j]-x[j])*gamma        
            cons_der = np.append(cons_der,[cons])
        return cons_der
    
    def mconstraint_deriv(x, N = N_point, b = B, beta = Betak, Y = Yk, X = Xk, gamma = Gamma):
        cons = 0
        for i in range(N):
            cons += beta[i]*Y[i]*Kernel_func(X[i,:],x,gamma)*2*(X[i,0]-x[0])*gamma
        cons_der = np.array([cons])
        for j in range(1, X.shape[1]):
            cons = 0
            for i in range(N):
                cons += beta[i]*Y[i]*Kernel_func(X[i,:],x,gamma)*2*(X[i,j]-x[j])*gamma       
            cons_der = np.append(cons_der,[cons])
        return (0 - cons_der)
    
    if (choice != 1):
        con1 = {'type': 'ineq', 'fun': constraint, 'jac': constraint_deriv}
        con2 = {'type': 'ineq', 'fun': mconstraint, 'jac': mconstraint_deriv}
        cons = ([con1,con2])
        
        solution = scp.optimize.minimize(objective, initial_guess, args=(random_point,), 
                                         method='COBYLA', 
                                         constraints=cons, options={'maxiter': iters, 'disp': False})
        
        #print('Success: ' + str(solution.success) + ' ~ Error: ' + str(solution.maxcv))
        return solution
    else:
        con1 = {'type': 'eq', 'fun': constraint, 'jac': constraint_deriv}
        cons = ([con1])
        solution = scp.optimize.minimize(objective, initial_guess, args=(random_point,),
                                         method='SLSQP', jac=objective_deriv, 
                                         constraints=cons, options={ 'maxiter': iters, 'disp': False})
        #print('Success: ' + str(solution.success) + ' ~ Error: ' + str(constraint(solution.x)))
        return solution
    
    
def SVM_Func(x, b, beta, Y, X, gamma):
        N = Y.shape[0]
        cons_func = b
        for i in range(N):
            cons_func = cons_func + beta[i]*Y[i]*Kernel_func(X[i,:],x,gamma)
        return cons_func