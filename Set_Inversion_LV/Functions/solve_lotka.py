# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 00:14:46 2018

Library to solve integrate ODEs with scipy, particular Lokta-Volterra model

@author: ManhDuy
"""

# We will have a look at the Lokta-Volterra model, also known as the
# predator-prey equations, which are a pair of first order, non-linear, differential
# equations frequently used to describe the dynamics of biological systems in
# which two species interact, one a predator and one its prey. They were proposed
# independently by Alfred J. Lotka in 1925 and Vito Volterra in 1926:
# du/dt =  a*u -   b*u*v
# dv/dt = -c*v +   d*u*v 
# 
# with the following notations:
# 
# *  u: number of preys (for example, rabbits)
# 
# *  v: number of predators (for example, foxes)  
#   
# * a, b, c, d are constant parameters defining the behavior of the population:    
# 
#   + a is the natural growing rate of rabbits, when there's no fox
# 
#   + b is the natural dying rate of rabbits, due to predation
# 
#   + c is the natural dying rate of fox, when there's no rabbit
# 
#   + d is the growing factor of foxes due to encouter with rabbits,
# 
# We will use X=[u, v] to describe the state of both populations.

import numpy as np
import scipy as scp

def dX_dt(X, t, a, b, c, d):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([ a*X[0] -   b*X[0]*X[1] ,  
                      -c*X[1] + d*X[0]*X[1] ])

    
def Solve_Lotka(X0, Time, Step, a, b, c, d):
    t = np.linspace(0, Time,  Step) 
    
    X, infodict = scp.integrate.odeint(dX_dt, X0, t, args=(a, b, c, d), full_output = 1)
    
    """
    Ans = X[-1,:]
    print("Numb of preys at time " + repr(Time) + " : " + repr(Ans[0]))
    print("Numb of predators at time " + repr(Time) + " : " + repr(Ans[1]))
    """
    #print(infodict)
    """
    preys, predators = X.T

    f1 = pyl.figure()
    pyl.plot(t, preys, 'r-', label='Preys')
    pyl.plot(t, predators, 'b-', label='Predators')
    pyl.grid()
    pyl.legend(loc='best')
    pyl.xlabel('time')
    pyl.ylabel('population')
    pyl.title('Evolution of predator and prey populations')
    """
    
    return X

def Decision_Lotka(X0, Time, Step, M0, a, b, c, d):
    R = Solve_Lotka(X0, Time, Step, a*1.0, b*1.0, c*1.0, d*1.0)
    if (np.min(R[:,0]) > M0):
        result = -1
    else:
        result = 1
    return result