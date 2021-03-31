# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:43:58 2021

@author: flori
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



def fIDot(iK, dA, dR, vT): 
    """
    Purpose:
        This function calculates the changing daily number of new infections.
    
    Inputs:
        iK:    integer, total number of people (uninfected) in population at t=0
        dA:    double, ratio of uninfected over infected at t = 0
        dR:    double, reproduction factor
        vT:    vector, time span
        
    Return: 
        vIDot: vector, length as vT, expected number of new infections at t=t
    """    
    
    vIDot = (dA * iK * dR * np.exp(-dR * vT)) / ((1 + dA * np.exp(-dR *  vT)) ** 2) 
    
    return(vIDot)

def simulate_arrivals(dR, iN0, iK, iT):
    """
    Purpose:
        This function simulates the arrivals of COVID patients that require
        treatment at the intensive care. All inputs, outputs and returns are 
        measured on a daily base.
    
    Inputs:
        dR:     number of new infections an infected person causes
        iN0:    number of infected people at t = 0
        iK:     number of people available to be infected
        iT:     integer, time horizon for simulation (number of days)
    
    Outputs: 
        vA:   float, arrival times
        (optional) vIDot:  expected number of new infections at t=t
        (optional) vDaily: vector, number of daily cases
        
   
        
    """
    dA = (iK - iN0) / iN0
    vT = np.linspace(1, iT, iT)
    vIDot = fIDot(iK = iK, dA = dA, dR = dR, vT = vT)
    
    
    t = 0
    dDaily = 0
    vA = []
    vDaily = []
    while t < iT:
        tInd = int(np.floor(t))
        vA.append(float(t + np.random.exponential(1 / vIDot[tInd],1)))
        dDaily += 1
        
        if np.floor(vA[-1]) - tInd > 0:
            
            # store total number of new ICU patients per day
            vDaily.append(dDaily)
            dDaily = 0
        t = vA[-1]

    return vA