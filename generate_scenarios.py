# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:34:54 2021

@author: flori
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:23:07 2021

@author: flori
"""
import numpy as np
from sim_functions import simulate_system_T_days
import pandas as pd
from pytictoc import TicToc
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from scipy.stats import beta as beta_distribution


# create instance of class
t = TicToc() 

## First parameters: probability of death after admission and rejection        
# based on the following paper: page 4 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7341703/pdf/10729_2020_Article_9511.pdf
prob_death_adm = 0.5
prob_death_rej = 0.9

# Second set: how much people currently in the ICU (starting), how many currently infected, how many could be in total,  the K (how many people can be infected)
# the following is based on RIVM ICU capacity 
starting_ICU_capacity = 865
N0 = 7000 * 7 # number of infected people at t = 0 - approx. number of cases in last two weeks

# get the K
population = 17500000
already_infected = 1250000
vaccinated = 2500000
K = population - already_infected - vaccinated # total number of people (uninfected) in population at t=0


# third set of parameters: the growth rate of the number of people in the ICU
T_c = 5  
                                         
# Parameter settings: SQ (Status Quo), Low, Med, or High scenario
R_low = 1.01
R_SQ = 1.06
R_high = 1.11

# ICU rate possibilities
ICU_rate_low = 0.0248                              
ICU_rate_SQ = 0.0348

# generate the small r
r_low = (R_low - 1) /T_c
r_SQ = (R_SQ - 1)/T_c
r_high = (R_high - 1)/T_c


# icu capacity
iICUCapacitySQ = 1150                               # FIND OUT
iICUCapacityHigh = 1300                             # FIND OUT

# get parameters for the beta distribution, gamma distribution
param_ICU_time_beta = {'mean': 17.4, 'var': 14.6**2, 'min': 1, 'max': 80}
param_ICU_time_gamma = {'scale': 1.66 , 'shape': 1/0.206}

# how many sims,for how many days
n_sim = 10
n_days = 30                                         


# define all the possible parameters
parameters = {'prob_death_adm':[prob_death_adm],
              'prob_death_rej':[ prob_death_rej],
              'starting_ICU_capacity': [starting_ICU_capacity],
              'total_ICU_capacity': [iICUCapacitySQ, iICUCapacityHigh],
              'K': [K],
              'N0': [N0],
              'ICU_rate': [ICU_rate_low, ICU_rate_SQ],
              'r': [r_low, r_SQ, r_high]}

# create the parameter grid
param_grid = list(ParameterGrid(parameters))

# parameter grid
df_param_grid = pd.DataFrame(param_grid)



t.tic()
vAlphabet = list(string.ascii_uppercase)[:len(param_grid)]

dResults = {}

# calculate for each scenario
for i in range(len(vAlphabet)):
    
    # which scenario?
    print('Scenario:', vAlphabet[i])
    
    # create key to be filled with results
    sName = vAlphabet[i]
    dResults['result_sim'+sName] = [None] * n_sim
    
    parameters_scenario = param_grid[i]
    
    # go through number of sims
    for j in range(n_sim):
        
        r_scenario = parameters_scenario['r'] * parameters_scenario['ICU_rate']
        
        dResults['result_sim'+sName][j] = simulate_system_T_days(T = n_days,
                                                                 starting_ICU_capacity = parameters_scenario['starting_ICU_capacity'], 
                                                                 total_ICU_capacity =parameters_scenario['total_ICU_capacity'], 
                                                                 prob_death_adm = parameters_scenario['prob_death_adm'], 
                                                                 prob_death_rej = parameters_scenario['prob_death_rej'],
                                                                 param_ICU_time_dict=param_ICU_time_beta,
                                                                 r = r_scenario,
                                                                 N0 = parameters_scenario['N0'],
                                                                 K = parameters_scenario['K'])
t.toc()


# save average results
dAverageResults = {}
vData = ['total_death', 'total_death_after_admission', 'total_death_after_rejection', 'arrivals_per_day']
for i in vAlphabet: 
    dAverageResults['result_sim'+i] = [[np.zeros(n_days)], [np.zeros(n_days)], [np.zeros(n_days)], [np.zeros(n_days)]]
    for j in range(n_sim):        
        for d in range(4):
            sData = vData[d] 
            dAverageResults['result_sim'+i][d] += (dResults['result_sim'+i][j][sData] / n_sim).reshape((1,n_days))

### From here on not working anymore
#plot the arrivals 
for i in vAlphabet:
    plt.figure()
    plt.title(str('Scenario '+i))
    plt.plot(range(1,n_days+1), dAverageResults['result_sim'+i][3].reshape((n_days,)), label = str('Daily New Arrivals at ICU'))
    
    # plot the deaths
    plt.plot(range(1,n_days+1), dAverageResults['result_sim'+i][0].reshape((n_days,)), label = 'Total death')
    plt.plot(range(1,n_days+1), dAverageResults['result_sim'+i][1].reshape((n_days,)), label = 'Death after admission')
    plt.plot(range(1,n_days+1), dAverageResults['result_sim'+i][2].reshape((n_days,)), label = 'Death after rejection')
    plt.legend()
    plt.savefig(str('Scenario'+i+'.eps'))