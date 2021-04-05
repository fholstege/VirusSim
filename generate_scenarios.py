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
import scipy.stats as stats    


# create instance of class
t = TicToc() 

## First parameters: probability of death after admission and rejection  
# taken from: https://stichting-nice.nl/covid-19-op-de-ic.jsp, first of april
total_death_after_ICU = 2278
total_alive_after_ICU = 6128
total_alive_after_ICU_but_in_hospital = 384
prob_death_adm = total_death_after_ICU/(total_death_after_ICU + total_alive_after_ICU + total_alive_after_ICU_but_in_hospital)


# based on the following paper: page 4 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7341703/pdf/10729_2020_Article_9511.pdf
prob_death_rej = 0.9

# Second set: how much people currently in the ICU (starting), how many currently infected, how many could be in total,  the K (how many people can be infected)
# the following is based on RIVM ICU capacity 
ICU_capacity_taken_nonCovid = 500
ICU_capacity_taken_Covid = 700
starting_ICU_capacity = ICU_capacity_taken_Covid + ICU_capacity_taken_nonCovid

# based on RIVM data, high and low capacity
iICUCapacitySQ = 1600                              
iICUCapacityHigh = 2400    
                         
# number of infected people at t = 0 - approx. number of cases in last two weeks
N0 = 7000 * 7 

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


# generate the small r
r_low = (R_low - 1) /T_c
r_SQ = (R_SQ - 1)/T_c
r_high = (R_high - 1)/T_c



# get data on ICU stay
data_ICU_stay = pd.read_excel('raw_IC_data.xlsx')

# save here the data
distribution_ICU_stay = []

# generate the data for the distribution
for index, row in data_ICU_stay.iterrows():
    
    n_patients = row['n_patients']
    days_ICU = row['days_ICU']
    
    data_to_add = n_patients * [days_ICU]
    
    distribution_ICU_stay.append(data_to_add)
 
# flatten list of lists
distribution_ICU_stay=  [val for sublist in distribution_ICU_stay for val in sublist]

# get distributions based on the historical data
alpha, beta, loc, scale = beta_distribution.fit(distribution_ICU_stay)

# get parameters for the beta distribution, gamma distribution
param_ICU_time_beta = {'alpha': alpha, 'beta': beta, 'loc': loc, 'scale': scale}
param_ICU_time_gamma = {'scale': 1.66 , 'shape': 1/0.206}


# define parameters for the age split
param_ageGroup_split_60 = {'split_at_60': True,
                  'perc_patients_below_60_ICU': 0.322,
                  'perc_patients_below_60_cases': 0.85,
                  'ICU_rate_below_60': 0.023,
                  'ICU_rate_above_60': 0.28,
                  'prob_death_adm_below_60': 0.129,
                  'prob_death_adm_above_60': 0.398,
                  'ICU_rate_overall': 0.062}



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
        
        
        dResults['result_sim'+sName][j] = simulate_system_T_days(T = n_days,
                                                                 starting_ICU_capacity = parameters_scenario['starting_ICU_capacity'], 
                                                                 total_ICU_capacity =parameters_scenario['total_ICU_capacity'], 
                                                                 prob_death_adm = parameters_scenario['prob_death_adm'], 
                                                                 prob_death_rej = parameters_scenario['prob_death_rej'],
                                                                 param_ICU_time_dict=param_ICU_time_beta,
                                                                 param_ageGroup_dict=param_ageGroup_split_60,
                                                                 r = parameters_scenario['r'],
                                                                 N0 = parameters_scenario['N0'],
                                                                 K = parameters_scenario['K']
                                                                )
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