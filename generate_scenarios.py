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
from sim_functions import simulate_system_T_days, sample_patient_ICU_time_gamma
from visual_functions import calc_CI_stats_variable_scenario, generate_plots
import pandas as pd
from pytictoc import TicToc
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from scipy.stats import beta as beta_distribution
from scipy.stats import gamma as gamma_distribution
from sklearn.neighbors import KernelDensity
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
prob_death_rej = 0.99

# Second set: how much people currently in the ICU (starting), how many currently infected, how many could be in total,  the K (how many people can be infected)
# the following is based on RIVM ICU capacity 
ICU_capacity_taken_nonCovid = 500
ICU_capacity_taken_Covid = 700
starting_ICU_capacity = ICU_capacity_taken_Covid + ICU_capacity_taken_nonCovid

# based on RIVM data, high and low capacity
iICUCapacityLow = 1400                              
iICUCapacityMedium = 1700    
iICUCapacityHigh = 2000    

                         
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
alpha_b, beta_b,loc_b, scale_b = beta_distribution.fit(distribution_ICU_stay)

# fit a kernel density
X_kernel = np.array(distribution_ICU_stay).reshape(-1,1)
kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(X_kernel)



# get parameters for the beta distribution, gamma distribution
param_ICU_time_beta = {'alpha': alpha_b, 'beta': beta_b, 'loc': loc_b, 'scale': scale_b}
param_ICU_time_gamma = {'scale': 1.66 , 'shape': 1/0.206} # informed by paper
param_ICU_time_kde = {'kde':kde }



# define parameters for the age split
param_ageGroup_split_60 = {'split_at_60': True,
                  'perc_patients_below_60_ICU': 0.322,
                  'ICU_rate_below_60': 0.023,
                  'ICU_rate_above_60': 0.28,
                  'prob_death_adm_below_60': 0.129,
                  'prob_death_adm_above_60': 0.398,
                  'ICU_rate_overall': 0.062}
# % of cases
perc_cases_below60_sq = 0.85
perc_cases_below60_improve = 0.9


# how many sims,for how many days
n_sim = 1000
n_days = 30


# define all the possible parameters
parameters = {'prob_death_adm':[prob_death_adm],
              'prob_death_rej':[ prob_death_rej],
              'starting_ICU_capacity': [starting_ICU_capacity],
              'total_ICU_capacity': [iICUCapacityLow, iICUCapacityMedium, iICUCapacityHigh], 
              'K': [K],
              'N0': [N0],
              'r': [r_low, r_SQ, r_high],
              'perc_cases_below60':[perc_cases_below60_sq, perc_cases_below60_improve]
              }

# create the parameter grid
param_grid = list(ParameterGrid(parameters))

# parameter grid
df_param_grid = pd.DataFrame(param_grid)
df_param_grid


t.tic()
vAlphabet = list(string.ascii_uppercase)[:len(param_grid)]

dResults = {}
dParam ={}

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
                                                                 param_ICU_time_dict=param_ICU_time_kde,
                                                                 ICU_time_distribution = 'kde',
                                                                 param_ageGroup_dict=param_ageGroup_split_60,
                                                                 perc_cases_below60=parameters_scenario['perc_cases_below60'] ,
                                                                 r = parameters_scenario['r'],
                                                                 N0 = parameters_scenario['N0'],
                                                                 K = parameters_scenario['K']
                                                                )
        
    dParam['param_sim'+sName] = parameters_scenario
t.toc()


   
# save average results
dOverallResults = {}
variables = ['total_death', 'total_death_after_admission', 'total_death_after_rejection', 'arrivals_per_day']


for scenario in dResults.keys():
    
   # get the letter of the scenario 
   letter_scenario = scenario[-1]
   
   # save here all the results per scenario
   total_death_results = [None]*n_sim
   total_death_after_admission_results = [None]*n_sim
   total_death_after_rejection_results = [None]*n_sim
   arrivals_per_day_results = [None]*n_sim
   
   # get results per scenario
   result_scenario = dResults[scenario]     
       
   for i_sim in range(0, n_sim):
      
       # add to respecitve lists
      total_death_results[i_sim] = result_scenario[i_sim]['total_death']
      total_death_after_admission_results[i_sim] = result_scenario[i_sim]['total_death_after_admission']
      total_death_after_rejection_results[i_sim] = result_scenario[i_sim]['total_death_after_rejection']
      arrivals_per_day_results[i_sim] = result_scenario[i_sim]['arrivals_per_day']
       
   
   # get results in dataframe, ready to calculate confidence intervals 
   df_results_total_death = pd.DataFrame(total_death_results).transpose()
   df_results_total_death_after_admission = pd.DataFrame(total_death_after_admission_results).transpose()
   df_results_total_death_after_rejection = pd.DataFrame(total_death_after_rejection_results).transpose()
   df_results_arrivals_per_day = pd.DataFrame(arrivals_per_day_results).transpose()
   
   # get confidence interval per stat
   df_CI_stats_total_death = calc_CI_stats_variable_scenario(df_results_total_death, 'total_death', letter_scenario)
   df_CI_stats_total_death_after_admission = calc_CI_stats_variable_scenario(df_results_total_death_after_admission, 'total_death_after_admission', letter_scenario)
   df_CI_stats_total_death_after_rejection = calc_CI_stats_variable_scenario(df_results_total_death_after_rejection, 'total_death_after_rejection', letter_scenario)
   df_CI_stats_arrivals_per_day = calc_CI_stats_variable_scenario(df_results_arrivals_per_day, 'arrivals_per_day', letter_scenario)

   # add to dict of overall results 
   dOverallResults[scenario] = {'total_death_overall':df_CI_stats_total_death,
                                 'total_death_after_admission_overall':df_CI_stats_total_death_after_admission, 
                                 'total_death_after_rejection_overall': df_CI_stats_total_death_after_rejection,
                                 'arrivals_per_day_overall': df_CI_stats_arrivals_per_day} 
       
      
       

generate_plots(dOverallResults, dParam,['arrivals_per_day','total_death','total_death_after_admission','total_death_after_rejection', ], CI = False)

