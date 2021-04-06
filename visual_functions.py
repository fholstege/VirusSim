# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:35:00 2021

@author: flori
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_CI_stats_variable_scenario(df_results_variable_scenario, variable, scenario):
    
    # calculate the mean and standard deviation
    mean_variable_scenario = df_results_variable_scenario.mean(axis=1)
    sd_variable_scenario = df_results_variable_scenario.std(axis=1)
    
    # calculate standard error to get CI
    SE_variable_scenario = sd_variable_scenario/np.sqrt(len(mean_variable_scenario))
    lower_variable_scenario =mean_variable_scenario - 1.96* SE_variable_scenario
    upper_variable_scenario =mean_variable_scenario + 1.96* SE_variable_scenario
    
    df_CI_stats_variable_scenario = pd.DataFrame({'mean_' + variable + '_' + scenario: mean_variable_scenario,
                                                  'sd_'+ variable + '_'+ scenario: sd_variable_scenario, 
                                                  'SE_'+ variable + '_'+ scenario: SE_variable_scenario,
                                                  'lower_'+ variable + '_'+ scenario: lower_variable_scenario,
                                                  'upper_'+ variable + '_'+ scenario: upper_variable_scenario,
                                                  })
    
    return df_CI_stats_variable_scenario


def generate_plots(dOverallResults,dParam,variables, save=False, CI=False, n_days=30, T_c=5):
        
    
    #plot the arrivals 
    for result_sim_scenario in dOverallResults.keys():
        
        letter_scenario = result_sim_scenario[-1]
        
        parameters_scenario = dParam['param_sim'+letter_scenario]
        
        # start with creating figure
        plt.figure()
        plt.title(str('Scenario '+ letter_scenario))
        
        # get per scenario the results
        results_scenario = dOverallResults[result_sim_scenario]
        
        for variable in variables:
            
            df_variable_overall = results_scenario[variable + '_overall']
            
            plt.plot(range(1,n_days+1), df_variable_overall['mean_'+variable +'_' +letter_scenario], label = variable)
            
            if CI:
                plt.plot(range(1,n_days+1), df_variable_overall['lower_'+variable+'_' +letter_scenario], '--', color = 'black')
                plt.plot(range(1,n_days+1), df_variable_overall['upper_'+variable+'_' +letter_scenario], '--', color = 'black')
            
        
        
        plt.gcf().text(0.02, 1,'Total ICU capacity: '+ str(parameters_scenario['total_ICU_capacity']) , fontsize=11)
        plt.gcf().text(0.02, 0.95,'R: ' + str((parameters_scenario['r']*T_c)+1) , fontsize=11)
        plt.gcf().text(0.02, 0.9,'% Cases below 60: ' + str(parameters_scenario['perc_cases_below60']*100) , fontsize=11)

        plt.legend()
        if save:
            plt.savefig(str('Scenario'+letter_scenario+'.eps'))
        