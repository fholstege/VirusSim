# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:51:59 2021

@author: flori
"""
import os
os.chdir('/Users/stanthijssen/Documents/Studie/TI - BDS/Simulation Analysis and Optimization/VirusSim-main')
import numpy as np
from generate_arrivals import simulate_arrivals
import matplotlib.pyplot as plt 
import pandas as pd
from pytictoc import TicToc
t = TicToc() #create instance of class
import string


def select_upcoming_event(dict_active_events):
    """
    

    Parameters
    ----------
    dict_active_events : dict with the active events 
        behind each key is a list of times.

    Returns
    -------
    upcoming_event : string
        the event which is most near in time.

    """
    
    # save here the upcoming
    upcoming_event_time = float("inf")
    upcoming_event = list(dict_active_events.keys())[0]
    
    # go through events in the dict
    for event in dict_active_events:
        
        # save the nearest time
        times_event = dict_active_events[event]
        nearest_time_event_type = min(times_event)
        
        # if earlier than previously saved, make this the upcoming
        if nearest_time_event_type <  upcoming_event_time:
            upcoming_event = event
            upcoming_event_time = nearest_time_event_type
    
    return upcoming_event

def sample_patient_ICU_time(shape, scale):
    """
    

    Parameters
    ----------
    shape : float
        shape parameter for distribution of patient time in ICU.
    scale : float
        scale parameter for distrition of patient time in ICUs.

    Returns
    -------
    sample : float
        how long patient staid in ICU

    """
    
    # use gamma distribution with shape (alpha) and scale (theta) parameters
    sample = np.random.gamma(shape, scale)
    return sample

def simulate_system_day(arrival_times, current_ICU_capacity, total_ICU_capacity, prob_death_adm, prob_death_rej,shape_ICU_time, scale_ICU_time ,active_events = None,hours_day = 24):
    """
    
    Parameters
    ----------
    arrival_times : list
        when the patients arrive, measured in days.
    current_ICU_capacity : int
        How much ICU beds are currently taken?
    total_ICU_capacity : int
        How much ICU beds are available in total?.
    prob_death_adm : float
        What is the probability of death, after being admitted to the ICU?.
    prob_death_rej : float
        What is the probability of death, after being reject from the ICU?.
    shape_ICU_time : float
        shape parameter for distribution for ICU time.
    scale_ICU_time : float
        scale parameter for distribution for ICU time.
    active events: dict
        if want to pass on active events to the function
    hours_day : int, optional
        how many hours is the hospital open. The default is 24.
    Returns
    -------
    total_deaths_today : int
        how many patients died today.
    current_ICU_capacity : int
        how many beds currently taken.
    """
    
    if active_events is None:
         
        # create a dict of the active events
        active_events = {'A': [], # A: arrival event, list of times when arrive
                         'D': []} # D: departure event, list of times when depart
        
    
    # count here the number of deaths
    total_deaths_after_admission = 0
    total_deaths_after_rejection = 0
    total_admitted_hospital = 0
    total_rejected_hospital = 0
    
    # set the current time to 0, and get total minutes per day
    current_time = 0
        
    # get arrivals 
    active_events['A'] = arrival_times
    
    # keep true while simulation ongoing
    simulate_day = True
    
    # go over each arrival
    while simulate_day:
        
                
        if not active_events['A'] and not active_events['D']:
            break 
        elif not active_events['D']:
            upcoming_event = 'A'
        elif not active_events['A']:
            upcoming_event = 'D'
        else:
            # check which event is first
            upcoming_event = select_upcoming_event(active_events)
        
        # sort the departure times
        active_events['D'] = sorted(active_events['D'])
        
        
        # if the upcoming event is past the day, stop the while loop
        if  active_events[upcoming_event][0] >= 1:
            simulate_day = False
        # otherwise continue simulating
        else:
            
            # get the event time and remove it from the dict    
            upcoming_event_time = active_events[upcoming_event].pop(0)
            
            # if the upcoming event is an arrival
            if upcoming_event == 'A':
            
                # if there is currently capacity; admit to ICU capacity
                if current_ICU_capacity < total_ICU_capacity:
                    
                    # add tot variable which saves current icu capacity
                    current_ICU_capacity += 1
                    
                    # save that a patient has been saved to the hospital
                    total_admitted_hospital += 1
                    
                    # generate a time when this patient will depart
                    time_in_ICU = sample_patient_ICU_time(shape_ICU_time, scale_ICU_time)
                    time_departure = upcoming_event_time + time_in_ICU
                    
                    # add departure to active events
                    active_events['D'].append(time_departure)
                
                # the patient is rejected from the ICU;
                elif current_ICU_capacity >= total_ICU_capacity:
                    
                    # add that rejected from hospital
                    total_rejected_hospital += 1
                    
                    # draw random uniform number between 0 and 1
                    u = np.random.random()
                    
                    # if below the probability of dying, add one to death count
                    if u < prob_death_rej:
                        total_deaths_after_rejection += 1
            
            # if a patient departs
            elif upcoming_event == 'D':
                
                # remove a person from the ICU
                current_ICU_capacity -= 1
                
                # draw a random number
                u = np.random.random()
                
                if u < prob_death_adm:
                    total_deaths_after_admission += 1
    
    # save events that remain active
    remaining_active_departures = active_events['D']
        
    # how many died after admission and rejection
    total_deaths_today = total_deaths_after_admission + total_deaths_after_rejection
    
    dict_results = {'total_death': total_deaths_today,
                    'total_death_after_admission': total_deaths_after_admission,
                    'total_death_after_rejection': total_deaths_after_rejection,
                    'total_admitted_hospital': total_admitted_hospital, 
                    'total_rejected_hospital': total_rejected_hospital,
                    'current_ICU_capacity': current_ICU_capacity,
                    'current_active_departures': remaining_active_departures}
            
    return dict_results

def simulate_system_T_days(T,starting_ICU_capacity, total_ICU_capacity, prob_death_adm, prob_death_rej,shape_ICU_time, scale_ICU_time,R,N0,K):
    """
    
    Parameters
    ----------
    T : int
        Number of days to run the system
    starting_ICU_capacity : int
        How much ICU beds are taken at the start of the system?.
    total_ICU_capacity : int
        How much ICU beds are available in total?.
    prob_death_adm : float
        What is the probability of death, after being admitted to the ICU?.
    prob_death_rej : float
        What is the probability of death, after being reject from the ICU?.
    shape_ICU_time : float
        shape parameter for distribution for ICU time.
    scale_ICU_time : float
        scale parameter for distribution for ICU time.
    R : float
        number of new infections an infected person causes (0.1  = 10 people create 11 sick people).
    N0 : int
        number of infected people at t = 0.
    K : int
        number of people available to be sick.
    Returns
    -------
    result_sim: dict
        dict with important metrics for a system
    """
    
    
    # save several parameters; how many death per day, how many arrived
    total_deaths_sim = np.zeros(T)
    total_deaths_after_admission_sim = np.zeros(T)
    total_deaths_after_rejection_sim = np.zeros(T)
    arrived_today = np.zeros(T)
    
    # update with each day - how much icu capacity is taken?
    ICU_capacity_today = starting_ICU_capacity
    
    # simulate the arrival date, save in which day it occurs
    arrival_times_days = np.array(simulate_arrivals(R, N0, K, T))
    which_day_arrival = np.array(list((map(int,arrival_times_days))))
    
    # initialy have no active events
    active_events = None
    
    
    # loop through the days
    for i_day in range(0, T):
        
        # get the arrival times for a particular day
        indeces_day = np.where(which_day_arrival == i_day)
        arrival_times_day = list(arrival_times_days[indeces_day] - i_day)
        
        # get how many arrived, and save
        total_arrived_today = len(arrival_times_day)
        arrived_today[i_day] = total_arrived_today
        
        # get results for a day
        result = simulate_system_day(arrival_times_day, ICU_capacity_today, total_ICU_capacity, prob_death_adm, prob_death_rej,shape_ICU_time, scale_ICU_time,active_events,hours_day = 24)
        
        # print results for checking [turn off during simulation]
        #print("Deaths today: ",result['total_death'])
        #print("Deaths today after admission: ", result['total_death_after_admission'])
        #print("Deaths today after rejection: ", result['total_death_after_rejection'])
        #print("Total today admitted hospital: ", result['total_admitted_hospital'])
        #print("Total today rejected hospital: ", result['total_rejected_hospital'])
        #print("Current ICU capacity: ", result['current_ICU_capacity'])
        
        # update the active vents for the next iteration
        active_events = {'A': [], 'D':list(np.array(result['current_active_departures']) - (i_day+1)) }
        
        # updat ICU capacity for the next iteration
        ICU_capacity_today = result['current_ICU_capacity']
        
        # update the relevant statistics
        total_deaths_sim[i_day] = result['total_death']
        total_deaths_after_admission_sim[i_day] = result['total_death_after_admission']
        total_deaths_after_rejection_sim[i_day] = result['total_death_after_rejection']

    
    results = {'total_death': total_deaths_sim,
               'total_death_after_admission': total_deaths_after_admission_sim,
               'total_death_after_rejection': total_deaths_after_rejection_sim,
               'arrivals_per_day': arrived_today}
        
        
    return results

def create_df_init_params(dRLow, dRSQ, dRHigh, dICURateLow, dICURateSQ, iICUCapacitySQ, iICUCapacityHigh, shape_gamma_ICULow, scale_gamma_ICULow, shape_gamma_ICUSQ, scale_gamma_ICUSQ):
    vScenarioNames = ['Scenario ' + _ for _ in 'ABCDEFGHIJKLMNOPQRSTUVWX']
    vColNames = ['R', 'ICU Rate', 'Available Capacity', 'Shape gamma', 'Scale gamma']
    
    mInitParams = np.nan * np.zeros((24, 5))
    mInitParams[:8, 0] = dRLow
    mInitParams[8:16, 0] = dRSQ
    mInitParams[16:, 0] = dRHigh
    mInitParams[(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19), 1] = dICURateLow
    mInitParams[(4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23), 1] = dICURateSQ
    mInitParams[(0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21), 2] = iICUCapacitySQ
    mInitParams[(2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23), 2] = iICUCapacityHigh
    mInitParams[(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22), 3:5] = [shape_gamma_ICULow, scale_gamma_ICULow]
    mInitParams[(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23), 3:5] = [shape_gamma_ICUSQ, scale_gamma_ICUSQ]
    # np.sort(np.mean(mInitParams, axis = 1)) # to check, should be al unique
    dfInitParams = pd.DataFrame(data = mInitParams, index = vScenarioNames, columns = vColNames)
    
    return(dfInitParams)
        
# based on the following paper: page 4 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7341703/pdf/10729_2020_Article_9511.pdf
#shape_gamma_ICU = 1.66  
#scale_gamma_ICU = 1/0.206
prob_death_adm = 0.5
prob_death_rej = 0.9

# the following is based on RIVM capacity 
starting_ICU_capacity = 865
total_ICU_capacity = 1150
N0 = 7000 * 7 # number of infected people at t = 0 - approx. number of cases in last two weeks

population = 17500000
already_infected = 1250000
vaccinated = 2500000
K = population - already_infected - vaccinated # total number of people (uninfected) in population at t=0

R = 1.07
T_c = 4
perc_ICU = 0.05
r = ((R -1) /T_c) * perc_ICU
n_days = 30                                         # change to 60

######################### Stans adds #########################
### Magic Numbers
iR = 10
T_c = 5                                             
# Parameter settings: SQ (Status Quo), Low, Med, or High scenario
dRLow = 1.01
dRSQ = 1.06
dRHigh = 1.11
dICURateLow = 0.0248                              
dICURateSQ = 0.0348

# iICUCapacityLow = 1150                            # Skip for now
iICUCapacitySQ = 1150                               # FIND OUT
iICUCapacityHigh = 1300                             # FIND OUT

# Note: initial mean was 1.66 * 1 / 0.26 = 6.3846
shape_gamma_ICULow = 1.66 * (10 / 6.3846)
scale_gamma_ICULow = 1/0.206 * (10 / 6.3846)
shape_gamma_ICUSQ = 1.66 * (14 / 6.3846)
scale_gamma_ICUSQ = 1/0.206 * (7 / 6.3846)

dfInitParams = create_df_init_params(dRLow, dRSQ, dRHigh, dICURateLow, dICURateSQ, iICUCapacitySQ, iICUCapacityHigh, shape_gamma_ICULow, scale_gamma_ICULow, shape_gamma_ICUSQ, scale_gamma_ICUSQ)
mInitParams = np.asmatrix(dfInitParams)


t.tic()
vAlphabet = list(string.ascii_uppercase)[:24]

dResults = {}
for i in range(len(vAlphabet)):
    print(i)
    sName = vAlphabet[i]
    dResults['result_sim'+sName] = [None] * iR
    for j in range(iR):
        dResults['result_sim'+sName][j] = simulate_system_T_days(T = n_days,
                                                                 starting_ICU_capacity = starting_ICU_capacity, 
                                                                 total_ICU_capacity = mInitParams[i,2], 
                                                                 prob_death_adm = prob_death_adm, 
                                                                 prob_death_rej = prob_death_rej,
                                                                 shape_ICU_time = mInitParams[i, 3], 
                                                                 scale_ICU_time = mInitParams[i, 4],
                                                                 R = ((mInitParams[i, 0] -1) /T_c) * mInitParams[i, 1], # Take care: R here represents r!!
                                                                 N0 = N0,
                                                                 K = K)
t.toc()

dAverageResults = {}
vData = ['total_death', 'total_death_after_admission', 'total_death_after_rejection', 'arrivals_per_day']
for i in vAlphabet: 
    dAverageResults['result_sim'+i] = [[np.zeros(n_days)], [np.zeros(n_days)], [np.zeros(n_days)], [np.zeros(n_days)]]
    for j in range(iR):        
        for d in range(4):
            sData = vData[d] 
            dAverageResults['result_sim'+i][d] += (dResults['result_sim'+i][j][sData] / iR).reshape((1,n_days))

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
    
######################### End Stans adds -> parallel #########################
    
