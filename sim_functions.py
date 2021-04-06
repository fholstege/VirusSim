# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:29:46 2021

@author: flori
"""

import numpy as np
from generate_arrivals import simulate_arrivals
import matplotlib.pyplot as plt 
from scipy.stats import beta as beta_distribution


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

def sample_patient_ICU_time_gamma(shape, scale):
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


def sample_patient_ICU_time_Beta(alpha, beta, loc, scale):
    """
    

    Parameters
    ----------
    mean : float
        mean beta distribution.
    var : float
        variance beta distribtuion.
    min_distribution : float
        minimum of beta distribution.
    max_distribution : float
        max of beta distribution.

    Returns
    -------
    sample : float
        single sample from beta.

    """
    

    # get sample
    sample = beta_distribution.rvs(a=alpha,b=beta,size = 1, loc = loc, scale = scale)
    
    return sample
    

def simulate_system_day(arrival_times, current_ICU_capacity, total_ICU_capacity, prob_death_adm, prob_death_rej, param_ICU_time_dict,ICU_time_distribution, param_ageGroup_dict, active_events = None,hours_day = 24):
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
                    
                    # save 0, then assign
                    time_in_ICU = 0
                    
                    # generate a time when this patient will depart - put in gamma or beta
                    if ICU_time_distribution == 'gamma':
                        time_in_ICU = sample_patient_ICU_time_gamma(param_ICU_time_dict['scale'], param_ICU_time_dict['shape'])
                    elif ICU_time_distribution == 'beta':
                        time_in_ICU = sample_patient_ICU_time_Beta(mean= param_ICU_time_dict['alpha'], var=param_ICU_time_dict['beta'],
                                                               min_distribution=param_ICU_time_dict['loc'], max_distribution=param_ICU_time_dict['scale'])
                    
                    # define moment of departure of this event
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
                u_1 = np.random.random()
                
                # if want to split at the age of 60
                if param_ageGroup_dict['split_at_60']:
                    
                    # generate another random number
                    u_2 = np.random.random()
                    
                    # if a patient below 60
                    if u_2 < param_ageGroup_dict['perc_patients_below_60_ICU']:
                    
                    # use a particular probability of death based on which group it is
                        prob_death_adm = param_ageGroup_dict['prob_death_adm_below_60']
                    else:
                        prob_death_adm = param_ageGroup_dict['prob_death_adm_above_60']
                    
                # check if would have died after admission
                if u_1 < prob_death_adm:
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

def simulate_system_T_days(T,starting_ICU_capacity, total_ICU_capacity, prob_death_adm, prob_death_rej, param_ICU_time_dict,ICU_time_distribution, param_ageGroup_dict, r, N0, K):
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
    r : float
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
    
    
    # check if want to split at age of 60
    if param_ageGroup_dict['split_at_60']:
        
        # define the ICU rate for below and above 60
        icu_rate_below_60 = param_ageGroup_dict['ICU_rate_below_60']
        icu_rate_above_60 = param_ageGroup_dict['ICU_rate_above_60']
        
        # using the percentage of people below 60
        perc_below_60 = param_ageGroup_dict['perc_patients_below_60_cases']
        
        # calculate icu rate of these two groups combined
        icu_rate_combined = (perc_below_60 * icu_rate_below_60) + ((1-perc_below_60) * icu_rate_above_60)
        
        r =  r * icu_rate_combined
    # if not, just have a single icu rate
    else:
        r = r * param_ageGroup_dict['ICU_rate_overall']

        
    
    # simulate the arrival date, save in which day it occurs
    arrival_times_days = np.array(simulate_arrivals(r, N0, K, T))
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
        result = simulate_system_day(arrival_times_day, ICU_capacity_today, total_ICU_capacity, prob_death_adm, prob_death_rej,param_ICU_time_dict,ICU_time_distribution,param_ageGroup_dict, active_events,hours_day = 24)
        
        
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