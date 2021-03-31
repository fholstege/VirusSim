# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:51:59 2021

@author: flori
"""


import numpy as np
from generate_arrivals import simulate_arrivals


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
    
    upcoming_event_time = float("inf")
    upcoming_event = list(dict_active_events.keys())[0]
    
    for event in dict_active_events:
        
        times_event = dict_active_events[event]
        nearest_time_event_type = min(times_event)
        
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
    
    # go over each arrival
    while current_time < 1:
                
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
        
        # get the event time and remove it from the dict    
        upcoming_event_time = active_events[upcoming_event].pop(0)
        
        
        # if the upcoming event is an arrival
        if upcoming_event == 'A':
        
            # if there is currently capacity; admit to ICU capacity
            if current_ICU_capacity <= total_ICU_capacity:
                
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
            elif current_ICU_capacity > total_ICU_capacity:
                
                # add that rejected from hospital
                total_rejected_hospital += 1
                
                # draw random uniform number between 0 and 1
                u = np.random.random()
                
                # if below the probability of dying, add one to death count
                if u < prob_death_rej:
                    total_deaths_after_rejection += 1
        
        # if a patient departs
        elif upcoming_event == 'D':
            
            
            print("patient removed from ICU at this time:  ", upcoming_event_time )
            
            # remove a person from the ICU
            current_ICU_capacity -= 1
            
            # draw a random number
            u = np.random.random()
            
            if u < prob_death_adm:
                total_deaths_after_admission += 1
        
        print("current time is: ", current_time)
        # update the current time
        current_time += upcoming_event_time
    
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
    
    
    # total deaths at start of the sim
    total_deaths_sim = 0
    
    # update with each day - how much icu capacity is taken?
    ICU_capacity_today = starting_ICU_capacity
    
    # simulate the arrival date, save in which day it occurs
    arrival_times_days = np.array(simulate_arrivals(R, N0, K, T))
    which_day_arrival = np.array(list((map(int,arrival_times_days))))
    
    active_events = None
    

    for i_day in range(0, T):
        
        # get the arrival times for a particular day
        indeces_day = np.where(which_day_arrival == i_day)
        arrival_times_day = list(arrival_times_days[indeces_day] - i_day)
        
        
        result = simulate_system_day(arrival_times_day, ICU_capacity_today, total_ICU_capacity, prob_death_adm, prob_death_rej,shape_ICU_time, scale_ICU_time,active_events,hours_day = 24)
        
        print("Total entered system: ", len(arrival_times_days[indeces_day]) )
        print("Deaths today: ",result['total_death'])
        print("Deaths after admission: ", result['total_death_after_admission'])
        print("Deaths after rejection: ", result['total_death_after_rejection'])
        print("Total admitted hospital: ", result['total_admitted_hospital'])
        print("Total rejected hospital: ", result['total_rejected_hospital'])
        print("Current ICU capacity: ", result['current_ICU_capacity'])
        print("Active departures: ", sorted(result['current_active_departures']))
        
        active_events = {'A': [], 'D':list(np.array(result['current_active_departures']) - (i_day+1)) }
        
        
        
        
        
        
    
    
    
    #total_deaths, ICU_capacity = simulate_system_day(arrival_times_days, starting_ICU_capacity, total_ICU_capacity, prob_death_adm, prob_death_rej,shape_ICU_time, scale_ICU_time ,hours_day = 24)
    
    #return total_deaths, ICU_capacity
    
   
                      
            
            
            
            
            
    

# based on the following paper: page 4 of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7341703/pdf/10729_2020_Article_9511.pdf
shape_gamma_ICU = 1.66  
scale_gamma_ICU = 1/0.206
prob_death_adm = 0.5
prob_death_rej = 0.9
starting_ICU_capacity = 100
total_ICU_capacity = 120
R = 0.1
N0 = 1000 # number of infected people at t = 0
K = 17000000 # total number of people (uninfected) in population at t=0

n_days = 3




simulate_system_T_days(n_days,
                           starting_ICU_capacity, 
                           total_ICU_capacity, 
                           prob_death_adm, 
                           prob_death_rej,
                           shape_gamma_ICU, 
                           scale_gamma_ICU,
                           R,
                           N0,
                           K)



t_dict = {'A': [0.1,0.2,0.3], 'D':[2.3]}

select_upcoming_event(t_dict)
