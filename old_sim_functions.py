# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:43:40 2021

@author: flori
"""
import numpy as np


def generate_patient_arrival_time_day(arrival_rate, hours_day = 24):
    """
    Input:
        arrival_rate: parameter for the exponential distribution with mean 1/arrival_rate
        hours_day: parameter for the number of hours a hospital is open
    
    """
    
    # set the current time to 0
    current_time = 0
    
    # the total number of minutes for a day 
    total_minutes_day = 60 * hours_day
    
    # save here when each patient arrives
    arrival_times = []
    
    # while the time is not up, keep adding patients
    while current_time < total_minutes_day:
        
        # get arrival time of patient; current time, plus the inter-arrival time
        
        arrival_time_patient = current_time + sample_inter_arrival_time(arrival_rate)
        
        # add to list with patient arrival times
        arrival_times.append(arrival_time_patient)
        
        # update the current time
        current_time =+ arrival_time_patient
    
    # add all except last to make sure it does not run over time
    return(arrival_times[:-1])



def sample_inter_arrival_time(arrival_rate):
    sample = np.exp(arrival_rate)
    return sample


def sample_patient_ICU_time(shape, scale):
    
    # use gamma distribution with shape (alpha) and scale (theta) parameters
    sample = np.random.gamma(shape, scale)
    return sample


def simulate_system_day(arrival_times, current_ICU_capacity, total_ICU_capacity, prob_death_adm, prob_death_rej,shape_ICU_time, scale_ICU_time ,hours_day = 24):
    
    # create a dict of the active events
    active_events = {'A': [], # A: arrival event, list of times when arrive
                     'D': []} # D: departure event, list of times when depart

    
    # count here the number of deaths
    total_deaths_today = 0
    
    # set the current time to 0, and get total minutes per day
    current_time = 0
    total_minutes_day = 60 * hours_day
    
    # generate all arrival times
   # arrival_times = generate_patient_arrival_time_day(arrival_rate, hours_day = hours_day)
    
    # get arrival of first event
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
            upcoming_event = min(active_events, key=active_events.get)
                
        upcoming_event_time = active_events[upcoming_event].pop(0)
        
        # if the upcoming event is an arrival
        if upcoming_event == 'A':
        
            # if there is currently capacity; admit to ICU capacity
            if current_ICU_capacity < total_ICU_capacity:
                
                # add tot variable which saves current icu capacity
                current_ICU_capacity += 1
                
                # generate a time when this patient will depart
                time_in_ICU = sample_patient_ICU_time(shape_ICU_time, scale_ICU_time)
                time_departure = upcoming_event_time + time_in_ICU
                
                # add departure to active events
                active_events['D'].append(time_departure)
            
            # the patient is rejected from the ICU;
            elif current_ICU_capacity >= total_ICU_capacity:
                
                # draw random uniform number between 0 and 1
                u = np.random.random()
                
                # if below the probability of dying, add one to death count
                if u < prob_death_rej:
                    total_deaths_today += 1
        
        # if a patient departs
        elif upcoming_event == 'D':
            
            # remove a person from the ICU
            current_ICU_capacity -= 1
            
            # draw a random number
            u = np.random.random()
            
            if u < prob_death_adm:
                total_deaths_today += 1
        
        # set current time to upcoming event time
        current_time = upcoming_event_time
    
    # save events that remain active
    remaining_active_events = active_events
            
    return total_deaths_today, current_ICU_capacity
