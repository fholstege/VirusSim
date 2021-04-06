# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:40:23 2021

@author: flori
"""

from scipy.stats import beta as beta_distribution
from scipy.stats import gamma as gamma_distribution
from sklearn.neighbors import KernelDensity
import scipy.stats as stats    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV


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
plt.hist(distribution_ICU_stay, bins = 60)


X = np.array(distribution_ICU_stay).reshape(-1,1)
X_plot = np.linspace(0, 60, 60)[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=1.5).fit(X)
density = np.exp( kde.score_samples(X_plot))

plt.hist(kde.sample(n_samples = 10000), bins = 60)



fig, ax1 = plt.subplots()
ax1.plot(X_plot, density)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.hist(distribution_ICU_stay, bins = 60, alpha = 0.2)

plt.show()



bandwidths =  np.linspace(0.1, 5, 10)
bandwidths

grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths})
grid.fit(X_plot)


grid.cv_results_['mean_test_score']


# get distributions based on the historical data
alpha_b, beta_b,loc_b, scale_b = beta_distribution.fit(distribution_ICU_stay)

# create a sample of fitted values
sample = beta_distribution.rvs(alpha_b, beta_b, loc_b, scale_b, size = 10000)
plt.hist(sample, bins = 60)

