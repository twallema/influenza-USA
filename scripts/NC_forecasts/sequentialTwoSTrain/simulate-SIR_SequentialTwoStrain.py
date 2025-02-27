"""
This script simulates the two-strain sequential infection SIR model for North Carolina, can also be used to simulate the one-strain model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."


import matplotlib.pyplot as plt
from datetime import datetime 
from influenza_USA.NC_forecasts.utils import initialise_model

##############
## Settings ##
##############

# model settings
start_sim = datetime(2014, 11, 1)
end_sim = datetime(2015, 5, 1)
state = 'north carolina'                # simulated US state
sr = 'states'                           # spatial resolution: 'states' or 'counties'
ar = 'full'                             # age resolution: 'collapsed' or 'full'
dd =  False                             # vary contact matrix by daytype

#################
## Setup model ##
#################

model = initialise_model(strains=True, spatial_resolution=sr, age_resolution=ar, state='north carolina', season='average', distinguish_daytype=dd) # `strains` to control one vs. two strain model

######################
## Visualize result ##
######################

# wrap timer
import timeit
def to_time():
    model.sim([start_sim, end_sim])
# Measure the execution time for 20 repetitions
execution_time = timeit.timeit(to_time, number=20)
print(f"Total execution time for 20 runs: {execution_time:.6f} seconds")
print(f"Average execution time per run: {execution_time / 20:.6f} seconds")

# Simulate model
out = model.sim([start_sim, end_sim])


# Visualize
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.3, 11.7/2))
# hospitalisations
ax[0].plot(out['date'], 7*out['I_inc'].sum(dim=['age_group', 'location']), color='green', alpha=1, linewidth=2, label='Infectious (strain I + II)')
ax[0].plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2, label='Hosp. strain I')
ax[0].plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']), color='red', alpha=1, linewidth=2, label='Hosp. strain II')
ax[0].plot(out['date'], 7*((out['H1_inc'] + out['H2_inc']).sum(dim=['age_group', 'location'])), color='black', alpha=1, linewidth=2, label = 'Hosp. (strain I + II)')
ax[0].set_ylabel('Weekly hospitalisations (-)')
ax[0].legend()
# ratios
ax[1].plot(out['date'], 100*out['H1_inc'].sum(dim=['age_group', 'location']) / (out['H1_inc'].sum(dim=['age_group', 'location'])+out['H2_inc'].sum(dim=['age_group', 'location'])),
           color='black', alpha=1, linewidth=2)
ax[1].set_ylabel('ratio strain I (%)')
ax[1].set_ylim([0,100])
plt.show()
plt.close()
