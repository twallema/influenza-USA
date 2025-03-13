"""
This script simulates the two-strain sequential infection SIR model for North Carolina, can also be used to simulate the one-strain model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import matplotlib.dates as mdates
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

model.parameters['f_I1'] = 5.0e-5
f_I2_list = [5.0e-5, 5.0e-5, 5.0e-5, 5.0e-5, 5.0e-5, 5.0e-5, 5.0e-5, 5.0e-5,  4.5e-5, 4.0e-5, 3.5e-5, 3.0e-5, 2.5e-5, 2.0e-5, 1.5e-5, 1e-5, 0.5e-5]
f_R1_list = [0.5, 0.475, 0.45, 0.425, 0.40, 0.375, 0.35, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325 ]


for i, (f_I2, f_R1) in enumerate(zip(f_I2_list, f_R1_list)):

    # set parameters
    model.parameters['f_I2'] = f_I2
    model.parameters['f_R1'] = f_R1
        
    # Simulate model
    out = model.sim([start_sim, end_sim])

    # Visualize
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8.3, 11.7/5))
    # hospitalisations
    ax.plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']), color='black', alpha=1, linestyle='-.', linewidth=1, label='Influenza A')
    ax.plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']), color='black', linestyle=':' , alpha=1, linewidth=1, label='Influenza B')
    ax.plot(out['date'], 7*((out['H1_inc'] + out['H2_inc']).sum(dim=['age_group', 'location'])), color='black', alpha=1, linewidth=3, label = 'Influenza A + B')
    ax.legend()
    # no spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # months only
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # no ticks
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'frame-{i}.png', dpi=150)
    #plt.show()
    plt.close()
