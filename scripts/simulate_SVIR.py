"""
This script simulates an age-stratified, spatially-explicit SIR model with a vaccine for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from influenza_USA.SVIR.utils import name2fips, \
                                            construct_coordinates_dictionary, \
                                                get_contact_matrix, \
                                                    get_mobility_matrix, \
                                                        get_vaccination_data, \
                                                            construct_initial_susceptible, \
                                                                construct_initial_infected, \
                                                                    load_initial_condition
    
#################
## Setup model ##
#################

# settings
sr = 'states'                       # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                         # age resolution: 'collapsed' or 'full'
distinguish_daytype = True          # vary contact matrix by daytype
stochastic = True                   # ODE vs. tau-leap
N = 20                              # number of stochastic realisations
processes = 20
start_sim = '2024-06-30'            # simulation start
end_sim = '2025-07-01'              # simulation end

# model
if stochastic:
    from influenza_USA.SVIR.model import TL_SVIR as SVIR
else:
    from influenza_USA.SVIR.model import ODE_SVIR as SVIR

# coordinates
coordinates = construct_coordinates_dictionary(spatial_resolution=sr, age_resolution=ar)

# parameters
params = {'beta': 0.023,                                                                                                        # infectivity (-)
          'gamma': 5,                                                                                                           # duration of infection (d)
          'f_v': 0.5,                                                                                                           # fraction of total contacts on visited patch
          'e_vacc': 0.4,                                                                                                        # vaccine efficacy
          'r_vacc': np.ones(shape=(len(coordinates['age_group']), len(coordinates['location'])),dtype=np.float64),              # vaccination rate
          'vaccine_rate_modifier': 1.0,                                                                                         # modify vaccination rate
          'N': tf.convert_to_tensor(get_contact_matrix(daytype='all', age_resolution=ar), dtype=float),                         # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
          'M': tf.convert_to_tensor(get_mobility_matrix(dataset='cellphone_03092020', spatial_resolution=sr), dtype=float)      # origin-destination mobility matrix
        }

# initial states
ic = load_initial_condition(season='17-18')
total_population = construct_initial_susceptible(spatial_resolution=sr, age_resolution=ar)
init_states = {}
for k,v in ic.items():
    init_states[k] = tf.convert_to_tensor(v * total_population)

del init_states['H']
del init_states['D']
del init_states['Iv']

# time-dependencies
TDPFs = {}
## contacts
if distinguish_daytype:
    from influenza_USA.SVIR.TDPF import make_contact_function
    TDPFs['N'] = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=ar),
                                             get_contact_matrix(daytype='week_holiday', age_resolution=ar),
                                             get_contact_matrix(daytype='weekend', age_resolution=ar)).contact_function
## vaccines
from influenza_USA.SVIR.TDPF import make_vaccination_function
TDPFs['r_vacc'] = make_vaccination_function(get_vaccination_data()).vaccination_function

# initialise model
model = SVIR(states=init_states, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)

####################
## simulate model ##
####################

import time
t0 = time.time()
out = model.sim(time=[start_sim, end_sim], N=N, processes=processes, tau=0.2)
t1 = time.time()
print(f'elapsed: {t1-t0} s')

################################
## visualise overall dynamics ##
################################

if ((sr == 'states') & (ar == 'full')):
    if stochastic == True:
            
        fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/2))
        ax[0].set_title('Overall')
        for i in range(N):
            ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), color='green', alpha=0.3, label='S')
            ax[0].plot(out['date'], out['V'].sum(dim=['age_group', 'location']).isel(draws=i), linestyle='--', color='green', alpha=0.3, label='V')
            ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']).isel(draws=i), color='red', alpha=0.3, label='I')
            ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']).isel(draws=i), color='black', alpha=0.3, label='R')
            if i == 0:
                ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Infected by spatial patch')
        for i in range(N):
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Alabama')}).isel(draws=i), linestyle = '-', color='black', alpha=0.3, label='Alabama')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Florida')}).isel(draws=i), linestyle = '-.', color='red', alpha=0.3, label='Florida')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Maryland')}).isel(draws=i), linestyle = '--', color='green', alpha=0.3, label='Maryland')
            if i ==0:
                ax[1].legend(loc=1, framealpha=1)

        ax[2].set_title('Infected by age group (Maryland)')
        for i in range(N):
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[0, 5)', 'location': name2fips('Maryland')}).isel(draws=i), linestyle = '-', color='red', alpha=0.3, label='0-5')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[5, 18)', 'location': name2fips('Maryland')}).isel(draws=i), linestyle = ':', color='red', alpha=0.3, label='5-18')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[18, 50)', 'location': name2fips('Maryland')}).isel(draws=i), linestyle = '--', color='green', alpha=0.3, label='18-50')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[50, 65)', 'location': name2fips('Maryland')}).isel(draws=i), linestyle = '-.', color='green', alpha=0.3, label='50-65')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[65, 100)', 'location': name2fips('Maryland')}).isel(draws=i), linestyle = '-', color='black', alpha=0.3, label='65-100')
            if i ==0:
                ax[2].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()

    else:

        fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/2))
        ax[0].set_title('Overall')
        ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']), color='green', label='S')
        ax[0].plot(out['date'], out['V'].sum(dim=['age_group', 'location']), linestyle='--', color='green', label='V')
        ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']), color='red', label='I')
        ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']), color='black', label='R')
        ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Infected by spatial patch')
        ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Alabama')}), linestyle = '-', color='black', label='Alabama')
        ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Florida')}), linestyle = '-.', color='red', label='Florida')
        ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Maryland')}), linestyle = '--', color='green', label='Maryland')
        ax[1].legend(loc=1, framealpha=1)

        ax[2].set_title('Infected by age group (Maryland)')
        ax[2].plot(out['date'], out['I'].sel({'age_group': '[0, 5)', 'location': name2fips('Maryland')}), linestyle = '-', color='red', label='0-5')
        ax[2].plot(out['date'], out['I'].sel({'age_group': '[5, 18)', 'location': name2fips('Maryland')}), linestyle = ':', color='red', label='5-18')
        ax[2].plot(out['date'], out['I'].sel({'age_group': '[18, 50)', 'location': name2fips('Maryland')}), linestyle = '--', color='green', label='18-50')
        ax[2].plot(out['date'], out['I'].sel({'age_group': '[50, 65)', 'location': name2fips('Maryland')}), linestyle = '-.', color='green', label='50-65')
        ax[2].plot(out['date'], out['I'].sel({'age_group': '[65, 100)', 'location': name2fips('Maryland')}), linestyle = '-', color='black', label='65-100')
        ax[2].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()

else:
    raise ValueError("script only works for spatial resolution 'states' and age resolution 'full'")

