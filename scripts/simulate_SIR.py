"""
This script simulates an age-stratified, spatially-explicit SIR model for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import tensorflow as tf
import matplotlib.pyplot as plt
from influenza_USA.SIR.utils import name2fips, \
                                            construct_coordinates_dictionary, \
                                                get_contact_matrix, \
                                                    get_mobility_matrix, \
                                                        construct_initial_susceptible, \
                                                            construct_initial_infected
    
#################
## Setup model ##
#################

# settings
sr = 'states'                       # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                         # age resolution: 'collapsed' or 'full'
distinguish_daytype = True          # vary contact matrix by daytype
stochastic = True                   # ODE vs. tau-leap
N = 30

# model
if stochastic:
    from influenza_USA.SIR.model import TL_SIR as SIR
else:
    from influenza_USA.SIR.model import ODE_SIR as SIR

# coordinates
coordinates = construct_coordinates_dictionary(spatial_resolution=sr, age_resolution=ar)

# parameters
params = {'beta': 0.015,                                                                                                      # infectivity (-)
        'gamma': 5,                                                                                                           # duration of infection (d)
        'f_v': 0.5,                                                                                                           # fraction of total contacts on visited patch
        'N': tf.convert_to_tensor(get_contact_matrix(daytype='all', age_resolution=ar), dtype=float),                         # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
        'M': tf.convert_to_tensor(get_mobility_matrix(dataset='cellphone_03092020', spatial_resolution=sr), dtype=float)      # origin-destination mobility matrix
        }

# initial states
I0 = construct_initial_infected(seed_loc=('alabama',''), n=10, agedist='demographic', spatial_resolution=sr, age_resolution=ar)
S0 = construct_initial_susceptible(I0, spatial_resolution=sr, age_resolution=ar)
init_states = {'S': tf.convert_to_tensor(S0, dtype=float),
               'I': tf.convert_to_tensor(I0, dtype=float)
                }

# time-dependencies
if distinguish_daytype:
    from influenza_USA.SIR.TDPF import make_contact_function
    contact_function = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=ar),
                                             get_contact_matrix(daytype='week_holiday', age_resolution=ar),
                                             get_contact_matrix(daytype='weekend', age_resolution=ar)).contact_function
    # initialize model
    model = SIR(states=init_states, parameters=params, coordinates=coordinates, time_dependent_parameters={'N': contact_function})
else:
    model = SIR(states=init_states, parameters=params, coordinates=coordinates)

####################
## simulate model ##
####################

import time
t0 = time.time()
out = model.sim(time=['2024-08-01', '2025-04-01'], tau=1, N=N, processes=15)
t1 = time.time()
print(f'elapsed: {t1-t0} s')

###################################################
## visualise six possible age/space combinations ##
###################################################

if sr == 'collapsed':

    if ar == 'collapsed':

        fig,ax=plt.subplots(figsize=(8.3,11.7/6))
        ax.set_title('Overall')
        for i in range(N):
            ax.plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), color='green', alpha=0.3, label='S')
            ax.plot(out['date'], out['I'].sum(dim=['age_group', 'location']).isel(draws=i), color='red', alpha=0.3, label='I')
            ax.plot(out['date'], out['R'].sum(dim=['age_group', 'location']).isel(draws=i), color='black', alpha=0.3, label='R')
            if i == 0:
                ax.legend(loc=1, framealpha=1)
        plt.tight_layout()
        plt.show()
        plt.close()

    elif ar == 'full':

        fig,ax=plt.subplots(nrows=2, figsize=(8.3,11.7/3))
        ax[0].set_title('Overall')
        for i in range(N):
            ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), color='green', alpha=0.3, label='S')
            ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']).isel(draws=i), color='red', alpha=0.3, label='I')
            ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']).isel(draws=i), color='black', alpha=0.3, label='R')
            if i == 0:
                ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Infected by age group (USA)')
        for i in range(N):
            ax[1].plot(out['date'], out['I'].sel({'age_group': '[0, 5)'}).sum(dim='location').isel(draws=i), linestyle = '-', color='red', alpha=0.3, label='0-5')
            ax[1].plot(out['date'], out['I'].sel({'age_group': '[5, 18)'}).sum(dim='location').isel(draws=i), linestyle = ':', color='red', alpha=0.3, label='5-18')
            ax[1].plot(out['date'], out['I'].sel({'age_group': '[18, 50)'}).sum(dim='location').isel(draws=i), linestyle = '--', color='green', alpha=0.3, label='18-50')
            ax[1].plot(out['date'], out['I'].sel({'age_group': '[50, 65)'}).sum(dim='location').isel(draws=i), linestyle = '-.', color='green', alpha=0.3, label='50-65')
            ax[1].plot(out['date'], out['I'].sel({'age_group': '[65, 100)'}).sum(dim='location').isel(draws=i), linestyle = '-', color='black', alpha=0.3, label='65-100')
            if i == 0:
                ax[1].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()

elif sr == 'states':

    if ar == 'collapsed':

        fig,ax=plt.subplots(nrows=2, figsize=(8.3,11.7/3))
        ax[0].set_title('Overall')
        for i in range(N):
            ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), color='green', alpha=0.3, label='S')
            ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']).isel(draws=i), color='red', alpha=0.3, label='I')
            ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']).isel(draws=i), color='black', alpha=0.3, label='R')
            if i == 0:
                ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Infected by spatial patch')
        for i in range(N):
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Alabama')}).isel(draws=i), linestyle = '-', color='black', alpha=0.3, label='Alabama')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Florida')}).isel(draws=i), linestyle = '-.', color='red', alpha=0.3, label='Florida')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Maryland')}).isel(draws=i), linestyle = '--', color='green', alpha=0.3, label='Maryland')
            if i == 0:
                ax[1].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()

    elif ar == 'full':

        fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/2))
        ax[0].set_title('Overall')
        for i in range(N):
            ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), color='green', alpha=0.3, label='S')
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

elif sr == 'counties':

    if ar == 'collapsed':
        fig,ax=plt.subplots(nrows=2, figsize=(8.3,11.7/3))

        ax[0].set_title('Overall')
        for i in range(N):
            ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), color='green', alpha=0.3, label='S')
            ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']).isel(draws=i), color='red', alpha=0.3, label='I')
            ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']).isel(draws=i), color='black', alpha=0.3, label='R')
            if i ==0:
                ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Infected by spatial patch')
        for i in range(N):
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Alabama', 'Montgomery County')}).isel(draws=i), linestyle = '-', color='red', alpha=0.3, label='Montgomery County (AL)')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Florida', 'Palm Beach County')}).isel(draws=i), linestyle = '-', color='red', alpha=0.3, label='Palm Beach (FL)')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Maryland', 'Baltimore City')}).isel(draws=i), linestyle = '--', color='red', alpha=0.3, label='Baltimore City (MD)')
            if i ==0:
                ax[1].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()

    elif ar == 'full':

        fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/2))

        ax[0].set_title('Overall')
        for i in range(N):
            ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), color='green', alpha=0.3, label='S')
            ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']).isel(draws=i), color='red', alpha=0.3, label='I')
            ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']).isel(draws=i), color='black', alpha=0.3, label='R')
            if i ==0:
                ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Infected by spatial patch')
        for i in range(N):
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Alabama', 'Montgomery County')}).isel(draws=i), linestyle = '-', color='red', alpha=0.3, label='Montgomery County (AL)')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Florida', 'Palm Beach County')}).isel(draws=i), linestyle = '-', color='red', alpha=0.3, label='Palm Beach (FL)')
            ax[1].plot(out['date'], out['I'].sum(dim='age_group').sel({'location': name2fips('Maryland', 'Baltimore City')}).isel(draws=i), linestyle = '--', color='red', alpha=0.3, label='Baltimore City (MD)')
            if i ==0:
                ax[1].legend(loc=1, framealpha=1)

        ax[2].set_title('Infected by age group (Baltimore City)')
        for i in range(N):
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[0, 5)', 'location': name2fips('Maryland', 'Baltimore City')}).isel(draws=i), linestyle = '-', color='red', alpha=0.3, label='0-5')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[5, 18)', 'location': name2fips('Maryland', 'Baltimore City')}).isel(draws=i), linestyle = ':', color='red', alpha=0.3, label='5-18')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[18, 50)', 'location': name2fips('Maryland', 'Baltimore City')}).isel(draws=i), linestyle = '--', color='green', alpha=0.3, label='18-50')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[50, 65)', 'location': name2fips('Maryland', 'Baltimore City')}).isel(draws=i), linestyle = '-.', color='green', alpha=0.3, label='50-65')
            ax[2].plot(out['date'], out['I'].sel({'age_group': '[65, 100)', 'location': name2fips('Maryland', 'Baltimore City')}).isel(draws=i), linestyle = '-', color='black', alpha=0.3, label='65-100')
            if i==0:
                ax[2].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()