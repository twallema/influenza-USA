"""
This script simulates an age-stratified, spatially-explicit SIR model for Belgium using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import matplotlib.pyplot as plt
from influenza_USA.models.SIR import ODE_SIR
from influenza_USA.models.utils import name2fips, \
                                            construct_coordinates_dictionary, \
                                                get_contact_matrix, \
                                                    get_mobility_matrix, \
                                                        construct_initial_susceptible, \
                                                            construct_initial_infected
                  
#################
## Setup model ##
#################

# spatial resolution
sr = 'states'

# coordinates
coordinates = construct_coordinates_dictionary(spatial_resolution=sr)

# parameters
params = {'beta': 0.03,                                                                             # infectivity (-)
          'gamma': 5,                                                                               # duration of infection (d)
          'f_v': 0.5,                                                                               # fraction of total contacts on visited patch
          'N': get_contact_matrix(),                                                                # contact matrix
          'M': get_mobility_matrix(dataset='cellphone_03092020', spatial_resolution=sr),            # origin-destination mobility matrix
          }

# initial states
I0 = construct_initial_infected(seed_loc=('alabama',''), n=1, agedist='demographic', spatial_resolution=sr)
S0 = construct_initial_susceptible(I0, spatial_resolution=sr)
init_states = {'S': S0,
               'I': I0
               }

# initialize model
model = ODE_SIR(states=init_states, parameters=params, coordinates=coordinates)

import time

####################
## simulate model ##
####################

t0 = time.time()
out = model.sim(120, method='RK23', rtol=1e-3)
t1 = time.time()
print(f'elapsed: {t1-t0} s')

#######################
## visualise results ##
#######################

if sr == 'states':

    fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/2))

    ax[0].set_title('Overall')
    ax[0].plot(out['time'], out['S'].sum(dim=['age_group', 'location']), color='green', alpha=0.8, label='S')
    ax[0].plot(out['time'], out['I'].sum(dim=['age_group', 'location']), color='red', alpha=0.8, label='I')
    ax[0].plot(out['time'], out['R'].sum(dim=['age_group', 'location']), color='black', alpha=0.8, label='R')
    ax[0].legend(loc=1, framealpha=1)

    ax[1].set_title('Infected by spatial patch (assorted)')
    ax[1].plot(out['time'], out['I'].sum(dim='age_group').sel({'location': name2fips('Florida')}), linestyle = '-', color='red', alpha=0.8, label='Florida')
    ax[1].plot(out['time'], out['I'].sum(dim='age_group').sel({'location': name2fips('Alaska')}), linestyle = ':', color='red', alpha=0.8, label='Alaska')
    ax[1].plot(out['time'], out['I'].sum(dim='age_group').sel({'location': name2fips('Maryland')}), linestyle = '--', color='red', alpha=0.8, label='Maryland')
    ax[1].legend(loc=1, framealpha=1)

    ax[2].set_title('Infected by age group (Maryland)')
    ax[2].plot(out['time'], out['I'].sel({'age_group': '[0, 5)', 'location': name2fips('Maryland')}), linestyle = '-', color='red', alpha=0.8, label='0-5')
    ax[2].plot(out['time'], out['I'].sel({'age_group': '[5, 18)', 'location': name2fips('Maryland')}), linestyle = ':', color='red', alpha=0.8, label='5-18')
    ax[2].plot(out['time'], out['I'].sel({'age_group': '[18, 50)', 'location': name2fips('Maryland')}), linestyle = '--', color='green', alpha=0.8, label='18-50')
    ax[2].plot(out['time'], out['I'].sel({'age_group': '[50, 65)', 'location': name2fips('Maryland')}), linestyle = '-.', color='green', alpha=0.8, label='50-65')
    ax[2].plot(out['time'], out['I'].sel({'age_group': '[65, 100)', 'location': name2fips('Maryland')}), linestyle = '-', color='black', alpha=0.8, label='65-100')
    ax[2].legend(loc=1, framealpha=1)

    plt.tight_layout()
    plt.show()
    plt.close()