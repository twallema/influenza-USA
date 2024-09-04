"""
This script simulates an age-stratified, spatially-explicit SIR model with a vaccine for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import matplotlib.pyplot as plt
from datetime import datetime as datetime
from influenza_USA.SVIR.utils import initialise_SVI2RHD, name2fips

#################
## Setup model ##
#################

# model settings
sr = 'states'                       # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                         # age resolution: 'collapsed' or 'full'
dd = True                           # vary contact matrix by daytype
stoch = False                       # ODE vs. tau-leap

# simulation settings
N = 1                               # number of stochastic realisations
processes = 1                       # number of cores to use
start_sim = datetime(2024, 6, 30)   # simulation start
end_sim = datetime(2025, 7, 1)      # simulation end

# initialise model
model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, distinguish_daytype=dd, stochastic=stoch, start_sim=start_sim)

####################
## simulate model ##
####################

import time
t0 = time.time()
out = model.sim(time=[start_sim, end_sim])
t1 = time.time()
print(f'elapsed: {t1-t0} s')

################################
## visualise overall dynamics ##
################################

if ((sr == 'states') & (ar == 'full')):
    if stoch == True:
            
        fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/2))

        out['I_sum'] = out['I'] + out['Iv']

        ax[0].set_title('Population immunity')
        for i in range(N):
            ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']).isel(draws=i), alpha=0.3, color='green', label='S')
            ax[0].plot(out['date'], out['V'].sum(dim=['age_group', 'location']).isel(draws=i), alpha=0.3, linestyle='--', color='green', label='V')
            ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']).isel(draws=i), alpha=0.3, color='red', label='I')
            ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']).isel(draws=i), alpha=0.3, color='black', label='R')
            if i == 0:
                ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Ascertained incidences (USA)')
        for i in range(N):
            ax[1].plot(out['date'], out['I_inc'].sum(dim=['age_group', 'location']).isel(draws=i), alpha=0.3, color='black', label='Cases')
            ax[1].plot(out['date'], out['H_inc'].sum(dim=['age_group', 'location']).isel(draws=i), alpha=0.3, color='orange', label='Hospitalisations')
            ax[1].plot(out['date'], out['D_inc'].sum(dim=['age_group', 'location']).isel(draws=i), alpha=0.3, color='red', label='Deaths')
            if i ==0:
                ax[1].legend(loc=1, framealpha=1)

        ax[2].set_title('Infected prevalence by age group (Maryland)')
        for i in range(N):
            ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[0, 5)', 'location': name2fips('Maryland')}).isel(draws=i), alpha=0.3, linestyle = '-', color='red', label='0-5')
            ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[5, 18)', 'location': name2fips('Maryland')}).isel(draws=i), alpha=0.3, linestyle = ':', color='red', label='5-18')
            ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[18, 50)', 'location': name2fips('Maryland')}).isel(draws=i), alpha=0.3, linestyle = '--', color='green', label='18-50')
            ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[50, 65)', 'location': name2fips('Maryland')}).isel(draws=i), alpha=0.3, linestyle = '-.', color='green', label='50-65')
            ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[65, 100)', 'location': name2fips('Maryland')}).isel(draws=i), alpha=0.3, linestyle = '-', color='black', label='65-100')
            if i ==0:
                ax[2].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()

    else:

        fig,ax=plt.subplots(nrows=3, figsize=(8.3,11.7/2))

        out['I_sum'] = out['I'] + out['Iv']

        ax[0].set_title('Population immunity')
        ax[0].plot(out['date'], out['S'].sum(dim=['age_group', 'location']), color='green', label='S')
        ax[0].plot(out['date'], out['V'].sum(dim=['age_group', 'location']), linestyle='--', color='green', label='V')
        ax[0].plot(out['date'], out['I'].sum(dim=['age_group', 'location']), color='red', label='I')
        ax[0].plot(out['date'], out['R'].sum(dim=['age_group', 'location']), color='black', label='R')
        ax[0].legend(loc=1, framealpha=1)

        ax[1].set_title('Ascertained incidences (USA)')
        ax[1].plot(out['date'], out['I_inc'].sum(dim=['age_group', 'location']), color='black', label='Cases')
        ax[1].plot(out['date'], out['H_inc'].sum(dim=['age_group', 'location']), color='orange', label='Hospitalisations')
        ax[1].plot(out['date'], out['D_inc'].sum(dim=['age_group', 'location']), color='red', label='Deaths')
        ax[1].legend(loc=1, framealpha=1)

        ax[2].set_title('Infected prevalence by age group (Maryland)')
        ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[0, 5)', 'location': name2fips('Maryland')}), linestyle = '-', color='red', label='0-5')
        ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[5, 18)', 'location': name2fips('Maryland')}), linestyle = ':', color='red', label='5-18')
        ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[18, 50)', 'location': name2fips('Maryland')}), linestyle = '--', color='green', label='18-50')
        ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[50, 65)', 'location': name2fips('Maryland')}), linestyle = '-.', color='green', label='50-65')
        ax[2].plot(out['date'], out['I_sum'].sel({'age_group': '[65, 100)', 'location': name2fips('Maryland')}), linestyle = '-', color='black', label='65-100')
        ax[2].legend(loc=1, framealpha=1)

        plt.tight_layout()
        plt.show()
        plt.close()

else:
    raise ValueError("script only works for spatial resolution 'states' and age resolution 'full'")

