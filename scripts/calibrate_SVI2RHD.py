"""
This script simulates an age-stratified, spatially-explicit SIR model with a vaccine for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime as datetime

# influenza model
from influenza_USA.SVIR.utils import initialise_SVI2RHD

# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson
from pySODM.optimization.utils import add_poisson_noise, assign_theta

##########################
## Calibration settings ##
##########################

n_pso = 100                                      # Number of PSO iterations
multiplier_pso = 10                              # PSO swarm size
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())) # Retrieve CPU count

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
start_sim = datetime(2017, 6, 30)   # simulation start
end_sim = datetime(2018, 7, 1)      # simulation end

# initialise model
model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, distinguish_daytype=dd, stochastic=stoch, start_sim=start_sim)

###############
## Load data ##
###############

# load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),'../data/raw/cases/2017_2018_Flu.csv'), index_col=0, parse_dates=True)
# prior to reporting --> force zero: #TODO: ability to choose startdate simulation in pySODM!!
df.fillna(0, inplace=True)
# convert to daily incidence
df /= 7
# pySODM convention: use 'date' as temporal index
df.index.rename('date', inplace=True)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # define dataset, states to match and observational model
    data=[df['Weekly_Cases'], df['Weekly_Hosp'], df['Weekly_Deaths']]
    states = ['I_inc', 'H_inc', 'D_inc']
    log_likelihood_fnc = [ll_poisson, ll_poisson, ll_poisson]
    log_likelihood_fnc_args = [[],[],[]]
    # calibated parameters and bounds
    pars = ['beta', 'T_s', 'T_h', 'rho_h', 'T_d', 'rho_d', 'asc_case']
    labels = ['$\\beta$', '$T_s$',  '$T_h$', '$\rho_h$',  '$T_d$', '$\rho_d$', '$\alpha_{case}$']
    bounds = [(0.01,0.10), (0,365), (0,10), (0,0.1), (0,10), (0,1), (0,0.1)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,labels=labels)
    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds                                   
    # Nelder-mead
    theta = [0.030, 165, 2.5, 0.002, 4.71, 0.05, 0.001]
    #theta = [2.54427113e-02, 6.42023924e+01, 2.57300199e+00, 1.95412054e-03, 5.53401936e+00, 4.93271196e-02, 1.23325157e-03] ## LL: 863
    #step = len(expanded_bounds)*[0.05,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_sim, end_sim])
    # Add poisson obervational noise
    out = add_poisson_noise(out)
    # Visualize
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8.3,11.7/2))
    ## Cases
    axs[0].scatter(df.index.unique().values, 7*df['Weekly_Cases'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    axs[0].plot(out['date'], 7*out['I_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    axs[0].set_title('Cases')
    ## Hospitalisations
    axs[1].scatter(df.index.unique().values, 7*df['Weekly_Hosp'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    axs[1].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    axs[1].set_title('Hospitalisations')
    ## Deaths
    axs[2].scatter(df.index.unique().values, 7*df['Weekly_Deaths'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    axs[2].plot(out['date'], 7*out['D_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    axs[2].set_title('Deaths')
    ## Formatting
    axs[2].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in axs[2].get_xticklabels():
        tick.set_rotation(30)
    axs[2].grid(False)
    ## Print to screen
    plt.tight_layout()
    plt.show()
    plt.close()
