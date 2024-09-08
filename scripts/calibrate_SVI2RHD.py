"""
This script simulates an age-stratified, spatially-explicit SIR model with a vaccine for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import random
import corner
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime as datetime
from datetime import date

# influenza model
from influenza_USA.SVIR.utils import initialise_SVI2RHD

# pySODM packages
from pySODM.optimization import pso, nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, ll_negative_binomial, ll_normal, ll_lognormal
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

##########################
## Calibration settings ##
##########################

# Frequentist
n_pso = 100                                         # Number of PSO iterations
multiplier_pso = 10                                 # PSO swarm size
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count())) # Retrieve CPU count

# Bayesian
identifier = 'normal_nodaytype'                     # Give any output of this script an ID
n_mcmc = 500                                        # Number of MCMC iterations
multiplier_mcmc = 10                                 # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 10                                        # Print diagnostics every print_n iterations
discard = 100                                        # Discard first `discard` iterations as burn-in
thin = 5                                            # Thinning factor emcee chains
n = 100                                              # Repeated simulations used in visualisations

###############
## Load data ##
###############

# load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),'../data/raw/cases/2017_2018_Flu.csv'), index_col=0, parse_dates=True)
# convert to daily incidence
df /= 7
# pySODM convention: use 'date' as temporal index
df.index.rename('date', inplace=True)
# determine data start and enddate
start_calibration = datetime(2017, 8, 1)
end_calibration = df.index.max()

#################
## Setup model ##
#################

# model settings
sr = 'states'                       # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                         # age resolution: 'collapsed' or 'full'
dd = False                           # vary contact matrix by daytype
stoch = False                       # ODE vs. tau-leap

# initialise model
model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, distinguish_daytype=dd, stochastic=stoch, start_sim=start_calibration)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## PSO/Nelder-Mead ##
    #####################

    # define dataset, states to match and observational model
    data=[df['Weekly_Cases'], df['Weekly_Hosp'], df['Weekly_Deaths'], df['Weekly_Hosp'][slice(datetime(2018,1,10),datetime(2018,2,10))], df['Weekly_Deaths'][slice(datetime(2018,1,10),datetime(2018,2,10))]]
    weights = [1/max(df['Weekly_Cases']), 1/max(df['Weekly_Hosp']), 1/max(df['Weekly_Deaths']), 1/max(df['Weekly_Hosp']), 1/max(df['Weekly_Deaths'])]
    states = ['I_inc', 'H_inc', 'D_inc', 'H_inc', 'D_inc']
    log_likelihood_fnc = [ll_normal, ll_normal, ll_normal, ll_normal, ll_normal] # poisson works about equally well --> just make sure to weigh datasets appropriately!
    log_likelihood_fnc_args = [
        (0.05/3)*df['Weekly_Cases'].values,     # 95% CI +/- 10 % 
        (0.05/3)*df['Weekly_Hosp'].values,      # 95% CI +/- 5 %
        (0.05/3)*df['Weekly_Deaths'].values,    # 95% CI +/- 5 %
        (0.05/3)*df['Weekly_Deaths'][slice(datetime(2018,1,10),datetime(2018,2,10))].values,    # 95% CI +/- 5 %
        (0.05/3)*df['Weekly_Deaths'][slice(datetime(2018,1,10),datetime(2018,2,10))].values,    # 95% CI +/- 5 %    
    ]
    #log_likelihood_fnc = [ll_poisson, ll_poisson, ll_poisson, ll_poisson, ll_poisson] # poisson works about equally well --> just make sure to weigh datasets appropriately!
    #log_likelihood_fnc_args = [[],[],[],[],[]]
    # calibated parameters and bounds
    pars = ['beta', 'rho_h', 'rho_d', 'asc_case']
    labels = [r'$\beta$', r'$\rho_h$', r'$\rho_d$', r'$\alpha_{case}$']
    bounds = [(0.01,0.10), (0.001,0.1), (0.001,1), (0.001,0.1)]
    # Setup objective function (no priors --> uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,
                                                   start_sim=start_calibration,weights=weights,labels=labels)
    # Extract expanded bounds and labels
    expanded_labels = objective_function.expanded_labels 
    expanded_bounds = objective_function.expanded_bounds                                   
    # Nelder-mead
    theta = [0.0252, 0.0023, 0.045, 0.0018] # With varying datypes + U-shaped severity --> very good fit    
    theta = [0.024, 0.0025, 0.04, 0.0018] # Without varying datypes + U-shaped severity --> very good fit
    #step = len(expanded_bounds)*[0.05,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_calibration])
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

    # Visualize
    fig, axs = plt.subplots(2,1,figsize=(8.3,11.7/3))

    axs[0].plot(out['date'], 7*out['H_inc'].sum(dim=['location']).sel({'age_group': '[0, 5)'}), color='black', alpha=1, linewidth=2, label='0-5')
    axs[0].plot(out['date'], 7*out['H_inc'].sum(dim=['location']).sel({'age_group': '[5, 18)'}), color='blue', alpha=1, linewidth=2, label='5-18')
    axs[0].plot(out['date'], 7*out['H_inc'].sum(dim=['location']).sel({'age_group': '[18, 50)'}), color='red', alpha=1, linewidth=2, label='18-50')
    axs[0].plot(out['date'], 7*out['H_inc'].sum(dim=['location']).sel({'age_group': '[50, 65)'}), color='orange', alpha=1, linewidth=2, label='50-65')
    axs[0].plot(out['date'], 7*out['H_inc'].sum(dim=['location']).sel({'age_group': '[65, 100)'}), color='green', alpha=1, linewidth=2, label='65-100')
    axs[0].legend()

    demo = np.array([18608139, 54722401, 141598551, 63172279, 60019216])
    axs[1].bar(out.coords['age_group'].values, 7*out['H_inc'].sum(dim=['location']).cumsum(dim='date').isel(date=-1)/demo * 100000, color='black', alpha=0.6, linewidth=2)
    axs[1].set_title('Cumulative hospitalisations')

    plt.tight_layout()
    plt.show()
    plt.close()

    ##########
    ## MCMC ##
    ##########

    # Variables
    samples_path='sampler_output/'
    fig_path='sampler_output/'
    run_date = str(date.today())
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=expanded_bounds)
    # Write some usefull settings to a pickle file (no pd.Timestamps or np.arrays allowed!)
    settings={'start_calibration': start_calibration.strftime('%d-%m-%Y'), 'end_calibration': end_calibration.strftime('%d-%m-%Y'),
              'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': expanded_labels}
    # Sample n_mcmc iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,  objective_function_kwargs={'simulation_kwargs': {'warmup': 0}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings)                                                                               
    # Generate a sample dictionary and save it as .json for long-term storage
    # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard, thin=thin)
    # Look at the resulting distributions in a cornerplot
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=discard, thin=2, flat=True), labels=expanded_labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################
 
    # Define draw function
    def draw_fcn(parameters, samples):
        # Sample model parameters
        idx, parameters['beta'] = random.choice(list(enumerate(samples['beta'])))
        parameters['rho_h'] = samples['rho_h'][idx]
        parameters['rho_d'] = samples['rho_d'][idx]
        parameters['asc_case'] = samples['asc_case'][idx]
        return parameters
    # Simulate model
    out = model.sim([start_calibration, end_calibration], N=n,
                    draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=1)
    # Add poisson observation noise
    #out_noise = add_poisson_noise(out)

    # Visualize
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8.3,11.7/2))
    ## Cases
    axs[0].scatter(df.index.unique().values, 7*df['Weekly_Cases'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    axs[0].plot(out['date'], 7*out['I_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    axs[0].fill_between(out['date'], 7*out['I_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['I_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    
    axs[0].set_title('Cases')
    ## Hospitalisations
    axs[1].scatter(df.index.unique().values, 7*df['Weekly_Hosp'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    axs[1].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    axs[1].fill_between(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    axs[1].set_title('Hospitalisations')
    ## Deaths
    axs[2].scatter(df.index.unique().values, 7*df['Weekly_Deaths'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    axs[2].plot(out['date'], 7*out['D_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    axs[2].fill_between(out['date'], 7*out['D_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['D_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
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