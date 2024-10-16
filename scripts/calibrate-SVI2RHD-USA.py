"""
This script simulates an age-stratified, spatially-explicit SVI2HRD model for the USA using pySODM
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
from influenza_USA.SVIR.utils import initialise_SVI2RHD, construct_initial_susceptible # influenza model
# pySODM packages
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

##############
## Settings ##
##############

# model settings
season = '2017-2018'                # season: '2017-2018' or '2019-2020'
waning = 'no_waning'                # 'no_waning' vs. 'waning_180'
sr = 'states'                       # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                         # age resolution: 'collapsed' or 'full'
dd = False                          # vary contact matrix by daytype
stoch = False                       # ODE vs. tau-leap

# optimization
## frequentist
n_pso = 50                                                         # Number of PSO iterations
multiplier_pso = 10                                                 # PSO swarm size
## bayesian
identifier = 'USA'                                                 # Use waning as identifier of script output
n_mcmc = 300                                                        # Number of MCMC iterations
multiplier_mcmc = 5                                                # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 10                                                        # Print diagnostics every `print_n`` iterations
discard = 100                                                       # Discard first `discard` iterations as burn-in
thin = 5                                                            # Thinning factor emcee chains
n = 100                                                             # Repeated simulations used in visualisations
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()))    # Retrieve CPU count

# dates
## season specific
if season == '2017-2018':
    start_calibration = datetime(2017, 8, 1)
    end_calibration = None
    start_slice = datetime(2018, 1, 10)
    end_slice = datetime(2018, 2, 10)
elif season == '2019-2020':
    start_calibration = datetime(2019, 8, 1)
    end_calibration = datetime(2020, 3, 22)
    start_slice = datetime(2020, 1, 1)
    end_slice = datetime(2020, 3, 1)

###############
## Load data ##
###############

# load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../data/raw/cases/{season}_Flu.csv'), index_col=0, parse_dates=True)
# convert to daily incidence
df /= 7
# pySODM convention: use 'date' as temporal index
df.index.rename('date', inplace=True)
# slice data until end
df = df.loc[slice(None, end_calibration)]
df_peak = df.loc[slice(start_slice, end_slice)]
df_bump = df.loc[slice(datetime(2017, 8, 1), datetime(2017, 8, 16))]
# replace `end_calibration` None --> datetime
end_calibration = df.index.max()

#################
## Setup model ##
#################

model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, season=season, distinguish_daytype=dd, stochastic=stoch, start_sim=start_calibration)

# set up right waning parameters
if waning == 'no_waning':
    model.parameters['e_i'] = 0.2
    model.parameters['e_h'] = 0.5
    model.parameters['T_v'] = 10*365
elif waning == 'waning_180':
    model.parameters['e_i'] = 0.2
    model.parameters['e_h'] = 0.75
    model.parameters['T_v'] = 365/2

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    ##################################
    ## Set up posterior probability ##
    ##################################

    # define datasets
    data=[df['Weekly_Cases'], df['Weekly_Hosp'], df['Weekly_Deaths'],                                                                                                                       # all data
          df_peak['Weekly_Hosp'], df_peak['Weekly_Deaths'],
          df_bump['Weekly_Hosp'], df_bump['Weekly_Deaths']]   # hospital/death peak counted double
    # use maximum value in dataset as weight
    weights = [1/max(df['Weekly_Cases']), 1/max(df['Weekly_Hosp']), 1/max(df['Weekly_Deaths']),
               5/max(df['Weekly_Hosp']), 5/max(df['Weekly_Deaths']),
               100/max(df['Weekly_Hosp']), 100/max(df['Weekly_Deaths'])]
    # states to match with datasets
    states = ['I_inc', 'H_inc', 'D_inc', 'H_inc', 'D_inc', 'H_inc', 'D_inc']
    # log likelihood function + arguments
    log_likelihood_fnc = [ll_poisson, ll_poisson, ll_poisson, ll_poisson, ll_poisson, ll_poisson, ll_poisson] 
    log_likelihood_fnc_args = [[],[],[],[],[],[],[]]
    # parameters to calibrate and bounds
    pars = ['beta', 'rho_h', 'rho_d', 'asc_case', 'f_I', 'f_R']
    labels = [r'$\beta$', r'$\rho_h$', r'$\rho_d$', r'$\alpha_{case}$', r'$f_I$', r'$f_R$']
    bounds = [(0.01,0.048), (1e-4,0.1), (1e-4,1), (1e-4,0.1), (1e-7,1), (0.01,1)]
    # Setup objective function (no priors defined = uniform priors based on bounds)
    objective_function = log_posterior_probability(model,pars,bounds,data,states,log_likelihood_fnc,log_likelihood_fnc_args,
                                                    start_sim=start_calibration, weights=weights, labels=labels)
    
    #################
    ## Nelder-Mead ##
    #################

    # Initial guess
    # season: 2017-2018
    theta = [0.0333, 0.00334, 0.0479, 0.00317, 2.33e-05, 0.603]
    
    # Perform optimization 
    #step = len(bounds)*[0.05,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, processes=processes, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_calibration])
    # Add poisson obervational noise
    #out = add_poisson_noise(out)
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
    import sys
    sys.exit()

    ##########
    ## MCMC ##
    ##########

    # Variables
    samples_path=fig_path=f'../data/interim/calibration/{season}/{identifier}/'
    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], multiplier=multiplier_mcmc, bounds=bounds)
    # Append some usefull settings to the samples dictionary
    settings={'start_calibration': start_calibration.strftime('%Y-%m-%d'), 'end_calibration': end_calibration.strftime('%Y-%m-%d'),
              'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': labels, 'season': season,
              'spatial_resolution': sr, 'age_resolution': ar, 'distinguish_daytype': dd, 'stochastic': stoch}
    # Sample n_mcmc iterations
    import emcee
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,  objective_function_kwargs={'simulation_kwargs': {'warmup': 0}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True,
                                    settings_dict=settings, moves=[(emcee.moves.DEMove(), 0.5),(emcee.moves.StretchMove(live_dangerously=True), 0.5)],
                                )                                                                               
    # Generate a sample dictionary and save it as .json for long-term storage
    # Have a look at the script `emcee_sampler_to_dictionary.py`, which does the same thing as the function below but can be used while your MCMC is running.
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, discard=discard, thin=thin)
    # Look at the resulting distributions in a cornerplot
    CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    fig = corner.corner(sampler.get_chain(discard=discard, thin=2, flat=True), labels=labels, **CORNER_KWARGS)
    for idx,ax in enumerate(fig.get_axes()):
        ax.grid(False)
    plt.savefig(fig_path+'corner.pdf')
    #plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################
 
    # Define draw function
    def draw_fcn(parameters, initial_states, samples):
        # sample model parameters
        idx, parameters['beta'] = random.choice(list(enumerate(samples['beta'])))
        parameters['rho_h'] = samples['rho_h'][idx]
        parameters['rho_d'] = samples['rho_d'][idx]
        parameters['asc_case'] = samples['asc_case'][idx]
        parameters['f_I'] = samples['f_I'][idx]
        parameters['f_R'] = samples['f_R'][idx]
        parameters['T_r'] = samples['T_r'][idx]

        return parameters, initial_states
    
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
    plt.savefig(fig_path+'goodness_fit.pdf')
    plt.tight_layout()
    plt.show()
    plt.close()