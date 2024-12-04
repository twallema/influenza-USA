"""
This script calibrates an age-stratified, spatially-explicit two-strain sequential infection model for Influenza in a US state using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import random
import emcee
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from datetime import datetime as datetime
from influenza_USA.SIR_SequentialTwoStrain_stateSlice.utils import initialise_SIR_SequentialTwoStrain_stateSlice, name2fips # influenza model
# pySODM packages
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import assign_theta, add_poisson_noise
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, log_prior_normal_L2, log_prior_uniform
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

##############
## Settings ##
##############

# model settings
season_start = 2023                     # start of season
season = '2023-2024'                    # season to calibrate
sr = 'states'                           # spatial resolution: 'states' or 'counties'
ar = 'full'                             # age resolution: 'collapsed' or 'full'
dd = False                              # vary contact matrix by daytype

# optimization parameters
## state
state = 'North Carolina'
## dates
start_calibration = datetime(season_start, 10, 15)                              # calibration data will be sliced starting on this date
end_calibration = datetime(season_start+1, 4, 27)                               # calibration data will be sliced ending on this date
end_validation = datetime(season_start+1, 5, 1)                                 # alternative: None
## frequentist optimization
n_pso = 500                                                                    # Number of PSO iterations
multiplier_pso = 10                                                             # PSO swarm size
## bayesian inference
identifier = 'SequentialTwoStrain_May_simple'                                   # ID of run
samples_path=fig_path=f'../../data/interim/calibration/{season}/{identifier}/'  # Path to backend
n_mcmc = 500                                                                   # Number of MCMC iterations
multiplier_mcmc = 5                                                            # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 500                                                                   # Print diagnostics every `print_n`` iterations
discard = 450                                                                     # Discard first `discard` iterations as burn-in
thin = 10                                                                        # Thinning factor emcee chains
n = 100                                                                         # Repeated simulations used in visualisations
processes = 16                                                                  # Retrieve CPU count
L1_weight = 5

## continue run
# run_date = '2024-12-03'                                                         # First date of run
# backend_identifier = 'SequentialTwoStrain_May_simple'
# backend_path = f"../../data/interim/calibration/{season}/{backend_identifier}/{backend_identifier}_BACKEND_{run_date}.hdf5"
## new run
backend_path = None
if not backend_path:
    # get run date
    run_date = datetime.today().strftime("%Y-%m-%d")
    # check if samples folder exists, if not, make it
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)
    # start from some ballpark estimates
    ## level 0
    rho_h = 0.0025
    beta1 = 0.0218
    beta2 = 0.0222
    f_R1_R2 = 0.5
    f_R1 = 0.43
    f_I1 = 5e-5
    f_I2 = 5e-5
    ## level 1 
    delta_beta_temporal = 0.01

##########################################
## Load and format hospitalisation data ##
##########################################

# load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/cases/hospitalisations_per_state.csv'), index_col=1, parse_dates=True, dtype={'season_start': str, 'location': str}).reset_index()
# slice right season (and state; if applicable)
df = df[((df['season_start'] == str(season_start)) & (df['location'] == name2fips(state)))][['date', 'H_inc']]
# set date as index --> this is a pySODM requirement
df = df.set_index('date').squeeze()
# convert to daily incidence
df /= 7

####################################################
## Make a flu A vs. flu B hospitalisation dataset ##
####################################################

# load subtype data flu A vs. flu B
df_subtype = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/cases/subtypes_NC_23-24.csv'), index_col=0, parse_dates=True)
df_subtype = df_subtype.fillna(50) # assume 50/50 mixed if no data
df_subtype['fraction_A'] = df_subtype['flu_A'] / (df_subtype['flu_A'] + df_subtype['flu_B']) # compute percent A
# make a combined dataframe
df_combined = pd.concat([(df * df_subtype['fraction_A']).fillna(0), (df * (1-df_subtype['fraction_A'])).fillna(0)], axis=1)
df_combined.columns = ['flu_A', 'flu_B']
# slice out calibration data
df_calibration = df_combined.loc[slice(start_calibration, end_calibration)]
# replace `end_calibration` None --> datetime
end_calibration = df_calibration.index.unique().max() + timedelta(days=1)
# slice out validation data
df_validation = df_combined.loc[slice(end_calibration, end_validation)]

#####################################################
## Load previous sampler and extract last estimate ##
#####################################################

if backend_path:
    # Load emcee backend
    backend_path = os.path.join(os.getcwd(), backend_path)
    backend = emcee.backends.HDFBackend(backend_path)
    # Get last position
    pos = backend.get_chain(discard=discard, thin=thin, flat=False)[-1, ...]
    # Average out all walkers/parameter
    theta = np.mean(pos, axis=0)

#################
## Setup model ##
#################

model = initialise_SIR_SequentialTwoStrain_stateSlice(spatial_resolution=sr, age_resolution=ar, state=state, season=season, distinguish_daytype=dd)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    ##################################
    ## Set up posterior probability ##
    ##################################

    # subtype data
    data = [df_calibration['flu_A'], df_calibration['flu_B'],]
    weights = [1/max(df_calibration['flu_A'],), 1/max(df_calibration['flu_B'],)]
    weights = np.array(weights) / np.mean(weights)
    states = ['H1_inc', 'H2_inc']
    log_likelihood_fnc = [ll_poisson, ll_poisson]
    log_likelihood_fnc_args = [[],[]]

    # parameters to calibrate
    pars = [
        'rho_h', 'beta1', 'beta2', 'f_R1_R2', 'f_R1', 'f_I1', 'f_I2',       # level 0
        'delta_beta_temporal',                                              # level 1                                                                                                                                                
            ]
    # labels in output figures
    labels = [
        r'$\rho_{h}$', r'$\beta_{1}$',  r'$\beta_{2}$', r'$f_{R1+R2}$', r'$f_{R1}$', r'$f_{I1}$', r'$f_{I2}$',      # level 0
        r'$\Delta \beta_{t}$',                                                                                      # level 1
                ]
    # parameter bounds
    bounds = [
        (1e-8,0.01), (0.01,0.06), (0.01,0.06), (0,0.99), (0,1), (1e-8,1e-3), (1e-8,1e-3),     # level 0
        (-0.5,0.5),                                                                           # level 1
              ]
    # priors
    log_prior_prob_fcn = [
        log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform,    # level 0
        log_prior_normal_L2,                                                                                                                    # level 1
              ]
    stdev = 0.10
    log_prior_prob_fcn_args = [
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5], bounds[6],    # level 0
        (0, stdev,  L1_weight),                                                         # level 1
    ]
    # Setup objective function (no priors defined = uniform priors based on bounds)
    objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                                   log_prior_prob_fnc=log_prior_prob_fcn, log_prior_prob_fnc_args=log_prior_prob_fcn_args,
                                                    start_sim=start_calibration, weights=weights, labels=labels)

    #################
    ## Nelder-Mead ##
    #################

    # Initial guess
    if not backend_path:
        # set ballpark theta
        theta = [rho_h, beta1, beta2, f_R1_R2, f_R1, f_I1, f_I2] + len(model.parameters['delta_beta_temporal']) * [delta_beta_temporal,]

        # perform optimization 
        step = len(objective_function.expanded_bounds)*[0.2,]
        theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs': {'method': 'RK23', 'rtol': 5e-3}},
                                        processes=1, max_iter=n_pso, no_improv_break=1000)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_validation])
    # Visualize
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8.3, 11.7/5*3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    ## State
    x_calibration_data = df_calibration.index.unique().values
    x_validation_data = df_validation.index.unique().values
    ax[0].scatter(x_calibration_data, 7*(df_calibration['flu_A'] + df_calibration['flu_B']), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[0].scatter(x_validation_data, 7*(df_validation['flu_A'] + df_validation['flu_B']), color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[0].grid(False)
    ax[0].set_title(f'{state}')
    ## Flu A
    ax[1].scatter(x_calibration_data, 7*df_calibration['flu_A'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[1].scatter(x_validation_data, 7*df_validation['flu_A'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[1].plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[1].grid(False)
    ax[1].set_title(f'{state} (Flu A)')
    ## Flu B
    ax[2].scatter(x_calibration_data, 7*df_calibration['flu_B'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[2].scatter(x_validation_data, 7*df_validation['flu_B'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[2].plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[2].grid(False)
    ax[2].set_title(f'{state} (Flu B)')
    ## format dates
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(30)
    ## Print to screen
    plt.tight_layout()
    fig_path=f'../../data/interim/calibration/{season}/{identifier}/'
    plt.savefig(fig_path+'goodness-fit-NM.pdf')
    #plt.show()
    plt.close()

    ##########
    ## MCMC ##
    ##########

    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=objective_function.expanded_bounds)
    # Append some usefull settings to the samples dictionary
    settings={'start_calibration': start_calibration.strftime('%Y-%m-%d'), 'end_calibration': end_calibration.strftime('%Y-%m-%d'),
              'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': labels, 'season': season,
              'spatial_resolution': sr, 'age_resolution': ar, 'distinguish_daytype': dd}
    # Sample n_mcmc iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=backend_path, processes=processes, progress=True, 
                                        moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1),
                                                (emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                        settings_dict=settings,
                                        objective_function_kwargs={'simulation_kwargs': {'method': 'RK23', 'rtol': 5e-3}}
                                        )                                                                               
    # Generate a sample dictionary and save it as .json for long-term storage
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, run_date=run_date, discard=discard, thin=thin)


    # average out 
    delta_beta_temporal_average = []
    for i, beta_delta_temporal_i in enumerate(samples_dict['delta_beta_temporal']):
        delta_beta_temporal_average.append(np.mean(beta_delta_temporal_i))
    
    # compute trajectory
    ## get function
    from influenza_USA.SIR_SequentialTwoStrain_stateSlice.TDPF import transmission_rate_function
    f = transmission_rate_function(sigma=2)
    ## pre-allocate x and y
    x = pd.date_range(start=start_calibration, end=end_validation, freq='d').tolist()
    y = []
    ## compute output
    for d in x:
        y.append(f(d, {}, 1, np.array(delta_beta_temporal_average)))
    ## visualise
    fig,ax=plt.subplots()
    ax.plot(x, y)
    plt.show()
    plt.close()

    ######################
    ## Visualize result ##
    ######################

    def draw_fcn(parameters, samples):
        # level 0
        idx, parameters['rho_h'] = random.choice(list(enumerate(samples['rho_h'])))
        parameters['beta1'] = samples['beta1'][idx]
        parameters['beta2'] = samples['beta2'][idx]
        parameters['f_R1_R2'] = samples['f_R1_R2'][idx]
        parameters['f_R1'] = samples['f_R1'][idx]
        parameters['f_I1'] = samples['f_I1'][idx]
        parameters['f_I2'] = samples['f_I2'][idx]
        # level 1
        parameters['delta_beta_temporal'] = np.array([slice[idx] for slice in samples['delta_beta_temporal']])
        return parameters
    
    # Simulate model
    out = model.sim([start_calibration, end_validation], N=n,
                        draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=1)
    # Add sampling noise
    out = add_poisson_noise(out)
    # Visualize
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8.3, 11.7/5*3))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    ## State
    x_calibration_data = df_calibration.index.unique().values
    x_validation_data = df_validation.index.unique().values
    ax[0].scatter(x_calibration_data, 7*(df_calibration['flu_A'] + df_calibration['flu_B']), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[0].scatter(x_validation_data, 7*(df_validation['flu_A'] + df_validation['flu_B']), color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    ax[0].fill_between(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    ax[0].grid(False)
    ax[0].set_title(f'{state}')
    ## Flu A
    ax[1].scatter(x_calibration_data, 7*df_calibration['flu_A'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[1].scatter(x_validation_data, 7*df_validation['flu_A'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[1].plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    ax[1].fill_between(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    ax[1].grid(False)
    ax[1].set_title(f'{state} (Flu A)')
    ## Flu B
    ax[2].scatter(x_calibration_data, 7*df_calibration['flu_B'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[2].scatter(x_validation_data, 7*df_validation['flu_B'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[2].plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    ax[2].fill_between(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    ax[2].grid(False)
    ax[2].set_title(f'{state} (Flu B)')
    ## format dates
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(30)
    ## Print to screen
    plt.tight_layout()
    fig_path=f'../../data/interim/calibration/{season}/{identifier}/'
    plt.savefig(fig_path+'goodness-fit-MCMC.pdf')
    #plt.show()
    plt.close()
