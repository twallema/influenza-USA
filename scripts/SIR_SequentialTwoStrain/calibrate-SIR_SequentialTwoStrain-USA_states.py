"""
This script simulates an age-stratified, spatially-explicit SVI2HRD model for the USA using pySODM
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
from influenza_USA.SIR_SequentialTwoStrain.utils import initialise_SIR_SequentialTwoStrain, fips2name # influenza model
# pySODM packages
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import assign_theta
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, log_prior_normal_L2, log_prior_uniform, log_prior_gamma
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

##############
## Settings ##
##############

# model settings
season_start = 2017                     # '2017' or '2019'
season = '2017-2018'                    # '2017-2018' or '2019-2020'
sr = 'states'                           # spatial resolution: 'states' or 'counties'
ar = 'full'                             # age resolution: 'collapsed' or 'full'
dd = False                              # vary contact matrix by daytype

# optimization
start_calibration = datetime(season_start, 10, 15)                              # simulations will start on this date
end_calibration = datetime(season_start+1, 4, 27)                               # 2017-2018: None, 2019-2020: datetime(2020,3,22) - exclude COVID
end_validation = datetime(season_start+1, 5, 1)                                 # alternative: None
start_slice = datetime(season_start+1, 1, 1)                                    # add in a part of the dataset twice: in this case the peak in hosp.
end_slice = datetime(season_start+1, 3, 1)
## frequentist
n_pso = 5000                                                                    # Number of PSO iterations
multiplier_pso = 10                                                             # PSO swarm size
## bayesian
identifier = 'SequentialTwoStrain_May_simple_holiday'                                       # ID of run
samples_path=fig_path=f'../../data/interim/calibration/{season}/{identifier}/'  # Path to backend
n_mcmc = 2000                                                                   # Number of MCMC iterations
multiplier_mcmc = 3                                                             # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 500                                                                  # Print diagnostics every `print_n`` iterations
discard = 0                                                                     # Discard first `discard` iterations as burn-in
thin = 1                                                                        # Thinning factor emcee chains
n = 500                                                                         # Repeated simulations used in visualisations
processes = 16                                                                  # Retrieve CPU count
## hierarchal hyperparameters                                                       
L1_weight = 2
rel_weight_level2 = 2
n_regions = 9
n_states = 52
n_temporal_modifiers = 10

## continue run
# run_date = '2024-11-27'                                                         # First date of run
# backend_identifier = 'SequentialTwoStrain_May_simple_temporal'
# backend_path = f"../../data/interim/calibration/{season}/{backend_identifier}/{backend_identifier}_BACKEND_{run_date}.hdf5"
## new run
backend_path = None
if not backend_path:
    # get run date
    run_date = datetime.today().strftime("%Y-%m-%d")
    # check if samples folder exists, if not, make it
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)
    # set some ballpark national estimates
    ## level 0
    rho_h = 0.0025
    beta1_US = 0.0218
    beta2_US = 0.0222
    f_R1_R2 = 0.5
    f_R1 = 0.43
    f_I1 = 5e-5
    f_I2 = 5e-5
    ## level 1
    delta_beta1_regions = 0.01
    delta_beta2_regions = 0.01
    delta_f_I1_regions = 0.01
    delta_f_I2_regions = 0.01
    delta_f_R1_regions = 0.01
    delta_f_R2_regions = 0.01
    delta_beta_holiday = -0.05
    delta_beta_temporal = 0.01
    ## level 2
    delta_beta1_states = 0.01
    delta_beta2_states = 0.01
    
##########################################
## Load and format hospitalisation data ##
##########################################

# load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/cases/hospitalisations_per_state.csv'), index_col=1, parse_dates=True, dtype={'season_start': str, 'location': str}).reset_index()
# slice right season
df = df[df['season_start'] == str(season_start)][['date', 'location', 'H_inc']]
# set a multiindex: 'date' + 'location' --> pySODM will align 'location' with model
df = df.groupby(by=['date', 'location']).sum().squeeze()
# convert to daily incidence
df /= 7
# slice data until end
df_calibration = df.loc[slice(start_calibration, end_calibration), slice(None)]
df_slice = df.loc[slice(start_slice, end_slice), slice(None)]
# replace `end_calibration` None --> datetime
end_calibration = df_calibration.index.get_level_values('date').unique().max() + timedelta(days=1)
# now slice out the remainder of the dataset
df_validation = df.loc[slice(end_calibration, end_validation), slice(None)]
# compute enddate of the dataset
end_sim = df_validation.index.get_level_values('date').unique().max()

############################################################
## Make a US-level flu A vs. flu B hospitalisation dataet ##
############################################################

# load subtype data flu A vs. flu B
subtypes_1718 = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/raw/cases/FluView_Subtypes_17-18.csv'))
# compute percent A
subtypes_1718['percent_A'] = subtypes_1718['TOTAL A'] / (subtypes_1718['TOTAL A'] + subtypes_1718['TOTAL B'])
# convert year + week to date (Saturday)
subtypes_1718['date'] = pd.to_datetime(subtypes_1718['YEAR'].astype(str) + subtypes_1718['WEEK'].astype(str) + '6', format='%Y%W%w')
# extract only necessary columns
subtypes_1718 = subtypes_1718[['date', 'percent_A']].set_index('date')
# slice from start to end calibration
subtypes_1718_calibration = subtypes_1718.loc[slice(start_calibration, end_calibration)].squeeze()
subtypes_1718_validation = subtypes_1718.loc[slice(end_calibration, end_validation)].squeeze()
# get national data
df_flu_A_calibration = df_calibration.groupby(by='date').sum() * subtypes_1718_calibration
df_flu_B_calibration = df_calibration.groupby(by='date').sum() * (1 - subtypes_1718_calibration)
df_flu_A_validation = df_validation.groupby(by='date').sum() * subtypes_1718_validation
df_flu_B_validation = df_validation.groupby(by='date').sum() * (1 - subtypes_1718_validation)

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

model = initialise_SIR_SequentialTwoStrain(spatial_resolution=sr, age_resolution=ar, season=season, distinguish_daytype=dd)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    ##################################
    ## Set up posterior probability ##
    ##################################

    # subtype data
    data = [df_flu_A_calibration, df_flu_B_calibration]
    weights = [1/max(df_flu_A_calibration), 1/max(df_flu_B_calibration)]
    states = ['H1_inc', 'H2_inc']
    log_likelihood_fnc = [ll_poisson, ll_poisson]
    log_likelihood_fnc_args = [[],[]]

    # weight of each state = 1/(max(hosp_i)) --> recentered around one using mean_i{max(hosp_i)} --> multiplied with a hyperweight 
    for d_star, rel_weight in zip([df_calibration,], [1,]):
        for state_fips in df_calibration.index.get_level_values('location').unique():
            d = d_star.reset_index()[d_star.reset_index()['location'] == state_fips].groupby(by=['date', 'location']).sum()
            data.append(d)
            weights.append(1/max(d.squeeze()))
            states.append('H_inc')
            log_likelihood_fnc.append(ll_poisson)
            log_likelihood_fnc_args.append([])
        weights = list(rel_weight * np.array(weights) / np.mean(weights))

    weights[0] = 1
    weights[1] = 1

    # parameters to calibrate
    pars = ['rho_h', 'beta1_US', 'beta2_US', 'f_R1_R2', 'f_R1', 'f_I1', 'f_I2',                                                                                     # level 0
            'delta_beta1_regions','delta_beta2_regions','delta_f_I1_regions','delta_f_I2_regions','delta_f_R1_regions','delta_f_R2_regions','delta_beta_holiday'    # level 1                                                                                                                                                
            ]
    # labels in output figures
    labels = [r'$\rho_{h}$', r'$\beta_{1, US}$',  r'$\beta_{2, US}$', r'$f_{R1+R2}$', r'$f_{R1}$', r'$f_{I1}$', r'$f_{I2}$',                                # level 0
                r'$\Delta \beta_{1, regions}$', r'$\Delta \beta_{2, regions}$',   r'$\Delta f_{I, 1, regions}$', r'$\Delta f_{I, 2, regions}$',             # level 1
                  r'$\Delta f_{R, 1, regions}$', r'$\Delta f_{R, 2, regions}$', r'$\Delta \beta_{holiday}$'      
                ]
    # parameter bounds
    bounds = [(1e-8,0.01), (0.01,0.06), (0.01,0.06), (0,0.99), (0,1), (1e-8,1e-3), (1e-8,1e-3),                                                 # level 0
              (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5),                                               # level 1
              ]
    # priors
    log_prior_prob_fcn = [
        log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform,    # level 0
        log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2  # level 1
    ]
    stdev = 0.10
    log_prior_prob_fcn_args = [
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5], bounds[6],                                                                        # level 0
        (0, stdev,  L1_weight), (0, stdev,  L1_weight), (0, stdev,  L1_weight), (0, stdev,  L1_weight), (0, stdev,  L1_weight), (0, stdev,  L1_weight), (0, stdev,  L1_weight) # level 1
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
        theta = [rho_h, beta1_US, beta2_US, f_R1_R2, f_R1, f_I1, f_I2] + \
                    n_regions*[delta_beta1_regions,] + n_regions*[delta_beta2_regions,] + n_regions*[delta_f_I1_regions,] + n_regions*[delta_f_I2_regions,] + \
                         n_regions*[delta_f_R1_regions,] + n_regions*[delta_f_R2_regions,] + [delta_beta_holiday,]
        # perform optimization 
        #step = len(objective_function.expanded_bounds)*[0.2,]
        #theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs': {'method': 'RK23', 'rtol': 5e-3}},
        #                          processes=1, max_iter=n_pso, no_improv_break=1000)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_sim])
    # Visualize
    fig, ax = plt.subplots(n_states+2, 1, sharex=True, figsize=(8.3, 11.7/4*(n_states)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    ## Overall
    x_calibration_data = df_calibration.index.get_level_values('date').unique().values
    x_validation_data = df_validation.index.get_level_values('date').unique().values
    ax[0].scatter(x_calibration_data, 7*df_calibration.groupby(by='date').sum(), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[0].scatter(x_validation_data, 7*df_validation.groupby(by='date').sum(), color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[0].grid(False)
    ax[0].set_title('USA')
    ## Overall flu A
    ax[1].scatter(x_calibration_data, 7*df_flu_A_calibration, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[1].scatter(x_validation_data, 7*df_flu_A_validation, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[1].plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[1].grid(False)
    ax[1].set_title('USA (Flu A)')
    ## Overall flu B
    ax[2].scatter(x_calibration_data, 7*df_flu_B_calibration, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[2].scatter(x_validation_data, 7*df_flu_B_validation, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[2].plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[2].grid(False)
    ax[2].set_title('USA (Flu B)')
    ## State-level
    for i,loc in enumerate(df.index.get_level_values('location').unique().values):
        ax[i+3].scatter(x_calibration_data, 7*df_calibration.loc[slice(None), loc], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_validation.empty:
            ax[i+3].scatter(x_validation_data, 7*df_validation.loc[slice(None), loc], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[i+3].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc), color='blue', alpha=1, linewidth=2)
        ax[i+3].set_title(f"{fips2name(loc)} ({loc})")
        ax[i+3].grid(False)
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

    ######################
    ## Visualize result ##
    ######################

    def draw_fcn(parameters, samples):
        # level 0
        idx, parameters['rho_h'] = random.choice(list(enumerate(samples['rho_h'])))
        parameters['beta1_US'] = samples['beta1_US'][idx]
        parameters['beta2_US'] = samples['beta2_US'][idx]
        parameters['f_R1_R2'] = samples['f_R1_R2'][idx]
        parameters['f_R1'] = samples['f_R1'][idx]
        parameters['f_I1'] = samples['f_I1'][idx]
        parameters['f_I2'] = samples['f_I2'][idx]
        # level 1
        parameters['delta_beta1_regions'] = np.array([slice[idx] for slice in samples['delta_beta1_regions']])
        parameters['delta_beta2_regions'] = np.array([slice[idx] for slice in samples['delta_beta2_regions']])
        parameters['delta_f_I1_regions'] = np.array([slice[idx] for slice in samples['delta_f_I1_regions']])
        parameters['delta_f_I2_regions'] = np.array([slice[idx] for slice in samples['delta_f_I2_regions']])
        parameters['delta_f_R1_regions'] = np.array([slice[idx] for slice in samples['delta_f_R1_regions']])
        parameters['delta_f_R2_regions'] = np.array([slice[idx] for slice in samples['delta_f_R2_regions']])
        #parameters['delta_beta_temporal'] = np.array([slice[idx] for slice in samples['delta_beta_temporal']])
        # level 2
        #parameters['delta_beta1_states'] = np.array([slice[idx] for slice in samples['delta_beta1_states']])
        #parameters['delta_beta2_states'] = np.array([slice[idx] for slice in samples['delta_beta2_states']])
        return parameters
    
    # Simulate model
    out = model.sim([start_calibration, end_sim], N=n,
                        draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=1)

    # Visualize
    fig, ax = plt.subplots(n_states+2, 1, sharex=True, figsize=(8.3, 11.7/4*(n_states)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    ## Overall
    x_calibration_data = df_calibration.index.get_level_values('date').unique().values
    x_validation_data = df_validation.index.get_level_values('date').unique().values
    ax[0].scatter(x_calibration_data, 7*df_calibration.groupby(by='date').sum(), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[0].scatter(x_validation_data, 7*df_validation.groupby(by='date').sum(), color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    ax[0].fill_between(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    ax[0].grid(False)
    ## Overall flu A
    ax[1].scatter(x_calibration_data, 7*df_flu_A_calibration, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[1].scatter(x_validation_data, 7*df_flu_A_validation, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[1].plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    ax[1].fill_between(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    ax[1].grid(False)
    ax[1].set_title('USA (Flu A)')
    ## Overall flu B
    ax[2].scatter(x_calibration_data, 7*df_flu_B_calibration, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[2].scatter(x_validation_data, 7*df_flu_B_validation, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[2].plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    ax[2].fill_between(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    ax[2].grid(False)
    ax[2].set_title('USA (Flu B)')
    ## per state
    for i,loc in enumerate(df.index.get_level_values('location').unique().values):
        ax[i+3].scatter(x_calibration_data, 7*df_calibration.loc[slice(None), loc], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_validation.empty:
            ax[i+3].scatter(x_validation_data, 7*df_validation.loc[slice(None), loc], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[i+3].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
        ax[i+3].fill_between(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).quantile(dim='draws', q=0.05/2),
                             7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
        ax[i+3].set_title(f"{fips2name(loc)} ({loc})")
        ax[i+3].grid(False)

    ## format dates
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(30)

    ## Print to screen
    plt.tight_layout()
    fig_path=f'../../data/interim/calibration/{season}/{identifier}/'
    plt.savefig(fig_path+'goodness-fit-MCMC.pdf')
    plt.close()