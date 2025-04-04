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
from influenza_USA.SVI2RHD.utils import initialise_SVI2RHD, fips2name # influenza model
# pySODM packages
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import assign_theta
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, log_prior_normal_L2, log_prior_uniform, log_prior_gamma
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

##############
## Settings ##
##############

# model settings
season_start = 2014                     # '2017' or '2019'
season = '2014-2015'                    # '2017-2018' or '2019-2020'
waning = 'no_waning'                    # 'no_waning' vs. 'waning_180'
sr = 'states'                           # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                             # age resolution: 'collapsed' or 'full'
dd = False                              # vary contact matrix by daytype
hierarchal_transmission_rate = True     # Hierarchal structure on transmission rate
hierarchal_immunity = True              # Hierarchal structure on the waning of the natural immunity

# optimization
start_calibration = datetime(season_start, 10, 15)                              # simulations will start on this date
end_calibration = datetime(season_start+1, 5, 2)                                # 2017-2018: None, 2019-2020: datetime(2020,3,22) - exclude COVID
start_slice = datetime(season_start+1, 1, 1)                                    # add in a part of the dataset twice: in this case the peak in hosp.
end_slice = datetime(season_start+1, 3, 1)
## frequentist
n_pso = 500                                                                     # Number of PSO iterations
multiplier_pso = 10                                                             # PSO swarm size
## bayesian
identifier = 'USA_regions_hierarchal_May-waning'                                # ID of run
samples_path=fig_path=f'../../data/interim/calibration/{season}/{identifier}/'  # Path to backend
n_mcmc = 5000                                                                   # Number of MCMC iterations
multiplier_mcmc = 3                                                             # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 1000                                                                  # Print diagnostics every `print_n`` iterations
discard = 0                                                                     # Discard first `discard` iterations as burn-in
thin = 100                                                                      # Thinning factor emcee chains
n = 500                                                                         # Repeated simulations used in visualisations
processes = 16                                                                  # Retrieve CPU count
## hierarchal hyperparameters                                                       
L1_weight = 2
rel_weight_level2 = 2
n_regions = 9
n_states = 52
n_temporal_modifiers = 10

## continue run
# run_date = '2024-10-30'                                                         # First date of run
# backend_identifier = 'USA_regions_hierarchal_May-waning'
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
    beta_US = 0.035
    delta_beta_regions = 0.01
    delta_beta_states = 0.01
    delta_beta_temporal = 0.01
    delta_beta_regions_Nov1 = 0.01
    delta_beta_regions_Nov2 = 0.01
    delta_beta_regions_Dec1 = 0.01
    delta_beta_regions_Dec2 = 0.01
    delta_beta_regions_Jan1 = 0.01
    delta_beta_regions_Jan2 = 0.01
    delta_beta_regions_Feb1 = 0.01
    delta_beta_regions_Feb2 = 0.01
    delta_beta_regions_Mar1 = 0.01
    delta_beta_regions_Mar2 = 0.01
    f_R = 0.50
    delta_f_R_states = 0.01
    delta_f_R_regions = 0.01
    T_r_US = 365/np.log(2)
    delta_T_r_regions = 0.01
    rho_h = 0.0026
    f_I = 2e-4

###############################
## Load hospitalisation data ##
###############################

# load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/cases/hospitalisations_per_state.csv'), index_col=1, parse_dates=True, dtype={'season_start': str, 'location': str}).reset_index()
# slice right season
df = df[df['season_start'] == str(season_start)][['date', 'location', 'H_inc']]
# set a multiindex: 'date' + 'location' --> pySODM will align 'location' with model
df = df.groupby(by=['date', 'location']).sum().squeeze()
# convert to daily incidence
df /= 7
# compute enddate of the dataset
end_sim = df.index.get_level_values('date').unique().max()
# slice data until end
df_calibration = df.loc[slice(start_calibration, end_calibration), slice(None)]
df_slice = df.loc[slice(start_slice, end_slice), slice(None)]
# replace `end_calibration` None --> datetime
end_calibration = df_calibration.index.get_level_values('date').unique().max() + timedelta(days=1)
# now slice out the remainder of the dataset
df_validation = df.loc[slice(end_calibration, None), slice(None)]

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

model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, season=season, hierarchal_transmission_rate=hierarchal_transmission_rate,
                           hierarchal_immunity=hierarchal_immunity, distinguish_daytype=dd, start_sim=start_calibration)

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

    # weight of each state = 1/(max(hosp_i)) --> recentered around one using mean_i{max(hosp_i)} --> multiplied with a hyperweight 
    data = []
    weights = []
    states = []
    log_likelihood_fnc = []
    log_likelihood_fnc_args = []
    for d_star, rel_weight in zip([df_calibration,], [1,]):
        for state_fips in df_calibration.index.get_level_values('location').unique():
            d = d_star.reset_index()[d_star.reset_index()['location'] == state_fips].groupby(by=['date', 'location']).sum()
            data.append(d)
            weights.append(1/max(d.squeeze()))
            states.append('H_inc')
            log_likelihood_fnc.append(ll_poisson)
            log_likelihood_fnc_args.append([])
        weights = list(rel_weight * np.array(weights) / np.mean(weights))
    # parameters to calibrate
    pars = ['rho_h', 'f_I', 'beta_US', 'f_R', 'T_r_US',                                                                                             # top level
            'delta_beta_regions', 'delta_beta_temporal', 'delta_f_R_regions', 'delta_T_r_regions',                                                  # mid level
            'delta_beta_states', 'delta_f_R_states',                                                                                                # bottom level
            'delta_beta_regions_Nov1', 'delta_beta_regions_Nov2',
            'delta_beta_regions_Dec1', 'delta_beta_regions_Dec2',
            'delta_beta_regions_Jan1', 'delta_beta_regions_Jan2',
            'delta_beta_regions_Feb1', 'delta_beta_regions_Feb2',
            'delta_beta_regions_Mar1', 'delta_beta_regions_Mar2',   
            ]
    # labels in output figures
    labels = [r'$\rho_h$', r'$f_I$', r'$\beta_{US}$', r'$f_R$', r'$T_{r, US}$',
                r'$\Delta \beta_{regions}$',  r'$\Delta \beta_{temporal}$', r'$\Delta f_{R, regions}$', r'$\Delta T_{r, regions}$',
                r'$\Delta \beta_{states}$', r'$\Delta f_{R, states}$',
                r'$\Delta \beta_{regions, Nov 1}$', r'$\Delta \beta_{regions, Nov 2}$',
                r'$\Delta \beta_{regions, Dec 1}$', r'$\Delta \beta_{regions, Dec 2}$', 
                r'$\Delta \beta_{regions, Jan 1}$', r'$\Delta \beta_{regions, Jan 2}$',
                r'$\Delta \beta_{regions, Feb 1}$', r'$\Delta \beta_{regions, Feb 2}$',
                r'$\Delta \beta_{regions, Mar 1}$', r'$\Delta \beta_{regions, Mar 2}$',
                ]
    # parameter bounds
    bounds = [(1e-9,0.01), (1e-8,1e-3), (0.01,0.060), (0.10,0.90), ((365*8/12)/np.log(2), 2000),
              (-0.5,0.5), (-0.5,0.5), (-0.5,0.5), (-0.5,0.5),
              (-0.5,0.5), (-0.5,0.5),
              (-0.5,0.5), (-0.5,0.5), 
              (-0.5,0.5), (-0.5,0.5), 
              (-0.5,0.5), (-0.5,0.5), 
              (-0.5,0.5), (-0.5,0.5), 
              (-0.5,0.5), (-0.5,0.5)
              ]
    # priors
    log_prior_prob_fcn = [
        log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_uniform, log_prior_gamma,
        log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2, log_prior_normal_L2,
        log_prior_normal_L2, log_prior_normal_L2,
        log_prior_normal_L2, log_prior_normal_L2,
        log_prior_normal_L2, log_prior_normal_L2,
        log_prior_normal_L2, log_prior_normal_L2,
        log_prior_normal_L2, log_prior_normal_L2,
        log_prior_normal_L2, log_prior_normal_L2,
    ]
    stdev = 0.10
    log_prior_prob_fcn_args = [
        # top level
        bounds[0], bounds[1], bounds[2], bounds[3], (12/8, np.log(2)/(365*8/12), (365*8/12)/np.log(2)),
        # mid level
        (0, stdev,  L1_weight), (0, stdev,  L1_weight), (0, stdev,  L1_weight), (0, stdev,  L1_weight),
        # bottom level
        (0, stdev,  rel_weight_level2*L1_weight), (0, stdev,  rel_weight_level2*L1_weight),
        (0, stdev,  rel_weight_level2*L1_weight), (0, stdev,  rel_weight_level2*L1_weight),
        (0, stdev,  rel_weight_level2*L1_weight), (0, stdev,  rel_weight_level2*L1_weight),
        (0, stdev,  rel_weight_level2*L1_weight), (0, stdev,  rel_weight_level2*L1_weight),
        (0, stdev,  rel_weight_level2*L1_weight), (0, stdev,  rel_weight_level2*L1_weight),
        (0, stdev,  rel_weight_level2*L1_weight), (0, stdev,  rel_weight_level2*L1_weight),
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
        theta = [rho_h, f_I, beta_US, f_R, T_r_US] + n_regions*[delta_beta_regions,] + n_temporal_modifiers*[delta_beta_temporal,] + \
                    n_regions*[delta_f_R_states,] + n_regions * [delta_T_r_regions,] + \
                        n_states * [delta_beta_states,] + n_states * [delta_f_R_states,] + \
                            n_regions*[delta_beta_regions_Nov1,] + n_regions*[delta_beta_regions_Nov2,] + \
                                n_regions*[delta_beta_regions_Dec1,] + n_regions*[delta_beta_regions_Dec2,] + \
                                    n_regions*[delta_beta_regions_Jan1] + n_regions*[delta_beta_regions_Jan2,] + \
                                        n_regions*[delta_beta_regions_Feb1] + n_regions*[delta_beta_regions_Feb2,] +\
                                            n_regions*[delta_beta_regions_Mar1] + n_regions*[delta_beta_regions_Mar2,]
        # perform optimization 
        #step = len(objective_function.expanded_bounds)*[0.2,]
        #theta = nelder_mead.optimize(objective_function, np.array(theta), step, kwargs={'simulation_kwargs': {'method': 'RK23', 'rtol': 5e-3}},
        #                          processes=1, max_iter=n_pso, no_improv_break=500)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_sim])
    # Visualize
    fig, ax = plt.subplots(n_states, 1, sharex=True, figsize=(8.3, 11.7/4*(n_states)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    ## Overall
    x_calibration_data = df_calibration.index.get_level_values('date').unique().values
    x_validation_data = df_validation.index.get_level_values('date').unique().values
    ax[0].scatter(x_calibration_data, 7*df_calibration.groupby(by='date').sum(), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    if not df_validation.empty:
        ax[0].scatter(x_validation_data, 7*df_validation.groupby(by='date').sum(), color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[0].grid(False)

    ## per state
    for i,loc in enumerate(df.index.get_level_values('location').unique().values):
        ax[i+1].scatter(x_calibration_data, 7*df_calibration.loc[slice(None), loc], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_validation.empty:
            ax[i+1].scatter(x_validation_data, 7*df_validation.loc[slice(None), loc], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[i+1].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc), color='blue', alpha=1, linewidth=2)
        ax[i+1].set_title(f"{fips2name(loc)} ({loc})")
        ax[i+1].grid(False)

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
    # Perturbation is relative --> zeros are difficult
    #pos[:, 5:] = np.random.normal(loc=0, scale=0.05, size=(nwalkers, pos[:, 5:].shape[1])) # --> modifiers
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
        # top level
        idx, parameters['rho_h'] = random.choice(list(enumerate(samples['rho_h'])))
        parameters['f_I'] = samples['f_I'][idx]
        parameters['beta_US'] = samples['beta_US'][idx]
        parameters['f_R'] = samples['f_R'][idx]
        parameters['T_r_US'] = samples['T_r_US'][idx]
        # mid level
        parameters['delta_f_R_regions'] = np.array([slice[idx] for slice in samples['delta_f_R_regions']])
        parameters['delta_beta_regions'] = np.array([slice[idx] for slice in samples['delta_beta_regions']])
        parameters['delta_beta_temporal'] = np.array([slice[idx] for slice in samples['delta_beta_temporal']])
        parameters['delta_T_r_regions'] = np.array([slice[idx] for slice in samples['delta_T_r_regions']])
        # bottom level
        parameters['delta_beta_states'] = np.array([slice[idx] for slice in samples['delta_beta_states']])
        parameters['delta_f_R_states'] = np.array([slice[idx] for slice in samples['delta_f_R_states']])
        parameters['delta_beta_regions_Nov1'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Nov1']])
        parameters['delta_beta_regions_Nov2'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Nov2']])
        parameters['delta_beta_regions_Dec1'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Dec1']])
        parameters['delta_beta_regions_Dec2'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Dec2']])
        parameters['delta_beta_regions_Jan1'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Jan1']])
        parameters['delta_beta_regions_Jan2'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Jan2']])
        parameters['delta_beta_regions_Feb1'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Feb1']])
        parameters['delta_beta_regions_Feb2'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Feb2']])
        parameters['delta_beta_regions_Mar1'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Mar1']])
        parameters['delta_beta_regions_Mar2'] = np.array([slice[idx] for slice in samples['delta_beta_regions_Mar2']])
        
        return parameters
    
    # Simulate model
    out = model.sim([start_calibration, end_sim], N=n,
                        draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=1)

    # Visualize
    fig, ax = plt.subplots(n_states, 1, sharex=True, figsize=(8.3, 11.7/4*(n_states)))
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

    ## per state
    for i,loc in enumerate(df.index.get_level_values('location').unique().values):
        ax[i+1].scatter(x_calibration_data, 7*df_calibration.loc[slice(None), loc], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_validation.empty:
            ax[i+1].scatter(x_validation_data, 7*df_validation.loc[slice(None), loc], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[i+1].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
        ax[i+1].fill_between(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).quantile(dim='draws', q=0.05/2),
                             7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
        ax[i+1].set_title(f"{fips2name(loc)} ({loc})")
        ax[i+1].grid(False)

    ## format dates
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(30)

    ## Print to screen
    plt.tight_layout()
    fig_path=f'../../data/interim/calibration/{season}/{identifier}/'
    plt.savefig(fig_path+'goodness-fit-MCMC.pdf')
    plt.close()