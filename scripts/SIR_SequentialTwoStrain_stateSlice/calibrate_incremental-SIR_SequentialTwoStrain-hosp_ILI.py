"""
This script calibrates an age-stratified spatially-explicit two-strain sequential infection SIR model to North Carolina hospital admission data
It automatically calibrates to incrementally larger datasets between `start_calibration` and `end_calibration`
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import random
import emcee
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import timedelta
import matplotlib.pyplot as plt
from datetime import datetime as datetime
from influenza_USA.shared.utils import name2fips
from influenza_USA.SIR_SequentialTwoStrain.utils import initialise_SIR_SequentialTwoStrain, get_NC_influenza_data # influenza model
# pySODM packages
from pySODM.optimization import nelder_mead, pso
from pySODM.optimization.utils import assign_theta, add_poisson_noise
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, log_prior_normal, log_prior_uniform, log_prior_gamma, log_prior_normal
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

##############
## Settings ##
##############

# model settings
state = 'North Carolina'                            # state we'd like to calibrate to
season = '2024-2025'                                # season to calibrate
sr = 'states'                                       # spatial resolution: 'states' or 'counties'
ar = 'full'                                         # age resolution: 'collapsed' or 'full'
dd = False                                          # vary contact matrix by daytype
season_start = int(season[0:4])                     # start of season
start_simulation = datetime(season_start, 10, 1)    # date simulation is started
L1_weight = 1                                       # Forcing strength on temporal modifiers 
stdev = 0.10                                        # Expected standard deviation on temporal modifiers

# optimization parameters
## dates
start_calibration = datetime(season_start+1, 1, 16)                             # incremental calibration will start from here
end_calibration = datetime(season_start+1, 5, 1)                                # and incrementally (weekly) calibrate until this date
end_validation = datetime(season_start+1, 5, 1)                                 # enddate used on plots
## frequentist optimization
n_pso = 2000                                                                  # Number of PSO iterations
multiplier_pso = 10                                                             # PSO swarm size
## bayesian inference
n_mcmc = 30000                                                                  # Number of MCMC iterations
multiplier_mcmc = 4                                                             # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 10000                                                                 # Print diagnostics every `print_n`` iterations
discard = 10000                                                                 # Discard first `discard` iterations as burn-in
thin = 1000                                                                     # Thinning factor emcee chains
processes = 16                                                                   # Number of CPUs to use
n = 1000                                                                         # Number of simulations performed in MCMC goodness-of-fit figure

# calibration parameters
pars = ['T_h', 'rho_i', 'rho_h1', 'rho_h2', 'beta1', 'beta2', 'f_R1_R2', 'f_R1', 'f_I1', 'f_I2', 'delta_beta_temporal']                                    # parameters to calibrate
bounds = [(1, 21), (1e-4,0.15), (1e-4,0.01), (1e-4,0.01), (0.005,0.06), (0.005,0.06), (0.01,0.99), (0.01,0.99), (1e-7,1e-3), (1e-7,1e-3), (-0.5,0.5)]        # parameter bounds
labels = [r'$T_h$', r'$\rho_{i}$', r'$\rho_{h,1}$', r'$\rho_{h,2}$', r'$\beta_{1}$',  r'$\beta_{2}$', r'$f_{R1+R2}$', r'$f_{R1}$', r'$f_{I1}$', r'$f_{I2}$', r'$\Delta \beta_{t}$'] # labels in output figures
# UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# log_prior_prob_fcn = 10*[log_prior_uniform,] + [log_prior_normal,]                                                                                   # prior probability functions
# log_prior_prob_fcn_args = [{'bounds':  bounds[0]}, {'bounds':  bounds[1]}, {'bounds':  bounds[2]}, {'bounds':  bounds[3]}, {'bounds':  bounds[4]},
#                            {'bounds':  bounds[5]}, {'bounds':  bounds[6]}, {'bounds':  bounds[7]}, {'bounds':  bounds[8]}, {'bounds':  bounds[9]},
#                            {'avg':  0, 'stdev': stdev, 'weight': L1_weight}]   # arguments prior functions
# INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
log_prior_prob_fcn = 6*[log_prior_gamma] + 2*[log_prior_normal] + 2*[log_prior_gamma] + 12*[log_prior_normal,] 
log_prior_prob_fcn_args = [{'a': 0.9, 'loc': 6.4e-01, 'scale': 3.2, 'weight': L1_weight},
                           {'a': 3.8, 'loc': -4.3e-05, 'scale': 4.9e-03, 'weight': L1_weight},
                           {'a': 4.8, 'loc': -2.5e-04, 'scale': 5.7e-04, 'weight': L1_weight},
                           {'a': 0.8, 'loc': 7.5e-04, 'scale': 1.9e-03, 'weight': L1_weight},
                           {'a': 3.0, 'loc': 1.3e-02, 'scale': 3.0e-03, 'weight': L1_weight},
                           {'a': 4.9, 'loc': 1.0e-02, 'scale': 2.3e-03, 'weight': L1_weight},
                           {'avg': 0.53, 'stdev': 0.23, 'weight': L1_weight},
                           {'avg': 0.47, 'stdev': 0.21, 'weight': L1_weight},
                           {'a': 1.9, 'loc': -3.0e-05, 'scale': 8.7e-05, 'weight': L1_weight},
                           {'a': 1.9, 'loc': 5.4e-05, 'scale': 9.0e-05, 'weight': L1_weight},
                           {'avg': -0.082, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': -0.054, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': -0.047, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': 0.0, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': 0.07, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': -0.11, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': 0.021, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': 0.112, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': 0.053, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': 0.061, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': 0.040, 'stdev': stdev, 'weight': L1_weight},
                           {'avg': -0.036, 'stdev': stdev, 'weight': L1_weight},
                           ]          # arguments of prior functions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## starting guestimate NM
T_h = 3.5
rho_i = 0.02
rho_h1 = 0.002
rho_h2 = 0.002
beta1 = beta2 = 0.0215
f_R1_R2 = f_R1 = 0.5
f_I1 = f_I2 = 5e-5
delta_beta_temporal = [-0.08, -0.05, -0.05, 0.001, 0.07, -0.11, 0.02, 0.11, 0.05, 0.06, 0.04, -0.04] # 0.01

#####################
## Load NC dataset ##
#####################

# load dataset
data_interim = get_NC_influenza_data(start_simulation, end_validation, season)

#################
## Setup model ##
#################

model = initialise_SIR_SequentialTwoStrain(spatial_resolution=sr, age_resolution=ar, state=state, season='average', distinguish_daytype=dd)

#####################
## Calibrate model ##
#####################

if __name__ == '__main__':

    #####################
    ## Loop over weeks ##
    #####################

    # compute the list of incremental calibration enddates between start_calibration and end_calibration
    incremental_enddates = data_interim.loc[slice(start_calibration, end_calibration)].index

    for end_date in incremental_enddates:

        print(f"Working on calibration ending on {end_date.strftime('%Y-%m-%d')}")

        # Make folder structure
        identifier = f'end-{end_date.strftime('%Y-%m-%d')}' # identifier
        samples_path=fig_path=f'../../data/interim/calibration/{season}/{name2fips(state)}/{identifier}/' # Path to backend
        run_date = datetime.today().strftime("%Y-%m-%d") # get current date
        # check if samples folder exists, if not, make it
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)

        ##################################
        ## Set up posterior probability ##
        ##################################

        # split data in calibration and validation dataset
        df_calib = data_interim.loc[slice(start_simulation, end_date)]
        df_valid = data_interim.loc[slice(end_date+timedelta(days=1), end_validation)]

        # prepare data-related arguments of posterior probability
        data = [df_calib['H_inc_A'], df_calib['H_inc_B'], df_calib['I_inc'].dropna()]
        weights = [1/max(df_calib['H_inc_A']), 1/max(df_calib['H_inc_B']), 1/max(df_calib['I_inc'].dropna())]
        weights = np.array(weights) / np.mean(weights)
        states = ['H1_inc', 'H2_inc', 'I_inc']
        log_likelihood_fnc = [ll_poisson, ll_poisson, ll_poisson]
        log_likelihood_fnc_args = [[],[],[]]

        # Setup objective function (no priors defined = uniform priors based on bounds)
        objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                                    log_prior_prob_fnc=log_prior_prob_fcn, log_prior_prob_fnc_args=log_prior_prob_fcn_args,
                                                        start_sim=start_simulation, weights=weights, labels=labels)

        #################
        ## Nelder-Mead ##
        #################

        # set ballpark theta
        theta = [T_h, rho_i, rho_h1, rho_h2, beta1, beta2, f_R1_R2, f_R1, f_I1, f_I2] + delta_beta_temporal # len(model.parameters['delta_beta_temporal']) * [delta_beta_temporal,]

        # perform optimization 
        ## PSO
        #theta, _ = pso.optimize(objective_function, swarmsize=multiplier_pso*len(pars), max_iter=100, processes=processes, debug=True)
        ## Nelder-Mead
        theta, _ = nelder_mead.optimize(objective_function, np.array(theta), len(objective_function.expanded_bounds)*[0.2,], kwargs={'simulation_kwargs': {'method': 'RK23', 'rtol': 5e-3}},
                                        processes=1, max_iter=n_pso, no_improv_break=1000)

        ######################
        ## Visualize result ##
        ######################

        # Assign results to model
        model.parameters = assign_theta(model.parameters, pars, theta)
        # Simulate model
        out = model.sim([start_simulation, end_validation])
        # Visualize
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8.3, 11.7/5*4))
        props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
        ## State
        x_calibration_data = df_calib.index.unique().values
        x_validation_data = df_valid.index.unique().values
        ax[0].scatter(x_calibration_data, 7*(df_calib['H_inc_A'] + df_calib['H_inc_B']), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[0].scatter(x_validation_data, 7*(df_valid['H_inc_A'] + df_valid['H_inc_B']), color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[0].plot(out['date'], 7*(out['H1_inc']+out['H2_inc']).sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
        ax[0].grid(False)
        ax[0].set_title(f'{state}\nHospitalisations')
        ## Flu A
        ax[1].scatter(x_calibration_data, 7*df_calib['H_inc_A'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[1].scatter(x_validation_data, 7*df_valid['H_inc_A'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[1].plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
        ax[1].grid(False)
        ax[1].set_title(f'{state} (Flu A)')
        ## Flu B
        ax[2].scatter(x_calibration_data, 7*df_calib['H_inc_B'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[2].scatter(x_validation_data, 7*df_valid['H_inc_B'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[2].plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
        ax[2].grid(False)
        ax[2].set_title(f'{state} (Flu B)')
        ## ILI
        ax[3].scatter(x_calibration_data, 7*df_calib['I_inc'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[3].scatter(x_validation_data, 7*df_valid['I_inc'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[3].plot(out['date'], 7*out['I_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
        ax[3].grid(False)
        ax[3].set_title('Influenza-like illness')
        ## format dates
        ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in ax[-1].get_xticklabels():
            tick.set_rotation(30)
        ## Print to screen
        plt.tight_layout()
        plt.savefig(fig_path+f'{identifier}_goodness-fit-NM.pdf')
        plt.close()

        ##########
        ## MCMC ##
        ##########

        # Perturbate previously obtained estimate
        ndim, nwalkers, pos = perturbate_theta(theta, pert=0.10*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=objective_function.expanded_bounds)
        # Append some usefull settings to the samples dictionary
        settings={'start_simulation': start_simulation.strftime('%Y-%m-%d'), 'start_calibration': start_calibration.strftime('%Y-%m-%d'), 'end_calibration': end_date.strftime('%Y-%m-%d'),
                'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': labels, 'season': season,
                'spatial_resolution': sr, 'age_resolution': ar, 'distinguish_daytype': dd}
        # Sample n_mcmc iterations
        sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True, 
                                            moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1),
                                                    (emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                            settings_dict=settings,
                                            objective_function_kwargs={'simulation_kwargs': {'method': 'RK23', 'rtol': 5e-3}}
                                            )                                                                               
        # Generate a sample dictionary and save it as .json for long-term storage
        samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, run_date=run_date, discard=discard, thin=thin)

        #######################
        ## Visualize results ##
        #######################

        # Simulate the model
        # ------------------

        def draw_fcn(parameters, samples):
            idx, parameters['T_h'] = random.choice(list(enumerate(samples['T_h'])))
            parameters['rho_i'] = samples['rho_i'][idx]
            parameters['rho_h1'] = samples['rho_h1'][idx]
            parameters['rho_h2'] = samples['rho_h2'][idx]
            parameters['beta1'] = samples['beta1'][idx]
            parameters['beta2'] = samples['beta2'][idx]
            parameters['f_R1_R2'] = samples['f_R1_R2'][idx]
            parameters['f_R1'] = samples['f_R1'][idx]
            parameters['f_I1'] = samples['f_I1'][idx]
            parameters['f_I2'] = samples['f_I2'][idx]
            parameters['delta_beta_temporal'] = np.array([slice[idx] for slice in samples['delta_beta_temporal']])
            return parameters
        
        # Simulate model
        out = model.sim([start_simulation, end_validation+timedelta(weeks=4)], N=n,
                            draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=1)
        
        # Add sampling noise
        out = add_poisson_noise(out)

        # Save as a .csv
        df = out.to_dataframe().reset_index()
        df.to_csv(samples_path+f'{identifier}_simulation-output.csv', index=False)

        # Construct delta_beta_temporal trajectory
        # ----------------------------------------

        # get function
        from influenza_USA.SIR_SequentialTwoStrain.TDPF import transmission_rate_function
        f = transmission_rate_function(sigma=2.5)
        # pre-allocate output
        y = []
        lower = []
        upper = []
        x = pd.date_range(start=start_simulation, end=end_validation, freq='d').tolist()
        # compute output
        for d in x:
            y.append(f(d, {}, 1, np.mean(np.array(samples_dict['delta_beta_temporal']), axis=1))[0])
            lower.append(f(d, {}, 1, np.quantile(np.array(samples_dict['delta_beta_temporal']), q=0.05/2, axis=1))[0])
            upper.append(f(d, {}, 1, np.quantile(np.array(samples_dict['delta_beta_temporal']), q=1-0.05/2, axis=1))[0])

        # Build figure
        # ------------

        # Visualize
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8.3, 11.7))
        props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
        ## State
        x_calibration_data = df_calib.index.unique().values
        x_validation_data = df_valid.index.unique().values
        ax[0].scatter(x_calibration_data, 7*(df_calib['H_inc_A'] + df_calib['H_inc_B']), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[0].scatter(x_validation_data, 7*(df_valid['H_inc_A'] + df_valid['H_inc_B']), color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[0].fill_between(out['date'], 7*(out['H1_inc'] + out['H2_inc']).sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                            7*(out['H1_inc']+out['H2_inc']).sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
        ax[0].fill_between(out['date'], 7*(out['H1_inc'] + out['H2_inc']).sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.50/2),
                            7*(out['H1_inc']+out['H2_inc']).sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
        ax[0].grid(False)
        ax[0].set_title(f'{state}\nHospitalisations')
        ax[0].set_ylabel('Weekly hospital inc. (-)')
        ax[0].set_ylim([0,2000])
        ## Flu A
        ax[1].scatter(x_calibration_data, 7*df_calib['H_inc_A'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[1].scatter(x_validation_data, 7*df_valid['H_inc_A'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[1].fill_between(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                            7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
        ax[1].fill_between(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.50/2),
                            7*out['H1_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
        ax[1].grid(False)
        ax[1].set_title('Influenza A')
        ax[1].set_ylabel('Weekly hospital inc. (-)')
        ## Flu B
        ax[2].scatter(x_calibration_data, 7*df_calib['H_inc_B'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[2].scatter(x_validation_data, 7*df_valid['H_inc_B'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[2].fill_between(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                            7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
        ax[2].fill_between(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.50/2),
                            7*out['H2_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)    
        ax[2].grid(False)
        ax[2].set_title('Influenza B')
        ax[2].set_ylabel('Weekly hospital inc. (-)')
        ## ILI incidences
        ax[3].scatter(x_calibration_data, 7*df_calib['I_inc'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        if not df_valid.empty:
            ax[3].scatter(x_validation_data, 7*df_valid['I_inc'], color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[3].fill_between(out['date'], 7*out['I_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                            7*out['I_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
        ax[3].fill_between(out['date'], 7*out['I_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.50/2),
                            7*out['I_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)    
        ax[3].grid(False)
        ax[3].set_title(f'Influenza-like illness')
        ax[3].set_ylabel('Weekly ILI inc. (-)')
        ## Temporal betas
        ax[4].plot(x, y, color='black')
        ax[4].fill_between(x, lower, upper, color='black', alpha=0.1)
        ax[4].grid(False)
        ax[4].set_title('Temporal modifiers transmission coefficient')
        ax[4].set_ylabel('$\\Delta \\beta (t)$')
        ax[4].set_ylim([0.70,1.30])
        ## format dates
        ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in ax[-1].get_xticklabels():
            tick.set_rotation(30)
        ## Print to screen
        plt.tight_layout()
        plt.savefig(fig_path+f'{identifier}_goodness-fit-MCMC.pdf')
        plt.close()
