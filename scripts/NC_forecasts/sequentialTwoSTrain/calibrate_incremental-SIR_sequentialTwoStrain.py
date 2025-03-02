"""
This script calibrates an age-stratified spatially-explicit two-strain sequential infection SIR model to North Carolina ED admission and ED visits data
It automatically calibrates to incrementally larger datasets between `start_calibration` and `end_calibration`
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import sys,os
import random
import emcee
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from datetime import datetime as datetime
from influenza_USA.NC_forecasts.utils import initialise_model, get_NC_influenza_data, pySODM_to_hubverse # influenza model
# pySODM packages
from pySODM.optimization import nelder_mead, pso
from pySODM.optimization.utils import assign_theta, add_poisson_noise
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, log_prior_normal, log_prior_uniform, log_prior_gamma, log_prior_normal, log_prior_beta
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler

#####################
## Parse arguments ##
#####################

import argparse
# helper function
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--use_ED_visits", type=str_to_bool, help="Use ED visit data (ILI) in addition to ED admission data (hosp. adm.).")
parser.add_argument("--informed", type=str_to_bool, default=None, help="Use priors informed by posterior hyperdistributions.")
parser.add_argument("--hyperparameters", type=str, default=None, help="Name of posterior hyperdistribution. Provide a valid column name in 'summary-hyperparameters.csv' to load the hyperdistributions.")
parser.add_argument("--season", type=str, help="Season to calibrate to. Format: '20XX-20XX'")
args = parser.parse_args()

# assign to desired variables
use_ED_visits = args.use_ED_visits
informed = args.informed
hyperparameters = args.hyperparameters
season = args.season

##############
## Settings ##
##############

# model settings
state = 'North Carolina'                            # state we'd like to calibrate to
sr = 'states'                                       # spatial resolution: 'states' or 'counties'
ar = 'full'                                         # age resolution: 'collapsed' or 'full'
dd = False                                          # vary contact matrix by daytype
season_start = int(season[0:4])                     # start of season
start_simulation = datetime(season_start, 10, 1)    # date simulation is started
L1_weight = 1                                       # Forcing strength on temporal modifiers 
stdev = 0.10                                        # Expected standard deviation on temporal modifiers

# optimization parameters
## dates
start_calibration = datetime(season_start, 12, 1)          # incremental calibration will start from here
end_calibration = datetime(season_start+1, 4, 7)            # and incrementally (weekly) calibrate until this date
end_validation = datetime(season_start+1, 5, 1)             # enddate used on plots
## frequentist optimization
n_pso = 2000                                                # Number of PSO iterations
multiplier_pso = 10                                         # PSO swarm size
## bayesian inference
n_mcmc = 15000                                              # Number of MCMC iterations
multiplier_mcmc = 3                                         # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 15000                                            # Print diagnostics every `print_n`` iterations
discard = 8000                                             # Discard first `discard` iterations as burn-in
thin = 500                                                  # Thinning factor emcee chains
processes = int(os.environ.get('NUM_CORES', '16'))          # Number of CPUs to use
n = 500                                                     # Number of simulations performed in MCMC goodness-of-fit figure

# calibration parameters
pars = ['rho_i', 'T_h', 'rho_h1', 'rho_h2', 'beta1', 'beta2', 'f_R1_R2', 'f_R1', 'f_I1', 'f_I2', 'delta_beta_temporal']                                      # parameters to calibrate
bounds = [(1e-4,0.05), (0.5, 7), (1e-4,5e-3), (1e-4,5e-3), (0.01,0.03), (0.01,0.03), (0.2,0.8), (0.2,0.7), (1e-7,5e-4), (1e-7,5e-4), (-0.25,0.25)]        # parameter bounds
labels = [r'$\rho_{i}$', r'$T_h$', r'$\rho_{h,1}$', r'$\rho_{h,2}$', r'$\beta_{1}$',  r'$\beta_{2}$', r'$f_{R1+R2}$', r'$f_{R1}$', r'$f_{I1}$', r'$f_{I2}$', r'$\Delta \beta_{t}$'] # labels in output figures
# UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
if not informed:
    # change name to build save path
    informed = 'uninformed'
    # assign priors
    log_prior_prob_fcn = 10*[log_prior_uniform,] + [log_prior_normal,]                                                                                   # prior probability functions
    log_prior_prob_fcn_args = [{'bounds':  bounds[0]}, {'bounds':  bounds[1]}, {'bounds':  bounds[2]}, {'bounds':  bounds[3]}, {'bounds':  bounds[4]},
                               {'bounds':  bounds[5]}, {'bounds':  bounds[6]}, {'bounds':  bounds[7]}, {'bounds':  bounds[8]}, {'bounds':  bounds[9]},
                               {'avg':  0, 'stdev': stdev, 'weight': L1_weight}]   # arguments prior functions
# INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
else:
    # change name to build save path
    informed='informed'
    # load and select priors
    priors = pd.read_csv('../../../data/interim/calibration/summary-hyperparameters.csv')
    priors = priors.loc[((priors['model'] == 'sequentialTwoStrain') & (priors['use_ED_visits'] == use_ED_visits)), ('parameter', f'{hyperparameters}')].set_index('parameter').squeeze()
    # assign values
    log_prior_prob_fcn = 4*[log_prior_gamma] + 2*[log_prior_normal] + 2*[log_prior_beta] + 2*[log_prior_gamma] + 12*[log_prior_normal,] 
    log_prior_prob_fcn_args = [ 
                            # ED visits
                            {'a': priors['rho_i_a'], 'loc': 0, 'scale': priors['rho_i_scale']},                             # rho_i
                            {'a': 1, 'loc': 0, 'scale': priors['T_h_rate']},                                                # T_h
                            # >>>>>>>>>
                            {'a': priors['rho_h1_a'], 'loc': 0, 'scale': priors['rho_h1_scale']},                           # rho_h1
                            {'a': priors['rho_h2_a'], 'loc': 0, 'scale': priors['rho_h2_scale']},                           # rho_h2
                            {'avg': priors['beta1_mu'], 'stdev': priors['beta1_sigma']},                                    # beta1
                            {'avg': priors['beta2_mu'], 'stdev': priors['beta2_sigma']},                                    # beta2
                            {'a': priors['f_R1_R2_a'], 'b': priors['f_R1_R2_b'], 'loc': 0, 'scale': 1},                     # f_R1_R2
                            {'a': priors['f_R1_a'], 'b': priors['f_R1_b'], 'loc': 0, 'scale': 1},                           # f_R1
                            {'a': priors['f_I1_a'], 'loc': 0, 'scale': priors['f_I1_scale']},                               # f_I1
                            {'a': priors['f_I2_a'], 'loc': 0, 'scale': priors['f_I2_scale']},                               # f_I2
                            {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                            {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
                            {'avg': priors['delta_beta_temporal_mu_2'], 'stdev': priors['delta_beta_temporal_sigma_2']},
                            {'avg': priors['delta_beta_temporal_mu_3'], 'stdev': priors['delta_beta_temporal_sigma_3']},
                            {'avg': priors['delta_beta_temporal_mu_4'], 'stdev': priors['delta_beta_temporal_sigma_4']},
                            {'avg': priors['delta_beta_temporal_mu_5'], 'stdev': priors['delta_beta_temporal_sigma_5']},
                            {'avg': priors['delta_beta_temporal_mu_6'], 'stdev': priors['delta_beta_temporal_sigma_6']},
                            {'avg': priors['delta_beta_temporal_mu_7'], 'stdev': priors['delta_beta_temporal_sigma_7']},
                            {'avg': priors['delta_beta_temporal_mu_8'], 'stdev': priors['delta_beta_temporal_sigma_8']},
                            {'avg': priors['delta_beta_temporal_mu_9'], 'stdev': priors['delta_beta_temporal_sigma_9']},
                            {'avg': priors['delta_beta_temporal_mu_10'], 'stdev': priors['delta_beta_temporal_sigma_10']},
                            {'avg': priors['delta_beta_temporal_mu_11'], 'stdev': priors['delta_beta_temporal_sigma_11']},
                            ]          # arguments of prior functions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## starting guestimate NM
rho_i = 0.02
T_h = 3.5
rho_h1 = 0.002
rho_h2 = 0.002
beta1 = beta2 = 0.0215
f_R1_R2 = f_R1 = 0.5
f_I1 = f_I2 = 5e-5
delta_beta_temporal = [-0.07, -0.04, -0.05, 0.01, 0.06, -0.12, 0.02, 0.10, 0.04, 0.06, 0.07, -0.03]
theta = [rho_i, T_h, rho_h1, rho_h2, beta1, beta2, f_R1_R2, f_R1, f_I1, f_I2] + delta_beta_temporal

## cut off 'rho_i' if not using ILI data
n_rows_figs = 4
if not use_ED_visits:
    pars = pars[1:]
    bounds = bounds[1:]
    labels = labels[1:]
    theta = theta[1:]
    log_prior_prob_fcn = log_prior_prob_fcn[1:]
    log_prior_prob_fcn_args = log_prior_prob_fcn_args[1:]
    n_rows_figs = 3

#####################
## Load NC dataset ##
#####################

# load dataset
data_interim = get_NC_influenza_data(start_simulation, end_validation, season)

#################
## Setup model ##
#################

model = initialise_model(strains=True, spatial_resolution=sr, age_resolution=ar, state=state, season='average', distinguish_daytype=dd)

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
        if use_ED_visits:
            samples_path=fig_path=f'../../../data/interim/calibration/incremental-calibration/sequentialTwoStrain/{informed}_{hyperparameters}/use_ED_visits/{season}/{identifier}/' # Path to backend
        else:
            samples_path=fig_path=f'../../../data/interim/calibration/incremental-calibration/sequentialTwoStrain/{informed}_{hyperparameters}/not_use_ED_visits/{season}/{identifier}/'
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
        states = ['H1_inc', 'H2_inc', 'I_inc']
        log_likelihood_fnc = [ll_poisson, ll_poisson, ll_poisson]
        log_likelihood_fnc_args = [[],[],[]]
        if not use_ED_visits:
            data = data[:-1]
            states = states[:-1]
            log_likelihood_fnc = log_likelihood_fnc[:-1]
            log_likelihood_fnc_args = log_likelihood_fnc_args[:-1]
        weights = [1/max(df) for df in data]
        weights = np.array(weights) / np.mean(weights)

        # Setup objective function (no priors defined = uniform priors based on bounds)
        objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args,
                                                        log_prior_prob_fnc=log_prior_prob_fcn, log_prior_prob_fnc_args=log_prior_prob_fcn_args,
                                                        start_sim=start_simulation, weights=weights, labels=labels,
                                                        simulation_kwargs={'method': 'RK23', 'rtol': 5e-3}
                                                        )

        #################
        ## Nelder-Mead ##
        #################

        # perform optimization 
        ## PSO
        #theta, _ = pso.optimize(objective_function, swarmsize=multiplier_pso*len(pars), max_iter=100, processes=processes, debug=True)
        ## Nelder-Mead
        theta, _ = nelder_mead.optimize(objective_function, np.array(theta), len(objective_function.expanded_bounds)*[0.2,],
                                        processes=1, max_iter=n_pso, no_improv_break=1000)

        ######################
        ## Visualize result ##
        ######################

        # Assign results to model
        model.parameters = assign_theta(model.parameters, pars, theta)
        # Simulate model
        out = model.sim([start_simulation, end_validation])
        # Visualize
        fig, ax = plt.subplots(n_rows_figs, 1, sharex=True, figsize=(8.3, 11.7/5*n_rows_figs))
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
        if use_ED_visits:
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
                  'season': season, 'starting_estimate': theta,
                  'spatial_resolution': sr, 'age_resolution': ar, 'distinguish_daytype': dd}
        # Sample n_mcmc iterations
        sampler, samples_xr = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function, fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True, 
                                                    moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1), (emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                                    settings_dict=settings
                                            )                                                                               

        #######################
        ## Visualize results ##
        #######################

        # Simulate the model
        # ------------------

        def draw_fcn(parameters, samples_xr):
            # get a random iteration and markov chain
            i = random.randint(0, len(samples_xr.coords['iteration'])-1)
            j = random.randint(0, len(samples_xr.coords['chain'])-1)
            # assign parameters
            for var in pars:
                parameters[var] = samples_xr[var].sel({'iteration': i, 'chain': j}).values
            return parameters
        
        # Simulate model
        out = model.sim([start_simulation, end_validation+timedelta(weeks=4)], N=n, processes=1, method='RK23',
                            draw_function=draw_fcn, draw_function_kwargs={'samples_xr': samples_xr})
        
        # Aggregate hospitalised for Flu A and Flu B
        out['H_inc'] = out['H1_inc'] + out['H2_inc']

        # Add sampling noise
        try:
            out = add_poisson_noise(out)
        except:
            print('no poisson resampling performed')
            sys.stdout.flush()
            pass

        # Save as a .csv in hubverse format / raw netcdf
        df = pySODM_to_hubverse(out, end_date+timedelta(weeks=1), 'wk inc flu hosp', 'H_inc', samples_path, quantiles=True)
        out.to_netcdf(samples_path+f'{identifier}_simulation-output.nc')

        # Construct delta_beta_temporal trajectory
        # ----------------------------------------

        # get function
        from influenza_USA.NC_forecasts.TDPF import transmission_rate_function
        f = transmission_rate_function(sigma=2.5)
        # pre-allocate output
        y = []
        lower = []
        upper = []
        x = pd.date_range(start=start_simulation, end=end_validation, freq='d').tolist()
        # compute output
        for d in x:
            y.append(f(d, {}, 1, samples_xr['delta_beta_temporal'].mean(dim=['chain', 'iteration']))[0])
            lower.append(f(d, {}, 1, samples_xr['delta_beta_temporal'].quantile(q=0.05/2, dim=['chain', 'iteration']))[0])
            upper.append(f(d, {}, 1, samples_xr['delta_beta_temporal'].quantile(q=1-0.05/2, dim=['chain', 'iteration']))[0])

        # Build figure
        # ------------

        # Visualize
        fig, ax = plt.subplots(n_rows_figs+1, 1, sharex=True, figsize=(8.3, 11.7/5*(n_rows_figs+1)))
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
        next_ax = 3
        if use_ED_visits:
            next_ax = 4
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
        ax[next_ax].plot(x, y, color='black')
        ax[next_ax].fill_between(x, lower, upper, color='black', alpha=0.1)
        ax[next_ax].grid(False)
        ax[next_ax].set_title('Temporal modifiers transmission coefficient')
        ax[next_ax].set_ylabel('$\\Delta \\beta (t)$')
        ax[next_ax].set_ylim([0.70,1.30])
        ## format dates
        ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
        for tick in ax[-1].get_xticklabels():
            tick.set_rotation(30)
        ## Print to screen
        plt.tight_layout()
        plt.savefig(fig_path+f'{identifier}_goodness-fit-MCMC.pdf')
        plt.close()
