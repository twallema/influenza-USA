"""
This script simulates an age-stratified, spatially-explicit SVI2HRD model for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import random
import corner
import emcee
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from datetime import datetime as datetime
from influenza_USA.SVIR.utils import initialise_SVI2RHD, fips2name # influenza model
# pySODM packages
from pySODM.optimization import nelder_mead
from pySODM.optimization.utils import add_poisson_noise, assign_theta
from pySODM.optimization.objective_functions import log_posterior_probability, ll_poisson, ll_negative_binomial
from pySODM.optimization.mcmc import perturbate_theta, run_EnsembleSampler, emcee_sampler_to_dictionary

##############
## Settings ##
##############

# model settings
season_start = 2017                 # '2017' or '2019'
season = '2017-2018'                # '2017-2018' or '2019-2020'
waning = 'no_waning'                # 'no_waning' vs. 'waning_180'
sr = 'states'                       # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                         # age resolution: 'collapsed' or 'full'
dd = False                          # vary contact matrix by daytype
stoch = False                       # ODE vs. tau-leap

# optimization
start_calibration = datetime(season_start, 8, 1)
end_calibration = None                                                          # 2017-2018: None, 2019-2020: datetime(2020,3,22) - exclude COVID
start_peakslice = datetime(season_start+1, 1, 1)
end_peakslice = datetime(season_start+1, 2, 14)
## frequentist
n_pso = 500                                                                     # Number of PSO iterations
multiplier_pso = 10                                                             # PSO swarm size
## bayesian
identifier = 'USA_states'                                                       # ID of run
samples_path=fig_path=f'../data/interim/calibration/{season}/{identifier}/'     # Path to backend
n_mcmc = 2000                                                                   # Number of MCMC iterations
multiplier_mcmc = 3                                                             # Total number of Markov chains = number of parameters * multiplier_mcmc
print_n = 100                                                                    # Print diagnostics every `print_n`` iterations
discard = 1000                                                                  # Discard first `discard` iterations as burn-in
thin = 50                                                                      # Thinning factor emcee chains
n = 100                                                                         # Repeated simulations used in visualisations
processes = int(os.getenv('SLURM_CPUS_ON_NODE', mp.cpu_count()))                # Retrieve CPU count
## get initial parameter estimate from backend
run_date = '2024-09-27'                                                         # First date of run
backend_identifier = 'beta_f_R'
backend_path = f"../data/interim/calibration/{season}/{backend_identifier}/{backend_identifier}_BACKEND_{run_date}.hdf5"

###############################
## Load hospitalisation data ##
###############################

# load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../data/interim/cases/hospitalisations_per_state.csv'), index_col=1, parse_dates=True, dtype={'season_start': str, 'location': str}).reset_index()
# slice right season
df = df[df['season_start'] == str(season_start)][['date', 'location', 'H_inc']]
# set a multiindex: 'date' + 'location' --> pySODM will align 'location' with model
df = df.groupby(by=['date', 'location']).sum().squeeze()
# convert to daily incidence
df /= 7
# slice data until end
df = df.loc[slice(None, end_calibration), slice(None)]
df_peak = df.loc[slice(start_peakslice, end_peakslice), slice(None)]
# replace `end_calibration` None --> datetime
end_calibration = df.index.get_level_values('date').unique().max()
# variables we need a lot
n_states = len(df.index.get_level_values('location').unique())

#####################################################
## Load previous sampler and extract last estimate ##
#####################################################

# Load emcee backend
backend_path = os.path.join(os.getcwd(), backend_path)
backend = emcee.backends.HDFBackend(backend_path)
# Get last position
pos = backend.get_chain(discard=0, thin=1, flat=False)[-1, ...]
# Average out all walkers/parameter
theta = np.mean(pos, axis=0)
# Function to get indices of a states fips
def get_pos_beta_f_R(fips, model_coordinates):
    n = len(model_coordinates)                  # number of states
    i = model_coordinates.index(fips)           # index of desired state
    return i, n+2+i

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

    data = []
    weights = []
    states = []
    log_likelihood_fnc = []
    log_likelihood_fnc_args = []
    for state_fips in df.index.get_level_values('location').unique():
        for d_star in [df, df_peak]:
            d = d_star.reset_index()[d_star.reset_index()['location'] == state_fips].groupby(by=['date', 'location']).sum()
            data.append(d)
            weights.append(1/max(d.squeeze()))
            states.append('H_inc')
            log_likelihood_fnc.append(ll_poisson)
            log_likelihood_fnc_args.append([])

    # define datasets
    #data=[df, df_peak]   # hospital/death peak counted double
    # use maximum value in dataset as weight
    #weights = [1, 1]
    # states to match with datasets
    #states = ['H_inc', 'H_inc']
    # log likelihood function + arguments
    #log_likelihood_fnc = [ll_negative_binomial, ll_negative_binomial] 
    #log_likelihood_fnc_args = [0.03*np.ones(n_states), 0.03*np.ones(n_states)]
    # parameters to calibrate
    pars = ['beta', 'rho_h', 'f_I', 'f_R']
    # labels in output figures
    labels = [r'$\beta$', r'$\rho_h$', r'$f_I$', r'$f_R$']
    # parameter bounds
    bounds = [(0.001,0.06), (0.0001,0.1), (1e-8,1), (0,1)]
    # Setup objective function (no priors defined = uniform priors based on bounds)
    objective_function = log_posterior_probability(model, pars, bounds, data, states, log_likelihood_fnc, log_likelihood_fnc_args, start_sim=start_calibration, weights=weights, labels=labels)

    #################
    ## Nelder-Mead ##
    #################

    # Initial guess
    # season: 2017-2018
    #theta = (n_states+1)*[0.0307,] + [3.34026895e-03, 8.72838050e-05] + (n_states+1)*[5.92385536e-01,] # --> no waning; state beta + f_R --> startpoint of 2024-09-27 fit

    # tweaking
    # tweak_fips = '09000'
    # pos_beta, pos_f_R = get_pos_beta_f_R(tweak_fips, model.coordinates['location'])
    # theta[pos_beta] = 0.035
    # theta[pos_f_R] = 0.67

    # Perform optimization 
    #step = len(bounds)*[0.05,]
    #theta = nelder_mead.optimize(objective_function, np.array(theta), step, processes=1, max_iter=n_pso)[0]

    ######################
    ## Visualize result ##
    ######################

    # Assign results to model
    model.parameters = assign_theta(model.parameters, pars, theta)
    # Simulate model
    out = model.sim([start_calibration, end_calibration])
    # Visualize
    fig, ax = plt.subplots(n_states+1, 1, sharex=True, figsize=(8.3, 11.7/4*(n_states+1)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    ## Overall
    x_data = df.index.get_level_values('date').unique().values
    ax[0].scatter(x_data, 7*df.groupby(by='date').sum(), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
    ax[0].set_title('USA')
    ax[0].grid(False)

    ## per state
    for i,loc in enumerate(df.index.get_level_values('location').unique().values):
        ax[i+1].scatter(x_data, 7*df.loc[slice(None), loc], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[i+1].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc), color='blue', alpha=1, linewidth=2)
        ax[i+1].set_title(f"{fips2name(loc)} ({loc})")
        pos_beta, pos_f_R = get_pos_beta_f_R(loc, model.coordinates['location'])
        ax[i+1].text(0.05, 0.95, f"$\\beta$: {theta[pos_beta]:.3f}, $f_R$: {theta[pos_f_R]:.2f}", transform=ax[i+1].transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
        ax[i+1].grid(False)

    ## format dates
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(30)

    ## Print to screen
    plt.tight_layout()
    fig_path=f'../data/interim/calibration/{season}/{identifier}/'
    plt.savefig(fig_path+'goodness-fit-NM.pdf')
    #plt.show()
    plt.close()

    ##########
    ## MCMC ##
    ##########

    # Perturbate previously obtained estimate
    ndim, nwalkers, pos = perturbate_theta(theta, pert=0.25*np.ones(len(theta)), multiplier=multiplier_mcmc, bounds=objective_function.expanded_bounds)
    # Append some usefull settings to the samples dictionary
    settings={'start_calibration': start_calibration.strftime('%Y-%m-%d'), 'end_calibration': end_calibration.strftime('%Y-%m-%d'),
              'n_chains': nwalkers, 'starting_estimate': list(theta), 'labels': labels, 'season': season,
              'spatial_resolution': sr, 'age_resolution': ar, 'distinguish_daytype': dd, 'stochastic': stoch}
    # Sample n_mcmc iterations
    sampler = run_EnsembleSampler(pos, n_mcmc, identifier, objective_function,  objective_function_kwargs={'simulation_kwargs': {'warmup': 0}},
                                    fig_path=fig_path, samples_path=samples_path, print_n=print_n, backend=None, processes=processes, progress=True, 
                                        moves=[(emcee.moves.DEMove(), 0.5*0.5*0.9),(emcee.moves.DEMove(gamma0=1.0),0.5*0.5*0.1),
                                                (emcee.moves.DESnookerMove(),0.5*0.5),(emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                        settings_dict=settings)                                                                               
    # Generate a sample dictionary and save it as .json for long-term storage
    samples_dict = emcee_sampler_to_dictionary(samples_path, identifier, run_date=run_date, discard=discard, thin=thin)
    # Look at the resulting distributions in a cornerplot
    #CORNER_KWARGS = dict(smooth=0.90,title_fmt=".2E")
    #fig = corner.corner(sampler.get_chain(discard=discard, thin=2, flat=True), labels=objective_function.expanded_labels , **CORNER_KWARGS)
    #for idx,ax in enumerate(fig.get_axes()):
    #    ax.grid(False)
    #plt.savefig(fig_path+'corner.pdf')
    #plt.show()
    #plt.close()

    ######################
    ## Visualize result ##
    ######################
 
    def draw_fcn(parameters, initial_states, samples):
        # sample posterior distribution
        idx, parameters['rho_h'] = random.choice(list(enumerate(samples['rho_h'])))
        parameters['f_I'] = samples['f_I'][idx] 
        parameters['beta'] = np.array([slice[idx] for slice in samples['beta']])
        parameters['f_R'] = np.array([slice[idx] for slice in samples['f_R']])

        return parameters, initial_states
    
    # Simulate model
    out = model.sim([start_calibration, end_calibration], N=n,
                        draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}, processes=1)

    # Visualize
    fig, ax = plt.subplots(n_states+1, 1, sharex=True, figsize=(8.3, 11.7/4*(n_states+1)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    ## Overall
    x_data = df.index.get_level_values('date').unique().values
    ax[0].scatter(x_data, 7*df.groupby(by='date').sum(), color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
    ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
    ax[0].fill_between(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=0.05/2),
                        7*out['H_inc'].sum(dim=['age_group', 'location']).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
    ax[0].set_title('USA')
    ax[0].grid(False)

    ## per state
    for i,loc in enumerate(df.index.get_level_values('location').unique().values):
        ax[i+1].scatter(x_data, 7*df.loc[slice(None), loc], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
        ax[i+1].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).mean(dim='draws'), color='blue', alpha=1, linewidth=2)
        ax[i+1].fill_between(out['date'], 7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).quantile(dim='draws', q=0.05/2),
                             7*out['H_inc'].sum(dim=['age_group']).sel(location=loc).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.2)
        ax[i+1].set_title(f"{fips2name(loc)} ({loc})")
        pos_beta, pos_f_R = get_pos_beta_f_R(loc, model.coordinates['location'])
        ax[i+1].text(0.05, 0.95, f"$\\beta$: {theta[pos_beta]:.3f}, $f_R$: {theta[pos_f_R]:.2f}", transform=ax[i+1].transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
        ax[i+1].grid(False)

    ## format dates
    ax[-1].xaxis.set_major_locator(plt.MaxNLocator(5))
    for tick in ax[-1].get_xticklabels():
        tick.set_rotation(30)

    ## Print to screen
    plt.tight_layout()
    fig_path=f'../data/interim/calibration/{season}/{identifier}/'
    plt.savefig(fig_path+'goodness-fit-MCMC.pdf')
    plt.close()