"""
This script does..
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import emcee
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import get_context
from pySODM.optimization.objective_functions import validate_calibrated_parameters
from influenza_USA.NC_forecasts.hierarchical_calibration_oneStrain import log_posterior_probability, dump_sampler_to_xarray, traceplot, plot_fit, hyperdistributions
from influenza_USA.NC_forecasts.utils import initialise_model, get_NC_influenza_data

##############
## Settings ##
##############

# model settings
state = 'North Carolina'                            # state we'd like to calibrate to
strains = False

# calibration settings
use_ED_visits = True                                                                                     # use both ED admission (hospitalisation) and ED visits (ILI) data 
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024']    # season to include in calibration excercise
start_calibration_month = 10                                                                             # start calibration on month 10, day 1
end_calibration_month = 5                                                                                # end calibration on month 5, day 1

# Define number of chains
max_n = 25000
multiplier_chains = 2
pert = 0.8
run_date = datetime.today().strftime("%Y-%m-%d")
identifier = 'exclude-2024-2025'
print_n = 50
backend = None
discard = 0
thin = 1
processes = int(os.environ.get('NUM_CORES', '16'))

# Make folder structure
if use_ED_visits:
    samples_path=fig_path=f'../../../data/interim/calibration/hierarchical-training/strains_{strains}/use_ED_visits/' # Path to backend
else:
    samples_path=fig_path=f'../../../data/interim/calibration/hierarchical-training/strains_{strains}/not_use_ED_visits/' # Path to backend
# check if samples folder exists, if not, make it
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

################
## Setup data ##
################

# convert to a list of start and enddates (datetime)
start_calibrations = [datetime(int(season[0:4]), start_calibration_month, 1) for season in seasons]
end_calibrations = [datetime(int(season[0:4])+1, end_calibration_month, 1) for season in seasons]

# gather datasets in a list
datasets = [get_NC_influenza_data(start_calibration, end_calibration, season) for start_calibration, end_calibration, season in zip(start_calibrations, end_calibrations, seasons)]

#################
## Setup model ##
#################

model = initialise_model(strains=strains, state=state, season='2014-2015')

##########################################
## Setup posterior probability function ##
##########################################

# define model states we want to calibrate to
states_model = ['I_inc', 'H_inc']
if strains:
    states_data = ['I_inc', 'H_inc_A', 'H_inc_B']
else:
    states_data = ['I_inc', 'H_inc']

# cut out 'I_inc'
if not use_ED_visits:
    states_model = states_model[1:]
    states_data = states_data[1:]

# define model parameters to calibrate to every season and their bounds
# not how we're not cutting out the parameters associated with the ED visit data
pars_model_names = ['rho_i', 'T_h', 'rho_h', 'beta', 'f_R_min1', 'f_R_min2', 'f_R_min3', 'f_I', 'delta_beta_temporal']
pars_model_bounds = [(1e-6,0.10), (0.5, 7), (1e-6,0.10), (0.01,1), (0,0.1), (0,0.1), (0,0.1), (1e-7,1e-3), (-0.5,0.5)]
_, pars_model_shapes = validate_calibrated_parameters(pars_model_names, model.parameters)
n_pars = sum([v[0] for v in pars_model_shapes.values()])

# define hyperparameters
if not strains: 
    hyperpars_shapes = {
        'rho_i_a': (1,), 'rho_i_scale': (1,),
        'T_h_rate': (1,),
        'rho_h_a': (1,), 'rho_h_scale': (1,),
        'beta_mu': (1,), 'beta_sigma': (1,),
        'f_R_min1_mu': (1,), 'f_R_min1_sigma': (1,),
        'f_R_min2_mu': (1,), 'f_R_min2_sigma': (1,),
        'f_R_min3_mu': (1,), 'f_R_min3_sigma': (1,),
        'f_I_a': (1,), 'f_I_scale': (1,),
        'delta_beta_temporal_mu': (len(model.parameters['delta_beta_temporal']),), 'delta_beta_temporal_sigma': (len(model.parameters['delta_beta_temporal']),),
    }
else:
    hyperpars_shapes = {
        'rho_i_a': (1,), 'rho_i_scale': (1,),
        'T_h_rate': (1,),
        'rho_h_a': (2,), 'rho_h_scale': (2,),
        'beta_mu': (2,), 'beta_sigma': (2,),
        'f_R_min1_mu': (2,), 'f_R_min1_sigma': (2,),
        'f_R_min2_mu': (2,), 'f_R_min2_sigma': (2,),
        'f_R_min3_mu': (2,), 'f_R_min3_sigma': (2,),
        'f_I_a': (2,), 'f_I_scale': (2,),
        'delta_beta_temporal_mu': (len(model.parameters['delta_beta_temporal']),), 'delta_beta_temporal_sigma': (len(model.parameters['delta_beta_temporal']),),
    }
####################################
## Fetch initial guess parameters ##
####################################

# get independent fit parameters
pars_model_0 = pd.read_csv('../../../data/interim/calibration/single-season-optimal-parameters.csv')
# filter out strains
pars_model_0 = pars_model_0[pars_model_0['strains'] == strains][seasons]
# parameters
pars_0 = list(pars_model_0.transpose().values.flatten())

# hyperparameters
if not strains:
    hyperpars_0 = [
                5.0, 1.0e-02,                                                                # rho_i
                1.7,                                                                         # T_h
                5.7, 1.1e-03,                                                                # rho_h
                0.50, 0.005,                                                                 # beta
                4.3e-5, 2.6e-5,                                                              # f_R_min1
                6.7e-5, 4.25e-5,                                                             # f_R_min2
                2e-4, 2.7e-4,                                                                # f_R_min3
                4.3, 2.8e-05,                                                                # f_I
                -0.06, -0.02, 0, 0.03, 0.14, -0.11, 0.03, 0.10, 0.03, 0.05, 0.03, -0.04,     # delta_beta_temporal_mu
                0.02, 0.06, 0.04, 0.07, 0.08, 0.11, 0.09, 0.08, 0.13, 0.10, 0.18, 0.08,      # delta_beta_temporal_sigma
                    ]
else:
    hyperpars_0 = [
                5.0, 1.0e-02,                                                                   # rho_i
                1.7,                                                                            # T_h
                5.7, 1.1e-03,                                                                   # rho_h
                0.491, 0.016,                                                                   # beta_0
                0.486, 0.025,                                                                   # beta_1
                4.34e-5, 2.91e-5,                                                               # f_R_min1_0
                6.5e-4, 5.9e-4,                                                                 # f_R_min1_1
                6.7e-5, 4.1e-5,                                                                 # f_R_min2_0
                1.2e-3, 2e-3,                                                                   # f_R_min2_1
                3e-4, 4.6e-4,                                                                   # f_R_min3_0
                1.1e-3, 1e-3,                                                                   # f_R_min3_1
                4.3, 2.8e-05,                                                                   # f_I_0
                4.3, 2.8e-05,                                                                   # f_I_1
                -0.07, -0.03, -0.01, 0.01, 0.13, -0.12, 0.04, 0.11, 0.05, 0.06, 0.04, -0.03,    # delta_beta_temporal_mu
                0.03, 0.06, 0.03, 0.07, 0.08, 0.10, 0.09, 0.08, 0.11, 0.11, 0.15, 0.07,         # delta_beta_temporal_sigma
                    ]
# combine
theta_0 = hyperpars_0 + pars_0
n_chains = multiplier_chains * len(theta_0)
print(f'The number of parameters is: {len(theta_0)}\n')

###################
## Setup sampler ##
###################

# Generate random perturbations from a normal distribution
perturbations = np.random.normal(
        loc=1, scale=pert, size=(n_chains, len(theta_0))
    )

# Apply perturbations to create the 2D array
pos = np.array(theta_0)[None, :] * perturbations
nwalkers, ndim = pos.shape

# By default: set up a fresh hdf5 backend in samples_path
if not backend:
    fn_backend = str(identifier)+'_BACKEND_'+run_date+'.hdf5'
    backend = emcee.backends.HDFBackend(samples_path+fn_backend)
# If user provides an existing backend: continue sampling 
else:
    try:
        backend = emcee.backends.HDFBackend(samples_path+backend)
        pos = backend.get_chain(discard=discard, thin=thin, flat=False)[-1, ...]
    except:
        raise FileNotFoundError("backend not found.")    

# setup sampler
if __name__ == '__main__':
    with get_context("spawn").Pool(processes=processes) as pool:
        # setup sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_probability, backend=backend, pool=pool,
                                        moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1),
                                               (emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                        args=(model, datasets, seasons, pars_model_names, pars_model_bounds, hyperpars_shapes, states_model, states_data)
                                        )
        # sample
        for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True, skip_initial_state_check=True):

            if sampler.iteration % print_n:
                continue
            else:
                # every print_n steps do..
                # ..dump samples
                samples = dump_sampler_to_xarray(sampler.get_chain(discard=discard, thin=thin), samples_path+str(identifier)+'_SAMPLES_'+run_date+'.nc', hyperpars_shapes, pars_model_shapes, seasons)
                # .. visualise hyperdistributions
                hyperdistributions(samples, samples_path+str(identifier)+'_HYPERDIST_'+run_date+'.pdf', pars_model_shapes, pars_model_bounds, 300)
                # ..generate goodness-of-fit
                #plot_fit(model, datasets, samples, pars_model_names, samples_path, identifier, run_date)
                # ..generate traceplots
                traceplot(samples, pars_model_shapes, hyperpars_shapes, samples_path, identifier, run_date)