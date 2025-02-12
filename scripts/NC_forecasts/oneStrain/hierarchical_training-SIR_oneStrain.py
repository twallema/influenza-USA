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
from influenza_USA.shared.utils import name2fips
from pySODM.optimization.objective_functions import validate_calibrated_parameters
from influenza_USA.NC_forecasts.hierarchical_calibration_oneStrain import log_posterior_probability, dump_sampler_to_xarray, traceplot, plot_fit, hyperdistributions
from influenza_USA.NC_forecasts.utils import initialise_model, get_NC_influenza_data

##############
## Settings ##
##############

# model settings
state = 'North Carolina'                            # state we'd like to calibrate to
season = '2024-2025'                                # season to calibrate
sr = 'states'                                       # spatial resolution: 'states' or 'counties'
ar = 'full'                                         # age resolution: 'collapsed' or 'full'
dd = False                                          # vary contact matrix by daytype

# calibration settings
use_ED_visits = True                                                                        # use both ED admission (hospitalisation) and ED visits (ILI) data 
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2023-2024']    # season to include in calibration excercise
start_calibration_month = 10                                                                # start calibration on month 10, day 1
end_calibration_month = 5                                                                   # end calibration on month 5, day 1

# Define number of chains
max_n = 10000
n_chains = 400
pert = 0.05
run_date = datetime.today().strftime("%Y-%m-%d")
identifier = 'test'
print_n = 1000
backend = None
discard = 0
thin = 1

# Make folder structure
if use_ED_visits:
    samples_path=fig_path=f'../../../data/interim/calibration/hierarchical-training/{name2fips(state)}/use_ED_visits/' # Path to backend
else:
    samples_path=fig_path=f'../../../data/interim/calibration/hierarchical-training/{name2fips(state)}/not_use_ED_visits/' # Path to backend
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

model = initialise_model(strains=False, spatial_resolution=sr, age_resolution=ar, state=state, season='average', distinguish_daytype=dd)

##########################################
## Setup posterior probability function ##
##########################################

# define model states we want to calibrate to
states_model = ['I_inc', 'H_inc']
states_data = ['I_inc', 'H_inc']

# cut out 'I_inc'
if not use_ED_visits:
    states_model = states_model[1:]
    states_data = states_data[1:]

# define model parameters to calibrate to every season and their bounds
# not how we're not cutting out the parameters associated with the ED visit data
pars_model_names = ['rho_i', 'T_h', 'rho_h', 'beta', 'f_R', 'f_I', 'delta_beta_temporal']
pars_model_bounds = [(1e-5,0.15), (0.1, 15), (1e-5,0.015), (0.001,0.06), (0.001,0.999), (1e-9,1e-3), (-1,1)]
_, pars_model_shapes = validate_calibrated_parameters(pars_model_names, model.parameters)
n_pars = sum([v[0] for v in pars_model_shapes.values()])

# define hyperparameters 
hyperpars_shapes = {
    'rho_i_a': (1,), 'rho_i_scale': (1,),
    'T_h_rate': (1,),
    'rho_h_a': (1,), 'rho_h_scale': (1,),
    'beta_mu': (1,), 'beta_sigma': (1,),
    'f_R_a': (1,), 'f_R_b': (1,),
    'f_I_a': (1,), 'f_I_scale': (1,),
    'delta_beta_temporal_mu': (len(model.parameters['delta_beta_temporal']),), 'delta_beta_temporal_sigma': (len(model.parameters['delta_beta_temporal']),),
}

####################################
## Fetch initial guess parameters ##
####################################

# get independent fit parameters
pars_model_0 = pd.read_csv('../../../data/interim/calibration/calibrated_parameters-oneStrain.csv', index_col=0)[seasons]

# parameters
pars_0 = list(pars_model_0.transpose().values.flatten())

# hyperparameters
hyperpars_0 = [
               4.2, 1.0e-02, # rho_i
               2.0, # T_h
               4.0, 1.5e-03, # rho_h
               0.025, 0.005, # beta
               5, 5, # f_R
               5, 2.6e-05, # f_I
               -0.070, -0.043, -0.023, 0.010, 0.12, -0.125, 0.024, 0.102, 0.033, 0.047, 0.036, -0.037, # delta_beta_temporal_mu
               0.026, 0.038, 0.026, 0.049, 0.067, 0.099, 0.086, 0.073, 0.122, 0.083, 0.164, 0.078, # delta_beta_temporal_sigma
                ]

# combine
theta_0 = hyperpars_0 + pars_0

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
    with get_context("spawn").Pool(processes=16) as pool:
        # setup sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_probability, backend=backend, pool=pool,
                                        moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1),
                                               (emcee.moves.StretchMove(live_dangerously=True), 0.50)],
                                        args=(model, datasets, pars_model_names, pars_model_bounds, hyperpars_shapes, states_model, states_data)
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
                plot_fit(model, datasets, samples, pars_model_names, samples_path, identifier, run_date)
                # ..generate traceplots
                traceplot(samples, pars_model_shapes, hyperpars_shapes, samples_path, identifier, run_date)