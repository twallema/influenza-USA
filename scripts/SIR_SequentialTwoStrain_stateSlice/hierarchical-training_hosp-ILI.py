"""
This script does..
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import emcee
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import get_context
from influenza_USA.shared.utils import name2fips
from pySODM.optimization.objective_functions import validate_calibrated_parameters
from influenza_USA.SIR_SequentialTwoStrain.hierarchical_calibration import log_posterior_probability, dump_sampler_to_xarray, traceplot, plot_fit, hyperdistributions
from influenza_USA.SIR_SequentialTwoStrain.utils import initialise_SIR_SequentialTwoStrain, get_NC_influenza_data

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
seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2023-2024']
start_calibration_month = 10 
end_calibration_month = 5

# Define number of chains
max_n = 5000
n_chains = 500
pert = 0.05
run_date = datetime.today().strftime("%Y-%m-%d")
identifier = 'test'
print_n = 2
backend = None
discard = 0
thin = 1

# Make folder structure
samples_path=fig_path=f'../../data/interim/calibration/hierarchical-training/{name2fips(state)}/' # Path to backend
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

model = initialise_SIR_SequentialTwoStrain(spatial_resolution=sr, age_resolution=ar, state=state, season='average', distinguish_daytype=dd)

##########################################
## Setup posterior probability function ##
##########################################

# define model states we want to calibrate to
states_model = ['I_inc', 'H1_inc', 'H2_inc']
states_data = ['I_inc', 'H_inc_A', 'H_inc_B']

# define model parameters to calibrate to every season and their bounds
pars_model_names = ['T_h', 'rho_i', 'rho_h1', 'rho_h2', 'beta1', 'beta2', 'f_R1_R2', 'f_R1', 'f_I1', 'f_I2', 'delta_beta_temporal']
pars_model_bounds = [(0.1, 15), (1e-5,0.05), (1e-5,0.01), (1e-5,0.01), (0.001,0.05), (0.001,0.05), (0.001,0.999), (0.001,0.999), (1e-7,5e-4), (1e-7,5e-4), (-0.5,0.5)]
_, pars_model_shapes = validate_calibrated_parameters(pars_model_names, model.parameters)
n_pars = sum([v[0] for v in pars_model_shapes.values()])

# define hyperparameters 
hyperpars_shapes = {
    'T_h_rate': (1,),
    'rho_i_a': (1,), 'rho_i_scale': (1,),
    'rho_h1_a': (1,), 'rho_h1_scale': (1,),
    'rho_h2_a': (1,), 'rho_h2_scale': (1,),
    'beta1_mu': (1,), 'beta1_sigma': (1,),
    'beta2_mu': (1,), 'beta2_sigma': (1,),
    'f_R1_R2_a': (1,), 'f_R1_R2_b': (1,),
    'f_R1_a': (1,), 'f_R1_b': (1,),
    'f_I1_a': (1,), 'f_I1_scale': (1,),
    'f_I2_a': (1,), 'f_I2_scale': (1,),
    'delta_beta_temporal_mu': (len(model.parameters['delta_beta_temporal']),), 'delta_beta_temporal_sigma': (len(model.parameters['delta_beta_temporal']),),
}

####################################
## Fetch initial guess parameters ##
####################################

# get independent fit parameters
pars_model_0 = pd.read_csv('../../data/interim/calibration/calibrated_parameters.csv', index_col=0)[seasons]

# parameters
pars_0 = list(pars_model_0.transpose().values.flatten())
# print avg/stdev to inform hyperparameters
#print(pd.concat([pars_model_0.transpose().mean().rename('avg'), pars_model_0.transpose().std().rename('stdev')], axis=1))

# hyperparameters
hyperpars_0 = [3.6, # T_h
               3.5, 5.2e-03, # rho_i
               4.8, 5.7e-04, # rho_h1
               4.8, 5.7e-04, # rho_h2
               0.023, 0.006, # beta1
               0.023, 0.004, # beta2
               5, 5, # f_R1_R2
               5, 5, # f_R1
               2, 9e-05, # f_I1
               2, 9e-05, # f_I2
               -0.08, -0.05, -0.05, 0.005, 0.07, -0.11, 0.02, 0.11, 0.05, 0.06, 0.04, -0.04, # delta_beta_temporal_mu
               0.04, 0.04, 0.04, 0.07, 0.07, 0.09, 0.08, 0.08, 0.09, 0.07, 0.16, 0.07, # delta_beta_temporal_sigma
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
                #hyperdistributions(samples, pars_model_shapes, pars_model_bounds, 300)
                # ..generate goodness-of-fit
                #plot_fit(model, datasets, samples, pars_model_names)
                # ..generate traceplots
                #traceplot(samples, pars_model_shapes, hyperpars_shapes)