"""
This script calibrates an age-stratified spatially-explicit two-strain sequential infection SIR model to North Carolina hospital admission data
It automatically calibrates to incrementally larger datasets between `start_calibration` and `end_calibration`
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import arviz as az
import pymc as pm
import pytensor as pt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as datetime
from influenza_USA.SIR_SequentialTwoStrain.utils import initialise_SIR_SequentialTwoStrain, get_NC_influenza_data # influenza model
from pySODM.optimization.utils import assign_theta, add_poisson_noise

##############
## Settings ##
##############

# model settings
state = 'North Carolina'                            # state we'd like to calibrate to
sr = 'states'                                       # spatial resolution: 'states' or 'counties'
ar = 'full'                                         # age resolution: 'collapsed' or 'full'
dd = False                                          # vary contact matrix by daytype
season = '2019-2020'                                # calibration to season
season_start = int(season[0:4])                     # start of season
start_calibration = datetime(season_start, 10, 1)    # date simulation is started
end_calibration = datetime(season_start+1, 5, 1)    # date calibration (comparison to ground thruth) is ended

#####################
## Load NC dataset ##
#####################

data_interim = get_NC_influenza_data(start_calibration, end_calibration, season)

#################
## Setup model ##
#################

epi_model = initialise_SIR_SequentialTwoStrain(spatial_resolution=sr, age_resolution=ar, state=state, season='average', distinguish_daytype=dd)

#####################
## Calibrate model ##
#####################

pars = ['T_h', 'rho_i', 'rho_h1', 'rho_h2', 'beta1', 'beta2', 'f_R1_R2', 'f_R1', 'f_I1', 'f_I2', 'delta_beta_temporal']

# decorator with input and output types a Pytensor double float tensors
## import packages
from pytensor.tensor import TensorType
from pytensor.compile.ops import as_op
## define wrapper input and output type
dscalar = TensorType(dtype='float64', broadcastable=())  # A scalar
dvector = TensorType(dtype='float64', broadcastable=(False,))  # A 1D vector
dmatrix = TensorType(dtype='float64', broadcastable=(False, False))  # A 2D matrix
## define forward simulation wrapper
@as_op(itypes=[dvector,], otypes=[dmatrix,])
def pytensor_simulate_model_extract_output(theta):
    # assign the parameters (flat list of parameters -> conversion to correct shape)
    epi_model.parameters = assign_theta(epi_model.parameters, pars, theta.tolist())
    # forward simulate the model
    simout = epi_model.sim([start_calibration, end_calibration], method='RK23', rtol=5e-3)
    # get right state, collapse age groups and locations, resample to dates in data
    simout = np.stack(
        [
            simout['H1_inc'].sum(dim=['age_group', 'location']).interp(date=data_interim.index).values,
            simout['H2_inc'].sum(dim=['age_group', 'location']).interp(date=data_interim.index).values,
            simout['I_inc'].sum(dim=['age_group', 'location']).interp(date=data_interim.index).values
        ],
        axis=-1
    )
    return simout

# define a custom log likelihood function
def gammaln_approx(x, num_terms=3):
    """ Approximate scipy.special.gammaln using a series expansion - validated
    """
    # Constants
    pi = 3.1415
    bernoulli_numbers = [1/6, -1/30, 1/42, -1/30, 5/66]  # Precomputed B_2, B_4, ..., B_10
    # Basic Stirling's approximation
    log_gamma = (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * pi)
    # Add higher-order terms
    for k in range(1, num_terms + 1):
        term = bernoulli_numbers[k-1] / (2 * k * (2 * k - 1) * x**(2 * k - 1))
        log_gamma += term
    return log_gamma

def weighted_ll(value, mu):
    """ Weighted Poisson likelihoods 
    """
    ll_0 = 1/pm.math.max(value[:,0]) * (- pm.math.sum(mu[:,0]) + pm.math.sum(value[:,0]*pm.math.log(mu[:,0])) - pm.math.sum(gammaln_approx(value[:,0])))
    ll_1 = 1/pm.math.max(value[:,1]) * (- pm.math.sum(mu[:,1]) + pm.math.sum(value[:,1]*pm.math.log(mu[:,1])) - pm.math.sum(gammaln_approx(value[:,1])))
    ll_2 = 1/pm.math.max(value[:,2]) * (- pm.math.sum(mu[:,2]) + pm.math.sum(value[:,2]*pm.math.log(mu[:,2])) - pm.math.sum(gammaln_approx(value[:,2])))
    return ll_0 + ll_1 + ll_2

# define inference model
with pm.Model() as model:
    ## Define priors
    T_h = pm.Uniform('T_h', lower=0.5, upper=14, initval=3.6)
    rho_i = pm.Uniform('rho_i', lower=1e-4, upper=0.15, initval=0.0188)
    rho_h1 = pm.Uniform('rho_h1', lower=1e-4, upper=0.01, initval=0.00228)
    rho_h2 = pm.Uniform('rho_h2', lower=1e-4, upper=0.01, initval=0.00153)
    beta1 = pm.Uniform('beta1', lower=5e-3, upper=0.06, initval=0.0214)
    beta2 = pm.Uniform('beta2', lower=5e-3, upper=0.06, initval=0.0213)
    f_R1_R2 = pm.Uniform('f_R1_R2', lower=0.01, upper=0.99, initval=0.50)
    f_R1 = pm.Uniform('f_R1', lower=0.01, upper=0.99, initval=0.49)
    f_I1 = pm.Uniform('f_I1', lower=1e-7, upper=1e-3, initval=1.9e-5)
    f_I2 = pm.Uniform('f_I2', lower=1e-7, upper=1e-3, initval=1.0e-4)
    delta_beta_temporal = pm.Normal('delta_beta_temporal', mu=0, sigma=0.10, initval=np.array([1.7e-02, 1.2e-02, 9.0e-04, 4.8e-03, 8.9e-03, -5.0e-02, 1.2e-01, 3.1e-03, 6.9e-03, 8.2e-03, -2.6e-02, 1.1e-02]), shape=12)
    ## Simulate the model
    simout = pytensor_simulate_model_extract_output(pm.math.concatenate([pm.math.stack([T_h, rho_i, rho_h1, rho_h2, beta1, beta2, f_R1_R2, f_R1, f_I1, f_I2]), delta_beta_temporal]))
    ## Compute likelihood using custom function
    dens = pm.DensityDist(
        'weight_ll',
        simout,
        logp=weighted_ll,
        observed=np.stack([data_interim['H_inc_A'].values, data_interim['H_inc_B'].values, data_interim['I_inc'].values], axis=-1),
    )

# tryout sampling (DEMetroplis didn't work)
vars_list = list(model.values_to_rvs.keys())[:-1]
chains = 44
tune = 3000
draws = 2000
with model:
    trace = pm.sample(step=[pm.DEMetropolisZ(vars_list)], draws=draws, tune=tune, chains=chains, cores=16)

# print diagnostics
print(az.summary(trace, round_to=2))

# show traceplot
ax = az.plot_trace(trace)
plt.savefig('trace.pdf')
plt.close()

# save trace
trace.to_netcdf("pyMC_inference.nc")

# define draw function
import random
def draw_function(parameters, posterior):
    # random chain and draw
    i = random.randint(0, len(posterior.coords['chain'])-1)
    j = random.randint(0, len(posterior.coords['draw'])-1)
    # assign parameters
    for var in list(posterior.data_vars):
        if var != 'sigma':
            parameters[var] = posterior[var].sel({'chain': i, 'draw': j}).values
    return parameters

# Simulate model
simout = epi_model.sim([start_calibration, end_calibration], N=500,
                    draw_function=draw_function, draw_function_kwargs={'posterior': trace.posterior}, processes=1)

# Add sampling noise
simout = add_poisson_noise(simout)

# visualise
fig,ax=plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.3, 11.7))
## ILI
ax[0].fill_between(simout['date'].values,
                   simout['I_inc'].sum(dim=['age_group', 'location']).quantile(q=1-0.5/2, dim='draws'),
                   simout['I_inc'].sum(dim=['age_group', 'location']).quantile(q=0.5/2, dim='draws'),
                   color='blue', alpha=0.20)
ax[0].fill_between(simout['date'].values,
                   simout['I_inc'].sum(dim=['age_group', 'location']).quantile(q=1-0.95/2, dim='draws'),
                   simout['I_inc'].sum(dim=['age_group', 'location']).quantile(q=0.95/2, dim='draws'),
                   color='blue', alpha=0.15)
ax[0].scatter(data_interim.index, data_interim['I_inc'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
ax[0].set_title('Influenza-like illness')
## Hosp Flu A
ax[1].fill_between(simout['date'].values,
                   simout['H1_inc'].sum(dim=['age_group', 'location']).quantile(q=1-0.5/2, dim='draws'),
                   simout['H1_inc'].sum(dim=['age_group', 'location']).quantile(q=0.5/2, dim='draws'),
                   color='blue', alpha=0.20)
ax[1].fill_between(simout['date'].values,
                   simout['H1_inc'].sum(dim=['age_group', 'location']).quantile(q=1-0.95/2, dim='draws'),
                   simout['H1_inc'].sum(dim=['age_group', 'location']).quantile(q=0.95/2, dim='draws'),
                   color='blue', alpha=0.15)
ax[1].scatter(data_interim.index, data_interim['H_inc_A'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
ax[1].set_title('Hospitalisations (Flu A)')
## Hosp Flu B
ax[2].fill_between(simout['date'].values,
                   simout['H2_inc'].sum(dim=['age_group', 'location']).quantile(q=1-0.5/2, dim='draws'),
                   simout['H2_inc'].sum(dim=['age_group', 'location']).quantile(q=0.5/2, dim='draws'),
                   color='blue', alpha=0.20)
ax[2].fill_between(simout['date'].values,
                   simout['H2_inc'].sum(dim=['age_group', 'location']).quantile(q=1-0.95/2, dim='draws'),
                   simout['H2_inc'].sum(dim=['age_group', 'location']).quantile(q=0.95/2, dim='draws'),
                   color='blue', alpha=0.15)
ax[2].scatter(data_interim.index, data_interim['H_inc_B'], color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
ax[2].set_title('Hospitalisations (Flu B)')
plt.savefig('goodness-fit.pdf')
plt.show()
plt.close()