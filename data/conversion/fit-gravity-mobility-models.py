"""
A script to calibrate a departure-diffusion power law gravity model to the long-format mobility data originating from the commuter survey or SafeGraph data.
Made because the R mobility package was unfeasibly slow during the inference.
"""

##############
## Settings ##
##############

n_mcmc = 100
n_burn = 50
n_thin = 2
multip_chains = 5
dataset_name = 'mobility_cellphone_09032020_counties_longform.csv'

############################
## Load required packages ##
############################

import sys, os
import emcee
import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt
from pySODM.optimization.mcmc import perturbate_theta

##################################################################
## Load long-format mobility data and convert into model inputs ##
##################################################################

data = pd.read_csv(os.path.join(os.getcwd(), f'../interim/mobility/intermediates/{dataset_name}'), dtype={'origin': str, 'destination': str})
M = data.pivot(index='origin', columns='destination', values='commuters').values                    # mobility data
D = data.pivot(index='origin', columns='destination', values='distance_km').values                  # distance matrix
N = data.drop_duplicates(subset='origin')['origin_population'].reset_index(drop=True).values        # population vector

##########################################################
## Define a departure-diffusion power law gravity model ##
##########################################################

def depdiff_powerlawgravity(alpha, beta, theta, omega, gamma, D_ij, N_i, stochastic=False):
    
    if stochastic == True:
        # sample probability of travel 
        tau_i = np.random.beta(alpha, beta, size=D_ij.shape[0])
    else:
        # use expectation value
        tau_i = np.ones(D_ij.shape[0]) * alpha / (alpha+beta)

    # compute pi_ij
    N_ij = np.tile(N, (D_ij.shape[0], 1))
    T = (N_ij ** omega) / (D_ij ** gamma)
    np.fill_diagonal(T, 0)
    sum_j = np.sum(T, axis=1)
    pi_ij = T / np.transpose(np.tile(sum_j, (D_ij.shape[0], 1)))

    # compute off-diagonal elements
    M_hat = theta * N_i[:, np.newaxis] * tau_i[:, np.newaxis] * pi_ij 

    # fill in diagonal
    np.fill_diagonal(M_hat, theta * N_i * (1-tau_i))

    return M_hat

###############################
## Define objective function ##
###############################

# uniform priors to contraint parameters
def log_prior_uniform(x, bounds):
    prob = 1/(bounds[1]-bounds[0])
    condition = bounds[0] <= x <= bounds[1]
    if condition == True:
        return 0
    else:
        return -np.inf

# poisson likelihood
def poisson_ll(ymodel, ydata):
    # offset negative values
    if ((np.min(ymodel) < 0) | (np.min(ydata) < 0)):
        offset_value = (-1 - 1e-6)*np.min([np.min(ymodel), np.min(ydata)])
        ymodel += offset_value
        ydata += offset_value
    elif ((np.min(ymodel) == 0) | (np.min(ydata) == 0)):
        ymodel += 1e-6
        ydata += 1e-6
    return - np.sum(ymodel) + np.sum(ydata*np.log(ymodel)) - np.sum(gammaln(ydata))

# RMSE likelihood
def RMSE_ll(ymodel, ydata):
    return - np.sqrt(len(ydata) * np.sum((ymodel-ydata)**2) )

def log_posterior_probability(theta):

    # simulate model
    ymodel = depdiff_powerlawgravity(theta[0], theta[1], theta[2], theta[3], theta[4], D, N) 

    # compute log posterior probability
    lp=0
    # prior probabilities
    for i,par in enumerate(theta):
        lp += log_prior_uniform(par, bounds[i])
    # likelihood
    lp += poisson_ll(ymodel[~np.isnan(ydata)], ydata[~np.isnan(ydata)])

    return lp

###################
## Run inference ##
###################

# define parameters
theta = np.array([20,20,2,1,1])
bounds = [(0, 100), (0, 100), (0, 10), (0, np.inf), (0, np.inf)] # alpha, beta, theta, omega, gamma
ydata = M

# perturbate initial estimate
ndim, nwalkers, pos = perturbate_theta(theta, pert=[0.10,0.10,0.10,0.10,0.10], multiplier=multip_chains, bounds=bounds)

# run inference in parallel
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior_probability, args=()
)
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_probability, pool=pool)
        sampler.run_mcmc(pos, n_mcmc, progress=True)

    # traceplot
    fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["alpha", "beta", "theta", 'omega', 'gamma']
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.show()
    plt.close()

    # autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
    except:
        pass

    # get flat samples
    flat_samples = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)

    # cornerplot
    import corner
    fig = corner.corner(
        flat_samples, labels=labels,
    )
    plt.show()
    plt.close()

    ################################################
    ## Compare estimated mobility matrix and data ##
    ################################################

    # simulate model
    theta = np.mean(flat_samples, axis=0)
    ymodel = depdiff_powerlawgravity(theta[0], theta[1], theta[2], theta[3], theta[4], D, N) 

    # normalise model prediction and data with number of inhabitants
    ymodel = ymodel / N[:, np.newaxis]
    ydata = ydata / N[:, np.newaxis]

    # slice where data is present
    ymodel = ymodel[~np.isnan(ydata)]
    ydata = ydata[~np.isnan(ydata)]

    # summary statistics
    print(f"alpha: {theta[0]:.2f}, beta: {theta[1]:.2f}, 'theta: {theta[2]:.2f}, 'omega': {theta[3]:.2f}, 'gamma': {theta[4]:.2f}")
    R2 = 1 - np.sum((ymodel-ydata)**2) / np.sum((np.mean(ydata) - ydata)**2)
    print(f"R-squared: {R2:.2f}")
    MAPE = 100*np.sum(np.abs(ymodel[ydata!=0]-ydata[ydata!=0])/ydata[ydata!=0])*(1/len(ydata[ydata!=0]))
    print(f"MAPE: {MAPE:.2f}")
    
    # transform data
    ymodel = np.log10(ymodel+1e-99)
    ydata = np.log10(ydata+1e-99)

    # compute density
    from scipy.stats import gaussian_kde
    density = gaussian_kde(ymodel)
    xs = np.linspace(-8, 1, 500)

    fig, ax = plt.subplots(nrows=2)
    # density comparison
    ax[0].set_title('Comparison data-model')
    ax[0].hist(ydata, bins=500, density=True, color='black', alpha=0.5)
    ax[0].plot(xs, density(xs), color='red')
    ax[0].set_xlim([-8,1])
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax[0].text(0.05, 0.95, f"$R^2 = $ {R2:.2f}\nMAPE = {MAPE:.1f}%", transform=ax[0].transAxes, fontsize=12,verticalalignment='top', bbox=props)
    
    # predicted versus data
    ax[1].scatter(ydata, ymodel, marker='o', s=10, color='black', alpha=0.2)
    ax[1].plot(range(-8,2), range(-8,2), color='red', linestyle='--')
    ax[1].set_xlim([-8,1])
    ax[1].set_ylim([-8,1])
    ax[1].set_xlabel('log10 norm. trips (data)')
    ax[1].set_ylabel('log10 norm. trips (model)')
    plt.savefig(os.path.join(os.getcwd(), f'../interim/mobility/fitted_models/departure_diffusion_power_gravitation/comparison.png'), dpi=200)
    plt.show()
    plt.close()


    ################################################
    ## Compare estimated mobility matrix and data ##
    ################################################

    # simulate model
    ymodel = depdiff_powerlawgravity(theta[0], theta[1], theta[2], theta[3], theta[4], D, N, stochastic=False) 

    # convert to pandas dataframe
    origin_idx, destination_idx = np.meshgrid(data['origin'].unique(), data['destination'].unique(), indexing='ij')

    # flatten indices and simulation
    origin_flat = origin_idx.flatten()
    destination_flat = destination_idx.flatten()
    ymodel_flat = ymodel.flatten()

    # fill in long format
    df_long = pd.DataFrame({'origin': origin_flat,'destination': destination_flat,'trips': ymodel_flat})

    # pivot to get indexed matrix
    matrix_counties = df_long.pivot(index='origin', columns='destination', values='trips')
    matrix_counties.to_csv(os.path.join(os.getcwd(), f'../interim/mobility/fitted_models/departure_diffusion_power_gravitation/matrix_counties.csv'), index=True)

    # aggregate to state level
    df_long['state_o'] = df_long['origin'].apply(lambda x: f"{x[0:2]:02}")          # add origin state code
    df_long['state_d'] = df_long['destination'].apply(lambda x: f"{x[0:2]:02}")     # add destination state code
    agg = df_long.groupby(by=['state_o', 'state_d'])['trips'].sum().reset_index()

    # pivot into a matrix
    matrix_states = agg.pivot(index='state_o', columns='state_d', values='trips')
    matrix_states.to_csv(os.path.join(os.getcwd(), f'../interim/mobility/fitted_models/departure_diffusion_power_gravitation/matrix_states.csv'), index=True)
