import json
import random
import numpy as np
import matplotlib.pyplot as plt
from influenza_USA.SVIR.utils import initialise_SVI2RHD # influenza model

##############
## Settings ##
##############

# calibration settings
season = '2017-2018'                        # season: '17-18' or '19-20'
waning = 'no_waning'                        # 'no_waning' vs. 'waning_180'
rundate = '2024-09-16'                      # calibration date

# scenario settings
N = 200                                         # number of repeated simulations
parameter_name = 'vaccine_rate_modifier'        # parameter to vary
parameter_values = [0.8, 1, 1.2]                # values to use
colors = ['red', 'black', 'green', 'blue']      # colors used in visualisation
labels = ['-20%', '0%', '+20%']                 # labels used in visualisation
start_sim = '2024-08-01'                        # start of visualisation
end_sim = '2025-06-07'                          # end of visualisation
conf_int = 0.05                                 # confidence level of visualisation

###############################
## Set up posterior sampling ##
###############################

# retrieve dictionary of samples
samples_dict = json.load(open(f'../data/interim/calibration/{season}/{waning}/{waning}_SAMPLES_{rundate}.json'))

# retrieve model settings
sr = samples_dict['spatial_resolution']
ar = samples_dict['age_resolution']
dd = samples_dict['distinguish_daytype']
stoch = samples_dict['stochastic']

# define draw function
def draw_fcn(parameters, samples):
    # Sample model parameters
    idx, parameters['beta'] = random.choice(list(enumerate(samples['beta'])))
    parameters['rho_h'] = samples['rho_h'][idx]
    parameters['rho_d'] = samples['rho_d'][idx]
    parameters['asc_case'] = samples['asc_case'][idx]
    return parameters
    
#################
## Setup model ##
#################

model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, season=season, distinguish_daytype=dd, stochastic=stoch, start_sim=start_sim)

if waning == 'no_waning':
    model.parameters['e_i'] = 0.2
    model.parameters['e_h'] = 0.5
    model.parameters['T_v'] = 10*365
elif waning == 'waning_180':
    model.parameters['e_i'] = 0.2
    model.parameters['e_h'] = 0.75
    model.parameters['T_v'] = 365/2

########################
## Simulate scenarios ##
########################

output = []
for val in parameter_values:
    # alter parameter value
    model.parameters[parameter_name] = val
    # simulate model
    simout = model.sim([start_sim, end_sim], N=N, draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict})
    output.append(simout)
    # save result
    simout.to_netcdf(f"{float(val)}_{waning}_{season}.nc")

#########################
## Visualise scenarios ##
#########################

date = output[0]['date']

fig, axs = plt.subplots(4,1, figsize=(8.3,11.7/2))
# overall
for i in range(len(parameter_values)):
    axs[0].plot(date, (7*output[i]['H_inc']).sum(dim=['age_group', 'location']).mean(dim='draws'), linewidth=2, color=colors[i], label=labels[i])
    axs[0].fill_between(date, (7*output[i]['H_inc']).sum(dim=['age_group', 'location']).quantile(q=conf_int/2, dim='draws'),
                        (7*output[i]['H_inc']).sum(dim=['age_group', 'location']).quantile(q=1-conf_int/2, dim='draws'), color=colors[i], alpha=0.2)
axs[0].legend()

# age breakdown of scenarios
demo = np.array([18608139, 54722401, 141598551, 63172279, 60019216])
for i in range(len(parameter_values)):
    axs[i+1].bar(output[i].coords['age_group'].values, (output[i]['H_inc'].mean(dim='draws').sum(dim=['location'])).cumsum(dim='date').isel(date=-1)/demo * 100000,
                 color='black', alpha=0.6, linewidth=2)
    axs[i+1].set_ylim([0, 400])
    axs[i+1].set_ylabel('per 100K inhab.')
    axs[i+1].set_title(f"Cumulative hosp. scenario {labels[i]}: {int(output[i]['H_inc'].mean(dim='draws').sum(dim=['location', 'age_group']).cumsum(dim='date').isel(date=-1).values)}")

plt.tight_layout()
plt.show()
plt.close()
