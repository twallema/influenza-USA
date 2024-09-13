import json
import random
import matplotlib.pyplot as plt
from influenza_USA.SVIR.utils import initialise_SVI2RHD # influenza model

##############
## Settings ##
##############

# model settings
season = '2017-2018'                        # season: '17-18' or '18-19'
sr = 'states'                               # spatial resolution: 'collapsed', 'states' or 'counties'
ar = 'full'                                 # age resolution: 'collapsed' or 'full'
dd = False                                  # vary contact matrix by daytype
stoch = False                               # ODE vs. tau-leap

# calibration settings
identifier = f'waning_180'  # identifier calibration
rundate = '2024-09-13'

# scenario settings
N = 100                                     # number of repeated simulations
parameter_name = 'vaccine_rate_modifier'    # parameter to vary
parameter_values = [0.8, 0, 1.2]            # values to use
colors = ['red', 'black', 'green']
labels = ['-20%', '0%', '20%']
start_sim = '2024-08-01'
end_sim = '2025-07-01'                      # start and end of simulation
conf_int = 0.05                             # confidence level used to visualise model fit

###############################
## Set up posterior sampling ##
###############################

samples_dict = json.load(open(f'../data/interim/calibration/{season}/{identifier}/{identifier}_SAMPLES_{rundate}.json'))

# todo: get season, sr, ar, dd, stoch from samples dict

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

model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, distinguish_daytype=dd, stochastic=stoch, start_sim=start_sim)

########################
## Simulate scenarios ##
########################

output = []
for val in parameter_values:
    # alter parameter value
    model.parameters[parameter_name] = val
    # simulate model
    output.append(model.sim([start_sim, end_sim], N=N, draw_function=draw_fcn, draw_function_kwargs={'samples': samples_dict}))

#########################
## Visualise scenarios ##
#########################

date = output[0]['date']

fig, axs = plt.subplots(3,1, figsize=(8.3,11.7/3))
# overall
for i in range(len(parameter_values)):
    axs[0].plot(date, (7*output[i]['H_inc']).sum(dim=['age_group', 'location']).mean(dim='draws'), linewidth=2, color=colors[i], label=labels[i])
    axs[0].fill_between(date, (7*output[i]['H_inc']).sum(dim=['age_group', 'location']).quantile(q=conf_int/2, dim='draws'),
                        (7*output[i]['H_inc']).sum(dim=['age_group', 'location']).quantile(q=1-conf_int/2, dim='draws'), color=colors[i], alpha=0.2)
axs[0].legend()

# vaccinated population
for i in range(len(parameter_values)):
    axs[1].plot(date, output[i]['V'].sum(dim=['age_group', 'location']).mean(dim='draws'), linewidth=2, color=colors[i], label=labels[i])
axs[1].legend()

# age breakdown of 0% scenario

# cumulative hospitalisations
# demo = np.array([18608139, 54722401, 141598551, 63172279, 60019216])
# axs[1].bar(out.coords['age_group'].values, (out['H_inc'].sum(dim=['location'])).cumsum(dim='date').isel(date=-1)/demo * 100000, color='black', alpha=0.6, linewidth=2)
# axs[1].set_ylabel('per 100K inhab.')
# axs[1].set_title('Cumulative hospitalisations')

plt.tight_layout()
plt.show()
plt.close()
