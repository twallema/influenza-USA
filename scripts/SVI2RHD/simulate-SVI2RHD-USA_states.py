"""
This script simulates an age-stratified, spatially-explicit SVI2HRD model for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import matplotlib.pyplot as plt
from datetime import datetime 
from influenza_USA.SVI2RHD.utils import initialise_SVI2RHD # influenza model

##############
## Settings ##
##############

# model settings
start_sim = datetime(2014, 11, 1)
end_sim = datetime(2015, 5, 1)
season = '2014-2015'                    # '2017-2018' or '2019-2020'
vaccine_waning = 'off'                  # 'no_waning' vs. 'waning_180'
sr = 'states'                           # spatial resolution: 'states' or 'counties'
ar = 'full'                             # age resolution: 'collapsed' or 'full'
dd = True                              # vary contact matrix by daytype

#################
## Setup model ##
#################

model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, season=season, vaccine_waning=vaccine_waning, distinguish_daytype=dd, start_sim=start_sim)

######################
## Visualize result ##
######################

# Simulate model
out = model.sim([start_sim, end_sim])

# Visualize
fig, ax = plt.subplots(2, 1, figsize=(8.3, 11.7/2))
ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
ax[1].plot(out['date'], out['V'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
ax[0].set_ylabel('Model state "H_inc"')
ax[1].set_ylabel('Model state "V"')
plt.show()
plt.close()
