"""
This script simulates an age-stratified, spatially-explicit SVI2HRD model for the USA using pySODM
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 
from influenza_USA.SIR_SequentialTwoStrain.utils import initialise_SIR_SequentialTwoStrain # influenza model

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
dd = False                              # vary contact matrix by daytype

#################
## Setup model ##
#################

model = initialise_SIR_SequentialTwoStrain(spatial_resolution=sr, age_resolution=ar, season=season, distinguish_daytype=dd)

######################
## Visualize result ##
######################

# Simulate model
out = model.sim([start_sim, end_sim])

# Visualize
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8.3, 11.7/2))
# hospitalisations
ax[0].plot(out['date'], 7*out['H1_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2, label='strain I')
ax[0].plot(out['date'], 7*out['H2_inc'].sum(dim=['age_group', 'location']), color='red', alpha=1, linewidth=2, label='strain II')
ax[0].plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='black', alpha=1, linewidth=2, label = 'sum')
ax[0].set_ylabel('Weekly hospitalisations (-)')
ax[0].legend()
# ratios
ax[1].plot(out['date'], 100*out['H1_inc'].sum(dim=['age_group', 'location']) / (out['H1_inc'].sum(dim=['age_group', 'location'])+out['H2_inc'].sum(dim=['age_group', 'location'])),
           color='black', alpha=1, linewidth=2)
ax[1].set_ylabel('ratio strain I (%)')
ax[1].set_ylim([0,100])
plt.show()
plt.close()
