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
from influenza_USA.SVI2RHD.utils import initialise_SVI2RHD # influenza model

##############
## Settings ##
##############

# model settings
start_sim = datetime(2014, 11, 1)
end_sim = datetime(2015, 5, 1)
season = '2014-2015'                    # '2017-2018' or '2019-2020'
waning = 'no_waning'                    # 'no_waning' vs. 'waning_180'
sr = 'counties'                           # spatial resolution: 'states' or 'counties'
ar = 'full'                             # age resolution: 'collapsed' or 'full'
dd = False                              # vary contact matrix by daytype

#################
## Setup model ##
#################

model = initialise_SVI2RHD(spatial_resolution=sr, age_resolution=ar, season=season, distinguish_daytype=dd, start_sim=start_sim)

# set up right waning parameters
if waning == 'no_waning':
    model.parameters['e_i'] = 0.2
    model.parameters['e_h'] = 0.5
    model.parameters['T_v'] = 10*365
elif waning == 'waning_180':
    model.parameters['e_i'] = 0.2
    model.parameters['e_h'] = 0.75
    model.parameters['T_v'] = 365/2

######################
## Visualize result ##
######################

# Simulate model
out = model.sim([start_sim, end_sim])

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(8.3, 11.7/4))
ax.plot(out['date'], 7*out['H_inc'].sum(dim=['age_group', 'location']), color='blue', alpha=1, linewidth=2)
plt.show()
plt.close()
