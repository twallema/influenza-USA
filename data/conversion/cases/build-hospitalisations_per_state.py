"""
A script to format the weekly flu hospitalisation incidence in the US states in `data/raw/cases/weekly_flu_incid_complete.csv`
"""

############################
## Load required packages ##
############################

import os
import pandas as pd
from influenza_USA.SVIR.utils import name2fips

########################
## Load & format data ##
########################

# load names of all demography files
data = pd.read_csv(os.path.join(os.getcwd(), '../../raw/cases/weekly_flu_incid_complete.csv'))

# drop invalid state + US as a whole
data = data[data['state'] != 'US']
data = data.dropna(subset=['state'])

# guarantee the data starts at week 40
data = data[data['mmwr_week'] >= 40]

# slice only what we need
data = data[['season_start', 'date', 'state', 'incidH']]

# rename 'incidH' to 'H_inc' (to match the model state)
data = data.rename(columns={'incidH': 'H_inc'})

# convert state name to fips code
data['location'] = data['state'].apply(name2fips)

# drop the state column
data = data[['season_start', 'date', 'location', 'H_inc']]

# drop the nan's
data = data.dropna()

# save dataframe
data.to_csv(os.path.join(os.getcwd(),'../../interim/cases/hospitalisations_per_state.csv'), index=False)
