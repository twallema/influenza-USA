"""
A script to format the weekly flu hospitalisation incidence in the US states in `data/raw/cases/weekly_flu_incid_complete.csv`
"""

############################
## Load required packages ##
############################

import os
import pandas as pd
from influenza_USA.SVI2RHD.utils import name2fips

########################
## Load & format data ##
########################

# load names of all demography files
data = pd.read_csv(os.path.join(os.getcwd(), '../../raw/cases/hosp-admissions_FluSurvNet_USA_09-24_raw.csv'))

# drop invalid state + US as a whole
data = data[data['state'] != 'US']
data = data.dropna(subset=['state'])

# guarantee the data starts at week 30 when simulation starts (August)
data = data[((data['mmwr_week'] < 18) | (data['mmwr_week'] > 30))]

# BUT: data actually starts in week 40 --> extend dataranges with zeros
data = data.fillna(0)

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
data.to_csv(os.path.join(os.getcwd(),'../../interim/cases/hosp-admissions_FluSurvNet_USA_09-24.csv'), index=False)
