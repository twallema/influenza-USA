"""
A script to aggregate the age-and space stratified vaccination rates for 2017-2018 into one dataframe
"""

############################
## Load required packages ##
############################

import os
import numpy as np
import pandas as pd

###################################################
## Construct a dataframe with the desired format ##
###################################################

# use demography for the age groups and spatial units
demography = pd.read_csv(os.path.join(os.getcwd(), '../../interim/demography/demography_states_2023.csv'), dtype={'fips': str}) # state, age, population
ages = demography['age'].unique()
states = demography['fips'].unique()
# use one of the datafiles for the dates
vaccination = pd.read_csv(os.path.join(os.getcwd(), '../../raw/vaccination/vacc_Flu_2024_R1_age18to49_dose1_reported_2017.csv'), parse_dates=True) 
dates = vaccination['date'].unique()

# pre-allocate output (date, age, state)
out = pd.Series(0, index=pd.MultiIndex.from_product([dates, ages, states], names=['date', 'age', 'state']), name='vaccination_rate', dtype=np.float64)

###############################
## Load & format county data ##
###############################

# load names of all vaccination files
file_names = os.listdir(os.path.join(os.getcwd(), '../../raw/vaccination/'))
# remove all hidden files (start with a .)
file_names = [file for file in file_names if not file.startswith('.')]
# sort to line up filenames with the age groups
file_names.sort()

# loop over datasets
for fn,age in zip(file_names,ages):
    # load dataset
    data = pd.read_csv(os.path.join(os.getcwd(), f'../../raw/vaccination/{fn}'), index_col=0) 
    # change column names to desired format (verified equality of used notations)
    data = data.rename(columns=dict(zip(data.columns, states)))
    # fill in data
    for state in states:
        out.loc[slice(None), age, state] = data[state].values

# vaccination data typically start in week 30
out = out.loc[slice('2017-08-01', None)]

#################
## Save result ##
#################

out.to_csv(os.path.join(os.getcwd(),'../../interim/vaccination/vaccination_rates_2017-2018.csv'))
