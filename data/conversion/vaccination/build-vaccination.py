"""
A script to aggregate the age-and space stratified vaccination rates for 2017-2018 into one dataframe
"""

############################
## Load required packages ##
############################

import sys,os
import pandas as pd

###################################################
## Construct a dataframe with the desired format ##
###################################################

# use demography for the age groups and spatial units
demography = pd.read_csv(os.path.join(os.getcwd(), '../../interim/demography/demography_states_2023.csv'), dtype={'state': str}) # state, age, population
ages = demography['age'].unique()
states = demography['state'].unique()
# use one of the datafiles for the dates
vaccination = pd.read_csv(os.path.join(os.getcwd(), '../../raw/vaccination/vacc_Flu_2024_R1_age18to49_dose1_reported_2017.csv')) 
dates = vaccination['date'].unique()
# pre-allocate output (date, age, state)
out = pd.Series(0, index=pd.MultiIndex.from_product([dates, ages, states], names=['date', 'age', 'state']), name='vaccination_rate')

###############################
## Load & format county data ##
###############################

# load names of all demography files
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
    # melt from wide to long format
    df_melted = data.melt(ignore_index=False, var_name='state', value_name='vaccination_rate').reset_index().set_index(['date', 'state'])
    df_melted = df_melted.squeeze()
    # fill in data
    out.loc[slice(None), age, slice(None)] = df_melted.values

#################
## Save result ##
#################

out.to_csv(os.path.join(os.getcwd(),'../../interim/vaccination/vaccination_rates_2017-2018.csv'))