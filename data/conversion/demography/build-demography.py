"""
A script to build US state and county level demography files, stratified into `desired_age_groups`
"""

############################
## Load required packages ##
############################

import os
import pandas as pd

#####################
## Script settings ##
#####################

# define model's age groups
desired_age_groups = pd.IntervalIndex.from_tuples([(0,5),(5,18),(18,50),(50,65),(65,100)], closed='left')
# exclude states
state_FIPS_exclude = []

###############################
## Load & format county data ##
###############################

# load names of all demography files
file_names = os.listdir(os.path.join(os.getcwd(), '../../raw/demography/counties/'))
# remove all hidden files (start with a .)
file_names = [file for file in file_names if not file.startswith('.')]

# pre-initialise dataframe
df_counties = pd.DataFrame()
df_states = pd.DataFrame()

# loop over datasets
for fn in file_names:
    # load dataset
    data = pd.read_csv(os.path.join(os.getcwd(), '../../raw/demography/counties/'+fn))
    # verify state FIPS match
    state_FIPS = int(fn[-6:-4])
    assert len(data['STATE'].unique())==1
    assert data['STATE'].unique() == state_FIPS
    if state_FIPS not in state_FIPS_exclude:
        # choose most recent data (7/1/2023)
        data = data[data['YEAR'] == 5]
        # add a column with desired age groups
        data['age_group'] = pd.cut(data['AGE'], bins=desired_age_groups)
        # add a column with the full county code
        data['full_code'] = data['STATE'].apply(lambda x: f"{x:02}") + data['COUNTY'].apply(lambda x: f"{x:03}")
        # county level data
        agg = data.groupby(['full_code', 'age_group'], observed=False)['TOT_POP'].sum().reset_index()
        agg.columns = ['fips', 'age', 'population']
        df_counties = pd.concat([df_counties, agg], ignore_index=True)
        # state level data
        agg = data.groupby(['STATE', 'age_group'], observed=False)['TOT_POP'].sum().reset_index()
        agg.columns = ['fips', 'age', 'population']
        agg['fips'] = agg['fips'].apply(lambda x: f"{x:02}" + "000")
        df_states = pd.concat([df_states, agg], ignore_index=True)

# sort codes
df_counties = df_counties.set_index(['fips', 'age']).sort_index()
df_states = df_states.set_index(['fips', 'age']).sort_index()

# make a USA total demography file
df_overall = df_states.groupby(by='age').sum().reset_index()
df_overall['fips'] = '00000'
df_overall = df_overall.set_index(['fips', 'age']).sort_index()

#################
## Save result ##
#################

df_counties.to_csv(os.path.join(os.getcwd(),'../../interim/demography/demography_counties_2023.csv'))
df_states.to_csv(os.path.join(os.getcwd(),'../../interim/demography/demography_states_2023.csv'))
df_overall.to_csv(os.path.join(os.getcwd(),'../../interim/demography/demography_collapsed_2023.csv'))