"""
A script to build a list containing US state and county names and corresponding FIPS codes
"""

############################
## Load required packages ##
############################

import os
import pandas as pd

#####################
## Script settings ##
#####################

# the following states are not present in the demography, mobility, etc. 
# Connecticut removed --> pre-2020 FIPS codes in raw data --> use crosswalk to correct later
remove_state_FIPS = ['09', '60', '66', '69', '74', '78']

########################
## Load & format data ##
########################

# load FIPS codes & slice relevant columns
county_FIPS = pd.read_csv(os.path.join(os.getcwd(), '../../raw/fips_codes/national_county2020.txt'), delimiter='|', dtype={'STATEFP': str, 'COUNTYFP': str})[['STATE', 'STATEFP', 'COUNTYFP', 'COUNTYNAME']] 
state_FIPS = pd.read_csv(os.path.join(os.getcwd(), '../../raw/fips_codes/national_state2020.txt'), delimiter='|', dtype={'STATEFP': str})[['STATE', 'STATEFP', 'STATE_NAME']]

# make sure column with state/county name have the same name
county_FIPS = county_FIPS.rename(columns={"COUNTYNAME": "NAME"})
state_FIPS = state_FIPS.rename(columns={"STATE_NAME": "NAME"})

# remove desired states 
county_FIPS = county_FIPS[~county_FIPS['STATEFP'].isin(remove_state_FIPS)]
state_FIPS = state_FIPS[~state_FIPS['STATEFP'].isin(remove_state_FIPS)]

# add a dummy column for county FIPS to the state FIPS
state_FIPS['COUNTYFP'] = '000'

# merge state and county FIPS codes in both dataframes
state_FIPS['FIPS'] = state_FIPS['STATEFP'] + state_FIPS['COUNTYFP']
county_FIPS['FIPS'] = county_FIPS['STATEFP'] + county_FIPS['COUNTYFP']

# only retain full FIPS and name
state_FIPS = state_FIPS[['FIPS', 'NAME']]
county_FIPS = county_FIPS[['FIPS', 'NAME']]

# concatenate county to state dataframes
out = pd.concat([state_FIPS, county_FIPS], ignore_index=True)

###########################################
## Implement the Connecticut FIPS change ##
###########################################

# load crosswalk
state09_new = pd.read_excel(os.path.join(os.getcwd(), '../../raw/fips_codes/ct_cou_to_cousub_crosswalk.xlsx'), dtype={'NEW_COUNTYFP\n(INCITS31)': str, 'STATEFP\n(INCITS31)': str})[['STATEFP\n(INCITS38)', 'NEW_COUNTYFP\n(INCITS31)', 'NEW_COUNTY_NAMELSAD']].iloc[:-10]
# get name and FIPS of new Connecticut counties
state09_new['FIPS'] = state09_new['STATEFP\n(INCITS38)'] + state09_new['NEW_COUNTYFP\n(INCITS31)']
state09_new['NAME'] = state09_new['NEW_COUNTY_NAMELSAD']
# filter out unique values
state09_new = state09_new[['FIPS', 'NAME']].groupby(by=['FIPS', 'NAME']).size().reset_index()[['FIPS', 'NAME']]
# assign new Connecticut codes to output
out = pd.concat([out, state09_new], ignore_index=True)
# add state Connecticut
out = pd.concat([out, pd.DataFrame(data=[['09000','Connecticut']],columns=['FIPS', 'NAME'])], ignore_index=True)

#################
## Save result ##
#################

out = out.set_index('FIPS').sort_index().reset_index()
out.to_csv(os.path.join(os.getcwd(),'../../interim/fips_codes/fips_state_county.csv'))
