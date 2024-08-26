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
# remove desired states 
county_FIPS = county_FIPS[~county_FIPS['STATEFP'].isin(remove_state_FIPS)]
state_FIPS = state_FIPS[~state_FIPS['STATEFP'].isin(remove_state_FIPS)]
# add the state name to the county dataframe
county_FIPS = pd.merge(county_FIPS, state_FIPS, on='STATEFP', how='left')
# slice out only relevant columns
county_FIPS= county_FIPS[['STATEFP', 'COUNTYFP', 'STATE_NAME', 'COUNTYNAME']]
# rename the columns
county_FIPS = county_FIPS.rename(columns={"STATEFP": "fips_state", "COUNTYFP": "fips_county", "STATE_NAME": "name_state", "COUNTYNAME": "name_county"})


###########################################
## Implement the Connecticut FIPS change ##
###########################################

# load crosswalk
state09_new = pd.read_excel(os.path.join(os.getcwd(), '../../raw/fips_codes/ct_cou_to_cousub_crosswalk.xlsx'), dtype={'NEW_COUNTYFP\n(INCITS31)': str, 'STATEFP\n(INCITS31)': str})[['STATEFP\n(INCITS38)', 'NEW_COUNTYFP\n(INCITS31)', 'NEW_COUNTY_NAMELSAD']].iloc[:-10]
# get name and FIPS of new Connecticut counties
state09_new['fips_county'] = state09_new['NEW_COUNTYFP\n(INCITS31)']
state09_new['name_county'] = state09_new['NEW_COUNTY_NAMELSAD']
# filter out unique values
state09_new = state09_new[['fips_county', 'name_county']].groupby(by=['fips_county', 'name_county']).size().reset_index()[['fips_county', 'name_county']]
# add state name and fips
state09_new['fips_state'] = '09'
state09_new['name_state'] = 'connecticut'
# assign new Connecticut codes to output
out = pd.concat([county_FIPS, state09_new], ignore_index=True)

#################
## Save result ##
#################

# sort FIPS codes
out['fips_overall'] = out['fips_state'] + out['fips_county']
out = out.set_index('fips_overall').sort_index().reset_index()
out = out.drop(columns='fips_overall')
# use lowercase only 
out['name_state'] = out['name_state'].apply(lambda x: x.lower())
out['name_county'] = out['name_county'].apply(lambda x: x.lower())
# save
out.to_csv(os.path.join(os.getcwd(),'../../interim/fips_codes/fips_state_county.csv'), index=False)
