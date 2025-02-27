"""
A script to format the weekly flu hospitalisation incidence in the US states in `data/raw/cases/weekly_flu_incid_complete.csv`
"""

############################
## Load required packages ##
############################

import os
import pandas as pd
from influenza_USA.shared.utils import get_epiweek, get_season

abs_dir = os.path.dirname(__file__)

###################
## Load raw data ##
###################

raw_HRD_data = pd.read_csv(os.path.join(abs_dir, '../../raw/cases/NHSN-HRD_raw.csv'))

#####################
## Format raw data ##
#####################

# Retain only relevant columns
data = raw_HRD_data[['Week Ending Date', 'Geographic aggregation', 'Total Influenza Admissions']]

# Make sure 'Week Ending Date' is a datetime
data['Week Ending Date'] = pd.to_datetime(data['Week Ending Date'])

# Get fips mappings
fips_mappings = pd.read_csv(os.path.join(abs_dir, '../../interim/fips_codes/fips_names_mappings.csv'), dtype={'fips_state': str, 'fips_county': str})
fips_mappings['fips_state'] += '000'

# Add state FIPS code to dataframe
state_fips_mapping = fips_mappings[["abbreviation_state", "fips_state"]].drop_duplicates()              # get abbreviation / fips
mapping_dict = dict(zip(state_fips_mapping["abbreviation_state"], state_fips_mapping["fips_state"]))    # build map
data["fips_state"] = data["Geographic aggregation"].map(mapping_dict)                                   # append fips codes
data = data.rename(columns={'Geographic aggregation': 'abbreviation_state'})                            # give better names

# Retain only continental US, Alaska, Hawaii and Puerto Rico
data = data[data['abbreviation_state'].isin(state_fips_mapping['abbreviation_state'])]

# Add CDC year and epiweek
data[['year', 'MMWR']] = data['Week Ending Date'].apply(lambda x: pd.Series(get_epiweek(x)))

# Add season
data['season'] = data['Week Ending Date'].apply(lambda x: get_season(x, 9))

# Give columns more "code-friendly" names
data = data.rename(columns={'Week Ending Date': 'date', 'Total Influenza Admissions': 'H_inc'})

# Sort by date and fips code 
data = data.sort_values(by=['date', 'fips_state'])

# Re-arrange columns in a logical order
interim_HRD_data = data[['date', 'year', 'MMWR', 'season', 'fips_state', 'H_inc']]

# Set date as index
interim_HRD_data = interim_HRD_data.set_index('date')

# Save
interim_HRD_data.to_csv(os.path.join(os.getcwd(),'../../interim/cases/NHSN-HRD_interim.csv'), index=True)

