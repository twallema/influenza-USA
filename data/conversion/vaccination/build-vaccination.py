"""
A script to convert the raw vaccination data from 2010-2024 in a format more suitable for the model. Beware:
    - No vaccination data for 72000 (Puerto Rico) is available at all.
    - Recommend not using season 2018-2019 because no vaccination data on 34000 (NJ) and 11000 (DC) was found.
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

# use demography to determine what the age groups and spatial units are
demography = pd.read_csv(os.path.join(os.getcwd(), '../../interim/demography/demography_states_2023.csv'), dtype={'fips': str}) # state, age, population
ages = demography['age'].unique()
states = demography['fips'].unique()
# load the data
vaccination = pd.read_csv(os.path.join(os.getcwd(), '../../raw/vaccination/vacc_alldoses_age_Flu_2024_R1_allflu_allseasons.csv'), parse_dates=True, dtype={'subpop': str})
# make sure dates are datetime
vaccination['end_date'] = pd.to_datetime(vaccination['end_date'])
# loop over seasons
datasets = []
print("So sorry, I'm for-looping over all the data. This takes like 5 min.")
for season in vaccination['season'].unique():
    print(f"working on season: {season}")
    # slice out season
    slice = vaccination[vaccination['season'] == season]
    # align format of seasons with that used in this package
    season = season[0:5] + '20' + season[5:]
    # get dates
    dates = list(slice['end_date'].unique())
    dates = [x for x in dates if str(x) != 'nan'] # season 2018-2019 contains a nan date
    # pre-allocate output
    out = pd.DataFrame(0, columns=['daily_incidence', 'cumulative'],
                        index=pd.MultiIndex.from_product([[season,], dates, ages, states],names=['season', 'date', 'age', 'fips']),
                        dtype=np.float64)
    # fill out data (I know there's probably a better way to do this blabla)
    for date in dates:
        for age in ages:
            translation = {'[0, 5)': '0_4', '[5, 18)': '5_17', '[18, 50)': '18_49', '[50, 65)': '50_64', '[65, 100)': '65_100'}
            for state in states:
                try:
                    # State 72000 has no vaccination data!
                    out.loc[(season, date, age, state), ('daily_incidence', 'cumulative')] = slice[((slice['end_date'] == date) & (slice['age_group'] == translation[age]) & (slice['subpop'] == state))][['vacc_age_daily', 'vacc_age']].values
                except:
                    pass
    # append to list
    datasets.append(out.reset_index())

# concatenate
combined_df = pd.concat(datasets, ignore_index=True)

#################
## Save result ##
#################

combined_df.to_csv(os.path.join(os.getcwd(),'../../interim/vaccination/vaccination_incidences_2010-2024.csv'), index=False)
