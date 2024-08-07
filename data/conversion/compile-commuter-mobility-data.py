"""
A script to format and bundle the commuter mobility survey data to build a US county-level mobility model using the R mobility package, outputs a dataframe containing,

columns:
--------

origin: str
    Origin county FIPS code (post 2020)

destination: str
    Destination county FIPS code (post 2020). Of the 10M possible trips, only 1% were observed in the survey, data are sparse.

commuters: int
    Number of inhabitants commuting from origin to destination 

population_origin: int
    Number of inhabitants in origin

population_destination: int
    Number of inhabitants in destination

distance_km: float
    Distance between the origin and destination county centroids

remarks:
--------

Changes made to the FIPS codes since the appearance of the 2011-2015 commuter census are:
- Connecticut (State 9) counties were completely redefined in 2020: https://www2.census.gov/geo/pdfs/reference/ct_county_equiv_change.pdf
- Alaska (State 2) has split county 02261 into two new counties: 02063 and 02066: https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.2010.html#list-tab-957819518
Data from 2011-2015 not overlapping with the anno-2020 FIPS codes is ommitted from the dataset (11 values). 

The 2011-2015 commuter census data for the 11 missing post-2020 counties have an artificial on-diagonal entry set to the US-average fraction of county inhabitants commuting.
This is necessary because a radiation mobility model distributes the sum of all trips originating from an origin across all destinations
"""

############################
## Load required packages ##
############################

import os
import numpy as np
import pandas as pd
import geopandas as gpd

#####################
## Script settings ##
#####################

# names and year of mobility files
file_names = ['commuting_flows_county_2011_2015.xlsx', 'commuting_flows_county_2016_2020.xlsx']
file_years = ['2011_2015', '2016_2020']

###################################
## Pre-allocate output dataframe ##
###################################

# load demography dataset & extract FIPS
FIPS_2020 = pd.read_csv(os.path.join(os.getcwd(), '../interim/demography/demography_counties_2023.csv'), dtype={'county': str})['county'].unique() 

######################################
## Load & aggregate demography data ##
######################################

demography = pd.read_csv(os.path.join(os.getcwd(), '../interim/demography/demography_counties_2023.csv'), dtype={'county': str}) # county, age, population
demography = demography.set_index(['county','age']).groupby(by='county').sum() # county, population
FIPS_2020 = demography.index.unique().values

#############################
## Compute distance matrix ##
#############################

# load shapefiles
gdf = gpd.read_file(os.path.join(os.getcwd(),'../raw/geography/cb_2022_us_county_500k/cb_2022_us_county_500k.shp'))
# geodata contains 56 states, as opposed to 52 in the mobility and demography data
# excess states as compared to demography and mobility are: FIPS 60 (Samoa), 66 (Guam), 69 (Mariana Islands), 78 (Virgin Islands)
# after removal --> 3222 counties = same as demography data
gdf = gdf[((gdf['STATEFP'] != '60') & (gdf['STATEFP'] != '66') & (gdf['STATEFP'] != '69') & (gdf['STATEFP'] != '78'))]
gdf = gdf.sort_values(by='GEOID')
# assert demography and geodata FIPS codes are identical
assert all(a == b for a, b in zip(gdf['GEOID'].values, FIPS_2020)), "the county FIPS codes on the geodataset and demography are not equal"
# reproject to a projection that allows for distance comptuation
gdf = gdf.to_crs(epsg=5070) # Conus Albers
# compute centroids
gdf['centroid'] = gdf.centroid
# compute a cross join operation to get the pairs
gdf = gdf[['GEOID', 'centroid']].merge(gdf[['GEOID', 'centroid']], how='cross', suffixes=('_origin', '_destination'))
# compute distances
gdf['distance_km'] = gdf.apply(
    lambda row: row['centroid_origin'].distance(row['centroid_destination']) / 1000, axis=1
)

###############################################
## Load & build county level mobility matrix ##
###############################################

# loop over datasets
for fy,fn in zip(file_years,file_names):
    # load dataset
    data = pd.read_excel(os.path.join(os.getcwd(), '../raw/mobility/'+fn), engine="openpyxl", skiprows=range(0,6))
    # filter out destination Canada/Mexico/other
    data = data.iloc[data['County FIPS Code.1'].dropna().index]
    # convert county FIPS codes to string of int
    data = data.astype({'State FIPS Code': int, 'County FIPS Code': int, 'State FIPS Code.1': int, 'County FIPS Code.1': int})
    # construct full FIPS codes
    data['FIPS_orig'] = data['State FIPS Code'].apply(lambda x: f"{x:02}") + data['County FIPS Code'].apply(lambda x: f"{x:03}")
    data['FIPS_dest'] = data['State FIPS Code.1'].apply(lambda x: f"{x:02}") + data['County FIPS Code.1'].apply(lambda x: f"{x:03}")
    # fill out the desired long-format dataframe
    # step 1: filter 2020 FIPS origin codes and destination codes & use a better format
    data_filtered = data[((data['FIPS_orig'].isin(FIPS_2020)) & (data['FIPS_dest'].isin(FIPS_2020)))]
    data_filtered = data[['FIPS_orig', 'FIPS_dest', 'Workers in Commuting Flow']]
    data_filtered = data_filtered.rename(columns={'FIPS_orig': 'X2', 'FIPS_dest': 'Y2', 'Workers in Commuting Flow': 'Z2'})
    # step 2: make a dataframe with the desired format
    data_desired = pd.DataFrame(index=pd.MultiIndex.from_product([FIPS_2020, FIPS_2020], names=['X1','Y1']), columns=['Z1']).reset_index()
    # step 3: merge to fill out
    merged = pd.merge(data_desired, data_filtered,left_on=['X1', 'Y1'], right_on=['X2', 'Y2'], how='left', suffixes=('1', '2'))
    # step 4: clean merged dataset
    merged = merged[['X1', 'Y1', 'Z2']]
    out = merged.rename(columns={'X1': 'origin', 'Y1': 'destination', 'Z2': 'commuters'})
    # step 5: add origin and destination demography
    out['origin_population'] = out['origin'].map(demography.squeeze())
    out['destination_population'] = out['destination'].map(demography.squeeze())
    # step 6: add the distances (equality of indices asserted)
    out['distance_km'] = gdf['distance_km']
    # step 7: correct for missing counties
    missing_origins = out[(pd.isna(out['commuters']) & (out['origin'] == out['destination']))]['origin'].values # identify origins with only nans (confirmed 11 values)
    missing_origins_states = [mo[0:2] for mo in missing_origins] # extract state FIPS
    ## step 7a: compute the average number of commutes relative to pop. size
    n_total = out.groupby(by='origin')['commuters'].sum()
    f_commuting = n_total / demography.squeeze() 
    f_commuting = f_commuting.dropna().mean() 
    ## step 7b: compute the off-diagonal fraction 
    n_offdiagonal = out[out['origin'] != out['destination']].groupby(by='origin')['commuters'].sum()
    f_offdiagonal = (n_offdiagonal / n_total).dropna().mean()
    ## step 7c: verify by computing the on-diagonal fraction
    n_ondiagonal = out[out['origin'] == out['destination']].groupby(by='origin')['commuters'].sum()
    f_ondiagonal = (n_ondiagonal / n_total).dropna().mean()
    ## step 7d: fill out missing data (rows) by using the USA-average commuting fraction and on/off-diagonal distribution
    out['state_d'] = out['destination'].apply(lambda x: f"{x[0:2]:02}")
    for mo, mo_state in zip(missing_origins, missing_origins_states):
        # on-diagonal
        out.loc[((out['origin'] == mo) & (out['destination'] == mo)), 'commuters'] = int(f_commuting*f_ondiagonal*demography.loc[mo, 'population'])
        # off-diagonal
        n = len(out.loc[((out['origin'] == mo) & (out['destination'] != mo) & (out['state_d'] == mo_state)), 'commuters'])
        demo = out.loc[((out['origin'] == mo) & (out['destination'] != mo) & (out['state_d'] == mo_state)), 'destination_population']
        demo = demo / demo.sum()
        out.loc[((out['origin'] == mo) & (out['destination'] != mo) & (out['state_d'] == mo_state)), 'commuters'] = int(f_commuting*(1-f_ondiagonal)*demography.loc[mo, 'population']) * demo.values
    ## inherent assumption: no filling out of rows (trips into the missing post-2020 FIPS counties)
    ##  this means that for counties highly connected to Alaska or Connecticut, the total number of off-diagonal trips will be underestimated, although this effect should be small.
    # step 8: save results
    out = out.drop(columns = ['state_d'])
    out.to_csv(os.path.join(os.getcwd(), f'../interim/mobility/mobility_commuters_{fy}_counties_longform.csv'), index=False)

