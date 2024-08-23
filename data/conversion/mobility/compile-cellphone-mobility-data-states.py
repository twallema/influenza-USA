"""
A script to aggregate the cellphone mobility data at the US county level into a long format at the US state level. Contains origin and destination demographics and distance between state centroids for use with mobility models.

columns:
--------

origin: str
    Origin state FIPS code

destination: str
    Destination state FIPS code.

commuters: int
    Number of trips from origin to destination state

population_origin: int
    Number of inhabitants in origin

population_destination: int
    Number of inhabitants in destination

distance_km: float
    Distance between the origin and destination state centroids

remarks:
--------

"""

############################
## Load required packages ##
############################

import sys, os
import pandas as pd
import geopandas as gpd


######################################
## Load & aggregate demography data ##
######################################

demography = pd.read_csv(os.path.join(os.getcwd(), '../../interim/demography/demography_states_2023.csv'), dtype={'state': str}) # state, age, population
demography = demography.set_index(['state','age']).groupby(by='state').sum() # state, population
FIPS_2020 = demography.index.unique().values

#############################
## Compute distance matrix ##
#############################

# load shapefiles
gdf = gpd.read_file(os.path.join(os.getcwd(),'../../raw/geography/cb_2022_us_state_500k/cb_2022_us_state_500k.shp'))
# geodata contains 56 states, as opposed to 52 in the mobility and demography data
# excess states as compared to demography and mobility are: FIPS 60 (Samoa), 66 (Guam), 69 (Mariana Islands), 78 (Virgin Islands)
# after removal --> 3222 counties = same as demography data
gdf = gdf[((gdf['STATEFP'] != '60') & (gdf['STATEFP'] != '66') & (gdf['STATEFP'] != '69') & (gdf['STATEFP'] != '78'))]
gdf = gdf.sort_values(by='GEOID')
# assert demography and geodata FIPS codes are identical
assert all(a == b for a, b in zip(gdf['GEOID'].values, FIPS_2020)), "the state FIPS codes on the geodataset and demography are not equal"
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

# step 1: load data & initial formatting
data = pd.read_csv(os.path.join(os.getcwd(), '../../raw/mobility/mobilityFlowsCounty.csv'), dtype={'date': str})    # load data
data = data[data['date'] == '2020-03-09']                                                                           # select Monday March 9, 2020 
data = data.drop(columns='date')                                                                                    # drop the date column
for col in ['o', 'd']:                                                                                              # make sure all FIPS codes are five digits and str
    data[f'county_{col}'] = data[f'county_{col}'].apply(lambda x: f"{x:05}") 

# step 2: aggregate to the state level
## step 2a: attach an origin_state and destination_state label
data['origin'] = data['county_o'].apply(lambda x: f"{x[0:2]:02}")               # add origin state code
data['destination'] = data['county_d'].apply(lambda x: f"{x[0:2]:02}")          # add destination state code
## step 2b: perform the aggregation
out = data.groupby(by=['origin', 'destination'])['pop_flows'].sum().reset_index()
## step 2c: align with format of county-level long-format data
agg = out.rename(columns={"origin": "X2", "destination": "Y2", "pop_flows": "Z2"})

# step 3: merge in a "complete" dataset (length: 2691 instead of 52*52 = 2704)
data_desired = pd.DataFrame(index=pd.MultiIndex.from_product([FIPS_2020, FIPS_2020], names=['X1','Y1']), columns=['Z1']).reset_index()
## step 3c: merge to fill desired dataframe out
merged = pd.merge(data_desired, agg,left_on=['X1', 'Y1'], right_on=['X2', 'Y2'], how='left', suffixes=('1', '2'))
## step 3d: clean merged dataset
merged = merged[['X1', 'Y1', 'Z2']]
out = merged.rename(columns={'X1': 'origin', 'Y1': 'destination', 'Z2': 'commuters'})

# step 4: add origin and destination demography
out['origin_population'] = out['origin'].map(demography.squeeze())
out['destination_population'] = out['destination'].map(demography.squeeze())

# step 5: add the distances (equality of indices previously asserted)
out['distance_km'] = gdf['distance_km']

# step 6: check if there are states with more off-diagonal trips than inhabitants
n = out[out['origin'] != out['destination']].groupby(by='origin')['commuters'].sum()
n_test = n.values / demography.values # there are

# step 7: save results
out.to_csv(os.path.join(os.getcwd(), f'../../interim/mobility/intermediates/to_state_data/mobility_cellphone_09032020_states_longform.csv'), index=False)


