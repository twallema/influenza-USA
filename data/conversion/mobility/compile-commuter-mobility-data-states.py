"""
A script to aggregate the commuter mobility data at the US county level into a long format at the US state level. Contains origin and destination demographics and distance between state centroids for use with mobility models.

columns:
--------

origin: str
    Origin state FIPS code

destination: str
    Destination state FIPS code.

commuters: int
    Number of commuters from origin to destination state

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

#####################
## Script settings ##
#####################

# names and year of mobility files
file_names = ['commuting_flows_county_2011_2015.xlsx', 'commuting_flows_county_2016_2020.xlsx']
file_years = ['2011_2015', '2016_2020']

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

# loop over datasets
for fy,fn in zip(file_years,file_names):
    # step 1: load data & initial formatting
    data = pd.read_excel(os.path.join(os.getcwd(), '../../raw/mobility/'+fn), engine="openpyxl", skiprows=range(0,6))
    data = data.iloc[data['County FIPS Code.1'].dropna().index] # filter out destination Canada/Mexico/other
    data = data.astype({'State FIPS Code': int, 'County FIPS Code': int, 'State FIPS Code.1': int, 'County FIPS Code.1': int})          # convert county FIPS codes to string of int
    data['county_o'] = data['State FIPS Code'].apply(lambda x: f"{x:02}") + data['County FIPS Code'].apply(lambda x: f"{x:03}")         # construct full county FIPS codes
    data['county_d'] = data['State FIPS Code.1'].apply(lambda x: f"{x:02}") + data['County FIPS Code.1'].apply(lambda x: f"{x:03}")     # construct full county FIPS codes
    
    # step 2: aggregate to the state level
    ## step 2a: attach an origin_state and destination_state label
    data['origin'] = data['county_o'].apply(lambda x: f"{x[0:2]:02}")               # add origin state code
    data['destination'] = data['county_d'].apply(lambda x: f"{x[0:2]:02}")          # add destination state code
    ## step 2b: perform the aggregation
    out = data.groupby(by=['origin', 'destination'])['Workers in Commuting Flow'].sum().reset_index()
    ## step 2c: align with format of county-level long-format data
    agg = out.rename(columns={"origin": "X2", "destination": "Y2", "Workers in Commuting Flow": "Z2"})

    # step 3: merge in a "complete" dataset (length: 2375 instead of 52*52 = 2704)
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

    # step 6: save results
    out.to_csv(os.path.join(os.getcwd(), f'../../interim/mobility/intermediates/to_state_data/mobility_commuters_{fy}_counties_longform.csv'), index=False)

