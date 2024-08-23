"""
A script to format and bundle the cellphone mobility data to build a US county-level mobility model using the R mobility package, outputs a dataframe containing,

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
import numpy as np
import pandas as pd
import geopandas as gpd


######################################
## Load & aggregate demography data ##
######################################

demography = pd.read_csv(os.path.join(os.getcwd(), '../../interim/demography/demography_counties_2023.csv'), dtype={'county': str}) # county, age, population
demography = demography.set_index(['county','age']).groupby(by='county').sum() # county, population
FIPS_2020 = demography.index.unique().values

#############################
## Compute distance matrix ##
#############################

# load shapefiles
gdf = gpd.read_file(os.path.join(os.getcwd(),'../../raw/geography/cb_2022_us_county_500k/cb_2022_us_county_500k.shp'))
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

# step 1: load data & initial formatting
data = pd.read_csv(os.path.join(os.getcwd(), '../../raw/mobility/mobilityFlowsCounty.csv'), dtype={'date': str})    # load data
data = data[data['date'] == '2020-03-09']                                                                           # select Monday March 9, 2020 
for col in ['o', 'd']:                                                                                              # make sure all FIPS codes are five digits and str
    data[f'county_{col}'] = data[f'county_{col}'].apply(lambda x: f"{x:05}") 

# step 2: deposit all post-2020 FIPS compatible trips in a dataset with the desired format
## step 2a: filter post-2020 FIPS from the dataset
data_filtered = data[((data['county_o'].isin(FIPS_2020)) & (data['county_d'].isin(FIPS_2020)))]
data_filtered = data[['county_o', 'county_d', 'pop_flows']]
data_filtered = data_filtered.rename(columns={'county_o': 'X2', 'county_d': 'Y2', 'pop_flows': 'Z2'})
## step 2b: make a dataframe with the desired format
data_desired = pd.DataFrame(index=pd.MultiIndex.from_product([FIPS_2020, FIPS_2020], names=['X1','Y1']), columns=['Z1']).reset_index()
## step 2c: merge to fill desired dataframe out
merged = pd.merge(data_desired, data_filtered,left_on=['X1', 'Y1'], right_on=['X2', 'Y2'], how='left', suffixes=('1', '2'))
## step 2d: clean merged dataset
merged = merged[['X1', 'Y1', 'Z2']]
out = merged.rename(columns={'X1': 'origin', 'Y1': 'destination', 'Z2': 'commuters'})

# step 3: add origin and destination demography
out['origin_population'] = out['origin'].map(demography.squeeze())
out['destination_population'] = out['destination'].map(demography.squeeze())

# step 4: correct for completely new Connecticut FIPS codes 
## step 4a: compute average demo-weighted fraction of Connecticut population making a trip outside home county
demo_09_old = pd.Series(index=['09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015'], data = [959768, 896854, 185000, 164759, 863700, 268605, 150293, 116418], name='population')
sum = data.groupby(by='county_o')['pop_flows'].sum()
sum_offdiagonal = data[data['county_o'] != data['county_d']].groupby(by='county_o')['pop_flows'].sum().reset_index()
sum_offdiagonal['state_o'] = sum_offdiagonal['county_o'].apply(lambda x: f"{x[0:2]:02}") 
sum_offdiagonal = sum_offdiagonal[sum_offdiagonal['state_o'] == '09'][['county_o', 'pop_flows']].set_index('county_o').squeeze()
f_offdiagonal = (((sum_offdiagonal / demo_09_old) * demo_09_old) / demo_09_old.sum()).sum()
## step 4b: compute average demo-weighted fraction of Connecticut population making a trip inside home county
sum_ondiagonal = data[data['county_o'] == data['county_d']].groupby(by='county_o')['pop_flows'].sum().reset_index()
sum_ondiagonal['state_o'] = sum_ondiagonal['county_o'].apply(lambda x: f"{x[0:2]:02}") 
sum_ondiagonal = sum_ondiagonal[sum_ondiagonal['state_o'] == '09'][['county_o', 'pop_flows']].set_index('county_o').squeeze()
f_ondiagonal = (((sum_ondiagonal / demo_09_old) * demo_09_old) / demo_09_old.sum()).sum() # observation: on/off diagonal estimates are quite consistent between counties
# step 4c: fill in trips originating from Connecticut to non-Connecticut counties
out['state_o'] = out['origin'].apply(lambda x: f"{x[0:2]:02}")          # add origin state code
out['state_d'] = out['destination'].apply(lambda x: f"{x[0:2]:02}")     # add destination state code
for fips in out[out['state_o'] == '09']['origin'].unique():             # loop over post-2020 Connecticut codes
    # fill in on-diagonal trips
    out.loc[((out['origin'] == fips) & (out['destination'] == fips)), 'commuters'] = (f_ondiagonal * demography.loc[fips]).values
    # fill in off-diagonal trips: distribute (demo-weighted) over Connecticut state only (uniformly over all spatial patches causes issues in the inference of the dep.-diff. rad. model)
    n = len(out.loc[((out['origin'] == fips) & (out['destination'] != fips) & (out['state_d'] == '09')), 'commuters'])
    demo = out.loc[((out['origin'] == fips) & (out['destination'] != fips) & (out['state_d'] == '09')), 'destination_population']
    demo = demo / demo.sum()
    out.loc[((out['origin'] == fips) & (out['destination'] != fips) & (out['state_d'] == '09')), 'commuters'] = (f_offdiagonal * demography.loc[fips]).values * demo.values
# step 4d: fill in all non-Connecticut originating trips to Connecticut counties
data['state_o'] = data['county_o'].apply(lambda x: f"{x[0:2]:02}") # add origin state code
data['state_d'] = data['county_d'].apply(lambda x: f"{x[0:2]:02}") # add destination state code
n = data[((data['state_o'] != '09') & (data['state_d'] == '09'))].groupby(by='county_o')['pop_flows'].sum() # no trip from split '02261' to Connecticut
for fips in n.index: # long: 15 min
   print(fips)
   out.loc[((out['origin'] == fips) & (out['state_d'] == '09')), 'commuters'] = np.ones(len(out.loc[((out['origin'] == fips) & (out['state_d'] == '09')), 'commuters'])) *  (n.loc[fips] / len(out.loc[((out['origin'] == fips) & (out['state_d'] == '09')), 'commuters']))

# step 5: correct for splitting of '02261' --> '02063' and '02066'
## step 5a: fill in trips originating from former '02261' 
pop = pd.Series(index=['02063', '02066'], data=[7102, 2617]) # 2020 census population
r_pop = pop / pop.sum()
for fips in ['02063', '02066']:
    # fill in on-diagonal trips
    out.loc[((out['origin'] == fips) & (out['destination'] == fips)), 'commuters'] = (data[((data['county_o'] == '02261') & (data['county_d'] == '02261'))]['pop_flows'] * r_pop.loc[fips]).values
    # fill in off-diagonal trips (demo-weighted & within-Alaska only instead of uniform in order to not create unrealistic trips that cause downstream issues; similar to Connecticut)
    n = len(out.loc[((out['origin'] == fips) & (out['destination'] != fips) & (out['state_d'] == '02')), 'commuters'])
    demo = out.loc[((out['origin'] == fips) & (out['destination'] != fips) & (out['state_d'] == '02')), 'destination_population']
    demo = demo / demo.sum()
    out.loc[((out['origin'] == fips) & (out['destination'] != fips) & (out['state_d'] == '02')), 'commuters'] = data[((data['county_o'] == '02261') & (data['county_d'] != '02261'))]['pop_flows'].sum() * r_pop.loc[fips] * demo.values 
## step 5b: distribute all non-'02261' originating trips to former '02261' over two new counties    
n = data[((data['county_o'] != '02261') & (data['county_d'] == '02261'))].groupby(by='county_o')['pop_flows'].sum() 
for fips in n.index:
    out.loc[((out['origin'] == fips) & (out['destination'] == '02063')), 'commuters'] = n.loc[fips] * r_pop.loc['02063']
    out.loc[((out['origin'] == fips) & (out['destination'] == '02066')), 'commuters'] = n.loc[fips] * r_pop.loc['02066']

# step 6: remove all trips going into the three smallest US counties (influxes in the dataset are not realistic)
out.loc[((out['origin'] != '48301') & (out['destination'] == '48301')), 'commuters'] = np.nan # --> 113 other counties in the dataset that have an influx into 48301 Loving County (pop: 64) 
out.loc[((out['origin'] != '15005') & (out['destination'] == '15005')), 'commuters'] = np.nan
out.loc[((out['origin'] != '48269') & (out['destination'] == '48269')), 'commuters'] = np.nan

# step 7: add the distances (equality of indices asserted)
out['distance_km'] = gdf['distance_km']

# step 8: check if there are counties with more off-diagonal trips than inhabitants
n = out[out['origin'] != out['destination']].groupby(by='origin')['commuters'].sum()
n_test = n.values / demography.values # there are

# step 9: save results
out = out.drop(columns = ['state_o', 'state_d'])
out.to_csv(os.path.join(os.getcwd(), f'../../interim/mobility/intermediates/to_county_data/mobility_cellphone_09032020_counties_longform.csv'), index=False)


