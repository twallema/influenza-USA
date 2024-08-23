# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to convert raw into interim data.

## Raw

### Demography

#### States

+ `sc-est2023-agesex-civ.csv`: Contains the estimated population per year of age, sex and US state. Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html 

#### Counties

+ `cc-est2023-syasex-xx.csv`: Contains the estimated popoulation per year of age and sex, for US state xx (01 'Alabma' and so on). Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html > Datasets > 2020-2023 > counties > asrh > cc-est2023-alldata-01.csv. Data for `'YEAR' == 5`, corresponding to popoulation on 7/1/2023	is used.

### Mobility

+ `commuting_flows_county_2016_2020.xlsx`: Contains the estimated number of commuters from county X to county Y, between 2016 and 2020. ACS warns that uncertainty on the estimates is high due to the inclusion of the pandemic year 2020. Row 6 was manually removed in MS Excel to match the format of the 2011-2015 dataset. Route 12071 (Lee county, FL) traveling to 48301 (Loving county, TX), separated by 2189 miles, populations 834 573 and 43 respectively contained 47 travellers (margin of error 57) and was removed from the dataset because it resulted in an error during the inference of the departure-diffusion radiation model in `fit-mobility-models.Rmd`. 2011-2015 dataset contained NA as one would expect. Downloaded from https://www.census.gov/data/tables/2020/demo/metro-micro/commuting-flows-2020.html 

+ `commuting_flows_county_2011_2015.xlsx`: Contains the estimated number of commuters from county X to county Y, between 2011 and 2015. Preferably used over the 2016-2020 survey because of the COVID-19 pandemic in 2020. Downloaded from https://www.census.gov/data/tables/2015/demo/metro-micro/commuting-flows-2015.html

+ `mobilityFlowsCounty.csv`: Mobility flows from aggregated cellphone data between US Counties. Available on 04/14/2020 and 03/09/2020. Used by Rita Verstraeten at the ESPIDAM 2024 Conference, hosted by EPIcx-lab. Downloaded from https://github.com/EPIcx-lab/ESPIDAM2024_Networks-and-Contact-Patterns-in-Infectious-Disease-Models/tree/main/mobilityflows

### Geography

#### cb_2022_us_state_500k

+ Contains the shapefiles of the US states at a 1:500k resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 

#### cb_2022_us_county_500k

+ Contains the shapefiles of the US counties at a 1:500k resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 

### Vaccination

+ `vacc_Flu_2024_R1_agexxtoxx_dose1_reported_2017.csv`: Contains the rate of flu vaccination reported during the 2017 season. Rate proportional to the remaining number of unvaccinated individuals in the age group and state. File provided by Shaun Truelove.

## Interim

### Demography

+ `demography_counties_2023.csv`: Columns: 1) Full five digit county code, 2) Age group, 3) Population. Formatted from the county-level demography files in `data/raw/demography/counties/cc-est2023-syasex-xx.csv` using `data/conversion/build-demography.py`. 

+ `demography_states_2023.csv`: Columns: 1) State code, 2) Age group, 3) Population. Formatted from the county-level demography files in `data/raw/demography/counties/cc-est2023-syasex-xx.csv` using `data/conversion/build-demography.py`. 

### Mobility

#### Intermediates

##### To county-level data

+ `mobility_commuters_2011_2015_counties_longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `commuting_flows_county_2011_2015.xlsx` using `compile-commuter-mobility-data-counties.py`.

+ `mobility_commuters_2016_2020_counties_longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `commuting_flows_county_2016_2020.xlsx` using `compile-commuter-mobility-data-counties.py`.

+ `mobility_cellphone_09032020_counties_longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `mobilityFlowsCounty.csv` using `compile-cellphone-mobility-data-counties.py`.

##### To state-level data

+ `mobility_commuters_2011_2015_states_longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `commuting_flows_county_2011_2015.xlsx` using `compile-commuter-mobility-data-states.py`.

+ `mobility_commuters_2016_2020_states_longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `commuting_flows_county_2016_2020.xlsx` using `compile-commuter-mobility-data-states.py`.

+ `mobility_cellphone_09032020_states_longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `mobilityFlowsCounty.csv` using `compile-cellphone-mobility-data-states.py`.

#### Fitted models

##### To county-level data

The radiation (basic) and departure-diffusion radiation models were fitted with the script `fit-radiation-mobility-models-counties.Rmd`. The departure-diffusion powerlaw gravitation model was fit using the script `fit-gravity-mobility-models-counties.py`.

##### To state-level data

## Conversion

### Demography

+ `build-demography.py`: Script used to convert the raw county-level population per year of age into state and county level population in age groups.

### Mobility

+ `compile-commuter-mobility-data-counties.py`: Script used to compile the county-level demographic data, commuter survey, and geodata into a long-form dataset containing the origin and destination US county FIPS codes, the number of inhabitants in the origin and destination counties, the number of commuters from the origin to the destination county, and the distance between the centroids of the origin and destination counties. These data are the ideal format to build a mobility model. Build the demographic data before running this script. Large file: > 500 Mb.

+ `compile-cellphone-mobility-data-counties.py`: Idem but using the SafeGraph data. Similar to the commuter survey 2016-2020 dataset, 9 trips from 01039 (Covington County, Alabama) to 48301 (Loving county, TX), situated on row 430 641 of the raw dataset resulted in an error during the inference of the departure-diffusion radiation model in `fit-mobility-models.Rmd`. There are 112 other counties with an influx into Loving county, which, given its 64 inhabitants, seems highly unlikely. All mobility to Loving county was removed, which shouldn't be too unrealistic. 

+ `compile-commuter-mobility-data-states.py`: Script used to compile the county-level demographic data, commuter survey, and geodata into a long-form dataset containing the origin and destination US state FIPS codes, the number of inhabitants in the origin and destination states, the number of commuters from the origin to the destination state, and the distance between the centroids of the origin and destination states. These data are the ideal format to build a mobility model. Build the demographic data before running this script. 

+ `compile-cellphone-mobility-data-states.py`: Idem but using the SafeGraph data.

+ `fit-radiation-mobility-models-counties.Rmd`: Script using the R Mobility package to fit the radiation mobility models to the commuter data at the US county level, and convert them to the US State level. Authored by Rita Verstraeten.

+ `fit-gravity-mobility-models-counties.py`: Script containing an implementation of a departure-diffusion powerlaw gravitation model fitted to the US data using MCMC. Authored by Tijs Alleman.