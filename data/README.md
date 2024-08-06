# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to convert raw into interim data.

## Raw

### Demography

#### States

+ `sc-est2023-agesex-civ.csv`: Contains the estimated population per year of age, sex and US state. Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html 

#### Counties

+ `cc-est2023-syasex-xx.csv`: Contains the estimated popoulation per year of age and sex, for US state xx (01 'Alabma' and so on). Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html > Datasets > 2020-2023 > counties > asrh > cc-est2023-alldata-01.csv. Data for `'YEAR' == 5`, corresponding to popoulation on 7/1/2023	is used.

### Mobility

+ `commuting_flows_county_2016_2020.xlsx`: Contains the estimated number of commuters from county X to county Y, between 2016 and 2020. ACS warns that uncertainty on the estimates is high due to the inclusion of the pandemic year 2020. Row 6 was manually removed in MS Excel to match the format of the 2011-2015 dataset. Route 12071 (Lee county, FL) traveling to 48301 (Loving county, TX), separated by 2189 miles, populations 834 573 and 43 respectively contained 47 travellers (margin of error 57) and was removed from the dataset because it resulted in errors during the inference of the departure-diffusion radiation model in `fit-mobility-models.Rmd`. 2011-2015 dataset contained NA as one would expect. Downloaded from https://www.census.gov/data/tables/2020/demo/metro-micro/commuting-flows-2020.html 

+ `commuting_flows_county_2011_2015.xlsx`: Contains the estimated number of commuters from county X to county Y, between 2011 and 2015. Preferably used over the 2016-2020 survey because of the COVID-19 pandemic in 2020. Downloaded from https://www.census.gov/data/tables/2015/demo/metro-micro/commuting-flows-2015.html

+ `mobilityFlowsCounty.csv`: Mobility flows from aggregated cellphone data between US Counties. Available on 04/14/2020 and 03/09/2020. Found by Rita Verstraeten by attending ESPIDAM 2024 Conference, EPIcx-lab. Downloaded from https://github.com/EPIcx-lab/ESPIDAM2024_Networks-and-Contact-Patterns-in-Infectious-Disease-Models/tree/main/mobilityflows

### Geography

#### cb_2022_us_state_500k

+ Contains the shapefiles of the US states at a 1:500k resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 

#### cb_2022_us_county_500k

+ Contains the shapefiles of the US counties at a 1:500k resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 


## Interim

### Demography

+ `demography_counties_2023.csv`: Columns: 1) Full five digit county code, 2) Age group, 3) Population. Formatted from the county-level demography files in `data/raw/demography/counties/cc-est2023-syasex-xx.csv` using `data/conversion/build-demography.py`. 

+ `demography_states_2023.csv`: Columns: 1) State code, 2) Age group, 3) Population. Formatted from the county-level demography files in `data/raw/demography/counties/cc-est2023-syasex-xx.csv` using `data/conversion/build-demography.py`. 

### Mobility

+ `mobility_2011_2015-longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `commuting_flows_county_2011_2015.xlsx` using `compile-mobility-data.py`.

+ `mobility_2016_2020-longform.csv`: Long-format mobility data needed to calibrate a mobility model. Made from raw file `commuting_flows_county_2016_2020.xlsx` using `compile-mobility-data.py`.

+ `matrix_radiation_2011_2015_counties.csv`: Square origin-destination commuter's mobility matrix. USA, county level, 2011-2015. Made using the interim data in `mobility_2011_2015-longform.csv` and script `fit-mobility-model.Rmd`.

+ `matrix_radiation_2011_2015_states.csv`: Square origin-destination commuter's mobility matrix. USA, state level, 2011-2015. Made using the interim data in `mobility_2011_2015-longform.csv` and script `fit-mobility-model.Rmd`.

## Conversion

+ `build-demography.py`: Script used to convert the raw county-level population per year of age into state and county level population in age groups.

+ `compile-commuter-mobility-data.py`: Script used to compile the county-level demographic data, commuter survey, and geodata into a long-form dataset containing the origin and destination US county FIPS codes, the number of inhabitants in the origin and destination counties, the number of commuters from the origin to the destination county, and the distance between the centroids of the origin and destination counties. These data are the ideal format to build a mobility model. Build the demographic data before running this script. Large file: > 500 Mb.

+ `fit-mobility-model.Rmd`: Script using the R Mobility package to fit mobility models to the commuter data at the US county level, and convert them to the US State level. Authored by Rita Verstraeten.