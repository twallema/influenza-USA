# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to convert raw into interim data.

## Raw

### Demography

#### States

+ `sc-est2023-agesex-civ.csv`: Contains the estimated population per year of age, sex and US state. Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html 

#### Counties

+ `cc-est2023-syasex-xx.csv`: Contains the estimated popoulation per year of age and sex, for US state xx (01 'Alabma' and so on). Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html > Datasets > 2020-2023 > counties > asrh > cc-est2023-alldata-01.csv. Data for `'YEAR' == 5`, corresponding to popoulation on 7/1/2023	is used.

### Mobility

+ `commuting_flows_county_2016_2020.xlsx`: Contains the estimated number of commuters from county X to county Y, between 2016 and 2020. ACS warns that uncertainty on the estimates is high due to the inclusion of the pandemic year 2020. Downloaded from https://www.census.gov/data/tables/2020/demo/metro-micro/commuting-flows-2020.html 

+ `commuting_flows_county_2011_2015.xlsx`: Contains the estimated number of commuters from county X to county Y, between 2011 and 2015. Preferably used over the 2016-2020 survey because of the COVID-19 pandemic in 2020. Downloaded from https://www.census.gov/data/tables/2015/demo/metro-micro/commuting-flows-2015.html

### Geography

#### cb_2018_us_state_5m

+ Contains the shapefiles of the US states at a 1:5000000 resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 

#### cb_2018_us_county_5m

+ Contains the shapefiles of the US counties at a 1:5000000 resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 


## Interim

### Demography

+ `demography_counties_2023.csv`: Columns: 1) Full five digit county code, 2) Age group, 3) Population. Formatted from the county-level demography files in `data/raw/demography/counties/cc-est2023-syasex-xx.csv` using `data/conversion/build-demography.py`. 

+ `demography_states_2023.csv`: Columns: 1) State code, 2) Age group, 3) Population. Formatted from the county-level demography files in `data/raw/demography/counties/cc-est2023-syasex-xx.csv` using `data/conversion/build-demography.py`. 

## Conversion

+ `build-demography.py`: Script used to convert the raw county-level population per year of age into state and county level population in age groups.