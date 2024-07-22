# Data readme

Contains an overview of the raw data sources, and the conversion scripts used to build interim data.

## Raw

### Demography

#### States

+ `sc-est2023-agesex-civ.csv`: Contains the estimated population per year of age, sex and US state. Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-detail.html 

#### Counties

+ `cc-est2023-syasex-xx.csv`: Contains the estimated popoulation per year of age and sex, for US state xx (01 'Alabma' and so on). Downloaded from https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html > Datasets > 2020-2023 > counties > asrh > cc-est2023-alldata-01.csv	

### Mobility

+ `table1_commuting_flows_county`: Contains the estimated number of commuters from county X to county Y. Downloaded from https://www.census.gov/data/tables/2020/demo/metro-micro/commuting-flows-2020.html 

### Geography

#### cb_2018_us_state_5m

Contains the shapefiles of the US states at a 1:5000000 resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 

#### cb_2018_us_county_5m

Contains the shapefiles of the US counties at a 1:5000000 resolution. Downloaded from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html 

## Interim


## Conversion