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

+ `vacc_alldoses_age_Flu_2024_R1_allflu_allseasons.csv`: Contains, from the 2010-2011 to the 2023-2024 season, the weekly number of administered vaccines ('vacc_age'), per age group ('age_group'), per US state ('subpop').

### Contacts

`locations-all_daytype-all_XX_polymod-2008.xlsx`: Contains social contact matrices for age groups 0-5, 5-18, 18-50, 50-65, 65+ in country XX from the 2008 Polymod study. Sum of all locations (home, work, school, leisure, others) and averaged over all daytypes (week, weekend, holiday). Physical and non-physical contacts. Available countries: UK, Germany and Finland. Downloaded from: https://lwillem.shinyapps.io/socrates_rshiny/ 

`locations-all_daytype-week_holiday_XX_polymod-2008.xlsx`: Contains social contact matrices for age groups 0-5, 5-18, 18-50, 50-65, 65+ in country XX from the 2008 Polymod study. Sum of all locations (home, work, school, leisure, others) and for weeks during a holiday. Physical and non-physical contacts. Available countries: UK, Germany and Finland. Downloaded from: https://lwillem.shinyapps.io/socrates_rshiny/ 

`locations-all_daytype-week_no-holiday_XX_polymod-2008.xlsx`: Contains social contact matrices for age groups 0-5, 5-18, 18-50, 50-65, 65+ in country XX from the 2008 Polymod study. Sum of all locations (home, work, school, leisure, others) and for non-holiday weeks. Physical and non-physical contacts. Available countries: UK, Germany and Finland. Downloaded from: https://lwillem.shinyapps.io/socrates_rshiny/ 

`locations-all_daytype-weekend_XX_polymod-2008.xlsx`: Contains social contact matrices for age groups 0-5, 5-18, 18-50, 50-65, 65+ in country XX from the 2008 Polymod study. Sum of all locations (home, work, school, leisure, others) and for weekends. Physical and non-physical contacts. Available countries: UK, Germany and Finland. Downloaded from: https://lwillem.shinyapps.io/socrates_rshiny/ 

### fips codes

+ `national_state2020.txt`: Contains the 2020 US state names and corresponding two-digit FIPS. Downloaded from https://www.census.gov/library/reference/code-lists/ansi.html

+ `national_county2020.txt`: Contains the 2020 US county names, and the corresponding two-digit state FIPS and three-digit county FIPS. Does not include the 2020 Connecticut county changes. Downloaded from https://www.census.gov/library/reference/code-lists/ansi.html

### cases

+ `hosp-admissions_FluSurvNet_USA_09-24_raw.csv`: Weekly hospital admissions for Influenza in the USA, from 2009-2010 until 2023-2024, based on FluSurvNet. Obtained from Shaun Truelove.

+ `influenza-surveillance-summary_NC_xx-xx.pdf`: End-of-season report on the xx-xx Influenza season in North Carolina. Of special interest is the figure at the bottom of page 6 titled 'Influenza Positive Tests Reported by PHE Facilities', whose data will be extracted using WebPlotDigitizer. Downloaded from https://flu.ncdhhs.gov/data.htm

+ `hosp-admissions_NC_15-24.csv`: Weekly hospital admissions in North Carolina (reported by PHE facilities) from 2015 to 2024. Indexed on Saturday. Downloaded from: https://ncdetect.org/respiratory-dashboard/ by Matthew Mietchen.

+ `ILI_NC_15-24.csv`: Weekly ILI counts in North Carolina (reported by emergency departments) from 2015 to 2024. Indexed on Saturday. Downloaded from: https://ncdetect.org/respiratory-dashboard/ by Matthew Mietchen.

### initial condition

+ `initial_condition_2017-2018.csv`: Contains the initial condition and model parameters of Josh' reference model, obtained by calibrating the model (no age groups or spatial patches) to the 2017-2018 Influenza season.

+ `initial_condition_2019-2020.csv`: Idem for the 2019-2020 Influenza season.

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

### Vaccination
 
+ `vaccination_incidences_2010-2024.csv`: Formats the raw vaccination data from 2010-2024, `vacc_alldoses_age_Flu_2024_R1_allflu_allseasons.csv`, to use the naming conventions used in this software.

### FIPS codes

+ `fips_names_mappings.csv`: Contains for a five-digit FIPS code for every US state and county. The first two numbers represent the state, the last three numbers represent the county. Post-2020 FIPS codes, contains the reshuffled Connecticut counties. Made using `build-FIPS-list.py`. After making the initial file, the corresponding region names were added, as well with a mapping that can be used to convert arrays where entries represent a US region (9) or US state (52) to an array of length US counties. The order of the US regions is: 0 (New England), 1 (Mid-Atlantic), 2 (East North Central), 3 (West North Central), 4 (South Atlantic), 5 (East South Central), 6 (West South Central), 7 (Mountain), 8 (Pacific).   

### Contacts

+ `locations-all_daytype-all_avg-UK-DE-FI_polymod-2008.csv`: Contains the social contact matrix for age groups 0-5, 5-18, 18-50, 50-65, 65+. Average of the UK, Germany and Finland from the 2008 Polymod study. Sum of contacts in all locations (home, work, school, leisure, others) and averaged over all daytypes (week, weekend, holiday). Physical and non-physical contacts. Number of contacts integrated with the duration of the contact.

+ `locations-all_daytype-week_holiday_avg-UK-DE-FI_polymod-2008.csv`: Contains the social contact matrix for age groups 0-5, 5-18, 18-50, 50-65, 65+. Average of the UK, Germany and Finland from the 2008 Polymod study. Sum of all locations (home, work, school, leisure, others) and for weeks during a holiday. Physical and non-physical contacts. Number of contacts integrated with the duration of the contact.

+ `locations-all_daytype-week_no-holiday_avg-UK-DE-FI_polymod-2008.csv`: Contains the social contact matrix for age groups 0-5, 5-18, 18-50, 50-65, 65+. Average of the UK, Germany and Finland from the 2008 Polymod study. Sum of all locations (home, work, school, leisure, others) and for non-holiday weeks. Physical and non-physical contacts. Number of contacts integrated with the duration of the contact.

+ `locations-all_daytype-weekend_avg-UK-DE-FI_polymod-2008.csv`: Contains the social contact matrix for age groups 0-5, 5-18, 18-50, 50-65, 65+. Average of the UK, Germany and Finland from the 2008 Polymod study. Sum of all locations (home, work, school, leisure, others) and for weekends. Physical and non-physical contacts. Number of contacts integrated with the duration of the contact.

### Cases

+ `hosp-rate-age_CDC_USA_10-23.csv`: Hospitalisation rate per age group as reported by the CDC.

+ `hosp-admissions_FluSurvNet_USA_09-24.csv`: Contains the weekly flu hospitalisation incidence from 2009-2024 at the US state level. Data start in week 40. Dataset has four columns: 'season_start', 'date', 'location' (fips), 'H_inc'.

+ `subtypes_NC_14-24.csv`: Number of positive tests reported by PHE facilities attributed to Influenza A versus Influenza B Influenza. Extracted from the end-of-season reports available in the raw data folder: `influenza-surveillance-summary_NC_xx-xx.pdf`. 

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

### Vaccination

+ `build-vaccination.py`: A script that formats the vaccination rates of the 2010-2024 season.

### FIPS codes

+ `build-FIPS-list.py`: A script formatting and merging the raw US state and county FIPS codes found in `national_state2020.txt` and `national_county2020.txt`. Corrects the Connecticut counties to the post 2020 counties and FIPS codes using the crosswalk file `ct_cou_to_cousub_crosswalk.xlsx`. Resulting file in interim folder contains a five-digit FIPS codes for both states and counties. State FIPS codes are assumed to have an 'xx000' format.

### Cases

+ `build-hospitalisations_per_state.py`: A script formatting the weekly flu hospitalisation incidence data per US state from 2009-2024 in `data/raw/cases/weekly_flu_incid_complete.csv` into `data/interim/hospitalisations_per_state.csv`. Swaps the US state name for it's FIPS code and guarantees the data always start in week 40 of the year.