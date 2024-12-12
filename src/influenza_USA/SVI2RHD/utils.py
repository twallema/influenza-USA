"""
This script contains usefull functions for the pySODM US Influenza model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
from datetime import datetime as datetime
from influenza_USA.shared.utils import construct_coordinates_dictionary, get_contact_matrix, get_mobility_matrix, compute_case_hospitalisation_rate, \
                                        check_age_resolution, check_spatial_resolution

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

def initialise_SVI2RHD(spatial_resolution='states', age_resolution='full', season='2017-2018', vaccine_waning='off',
                       distinguish_daytype=True, start_sim=datetime(2024,8,1)):

    # model works at US state or county level
    if ((spatial_resolution != 'states') & (spatial_resolution != 'counties')):
        raise ValueError("this model was designed to work at the US state or county level. valid 'spatial_resolution' are 'states' or 'counties'. found: '{spatial_resolution}'.")

    # load model object
    from influenza_USA.SVI2RHD.model import ODE_SVI2RHD as SVI2RHD

    # construct coordinates
    N, G, coordinates = construct_coordinates_dictionary(spatial_resolution=spatial_resolution, age_resolution=age_resolution)

    # define parameters
    params = {
            # core parameters
            'beta': 0.03*np.ones(G),                                                                                                # infectivity (-)
            'f_v': 0.5,                                                                                                             # fraction of total contacts on visited patch
            'N': get_contact_matrix(daytype='all', age_resolution=age_resolution),                                                  # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
            'M': get_mobility_matrix(spatial_resolution=spatial_resolution, dataset='cellphone_03092020'),                          # origin-destination mobility matrix          
            'n_vacc': np.zeros(shape=(N, G),dtype=np.float64),                                                                      # vaccination incidence (dummy)
            'e_i': 0.2,                                                                                                             # vaccine efficacy against infection
            'e_h': 0.5,                                                                                                             # vaccine efficacy against hospitalisation
            'T_r': 365/np.log(2),                                                                                                   # average time to waning of natural immunity
            'T_v': 10*365,                                                                                                          # average time to waning of vaccine immunity
            'rho_h': 0.014,                                                                                                         # hospitalised fraction (source: Josh)
            'T_h': 3.5,                                                                                                             # average time to hospitalisation (= length infectious period, source: Josh)
            'rho_d': 0.06,                                                                                                          # deceased in hospital fraction (source: Josh)
            'T_d': 5.0,                                                                                                             # average time to hospital outcome (source: Josh)
            'CHR': compute_case_hospitalisation_rate(season, age_resolution=age_resolution),                                        # case hosp. rate corrected for social contact and expressed relative to [0,5) yo
            # time-dependencies
            'vaccine_incidence_modifier': 1.0,                                                                                      # used to modify vaccination incidence
            'vaccine_incidence_timedelta': 0,                                                                                       # shift the vaccination season
            # initial condition function
            'f_I': 1e-4,                                                                                                            # initial fraction of infected
            'f_R': 0.5,                                                                                                             # initial fraction of recovered (USA)
            # outcomes
            'asc_case': 0.004,
            }
    
    # vaccine waning on/off
    if vaccine_waning == 'on':
        params.update({'e_i': 0.2, 'e_h': 0.75, 'T_v': 365/2})

    # initial condition function
    from influenza_USA.SVI2RHD.TDPF import make_initial_condition_function
    initial_condition_function = make_initial_condition_function(spatial_resolution, age_resolution, coordinates['location'], start_sim, season).initial_condition_function
    params.update({
        'delta_f_R_regions': np.zeros(9),
        'delta_f_R_states': np.zeros(52),        
    })
                                                                                 
    # time-dependencies
    TDPFs = {}
    ## contacts
    if distinguish_daytype:
        from influenza_USA.shared.TDPF import make_contact_function
        TDPFs['N'] = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='week_holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='weekend', age_resolution=age_resolution)).contact_function
    ## vaccine uptake
    from influenza_USA.SVI2RHD.TDPF import make_vaccination_function
    TDPFs['n_vacc'] = make_vaccination_function(season, spatial_resolution, age_resolution).vaccination_function

    ## hierarchal transmission rate
    from influenza_USA.SVI2RHD.TDPF import hierarchal_transmission_rate_function
    TDPFs['beta'] = hierarchal_transmission_rate_function(spatial_resolution, sigma=2.5)
    # append its parameters
    params.update(
        {
            'beta_US': 0.03,
            'delta_beta_regions': np.zeros(9),
            'delta_beta_states': np.zeros(52),
            'delta_beta_temporal': np.zeros(10),
            'delta_beta_spatiotemporal': np.zeros(shape=[10,9])
        }
    )

    # hierarchal natural immunity
    from influenza_USA.SVI2RHD.TDPF import hierarchal_waning_natural_immunity
    TDPFs['T_r'] = hierarchal_waning_natural_immunity(spatial_resolution)
    # append its parameters
    params.update({'T_r_US': 365/np.log(2), 'delta_T_r_regions': np.zeros(9)})

    return SVI2RHD(initial_states=initial_condition_function, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)


def get_vaccination_data():
    """
    A function to retrieve the 2010-2024 vaccination incidence data per age group and US state

    output
    ------

    data: pd.DataFrame
        Index: 'season' (str; '20xx-20xx')
        Columns: 'date' (str; 'yyyy-mm-dd'), 'age' (str; '[0, 5('), 'fips' (str; 'xx000'), 'vaccination_incidence' (float)
    """
    rel_dir = f'../../../data/interim/vaccination/vaccination_incidences_2010-2024.csv'
    data = pd.read_csv(os.path.join(abs_dir,rel_dir), dtype={'season': str, 'age': str, 'fips': str, 'daily_incidence': float, 'cumulative': float})
    data['date'] = pd.to_datetime(data['date'])
    return data


def convert_vaccination_data(vaccination_data, spatial_resolution, age_resolution):
    """
    A function converting the vaccination data to the right spatial and age resolution

    Native resolution of the data: 5 age groups, US states

    input
    -----

    vaccination_data: pd.DataFrame
        weekly vaccination incidence from 2010-2024 per age group and US state

    spatial_resolution: str
        'collapsed', 'states' or 'counties'
    
    age_resolution: str
        'collapsed' or 'full'

    output
    ------

    vaccination_data: pd.Dataframe
        weekly vaccination incidence from 2010-2024 with/without age groups, and for the US, US states or US counties
    """
    check_spatial_resolution(spatial_resolution)
    check_age_resolution(age_resolution)
    
    # aggregation or expansion of spatial units
    if spatial_resolution == 'collapsed':
        # sum the vaccination data
        vaccination_data = vaccination_data.groupby(['season', 'date', 'age'], as_index=False).agg({'daily_incidence': 'sum', 'cumulative': 'sum'})
        # re-insert the USA fips code
        vaccination_data['fips'] = '00000'
    elif spatial_resolution == 'counties':
        # perform an expansion from state --> county based on the demography
        # 1. load demography
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_counties_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
        # 2. compute each county+age population's share in the state+age population
        # add state fips
        demography['state_fips'] = demography['fips'].str[:2] + '000'
        # aggregate state-level population by age group
        state_population = demography.groupby(['state_fips', 'age'], as_index=False)['population'].sum()
        state_population.rename(columns={'population': 'state_population'}, inplace=True)
        # merge state population back to the county-level demography data
        demography = demography.merge(state_population, on=['state_fips', 'age'])
        # compute each county's proportional share in that age group
        demography['county_share'] = demography['population'] / demography['state_population']
        # 3. expand vaccination data by merging
        expanded_df = vaccination_data.merge(demography, left_on=['fips', 'age'], right_on=['state_fips', 'age'])
        # 4. multiply the county+age population's share with the state+age vaccination incidence/cumulative vaccinations
        expanded_df['county_daily_incidence'] = expanded_df['daily_incidence'] * expanded_df['county_share']
        expanded_df['county_cumulative'] = expanded_df['cumulative'] * expanded_df['county_share']
        # 5. drop unnecessary columns and clean up
        expanded_df = expanded_df[['season', 'date', 'age', 'fips_y', 'county_daily_incidence', 'county_cumulative']]
        vaccination_data = expanded_df.rename(columns={'fips_y': 'fips', 'county_daily_incidence': 'daily_incidence', 'county_cumulative': 'cumulative'})

    # aggregation of age groups
    if age_resolution == 'collapsed':
        # sum the vaccination data
        vaccination_data = vaccination_data.groupby(['season', 'date', 'fips'], as_index=False).agg({'daily_incidence': 'sum', 'cumulative': 'sum'})
        # re-insert the age group
        vaccination_data['age'] = '[0, 100)'

    return vaccination_data


def get_cumulative_vaccinated(t, season, vaccination_data):
    """
    A function returning the cumulative number of vaccinated individuals at date 't' in season 'season'

    input
    -----
    season: str
        '20xx-20xx': a specific season
        'average': the average vaccination incidence across all seasons

    vaccination_data: pd.DataFrame
        weekly vaccination incidence per age group and US state

    output
    ------
    cumulative_vaccinated: np.ndarray
        number of individuals vaccinated per age group and US state; shape: (5,52)
    """

    # get week number
    week_number = t.isocalendar().week

    # compute state sizes
    n_age = len(vaccination_data['age'].unique())
    n_loc = len(vaccination_data['fips'].unique())

    # check input season
    if ((season not in vaccination_data.season.unique()) & (season != 'average')):
        raise ValueError(f"season '{season}' vaccination data not found. provide a valid season (format '20xx-20xx') or 'average'.")
    
    # drop index
    vaccination_data = vaccination_data.reset_index()

    if season != 'average':
        # slice out correct season
        vaccination_data = vaccination_data[vaccination_data['season'] == season]
        # add week number & remove date
        vaccination_data['week'] = vaccination_data['date'].dt.isocalendar().week.values
        vaccination_data = vaccination_data[['week', 'age', 'fips', 'cumulative']]
        # sort age groups / spatial units --> are sorted in the model
        vaccination_data = vaccination_data.groupby(by=['week', 'age', 'fips']).last().sort_index().reset_index()
    else:
        # add week number & remove date
        vaccination_data['week'] = vaccination_data['date'].dt.isocalendar().week.values
        vaccination_data = vaccination_data[['week', 'age', 'fips', 'cumulative']]
        # average out + sort
        vaccination_data = vaccination_data.groupby(by=['week', 'age', 'fips']).mean('cumulative').sort_index().reset_index()

    try:
        return np.array(vaccination_data[vaccination_data['week'] == week_number]['cumulative'].values, np.float64).reshape(n_age, n_loc) 
    except:
        return np.zeros([n_age, n_loc], np.float64)


def get_spatial_mappings(spatial_resolution):
    """
    A function retrieving spatial mappings for regional/state parameters
        - For a model running at the US state level (52):
            - For a parameter at the US regional level (9) --> maps parameter in a region to all US states within that region
            - For a parameter at the US state level(52) --> does nothing
        - For a model running at the US county level (3222):
            - For a parameter at the US regional level (9) --> maps parameter in a US region to all US counties within that region
            - For a parameter at the US state level(52) --> maps parameter in a US state to all US counties within that region
    
    input
    -----

    spatial_resolution: str
        'states' or 'counties'; 'collapsed' will return an error
    
    output
    ------

    region_mapping: np.ndarray
        A (52,) or (3222,) map of regions to states or counties

    region_mapping: np.ndarray
        A (52,) or (3222,) map of states to states or counties
    """

    # input check on spatial resolution
    check_spatial_resolution(spatial_resolution)
    # get mapping data file
    mapping = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../../data/interim/fips_codes/fips_names_mappings.csv'), dtype={'fips_state': str, 'fips_county': str})
    if spatial_resolution == 'states':
        region_mapping = mapping.groupby(by=['fips_state']).last()['region_mapping'].values
        state_mapping = mapping.groupby(by=['fips_state']).last()['state_mapping'].values
        assert region_mapping.shape == state_mapping.shape
    elif spatial_resolution == 'counties':
        region_mapping = mapping['region_mapping'].values
        state_mapping = mapping['state_mapping'].values
        assert region_mapping.shape == state_mapping.shape
    else:
        raise ValueError("mapping is nonsensical for spatial_resolution='collapsed'.")
    
    return region_mapping, state_mapping