"""
This script contains usefull functions for the pySODM US Influenza model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
from datetime import datetime as datetime

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

def initialise_SIR_SequentialTwoStrain(spatial_resolution='states', age_resolution='full', season='2017-2018', distinguish_daytype=True):

    # model works at US state or county level
    if ((spatial_resolution != 'states') & (spatial_resolution != 'counties')):
        raise ValueError("this model was designed to work at the US state or county level. valid 'spatial_resolution' are 'states' or 'counties'. found: '{spatial_resolution}'.")

    # load model object
    from influenza_USA.SIR_SequentialTwoStrain.model import ODE_SIR_SequentialTwoStrain as SIR_SequentialTwoStrain

    # construct coordinates
    N, G, coordinates = construct_coordinates_dictionary(spatial_resolution=spatial_resolution, age_resolution=age_resolution)

    # define parameters
    params = {
            # core parameters
            'beta1': 0.028*np.ones(G),                                                                                              # infectivity strain 1 (-)
            'beta2': 0.028*np.ones(G),                                                                                              # infectivity strain 2 (-)
            'N': get_contact_matrix(daytype='all', age_resolution=age_resolution),                                                  # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
            'T_r': 3.5,                                                                                                             # average time to recovery 
            'CHR': compute_case_hospitalisation_rate(season, age_resolution=age_resolution),                                        # case hosp. rate corrected for social contact and expressed relative to [0,5) yo
            # outcomes
            'rho_h1': 0.001,                                                                                                        # hospitalised fraction (source: Josh)
            'rho_h2': 0.001,                                                                                                        # hospitalised fraction (source: Josh)
            # initial condition function
            'f_I1': 1e-4,                                                                                                           # initial fraction of infected with strain 1
            'f_I2': 1e-5,                                                                                                           # initial fraction of infected with strain 2
            'f_R1_R2': 0.75,                                                                                                        # sum of the initial fraction recovered from strain 1 and strain 2 --> needed to constraint initial R between 0 and 1 during calibration
            'f_R1': 0.45,                                                                                                           # fraction of f_R1_R2 recovered from strain 1
            }
    
    # initial condition function
    from influenza_USA.SIR_SequentialTwoStrain.TDPF import make_initial_condition_function
    initial_condition_function = make_initial_condition_function(spatial_resolution, age_resolution).initial_condition_function
    params.update({
        'delta_f_I1_regions': np.zeros(9),
        'delta_f_I2_regions': np.zeros(9),
        'delta_f_R1_regions': np.zeros(9),
        'delta_f_R2_regions': np.zeros(9),
    })
                                                                                 
    # time-dependencies
    TDPFs = {}
    ## contacts
    if distinguish_daytype:
        from influenza_USA.SIR_SequentialTwoStrain.TDPF import make_contact_function
        TDPFs['N'] = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='week_holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='weekend', age_resolution=age_resolution)).contact_function
    ## hierarchal transmission rate
    from influenza_USA.SIR_SequentialTwoStrain.TDPF import hierarchal_transmission_rate_function
    TDPFs['beta1'] = hierarchal_transmission_rate_function(spatial_resolution).strain1_function
    TDPFs['beta2'] = hierarchal_transmission_rate_function(spatial_resolution).strain2_function
    # append its parameters
    params.update(
        {
            'beta1_US': 0.03,
            'beta2_US': 0.03,
            'delta_beta1_regions': np.zeros(9),
            'delta_beta2_regions': np.zeros(9),
            'delta_beta1_states': np.zeros(52),
            'delta_beta2_states': np.zeros(52),
            'delta_beta_temporal': np.zeros(10)
        }
    )

    return SIR_SequentialTwoStrain(initial_states=initial_condition_function, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)

def construct_coordinates_dictionary(spatial_resolution, age_resolution):
    """
    A function returning the model's coordinates for the dimension 'age_group' and 'location', as well as the number of coordinates for each dimension

    input
    -----

    spatial_resolution: str
        USA 'collapsed', 'states' or 'counties'

    age_resolution: str
        'collapsed' or 'full' (0-5, 5-18, 18-50, 50-65, 65+)

    output
    ------
    N: int
        number of age groups
    
    G: int
        number of spatial units

    coordinates: dict
        Keys: 'age_group', 'location'. Values: Str representing age groups, Str representing US FIPS code of spatial unit
    """

    check_spatial_resolution(spatial_resolution)
    check_age_resolution(age_resolution)
    # space
    if spatial_resolution == 'collapsed':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_collapsed_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    elif spatial_resolution == 'states':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_states_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    else:
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_counties_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    # age
    if age_resolution == 'collapsed':
        age_groups = ['[0, 100)']
    else:
        age_groups = list(demography['age'].unique())

    return len(age_groups), len(list(demography['fips'].unique())), {'age_group': age_groups, 'location': list(demography['fips'].unique())}

def compute_case_hospitalisation_rate(season, age_resolution):
    """
    A function to compute the influenza case hospitalisation rate per age group
    
    - corrected for the differences in the number of social contacts
    - expressed relative to [0, 5) years old (= 1)

    input
    -----
    season: str
        '20xx-20xx': a specific season
        'average': the average case hospitalisation rate across all seasons
    
    age_resolution: str
        'collapsed': (0-100)
        'full': (0-5, 5-18, 18-50, 50-65, 65+) 

    output
    ------

    CHR: np.ndarray
        case hospitalisation rate
    """

    # get case hospitalisation rates published by CDC
    CDC_estimated_hosp = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/cases/CDC_hosp-rate-age_2010-2023.csv'),
                                        dtype={'season': str, 'age': str, 'hospitalisation_rate': float})

    # check input season
    if ((season not in CDC_estimated_hosp['season'].unique()) & (season != 'average')):
        raise ValueError(f"season '{season}' case hospitalisation data not found. provide a valid season (format '20xx-20xx') or 'average'.")

    # slice right season out / average over all seasons
    if season != 'average':
        CDC_estimated_hosp = CDC_estimated_hosp[CDC_estimated_hosp['season'] == season]['hospitalisation_rate'].values
    else:
        CDC_estimated_hosp = CDC_estimated_hosp.groupby(by=['age']).mean('hospitalisation_rate')['hospitalisation_rate'].values
    
    # get demography per age group
    demography = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/demography/demography_collapsed_2023.csv'),
                                dtype={'fips': str, 'age': str, 'population': int})['population'].values

    # get social contact matrix
    contacts = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/contacts/locations-all_daytype-all_avg-UK-DE-FI_polymod-2008.csv'), index_col=0, header=0)
    
    # normalise contacts per age group
    rel_contacts = (np.mean(np.sum(contacts, axis=1)) / np.sum(contacts, axis=1)).values

    # account for difference in contact rates
    CHR = rel_contacts * (CDC_estimated_hosp/demography) / (rel_contacts * (CDC_estimated_hosp/demography))[0]

    # collapse age groups if necessary
    check_age_resolution(age_resolution)
    if age_resolution == 'collapsed':
        return np.ones(shape=(1,1)) * np.sum(CHR * demography / np.sum(demography))

    return CHR

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

def get_contact_matrix(daytype, age_resolution):
    """
    A function to retrieve a Polymod 2008 contact matrix, averaged for the UK, DE and FI and used as a proxy for American contacts

    input
    -----

    daytype: str
        Valid arguments are: 'all', 'week_holiday', 'week_no-holiday', 'weekend'
    
    age_resolution: str
        'collapsed', 'full' (0-5, 5-18, 18-50, 50-65, 65+)

    output
    ------

    contacts: np.ndarray
        'age_resolution' == 'full': shape = (5,5)
        'age_resolution' == 'collapsed': shape = (1,1)
    """

    check_age_resolution(age_resolution)
    check_contact_daytype(daytype)

    # get contacts
    rel_dir = f'../../../data/interim/contacts/locations-all_daytype-{daytype}_avg-UK-DE-FI_polymod-2008.csv'
    contacts = pd.read_csv(os.path.join(abs_dir,rel_dir), index_col=0, header=0)

    # get overall demography
    demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_collapsed_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})['population']

    # aggregate contact matrix with demography
    if age_resolution == 'collapsed':
        return np.ones(shape=(1,1)) * np.sum(np.sum(contacts.values, axis=0) * demography.values / np.sum(demography.values))
    else:
        return contacts.values

def get_mobility_matrix(spatial_resolution, dataset='cellphone_03092020'):
    """
    A function to extract and format the mobility matrix

    input
    -----

    dataset: str
        'cellphone_03092020', 'commuters_2011-2015', or 'commuters_2016-2020'
    
    spatial_resolution: str
        USA 'collapsed', 'states' or 'counties'

    output
    ------

    mobility_matrix: np.ndarray
        2D origin-destination mobility matrix. Size: (states) 52 x 52, Counties: 3222 x 3222.

    remarks:
    --------

    At the US County level, use the 'cellphone_03092020' dataset over the 'commuters_2011-2015' or 'commuters_2016-2020' dataset, as the latter are inaccurate. The departure-diffusion powerlaw gravitation model was used for all county-to-county matrices. 
    At the US State level, the `cellphone_03092020' dataset was fit with a departure-diffusion powerlaw gravitation model, while the 'commuters_2011-2015' and 'commuters_2016-2020' datasets were fit with a departure-diffusion radiation model.
    """

    # retrieve appropriate demography and mobility matrix
    check_spatial_resolution(spatial_resolution)
    if spatial_resolution == 'collapsed':
        return np.array([[1]])
    elif spatial_resolution == 'states':
        # retrieve demography & aggregate age groups & return as numpy array
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_states_2023.csv'), dtype={'fips': str, 'age': str, 'population': int}).groupby(by='fips').sum()['population'].values
        # retrieve mobility matrix & return as numpy array
        if dataset == 'cellphone_03092020':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_state_data/departure_diffusion_power_gravitation/matrix_cellphone_03092020_departure-diffusion-powerlaw-gravitation_states.csv'), index_col=0).values
        elif dataset == 'commuters_2011-2015':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_state_data/departure_diffusion_radiation/matrix_commuters_2011-2015_departure-diffusion-radiation_states.csv'), index_col=0).values
        elif dataset == 'commuters_2016-2020':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_state_data/departure_diffusion_radiation/matrix_commuters_2011-2015_departure-diffusion-radiation_states.csv'), index_col=0).values
        else:
            raise ValueError("valid datasets are 'cellphone_03092020', 'commuters_2011-2015' or 'commuters_2016-2020'")
    else:
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_counties_2023.csv'), dtype={'fips': str, 'age': str, 'population': int}).groupby(by='fips').sum()['population'].values
        if dataset == 'cellphone_03092020':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_county_data/departure_diffusion_power_gravitation/matrix_cellphone_03092020_departure-diffusion-powerlaw-gravitation_counties.csv'), index_col=0).values
        elif dataset == 'commuters_2011-2015':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_county_data/departure_diffusion_power_gravitation/matrix_commuters_2011-2015_departure-diffusion-powerlaw-gravitation_counties.csv'), index_col=0).values
        elif dataset == 'commuters_2016-2020':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_county_data/departure_diffusion_power_gravitation/matrix_commuters_2016-2020_departure-diffusion-powerlaw-gravitation_counties.csv'), index_col=0).values
        else:
            raise ValueError("valid datasets are 'cellphone_03092020', 'commuters_2011-2015' or 'commuters_2016-2020'")

    # normalise mobility matrix
    mobility_matrix /= demography[:, None]

    # assume on-diagonal = 1 - traveling --> number of on-diagonal trips in cellphone data always exceed the number of inhabitants (people make multiple within-patch trips; but we're not interested in that at all) 
    np.fill_diagonal(mobility_matrix, 0)
    rowsum = mobility_matrix.sum(axis=1)
    np.fill_diagonal(mobility_matrix, 1-rowsum)

    # diagonal elements smaller than one?
    assert np.all(rowsum <= 1), f"there are {spatial_resolution} where the number of people traveling is higher than the population"

    # negative elements?
    assert np.all(mobility_matrix >= 0)

    return mobility_matrix

def construct_initial_susceptible(spatial_resolution, age_resolution, *subtract_states):
    """
    A function to construct the initial number of susceptible individuals, computed as the number of susceptibles 'S' derived from the demographic data, minus any individiduals present in `subtract_states`

    input
    -----

    *subtract_states: np.ndarray
        Other states that contain individuals at the start of the simulation.
        Substracted from the demographic data to compute the number of initial susceptible individuals
        If not provided: function returns demography

    spatial_resolution: str
        US 'states' or 'counties'

    age_resolution: str
        'collapsed', 'full' (0-5, 5-18, 18-50, 50-65, 65+)

    output
    ------

    S0: np.ndarray
        Initial number of susceptible. Shape n_age x n_loc. 
    """

    # load demography
    check_spatial_resolution(spatial_resolution)
    if spatial_resolution == 'collapsed':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_collapsed_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    elif spatial_resolution == 'states':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_states_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    else:
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_counties_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})

    # derive shape model states
    n_age = len(demography['age'].unique())
    n_loc = len(demography['fips'].unique())

    # convert to numpy array
    S0 = demography.set_index(['fips', 'age'])
    S0 = np.transpose(S0.values.reshape(n_loc, n_age))

    # collapse age groups if necessary
    if age_resolution == 'collapsed':
        n_age = 1
        S0 = np.sum(S0, axis=0)[np.newaxis,:]

    # there exist subpopulations with no susceptibles at the US county level
    S0 = np.where(S0 == 0, 1e-3, S0)

    # subtract other states
    for ss in subtract_states:
        # input checks
        if not isinstance(ss, np.ndarray):
            raise TypeError("input `subtract_states` should be of type np.ndarray")
        if ss.shape != (n_age, n_loc):
            raise ValueError(f"input `subtract_states` should be an {n_age}x{n_loc} np.ndarray")
        # subtraction
        S0 = S0 - ss

    # assert if there are negative susceptibles
    assert np.all(S0 >= 0), "the number of susceptibles is negative."

    return S0

import random
import warnings
def construct_initial_infected(spatial_resolution, age_resolution, seed_loc=('',''), n=1, agedist='demographic'):
    """
    A function to seed an initial number of infected

    input
    -----
    seed_loc: tuple containing two strings
        Location of initial infected. First string represents the name of the state. Second string represents the name of the county.

        Behavior:
        ---------

        Spatial resolution 'states':
            - seed_loc=('',''): seed in random state
            - seed_loc=('state_name',''): seed in state 'state_name'

        Spatial resolution 'states':
            - seed_loc=('',''): seed in random state, random county
            - seed_loc=('state_name',''): seed in state 'state_name', random county
            - seed_loc=('state_name','county_name'): seed in state 'state_name', county 'county_name'

    n: int/float
        The number of infected dividuals present in the select location. 
    
    agedist: str
        The distribution of the initial number of infected over the model's age groups. Either 'uniform', 'random' or 'demographic'.

    spatial_resolution: str
        US 'states' or 'counties'

    age_resolution: str
        'collapsed', 'full': (0-5, 5-18, 18-50, 50-65, 65+)

    output
    ------
    I0: np.ndarray
        Initial number of susceptible. Shape n_age x n_loc. 
    """

    # load demography
    check_spatial_resolution(spatial_resolution)
    if spatial_resolution == 'collapsed':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_collapsed_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    elif spatial_resolution == 'states':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_states_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    else:
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_counties_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})

    # load fips codes
    fips_codes = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/fips_codes/fips_names_mappings.csv'), dtype={'fips_state': str, 'fips_county': str})

    # input checks on seed location
    assert isinstance(seed_loc, tuple), "`seed_loc` must be of type 'tpl'"
    assert all(isinstance(item, str) for item in seed_loc), "entries in `seed_loc` must be of type 'str'"

    # interpret the seed location and get the right fips code
    if spatial_resolution == 'collapsed':
        seed_fips = '00000'
    elif spatial_resolution == 'states':
        # random 
        if ((seed_loc[0] == '') & (seed_loc[1] == '')):
            seed_fips = random.choice(list(fips_codes['fips_state'].unique())) + '000'
        # specified
        elif ((seed_loc[0] != '') & (seed_loc[1] == '')):
            seed_fips = name2fips(seed_loc[0])
        # specified
        elif ((seed_loc[0] != '') & (seed_loc[1] != '')):
            seed_fips = name2fips(seed_loc[0])
            warnings.warn("\nusing the first entry in `seed_loc` to specify the seed state", stacklevel=2)
        elif ((seed_loc[0] == '') & (seed_loc[1] != '')):
            seed_fips = random.choice(list(fips_codes['fips_state'].unique())) + '000'
            warnings.warn("\nthe first entry in `seed_loc` is empty. sampling a random seed state.", stacklevel=2)
    else:
        # fully random
        if ((seed_loc[0] == '') & (seed_loc[1] == '')):
            seed_fips = random.choice(list(demography['fips'].unique()))
        # fully specified 
        elif ((seed_loc[0] != '') & (seed_loc[1] != '')):
            seed_fips = name2fips(seed_loc[0], seed_loc[1])
        # state specified, county random
        elif ((seed_loc[0] != '') & (seed_loc[1] == '')):
            seed_fips = name2fips(seed_loc[0])[0:2] + random.choice(list(fips_codes[fips_codes['fips_state'] == name2fips(seed_loc[0])[0:2]]['fips_county'].unique()))
        # invalid
        elif ((seed_loc[0] == '') & (seed_loc[1] != '')):
            raise ValueError("input argument `seed_loc`. specifying only a county name is not valid.")

    # distribute over age groups
    demography_fips = demography[demography['fips'] == seed_fips]['population'].values
    if agedist == 'demographic':
        agedist_n = np.random.multinomial(n, demography_fips / sum(demography_fips))
    elif agedist == 'uniform':
        agedist_n = n/len(demography_fips) * np.ones(len(demography_fips))
    elif agedist == 'random':
        agedist_n = np.zeros(len(demography_fips))
        agedist_n[random.randint(0, len(demography_fips)-1)] = n
    else:
        raise ValueError(
            f"invalid input {agedist} for input argument 'agedist'. valid options are 'random', 'uniform', 'demographic'"
        )

    # build initial infected
    I0 = demography.set_index(['age', 'fips'])
    I0['population'] = 0.0
    I0.loc[(slice(None), seed_fips), 'population'] = agedist_n

    # convert to numpy array
    n_age = len(demography['age'].unique())
    n_fips = len(demography['fips'].unique())
    I0 = np.transpose(I0.values.reshape(n_fips, n_age))

    # collapse age groups if necessary
    if age_resolution == 'collapsed':
        I0 = np.sum(I0, axis=0)[np.newaxis,:]

    return I0 

def check_spatial_resolution(spatial_resolution):
    """
    A function to check the validity of the spatial resolution

    input
    -----

    spatial_resolution: str
        Valid arguments are: 'collapsed', 'states' or 'counties'
    """
    assert isinstance(spatial_resolution, str), "spatial_resolution must be a str"
    assert spatial_resolution in ['collapsed', 'states', 'counties'], f"invalid 'spatial_resolution' {spatial_resolution}. valid options are: 'collapsed', 'states', 'counties'"

def check_age_resolution(age_resolution):
    """
    A function to check the validity of the age resolution

    input
    -----

    age_resolution: str
        Valid arguments are: 'collapsed', 'full'
    """

    assert isinstance(age_resolution, str), "age_resolution must be a str"
    assert age_resolution in ['collapsed', 'full'], f"invalid age_resolution '{age_resolution}'. valid options are: 'collapsed', 'full'"

def check_contact_daytype(daytype):
    """
    A function to check the validity of the daytype

    input
    -----

    daytype: str
        Valid arguments are: 'all', 'week_holiday', 'week_no-holiday', 'weekend'
    """

    assert isinstance(daytype, str), "daytype must be a str"
    assert daytype in ['all', 'week_holiday', 'week_no-holiday', 'weekend'], f"invalid daytype '{daytype}'. valid options are: 'all', 'week_holiday', 'week_no-holiday', 'weekend'"

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

def name2fips(name_state, name_county=None):
    """
    A function to convert a US state, or US state and county name into a FIPS code
    US FIPS codes have the following format: SSCCC 1) The first two digits represent the state, 2) the last three digits represent the county. A state is assigned the county code '000'.

    input:
    ------

    name_state: str
        Name of the US state
    name_county: None/str
        Name of the US county. (Default: 'None').

    output:
    -------

    fips: str
        Five digit US FIPS code. If name_county=None, the fips code of the US state is returned in the format 'xx000'.
    """

    # load FIPS-name list (fips_state (2 digits), fips_county (3 digits), name_state, name_county)
    df = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/fips_codes/fips_names_mappings.csv'), dtype={'fips_state': str, 'fips_county': str})

    # state name only
    if not isinstance(name_state, str):
        raise TypeError('provided state name should be a string')
    if name_state.lower() not in df['name_state'].values:
        raise ValueError(
            f"name '{name_state}' is not a valid US state name"
        )
    fips_state = df[df['name_state'] == name_state.lower()]['fips_state'].unique()[0]
    if not name_county:
        return fips_state + '000'
    # state and county
    else:
        if not isinstance(name_county, str):
            raise TypeError('provided county name should be a string')
        if name_county.lower() not in df['name_county'].values:
            raise ValueError(
                f"name '{name_county}' is not a valid US county name;\ndon't forget a suffix like 'county'/'parish'/'planning region'"
            )
        fips_county = df[((df['name_state'] == name_state.lower()) & (df['name_county'] == name_county.lower()))]['fips_county'].unique()[0]
        return fips_state + fips_county

def fips2name(fips_code):
    """
    A function to convert a five digit US state or US county FIPS code into a name
    US FIPS codes have the following format: SSCCC 1) The first two digits represent the state, 2) the last three digits represent the county. A state is assigned the county code '000'.

    input:
    ------

    fips_code: str
        Five digit FIPS code. 

    output:
    -------

    name_state: str
        State name. 
    
    name_county: str
        County name. No county name is returned if the five-digit FIPS code represents a US state (format 'xx000')
    """

    # check input
    if not isinstance(fips_code, str):
        raise TypeError("fips code must be of type 'str'")
    if len(fips_code) != 5:
        raise ValueError("fips codes consist of five digits. state fips codes must be extended from 2 to 5 digits with zeros")
    
    # split input
    fips_state = fips_code[0:2]
    fips_county = fips_code[2:]

    # load FIPS-name list
    df = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/fips_codes/fips_names_mappings.csv'), dtype={'fips_state': str, 'fips_county': str})

    # look up name
    if fips_county == '000':
        return df[df['fips_state'] == fips_state]['name_state'].unique()[0]
    else:
        lu = df[((df['fips_state'] == fips_state) & (df['fips_county'] == fips_county))][['name_state', 'name_county']]
        return lu['name_state'].values[0], lu['name_county'].values[0]
    