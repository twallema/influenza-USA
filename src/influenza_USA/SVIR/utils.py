"""
This script contains usefull functions for the pySODM US Influenza model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime as datetime

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

def initialise_SVI2RHD(spatial_resolution='states', age_resolution='full', season='2017-2018', distinguish_daytype=True, stochastic=False, start_sim=datetime(2024,8,1)):

    # model
    if stochastic:
        from influenza_USA.SVIR.model import TL_SVI2RHD as SVI2RHD
    else:
        from influenza_USA.SVIR.model import ODE_SVI2RHD as SVI2RHD

    # coordinates
    coordinates = construct_coordinates_dictionary(spatial_resolution=spatial_resolution, age_resolution=age_resolution)

    # parameters
    params = {
            # core parameters
            'beta': 0.03*np.ones(52),                                                                                               # infectivity (-)
            'f_v': 0.5,                                                                                                             # fraction of total contacts on visited patch
            'N': tf.convert_to_tensor(get_contact_matrix(daytype='all', age_resolution=age_resolution), dtype=float),               # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
            'M': tf.convert_to_tensor(get_mobility_matrix(spatial_resolution=spatial_resolution, dataset='cellphone_03092020'), dtype=float),    # origin-destination mobility matrix          
            'r_vacc': np.ones(shape=(len(coordinates['age_group']), len(coordinates['location'])),dtype=np.float64),                # vaccination rate (dummy)
            'e_i': 0.2,                                                                                                             # vaccine efficacy against infection
            'e_h': 0.5,                                                                                                             # vaccine efficacy against hospitalisation
            'T_r': 365,                                                                                                             # average time to waning of natural immunity
            'T_v': 10*365/2,                                                                                                        # average time to waning of vaccine immunity
            'rho_h': 0.014,                                                                                                         # hospitalised fraction (source: Josh)
            'T_h': 3.5,                                                                                                             # average time to hospitalisation (= length infectious period, source: Josh)
            'rho_d': 0.06,                                                                                                          # deceased in hospital fraction (source: Josh)
            'T_d': 5.0,                                                                                                             # average time to hospital outcome (source: Josh)
            # time-dependencies
            'vaccine_rate_modifier': 1.0,                                                                                           # used to modify vaccination rate
            'vaccine_rate_timedelta': 0,                                                                                            # shift the vaccination season
            # initial condition
            'f_I': 1e-4,                                                                                                            # initial fraction of infected
            'f_R': 0.5*np.ones(52),                                                                                                 # initial fraction of recovered
            # outcomes
            'asc_case': 0.004,
            }

    # season-specific case hospitalisation rate
    if season == '2017-2018':
        CDC_estimated_hosp = np.array([23750, 19636, 76819, 123601, 466766])                            # https://archive.cdc.gov/#/details?url=https://www.cdc.gov/flu/about/burden/2017-2018.htm
    elif season == '2019-2020':
        CDC_estimated_hosp = np.array([26376, 19276, 80866, 92391, 173012])                             # https://www.cdc.gov/flu-burden/php/data-vis/2019-2020.html?CDC_AAref_Val=https://www.cdc.gov/flu/about/burden/2019-2020.html
    demo = np.array([18608139, 54722401, 141598551, 63172279, 60019216])                                # US demography
    rel_contacts = np.array([17.4/19.8, 17.4/29.1, 17.4/18.6, 17.4/13.2, 17.4/7.8])                     # contacts in age groups relative to the population average
    params.update({'CHR': rel_contacts * (CDC_estimated_hosp/demo) / (rel_contacts * (CDC_estimated_hosp/demo))[0]})

    # initial condition
    # OLD >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    ## states
    ic = load_initial_condition(season=season)
    total_population = construct_initial_susceptible(spatial_resolution, age_resolution)
    init_states = {}
    for k,v in ic.items():
        # no vaccines initially
        if k != 'V':
            init_states[k] = tf.convert_to_tensor(v * total_population)
    ## outcomes
    init_states['I_inc'] = 0 * total_population
    init_states['H_inc'] = 0 * total_population
    init_states['D_inc'] = 0 * total_population
    # NEW >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    from influenza_USA.SVIR.TDPF import make_initial_condition_function
    initial_condition_function = make_initial_condition_function(spatial_resolution, age_resolution).initial_condition_function
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # time-dependencies
    TDPFs = {}
    ## contacts
    if distinguish_daytype:
        from influenza_USA.SVIR.TDPF import make_contact_function
        TDPFs['N'] = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='week_holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='weekend', age_resolution=age_resolution)).contact_function
    ## vaccines
    ### vaccine uptake
    from influenza_USA.SVIR.TDPF import make_vaccination_function
    TDPFs['r_vacc'] = make_vaccination_function(get_vaccination_data()).vaccination_function

    return SVI2RHD(initial_states=initial_condition_function, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)

def construct_coordinates_dictionary(spatial_resolution, age_resolution):
    """
    A function returning the model's coordinates for the dimension 'age_group' and 'location'.

    input
    -----

    spatial_resolution: str
        USA 'collapsed', 'states' or 'counties'

    age_resolution: str
        'collapsed', 'full' (0-5, 5-18, 18-50, 50-65, 65+)

    output
    ------

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

    return {'age_group': age_groups, 'location': list(demography['fips'].unique())}

def get_vaccination_data():
    """
    A function to retrieve the 2017-2018 vaccination data
    """
    rel_dir = f'../../../data/interim/vaccination/vaccination_rates_2017-2018.csv'
    return pd.read_csv(os.path.join(abs_dir,rel_dir), index_col=0, header=0, parse_dates=True).reset_index()

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

def load_initial_condition(season='2017-2018'):
    """
    A function to load the initial condition of the Influenza model, as calibrated by Josh to the 17-18 and 18-19 seasons
    """
    # get initial condition
    rel_dir = f'../../../data/raw/initial_condition/initial_condition_{season}.csv'
    ic = pd.read_csv(os.path.join(abs_dir,rel_dir), header=0)[['S_0', 'I_0', 'IV_0', 'V_0', 'H_0', 'R_0', 'D_0']]
    # rename columns
    ic = ic.rename(columns={'S_0': 'S', 'I_0': 'I', 'IV_0': 'Iv', 'V_0': 'V', 'H_0': 'H', 'R_0': 'R', 'D_0': 'D'})
    # compute average
    ic = ic.div(ic.sum(axis=1), axis=0)
    ic = ic.mean(axis=0)
    # verify sum of average is one
    #assert ic.sum(axis=0) == 1.0, 'initial condition should sum to one'
    # we start without any vaccines administered
    ic['S'] += (ic['S'] / (ic['S'] + ic['R'])) * ic['V']
    ic['R'] += (ic['R'] / (ic['S'] + ic['R'])) * ic['V']
    ic = ic.drop(columns='V')
    # return output
    return ic

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
    fips_codes = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/fips_codes/fips_state_county.csv'), dtype={'fips_state': str, 'fips_county': str})

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
    df = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/fips_codes/fips_state_county.csv'), dtype={'fips_state': str, 'fips_county': str})

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
    df = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/fips_codes/fips_state_county.csv'), dtype={'fips_state': str, 'fips_county': str})

    # look up name
    if fips_county == '000':
        return df[df['fips_state'] == fips_state]['name_state'].unique()[0]
    else:
        lu = df[((df['fips_state'] == fips_state) & (df['fips_county'] == fips_county))][['name_state', 'name_county']]
        return lu['name_state'].values[0], lu['name_county'].values[0]
    