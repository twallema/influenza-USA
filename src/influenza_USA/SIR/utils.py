"""
This script contains usefull functions for the pySODM US Influenza model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

def initialise_SIR(spatial_resolution='states', age_resolution='full', distinguish_daytype=False, stochastic=False):
    """
    Initialises an age- and space structured SIR model for the USA

    input
    -----
    spatial_resolution: str
        'collapsed', 'states' or 'counties'. 

    age_resolution: str
        'collapsed' or 'full'. 

    distinguish_daytype: bool
        Differ contacts by weekday, weekendday and holiday.

    stochastic: bool
        Deterministic ODE model (False) or Gillespie Tau-leaping stochastic model (True)

    output
    ------

    model: pySODM model
        Initialised pySODM SIR model
    """

    # choose right model
    if stochastic:
        from influenza_USA.SIR.model import TL_SIR as SIR
    else:
        from influenza_USA.SIR.model import ODE_SIR as SIR

    # initialise coordinates
    coordinates = construct_coordinates_dictionary(spatial_resolution, age_resolution)

    # initialise parameters
    params = {'beta': 0.015,                                                                                                      # infectivity (-)
            'gamma': 5,                                                                                                           # duration of infection (d)
            'f_v': 0.5,                                                                                                           # fraction of total contacts on visited patch
            'N': tf.convert_to_tensor(get_contact_matrix(daytype='all', age_resolution=age_resolution), dtype=float),                         # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
            'M': tf.convert_to_tensor(get_mobility_matrix(dataset='cellphone_03092020', spatial_resolution=spatial_resolution), dtype=float)      # origin-destination mobility matrix
            }

    # initialise initial states
    I0 = construct_initial_infected(seed_loc=('alabama',''), n=10, agedist='demographic', spatial_resolution=spatial_resolution, age_resolution=age_resolution)
    S0 = construct_initial_susceptible(I0, spatial_resolution=spatial_resolution, age_resolution=age_resolution)
    init_states = {'S': tf.convert_to_tensor(S0, dtype=float),
                'I': tf.convert_to_tensor(I0, dtype=float)
                    }

    # initialise time-dependencies
    TDPFs={}
    if distinguish_daytype:
        from influenza_USA.SIR.TDPF import make_contact_function
        TDPFs['N'] = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='week_holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='weekend', age_resolution=age_resolution)).contact_function
        
    return SIR(initial_states=init_states, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)

def construct_coordinates_dictionary(spatial_resolution='states', age_resolution='full'):
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

def get_contact_matrix(daytype='all', age_resolution='full'):
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

def get_mobility_matrix(dataset='cellphone_03092020', spatial_resolution='states'):
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

def construct_initial_susceptible(*subtract_states, spatial_resolution='states', age_resolution='full'):
    """
    A function to construct the initial number of susceptible individuals, computed as the number of susceptibles 'S' derived from the demographic data, minus any individiduals present in `subtract_states`

    input
    -----

    *subtract_states: np.ndarray
        Other states that contain individuals at the start of the simulation.
        Must must be substracted from the demographic data to compute the number of initial susceptible individuals

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
def construct_initial_infected(seed_loc=('',''), n=1, agedist='demographic', spatial_resolution='states', age_resolution='full'):
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
    