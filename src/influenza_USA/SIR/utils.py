"""
This script contains usefull functions for the pySODM US Influenza model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from influenza_USA.shared.utils import construct_coordinates_dictionary, get_contact_matrix, get_mobility_matrix, \
                                        construct_initial_susceptible, check_spatial_resolution, name2fips

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
    _, _, coordinates = construct_coordinates_dictionary(spatial_resolution, age_resolution)

    # initialise parameters
    params = {'beta': 0.015,                                                                                                      # infectivity (-)
            'gamma': 5,                                                                                                           # duration of infection (d)
            'f_v': 0.5,                                                                                                           # fraction of total contacts on visited patch
            'N': tf.convert_to_tensor(get_contact_matrix(daytype='all', age_resolution=age_resolution), dtype=float),                         # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
            'M': tf.convert_to_tensor(get_mobility_matrix(dataset='cellphone_03092020', spatial_resolution=spatial_resolution), dtype=float)      # origin-destination mobility matrix
            }

    # initialise initial states
    I0 = construct_initial_infected(seed_loc=('alabama',''), n=10, agedist='demographic', spatial_resolution=spatial_resolution, age_resolution=age_resolution)
    S0 = construct_initial_susceptible(spatial_resolution, age_resolution, coordinates['location'], I0)
    init_states = {'S': tf.convert_to_tensor(S0, dtype=float),
                'I': tf.convert_to_tensor(I0, dtype=float)
                    }

    # initialise time-dependencies
    TDPFs={}
    if distinguish_daytype:
        from influenza_USA.shared.TDPF import make_contact_function
        TDPFs['N'] = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='week_holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='weekend', age_resolution=age_resolution)).contact_function
        
    return SIR(initial_states=init_states, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)


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