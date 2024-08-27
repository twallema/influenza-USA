"""
This script contains usefull functions for the pySODM US Influenza model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

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
    df = pd.read_csv(os.path.join(abs_dir,'../data/interim/fips_codes/fips_state_county.csv'), dtype={'fips_state': str, 'fips_county': str})

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
    df = pd.read_csv(os.path.join(abs_dir,'../data/interim/fips_codes/fips_state_county.csv'), dtype={'fips_state': str, 'fips_county': str})

    # look up name
    if fips_county == '000':
        return df[df['fips_state'] == fips_state]['name_state'].unique()[0]
    else:
        lu = df[((df['fips_state'] == fips_state) & (df['fips_county'] == fips_county))][['name_state', 'name_county']]
        return lu['name_state'].values[0], lu['name_county'].values[0]
    
def construct_coordinates_dictionary(spatial_resolution='states'):
    """
    A function returning the model's coordinates for the dimension 'age_group' and 'location'. Coordinates derived from the interim demography dataset.

    input
    -----

    spatial_resolution: str
        US 'states' or 'counties'
    
    output
    ------

    coordinates: dict
        Keys: 'age_group', 'location'. Values: Str representing age groups, Str representing US FIPS code of spatial unit
    """

    if spatial_resolution == 'states':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_states_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    elif spatial_resolution == 'counties':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_counties_2023.csv'), dtype={'fips': str, 'age': str, 'population': int})
    else:
        raise ValueError("input argument 'spatial_resolution' must be 'states' or 'counties'")

    return {'age_group': list(demography['age'].unique()), 'location': list(demography['fips'].unique())}

def get_contact_matrix():
    """
    A function to retrieve the total contact matrix for Belgium. Used as a proxy for American contacts.

    # TODO: Average over multiple POLYMOD countries
    # TODO: Retrieve and return weekend, weekday and holiday matrices to build a more realistic contact function
    """
    rel_dir = '../../../data/raw/contacts/locations-all_daytype-all_BE-2008.xlsx'
    contacts = pd.read_excel(os.path.join(abs_dir,rel_dir), sheet_name='time_integrated', index_col=0, header=0)
    return contacts.values

def get_mobility_matrix(dataset='cellphone_03092020', spatial_resolution='states'):
    """
    A function to extract and format the mobility matrix

    input
    -----

    dataset: str
        'cellphone_03092020', 'commuters_2011-2015', or 'commuters_2016-2020'
    
    spatial_resolution: str
        US 'states' or 'counties'

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
    if spatial_resolution == 'states':
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

    elif spatial_resolution == 'counties':
        demography = pd.read_csv(os.path.join(abs_dir,'../../../data/interim/demography/demography_counties_2023.csv'), dtype={'fips': str, 'age': str, 'population': int}).groupby(by='fips').sum()['population'].values
        if dataset == 'cellphone_03092020':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_county_data/departure_diffusion_power_gravitation/matrix_cellphone_03092020_departure-diffusion-powerlaw-gravitation_counties.csv'), index_col=0).values
        elif dataset == 'commuters_2011-2015':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_county_data/departure_diffusion_power_gravitation/matrix_commuters_2011-2015_departure-diffusion-powerlaw-gravitation_counties.csv'), index_col=0).values
        elif dataset == 'commuters_2016-2020':
            mobility_matrix = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/mobility/fitted_models/to_county_data/departure_diffusion_power_gravitation/matrix_commuters_2016-2020_departure-diffusion-powerlaw-gravitation_counties.csv'), index_col=0).values
        else:
            raise ValueError("valid datasets are 'cellphone_03092020', 'commuters_2011-2015' or 'commuters_2016-2020'")
    else:
        raise ValueError("valid 'spatial_resolution' is 'states' or 'counties'")

    # normalise mobility matrix
    mobility_matrix /= demography[:, None]

    # assume on-diagonal = 1 - traveling --> number of on-diagonal trips in cellphone data always exceed the number of inhabitants (people make multiple within-patch trips; but we're not interested in that at all) 
    np.fill_diagonal(mobility_matrix, 0)
    rowsum = mobility_matrix.sum(axis=1)
    np.fill_diagonal(mobility_matrix, 1-rowsum)

    # diagonal elements smaller than one?
    assert np.all(rowsum <= 1), f"there are {spatial_resolution} where the number of people traveling is higher than the population"

    return mobility_matrix