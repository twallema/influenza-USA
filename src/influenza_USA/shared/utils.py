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
    CDC_estimated_hosp = pd.read_csv(os.path.join(abs_dir, '../../../data/interim/cases/hosp-rate-age_CDC_USA_10-23.csv'),
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


def construct_initial_susceptible(spatial_resolution, age_resolution, spatial_coordinates, *subtract_states):
    """
    A function to construct the initial number of susceptible individuals, computed as the number of susceptibles 'S' derived from the demographic data, minus any individiduals present in `subtract_states`

    input
    -----

    *subtract_states: np.ndarray
        Other states that contain individuals at the start of the simulation.
        Substracted from the demographic data to compute the number of initial susceptible individuals
        If not provided: function returns demography

    spatial_resolution: str
        US 'states' or 'counties' or 'collapsed'

    age_resolution: str
        'collapsed', 'full' (0-5, 5-18, 18-50, 50-65, 65+)

    spatial_coordiantes: list
        A list containing the spatial coordinates of the model (fips codes)

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

    # slice spatial coordinates from the demography
    demography = demography[demography['fips'].isin(spatial_coordinates)]

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


from datetime import datetime
from scipy.ndimage import gaussian_filter1d
def get_smooth_temporal_modifier(modifier_vector, simulation_date, sigma):
    """
    A function returning the value of a temporal modifier on `simulation_date` after smoothing with a gaussian filter

    input
    -----

    modifier_vector: np.ndarray
        1D numpy array (time) or 2D numpy array (time x space).
        Each entry represents a value of the modifier in a time interval, with time intervals divided between Nov 1 and Apr 1.

    simulation_date: datetime
        current simulation date

    sigma: float
        smoother standard deviation. higher values represent more smooth trajectories.

    output
    ------

    smooth_temporal_modifier: float
        smoothed modifier at `simulation_date`
        1D array of smoothed modifiers at `simulation_date`. If the input is 1D, the output will be a single-element array. If the input is 2D, the output will have one value for each spatial dimension.
    """

    # Ensure the input is at least 2D
    if modifier_vector.ndim == 1:
        modifier_vector = modifier_vector[:, np.newaxis]
    _, num_space = modifier_vector.shape

    # Define number of days between Nov 1 and Apr 1
    num_days = 152

    # Step 1: Project the input vector onto the daily time scale
    interval_size = num_days / len(modifier_vector)
    positions = (np.arange(num_days) // interval_size).astype(int)
    expanded_vector = modifier_vector[positions, :]

    # Step 2: Prepend and append 31 days of ones
    padding = np.ones((31, num_space))
    padded_vector = np.vstack([padding, expanded_vector, padding])

    # Step 3: Apply a gaussian 1D smoother
    smoothed_series = gaussian_filter1d(padded_vector, sigma=sigma, axis=0)

    # Step 4: Compute the number of days since the last October 1
    year = simulation_date.year
    # Compute the last October 1
    oct1_this_year = datetime(year, 10, 1)
    if simulation_date >= oct1_this_year:
        last_oct1 = oct1_this_year
    else:
        last_oct1 = datetime(year - 1, 10, 1)
    # Calculate the difference in days
    days_difference = (simulation_date - last_oct1).days

    # Step 5: Return the smoothed value(s) for the specified day
    if 0 <= days_difference < smoothed_series.shape[0]:
        return smoothed_series[days_difference, :]  # Always returns a 1D array
    return np.ones(num_space)  # Default value if index is out of bounds