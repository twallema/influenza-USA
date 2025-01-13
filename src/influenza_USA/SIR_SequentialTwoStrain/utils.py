"""
This script contains usefull functions for the age-stratified spatially-explicit two-strain sequential infection SIR model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
from datetime import datetime as datetime
from influenza_USA.shared.utils import construct_coordinates_dictionary, name2fips, get_contact_matrix, compute_case_hospitalisation_rate

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

def initialise_SIR_SequentialTwoStrain(spatial_resolution='states', age_resolution='full', state=None, season='average', distinguish_daytype=True):
    """
    Initialises a two-strain sequential infection model. Optionally simulate a single state.

    input
    -----

    spatial_resolution: str
        'collapsed', 'states' or 'counties'. 

    age_resolution: str
        'collapsed' or 'full'. 

    state: str
        valid US state name.

    season: str
        influenza season. used to set the model's U-shaped severity curve.
        
    distinguish_daytype: bool
        Differ contacts by weekday, weekendday and holiday.

    output
    ------

    model: pySODM model
        Initialised pySODM two-strain sequential infection SIR model
    """
    
    # model works at US state or county level
    if ((spatial_resolution != 'states') & (spatial_resolution != 'counties')):
        raise ValueError("this model was designed to work at the US state or county level. valid 'spatial_resolution' are 'states' or 'counties'. found: '{spatial_resolution}'.")

    # load model object
    from influenza_USA.SIR_SequentialTwoStrain.model import ODE_SIR_SequentialTwoStrain as SIR_SequentialTwoStrain

    # construct coordinates
    _, G, coordinates = construct_coordinates_dictionary(spatial_resolution=spatial_resolution, age_resolution=age_resolution)

    # slice state out of the coordinates
    if state:
        # get fips code (performs checks)
        fips = name2fips(state)
        # extract coordinates
        coordinates['location'] = [coord for coord in coordinates['location'] if coord[0:2] == fips[0:2]]
        # update spatial dimension size
        G = len(coordinates['location'])

    # define parameters
    params = {
            # core parameters
            'beta1': 0.028*np.ones(G),                                                                                              # infectivity strain 1 (-)
            'beta2': 0.028*np.ones(G),                                                                                              # infectivity strain 2 (-)
            'N': get_contact_matrix(daytype='all', age_resolution=age_resolution),                                                  # contact matrix (overall: 17.4 contact * hr / person, week (no holiday): 18.1, week (holiday): 14.5, weekend: 16.08)
            'T_r': 3.5,                                                                                                             # average time to recovery 
            'CHR': compute_case_hospitalisation_rate(season, age_resolution=age_resolution),                                        # case hosp. rate corrected for social contact and expressed relative to [0,5) yo
            # outcomes
            'rho_h1': 0.002,                                                                                                        # hospitalised fraction (source: Josh)
            'rho_h2': 0.002,                                                                                                        # hospitalised fraction (source: Josh)
            # initial condition function
            'f_I1': 1e-4,                                                                                                           # initial fraction of infected with strain 1
            'f_I2': 1e-5,                                                                                                           # initial fraction of infected with strain 2
            'f_R1_R2': 0.75,                                                                                                        # sum of the initial fraction recovered from strain 1 and strain 2 --> needed to constraint initial R between 0 and 1 during calibration
            'f_R1': 0.45,                                                                                                           # fraction of f_R1_R2 recovered from strain 1
            }
    
    # initial condition function
    from influenza_USA.SIR_SequentialTwoStrain.TDPF import make_initial_condition_function
    initial_condition_function = make_initial_condition_function(spatial_resolution, age_resolution, coordinates['location']).initial_condition_function
                                                                                 
    # time-dependencies
    TDPFs = {}
    ## contacts
    if distinguish_daytype:
        from influenza_USA.shared.TDPF import make_contact_function
        TDPFs['N'] = make_contact_function(get_contact_matrix(daytype='week_no-holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='week_holiday', age_resolution=age_resolution),
                                                get_contact_matrix(daytype='weekend', age_resolution=age_resolution)).contact_function
    ## transmission rate
    from influenza_USA.SIR_SequentialTwoStrain.TDPF import transmission_rate_function
    TDPFs['beta1'] = transmission_rate_function(sigma=2.5)      # initialise TDPF
    TDPFs['beta2'] = transmission_rate_function(sigma=2.5)      # initialise TDPF
    params['delta_beta_temporal'] = np.zeros(12)                # initialise parameter of TDPF
    
    # initalise pySODM model
    return SIR_SequentialTwoStrain(initial_states=initial_condition_function, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)