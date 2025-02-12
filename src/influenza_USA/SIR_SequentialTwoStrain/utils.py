"""
This script contains usefull functions for the age-stratified spatially-explicit two-strain sequential infection SIR model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from influenza_USA.shared.utils import construct_coordinates_dictionary, name2fips, get_contact_matrix, compute_case_hospitalisation_rate

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

def initialise_SIR_SequentialTwoStrain(spatial_resolution='states', age_resolution='full', state=None, season='average', distinguish_daytype=True):
    """
    Initialises the two-strain sequential infection model. Optionally simulate a single state.

    input
    -----

    - spatial_resolution: str
        - 'collapsed', 'states' or 'counties'. 

    - age_resolution: str
        - 'collapsed' or 'full'. 

    - state: str
        - valid US state name.

    - season: str
        - influenza season. used to set the model's U-shaped severity curve.
        
    - distinguish_daytype: bool
        - differ contacts by weekday, weekendday and holiday.

    output
    ------

    - model: pySODM model
        - initialised pySODM two-strain sequential infection SIR model
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
            'T_h': 5,                                                                                                               # delay hospitalisations
            'rho_i': 0.02,                                                                                                        # detected fraction infected
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

def get_NC_influenza_data(startdate, enddate, season):
    """
    Get the North Carolina Influenza dataset -- containing ED visits, hospitalisation and subtype information -- for a given season

    input
    -----

    - startdate: str/datetime
        - start of dataset
    
    - enddate: str/datetime
        - end of dataset

    - season: str
        - influenza season

    output
    ------

    - data: pd.DataFrame
        - index: 'date' [datetime], columns: 'H_inc', 'I_inc', 'H_inc_A', 'H_inc_B' (frequency: weekly, converted to daily)
    """

    # load raw Hospitalisation and ILI data + convert to daily incidence
    data_raw = [
        pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../../data/raw/cases/hosp-admissions_NC_10-25.csv'), index_col=0, parse_dates=True)[['flu_hosp']].squeeze()/7,  # hosp
        pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../../data/raw/cases/ED-visits_NC_10-25.csv'), index_col=0, parse_dates=True)[['flu_ED']].squeeze()/7               # ILI
            ]   
    # rename 
    data_raw[0] = data_raw[0].rename('H_inc')
    data_raw[1] = data_raw[1].rename('I_inc')
    # merge
    data_raw = pd.concat(data_raw, axis=1)
    # change index name
    data_raw.index.name = 'date'
    # slice right dates
    data_raw = data_raw.loc[slice(startdate,enddate)]
    # load subtype data flu A vs. flu B
    df_subtype = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../../data/interim/cases/subtypes_NC_14-25.csv'), index_col=1, parse_dates=True)
    # load right season
    df_subtype = df_subtype[df_subtype['season']==season][['flu_A', 'flu_B']]
    # merge with the epi data
    df_merged = pd.merge(data_raw, df_subtype, how='outer', left_on='date', right_on='date')
    # assume a 50/50 ratio where no subtype data is available
    df_merged[['flu_A', 'flu_B']] = df_merged[['flu_A', 'flu_B']].fillna(1)
    # compute fraction of Flu A
    df_merged['fraction_A'] = df_merged['flu_A'] / (df_merged['flu_A'] + df_merged['flu_B']) # compute percent A
    # re-compute flu A and flu B cases
    df_merged['H_inc_A'] = df_merged['H_inc'] * df_merged['fraction_A']
    df_merged['H_inc_B'] = df_merged['H_inc'] * (1-df_merged['fraction_A'])
    # throw out rows with na
    df_merged = df_merged.dropna()
    # throw out `fraction_A`
    return df_merged[['H_inc', 'I_inc', 'H_inc_A', 'H_inc_B']].loc[slice(startdate,enddate)]


def pySODM_to_hubverse(simout: xr.Dataset,
                        reference_date: datetime,
                        target: str,
                        model_state: str,
                        path: str=None) -> pd.DataFrame:
    """
    Convert pySODM simulation result to Hubverse format

    Parameters
    ----------
    - simout: xr.Dataset
        - pySODM simulation output. must contain `model_state`.

    - reference_date: datetime
        - when using data until a Saturday `x` to calibrate the model, `reference_date` is the date of the next saturday `x+1`.

    - target: str
        - simulation target, typically 'wk inc flu hosp'.

    - path: str
        - path to save result in. if no path provided, does not save result.

    Returns
    -------

    - hubverse_df: pd.Dataframe
        - forecast in hubverse format

    Reference
    ---------

    https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md#Forecast-file-format
    """

    # deduce information from simout
    location = list(simout.coords['location'].values)
    output_type_id = simout.coords['draws'].values
    # fixed metadata
    horizon = range(-1,3)
    output_type = 'samples'
    # derived metadata
    target_end_date = [reference_date + timedelta(weeks=h) for h in horizon]

    # pre-allocate dataframe
    idx = pd.MultiIndex.from_product([[reference_date,], [target,], horizon, location, [output_type,], output_type_id],
                                        names=['reference_date', 'target', 'horizon', 'location', 'output_type', 'output_type_id'])
    df = pd.DataFrame(index=idx, columns=['value'])
    # attach target end date
    df = df.reset_index()
    df['target_end_date'] = df.apply(lambda row: row['reference_date'] + timedelta(weeks=row['horizon']), axis=1)

    # fill in dataframe
    for loc in location:
        for draw in output_type_id:
            df.loc[((df['output_type_id'] == draw) & (df['location'] == loc)), 'value'] = 7*simout[model_state].sum(dim='age_group').sel({'location': loc, 'draws': draw}).interp(date=target_end_date).values
    
    # save result
    if path:
        df.to_csv(path+reference_date.strftime('%Y-%m-%d')+'-JHU_IDD'+'-hierarchSIM.csv', index=False)

    return df