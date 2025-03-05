"""
This script contains usefull functions for the North Carolina influenza forecasting models
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from influenza_USA.shared.utils import construct_coordinates_dictionary, name2fips, get_contact_matrix, compute_case_hospitalisation_rate

# all paths relative to the location of this file
abs_dir = os.path.dirname(__file__)

def initialise_model(strains=True, state=None, season='2023-2024'):
    """
    Initialises the two-strain sequential infection model. Optionally simulate a single state.

    input
    -----

    - strains: bool
        - True: loads two-strain sequential infection model, False: loads single strain model

    - state: str
        - valid US state name.

    - season: str
        - influenza season. needed for the immunity link function.
        
    output
    ------

    - model: pySODM model
        - initialised pySODM two-strain sequential infection SIR model

    remarks
    -------

    The `spatial_resolution` is set to 'states' while the `age_resolution` is set to 'collapsed'. 
    """
    
    # preset `spatial_resolution` and `age_resolution`
    spatial_resolution = 'states'
    age_resolution = 'collapsed'
    state_fips = [name2fips(state),]

    # load model
    from influenza_USA.NC_forecasts.model import SIR_strains as model
    # load initial condition function
    from influenza_USA.NC_forecasts.TDPF import make_initial_condition_function
    # load and init time dependencies
    TDPFs={}
    from influenza_USA.NC_forecasts.TDPF import transmission_rate_function
    TDPFs['delta_beta_t'] = transmission_rate_function(sigma=2.5)   
    
    # load right model and parameters depending on strain
    if not strains:
        # initialise IC function
        historic_cumulative_incidence = get_NC_cumulatives_per_season()['H_inc'].to_frame()
        initial_condition_function = make_initial_condition_function(spatial_resolution, age_resolution, state_fips, historic_cumulative_incidence).initial_condition_function      
        # parameters
        params = {
            ## core parameters
            'beta': np.array([0.030*17.4]),                                                                             # infectivity (-)
            'delta_beta_t': 1,                                                                              # pre-allocation of modifier of transmission rate
            'T_r': 3.5,                                                                                     # average time to recovery 
            'T_h': 5,                                                                                       # delay hospitalisations
            'rho_i': 0.02,                                                                                  # detected fraction infected
            'rho_h': np.array([0.002,]),                                                                                 # hospitalised fraction
            ## initial condition function
            'f_I': 1e-4,                                                                                    # initial fraction of infected
            'season': season,                                                                               # current season 
            'f_R_min1': 5e-5,                                                                               # importance of season - 1 on immunity
            'f_R_min2': 5e-5,                                                                               # importance of season - 2 on immunity
            'f_R_min3': 5e-5,                                                                               # importance of season - 3 on immunity
            'season': season
            }
        # coordinates
        coordinates = {'strain': ['A+B',]}
    if strains:
        # initialise IC function
        historic_cumulative_incidence = get_NC_cumulatives_per_season()[['H_inc_A', 'H_inc_B']]
        initial_condition_function = make_initial_condition_function(spatial_resolution, age_resolution, state_fips, historic_cumulative_incidence).initial_condition_function
        # parameters
        params = {
            ## core parameters
            'beta': np.array([0.030*17.4, 0.030*17.4]),                                                     # infectivity (-)
            'delta_beta_t': 1,                                                                              # pre-allocation of modifier of transmission rate
            'T_r': 3.5,                                                                                     # average time to recovery 
            'T_h': 5,                                                                                       # delay hospitalisations
            'rho_i': 0.02,                                                                                  # detected fraction infected
            'rho_h': np.array([0.002, 0.002]),                                                              # hospitalised fraction
            ## initial condition function
            'f_I': np.array([1e-4, 1e-4]),                                                                  # initial fraction of infected
            'season': season,                                                                               # current season
            'f_R_min1': np.array([5e-5, 5e-5]),                                                             # importance of season - 1 on immunity
            'f_R_min2': np.array([5e-5, 5e-5]),                                                             # importance of season - 2 on immunity
            'f_R_min3': np.array([5e-5, 5e-5])                                                              # importance of season - 3 on immunity
            }
        # coordinates
        coordinates = {'strain': ['A', 'B']}
    # add parameter of TDPF
    params['delta_beta_temporal'] = np.zeros(12)              

    # initalise pySODM model
    return model(initial_states=initial_condition_function, parameters=params, coordinates=coordinates, time_dependent_parameters=TDPFs)

def get_NC_influenza_data(startdate, enddate, season):
    """
    Get the North Carolina Influenza dataset -- containing ED visits, ED admissions and subtype information -- for a given season

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


def get_NC_NHSN_data(startdate, enddate):
    """
    Get the North Carolina Influenza hospital admissions from the NHSN HRD dataset

    input
    -----

    - startdate: str/datetime
        - start of dataset
    
    - enddate: str/datetime
        - end of dataset

    output
    ------

    - data: pd.DataFrame
        - index: 'date' [datetime], columns: 'H_inc', 'I_inc', 'H_inc_A', 'H_inc_B' (frequency: weekly, converted to daily)
    """

    # get data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../../data/interim/cases/NHSN-HRD_interim.csv'), index_col=0, parse_dates=True, dtype={'fips_state': str})

    # slice NC out
    data = data[data['fips_state'] == '37000']['H_inc']

    return data.loc[slice(startdate,enddate)]

def get_NC_cumulatives_per_season():
    """
    A function that returns, for each season, the cumulative total H_inc, I_inc, H_inc_A and H_inc_B in the season - 0, season - 1 and season - 2.
    """
    # define seasons we want output for
    seasons = ['2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020', '2023-2024', '2024-2025']

    # loop over them
    seasons_collect = []
    for season in seasons:
        # get the season start
        season_start = int(season[0:4])
        # go back two seasons
        horizons_collect = []
        for i in [0, -1, -2, -3]:
            # get the data
            data = get_NC_influenza_data(datetime(season_start+i,10,1), datetime(season_start+1+i,5,1), f'{season_start+i}-{season_start+1+i}')*7
            # calculate cumulative totals
            column_sums = {
                "horizon": i,
                "H_inc": data["H_inc"].sum(),
                "I_inc": data["I_inc"].sum(),
                "H_inc_A": data["H_inc_A"].sum(),
                 "H_inc_B": data["H_inc_B"].sum(),
            }
            # create the DataFrame
            horizons_collect.append(pd.DataFrame([column_sums]))
        # concatenate data
        data = pd.concat(horizons_collect)
        # add current season
        data['season'] = season    
        # add to archive
        seasons_collect.append(data)
    # concatenate across seasons
    data = pd.concat(seasons_collect).set_index(['season', 'horizon'])

    return data

def pySODM_to_hubverse(simout: xr.Dataset,
                        reference_date: datetime,
                        target: str,
                        model_state: str,
                        path: str=None,
                        quantiles: bool=False,
                        location: str='37') -> pd.DataFrame:
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

    - quantiles: str
        - save quantiles instead of individual trajectories.
    
    - location: str
        - US state FIPS code. Defaults to North Carolina '37'.

    Returns
    -------

    - hubverse_df: pd.Dataframe
        - forecast in hubverse format

    Reference
    ---------

    https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-output/README.md#Forecast-file-format
    """

    # deduce information from simout
    location = [location,]
    output_type_id = simout.coords['draws'].values if not quantiles else [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    # fixed metadata
    horizon = range(-1,4)
    output_type = 'samples' if not quantiles else 'quantile'
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
        if not quantiles:
            for draw in output_type_id:
                df.loc[((df['output_type_id'] == draw) & (df['location'] == loc)), 'value'] = \
                    7*simout[model_state].sel({'draws': draw}).interp(date=target_end_date).values
        else:
            for q in output_type_id:
                df.loc[((df['output_type_id'] == q) & (df['location'] == loc)), 'value'] = \
                    7*simout[model_state].quantile(q=q, dim='draws').interp(date=target_end_date).values
    
    # hubverse uses
    df['location'] = df['location'].apply(lambda x: x[:2])

    # save result
    if path:
        df.to_csv(path+reference_date.strftime('%Y-%m-%d')+'-JHU_IDD'+'-hierarchSIM.csv', index=False)

    return df