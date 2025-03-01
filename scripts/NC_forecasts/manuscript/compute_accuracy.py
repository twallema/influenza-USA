"""
A script to compute the Weighted Interval Score (WIS) accuracy metric of a model

To use this script, place it in a folder containing the following structure:

MY_FOLDER
|--- compute_accuracy.py
|--- 2023-2024
    |--- end-2023-12-16
    |--- end-2023-12-23
    |--- ...
|--- 2019-2020
    |--- ...
"""

# packages needed
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from influenza_USA.NC_forecasts.utils import get_NC_influenza_data

# settings
norm = True
prediction_horizon_weeks = 4
quantiles_WIS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
start_month = 12
start_day = 1
end_month = 4
end_day = 7

# finding the right simulations
location = 37                           # NC FIPS code
model_name = 'JHU_IDD-hierarchSIM'      # 
seasons = [entry for entry in os.listdir(os.path.dirname(__file__)) if os.path.isdir(os.path.join(os.path.dirname(__file__), entry))]
seasons.sort()

# extract and store simulations as an xarray dataset
gather_seasons = []
for season in seasons:

    # derive start of season year
    season_start = int(season[:4])

    # compute maximum of season for normalisation
    max_data = max(get_NC_influenza_data(datetime(season_start,9,1), datetime(season_start+1,9,1), season)['H_inc']*7)

    # get all enddates of forecasts in a given season from the folder names
    subdirectories = [entry for entry in os.listdir(os.path.join(os.path.dirname(__file__), season)) if os.path.isdir(os.path.join(os.path.join(os.path.dirname(__file__), season), entry))]
    subdirectories.sort()
    reference_dates = [datetime.strptime(subdir[4:], '%Y-%m-%d')+timedelta(days=7) for subdir in subdirectories]
    data_ends = [datetime.strptime(subdir[4:], '%Y-%m-%d') for subdir in subdirectories] 

    # only evaluate between user-supplied start and enddate
    filtered_subdirectories = []
    filtered_reference_dates = []
    for subdir, data_end, reference_date in zip(subdirectories, data_ends, reference_dates):
        if datetime(season_start, start_month, start_day) <= data_end <= datetime(season_start+1, end_month, end_day):
            filtered_subdirectories.append(subdir)
            filtered_reference_dates.append(reference_date)
    subdirectories = filtered_subdirectories
    reference_dates = filtered_reference_dates

    # loop over directories to collect groundtruth and forecasts
    datas = []
    simouts = []
    for subdir, reference_date in zip(subdirectories, reference_dates):
        ## get forecast
        tmp = pd.read_csv(os.path.join(season, subdir, f'{reference_date.date()}-{model_name}.csv'), parse_dates=True, date_format='%Y-%m-%d')
        ## slice location, quantiles and right target
        tmp = tmp[((tmp['location'] == location) & (tmp['output_type'] == 'quantile') & (tmp['target'] == 'wk inc flu hosp'))]
        ## only need horizon numbers 0, 1, 2 and 3
        tmp = tmp[tmp['horizon'] != -1]
        ## make sure ref date is datetime
        tmp['target_end_date'] = pd.to_datetime(tmp['target_end_date'])
        ## save only horizon, output_type_id and value
        tmp = tmp[['target_end_date', 'output_type_id', 'value']]
        ## normalize if needed
        if norm:
            tmp['value'] /= max_data
        ## append to list
        simouts.append(tmp)
        ## get groundthruth data (+ the forecast horizon of four weeks)
        data = get_NC_influenza_data(reference_date, reference_date+timedelta(weeks=3), season)['H_inc']*7
        if norm:
            data /= max_data
        datas.append(data)

    # make a dataframe for the output of the season
    idx = pd.MultiIndex.from_product([[season,], reference_dates, range(0,prediction_horizon_weeks)], names=['season', 'reference_date', 'horizon'])
    season_accuracy = pd.Series(index=idx, name='WIS')

    # loop over weeks
    for reference_date, simout, data in zip(reference_dates, simouts, datas):
        # WIS
        for n in range(0,prediction_horizon_weeks):
            ## get date
            date = reference_date+timedelta(weeks=n)
            ## get data
            y = data.loc[date]
            ## compute IS
            IS_alpha = []
            for q in quantiles_WIS:
                # get quantiles
                try:
                    l = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == q/2))]['value'].values[0]
                    u = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == 1-q/2))]['value'].values[0]
                except:
                    l = np.nan
                    u = np.nan
                # compute IS
                IS = (u - l)
                if y < l:
                    IS += 2/q * (l-y)
                elif y > u:
                    IS += 2/q * (y-u)
                IS_alpha.append(IS)
            IS_alpha = np.array(IS_alpha)
            ## compute WIS & assign
            try:
                m = simout[((simout['target_end_date'] == reference_date+timedelta(weeks=n)) & (simout['output_type_id'] == 0.50))]['value'].values[0]
            except:
                m = np.nan
            season_accuracy.loc[season, reference_date, n] = (1 / (len(quantiles_WIS) + 0.5)) * (0.5 * np.abs(y-m) + np.sum(0.5*np.array(quantiles_WIS) * IS_alpha))
    # collect season results
    gather_seasons.append(season_accuracy)

# concatenate season results
output = pd.concat(gather_seasons, axis=0)

print(output.mean())
print(output.groupby(by=['season']).mean())
print(output.groupby(by=['season', 'horizon']).mean())

# output to csv
output.to_csv('accuracy.csv')