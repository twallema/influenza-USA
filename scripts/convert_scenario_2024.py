import xarray as xr
import pandas as pd
from datetime import datetime
from fastparquet import write

# settings
## projection start & end
projection_start = datetime(2024, 8, 11)
projection_end = datetime(2025, 6, 7)

## target 
targets = ['inc hosp',]     # scenariohub 'target' name
states = ['H_inc',]         # corresponding model state

## scenarios
scenario_ids = ['E-2024-08-01', 'C-2024-08-01', 'A-2024-08-01', 'F-2024-08-01', 'D-2024-08-01', 'B-2024-08-01']
filenames = ['0.8_no_waning_2017-2018.nc', '1.0_no_waning_2017-2018.nc', '1.2_no_waning_2017-2018.nc',
             '0.8_no_waning_2019-2020.nc', '1.0_no_waning_2019-2020.nc', '1.2_no_waning_2019-2020.nc']

# LOOP scenarios:
gather_scenarios=[]
for i, (scenario_id, filename) in enumerate(zip(scenario_ids,filenames)):

    print(f'Working on scenario: {scenario_id}')

    # 1) load right xarray dataset
    simout = xr.open_dataset(filename)

    # 2) slice between start and end of projection
    simout = simout.sel(date=slice(projection_start, projection_end))

    # 3) convert daily incidence to weekly incidence (verified thru plotting; index on end of epi-week Sun-Sa so Sa)
    simout = simout.resample(date='W-SAT').sum()

    # 4) pre-allocate pandas dataframe in the right format
    horizon = list(range(len(simout['date'])))                              # week numbers
    age_groups_model = simout.coords['age_group'].values                    # age groups (model)
    age_groups_scenariohub = ['0-4', '5-17', '18-49', '50-64', '65-130']    # age groups (Scenariohub)
    locations = simout.coords['location'].values                            # locations
    stochastic_runs = simout.coords['draws'].values                         # draws
    df = pd.DataFrame(index=pd.MultiIndex.from_product([horizon, targets, age_groups_scenariohub, locations, stochastic_runs], names=['horizon', 'target', 'age_group', 'location', 'stochastic_run']),
                        columns=['origin_date', 'scenario_id', 'output_type', 'output_type_ID', 'run_grouping', 'value'])
    
    # 5) fill in constant values
    df['origin_date'] = projection_start.strftime('%Y-%m-%d')
    df['scenario_id'] = scenario_id
    df['output_type'] = 'sample'
    df['output_type_ID'] = 'NA'
    df['run_grouping'] = 1

    # 6) fill in pre-allocated dataframe
    for target,state in zip(targets,states):
        for (age_group_model, age_group_scenariohub) in zip(age_groups_model, age_groups_scenariohub):
            for location in locations:
                for draw in stochastic_runs:
                    df.loc[(slice(None), target, age_group_scenariohub, location, draw), 'value'] = simout[state].sel({'age_group': age_group_model, 'location': location, 'draws': draw}).values
    gather_scenarios.append(df.reset_index())

# 7) append scenarios together
output = pd.concat(gather_scenarios)

# 8) save as a parquet, gz.parquet and csv
write(f'{projection_start.strftime('%Y-%m-%d')}-JHU_IDD-SVI2RHD.gz.parquet', output, compression='GZIP')
output.to_parquet(f'{projection_start.strftime('%Y-%m-%d')}-JHU_IDD-SVI2RHD.parquet', index=False)
output.to_csv(f'{projection_start.strftime('%Y-%m-%d')}-JHU_IDD-SVI2RHD.csv', index=False)
