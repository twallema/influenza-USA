"""
This script contains the time-dependent parameter functions associated with the age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
from datetime import timedelta
from functools import lru_cache
from influenza_USA.shared.utils import get_smooth_temporal_modifier

##################################
## Hierarchal transmission rate ##
##################################

class hierarchal_waning_natural_immunity():

    def __init__(self, spatial_resolution):
        # retrieve region/state --> state/county parameter mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)
        pass

    def __call__(self, t, states, param, T_r_US, delta_T_r_regions):
        return T_r_US * (1 + delta_T_r_regions[self.region_mapping]) 

class hierarchal_transmission_rate_function():

    def __init__(self, spatial_resolution, sigma):
        # set smoother length
        self.sigma = sigma
        # retrieve region/state --> state/county parameter mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)
        pass

    def __call__(self, t, states, param, beta_US, delta_beta_regions, delta_beta_states, delta_beta_temporal, delta_beta_spatiotemporal):
        """
        A function constructing a spatio-temporal hierarchal transmission rate 'beta'

        input
        -----

        t: datetime.datetime
            current timestep in simulation

        states: dict
            current values of all model states
        
        param: dict
            current values of all model parameters

        beta_US: float
            overall transmission rate. hierarchal level 0.

        delta_beta_regions: np.ndarray (len: 9)
            a spatial modifier on the overall transmision rate for every US region. hierarchal level 1.

        delta_beta_states: np.ndarray (len: 52)
            a spatial modifier on the overall transmision rate for every US state. hierarchal level 2.

        delta_beta_temporal: np.ndarray (len: 4)
            a temporal modifier on the overall transmission rate for Dec, Jan, Feb, Mar. hierarchal level 1.

        delta_beta_spatiotemporal: np.ndarray (shape: 10, 9)
            a spatio-temporal modifier for every US region in 1-15 Nov. hierarchal level 2.

        output
        ------

        beta: np.ndarray
            transmission rate per US state
        """

        # state parameter mapping
        delta_beta_states = 1 + delta_beta_states[self.state_mapping]
        # regional parameter mapping
        delta_beta_regions = 1 + delta_beta_regions[self.region_mapping]
        # spatiotemporal betas
        delta_beta_spatiotemporal = 1 + delta_beta_spatiotemporal[:, self.region_mapping] # --> if spatiotemporal components at regional level
        # temporal betas
        delta_beta_temporal = 1 + delta_beta_temporal
        # get smoothed temporal components
        to_smooth = delta_beta_spatiotemporal * delta_beta_temporal[:, np.newaxis]
        temporal_modifiers_smooth = []
        for i in range(delta_beta_spatiotemporal.shape[1]):
            temporal_modifiers_smooth.append(get_smooth_temporal_modifier(to_smooth[:,i], t, self.sigma))
        temporal_modifiers_smooth = np.array(temporal_modifiers_smooth)
        # construct modifiers
        return beta_US * temporal_modifiers_smooth * delta_beta_regions * delta_beta_states
    
##############
## Vaccines ##
##############

class make_vaccination_function():

    def __init__(self, season, spatial_resolution, age_resolution):
        """ Format the vaccination data
        """

        # retrieve the vaccination data
        vaccination_data = get_vaccination_data()
        # convert the vaccination data to the right age and spatial resolution
        vaccination_data = convert_vaccination_data(vaccination_data, spatial_resolution, age_resolution)

        # check input season
        if ((season not in vaccination_data['season'].unique()) & (season != 'average')):
            raise ValueError(f"season '{season}' vaccination data not found. provide a valid season (format '20xx-20xx') or 'average'.")

        if season != 'average':
            # slice out correct season
            vaccination_data = vaccination_data[vaccination_data['season'] == season]
            # add week number & remove date
            vaccination_data['week'] = vaccination_data['date'].dt.isocalendar().week.values
            vaccination_data = vaccination_data[['week', 'age', 'fips', 'daily_incidence']]
            # sort age groups / spatial units --> are sorted in the model
            vaccination_data = vaccination_data.groupby(by=['week', 'age', 'fips']).last().sort_index().reset_index()
            # remove negative entries (there may be some in the first week(s) of the season)
            vaccination_data['daily_incidence'] = np.where(vaccination_data['daily_incidence'] < 0, 0, vaccination_data['daily_incidence'])
        else:
            # add week number & remove date
            vaccination_data['week'] = vaccination_data['date'].dt.isocalendar().week.values
            vaccination_data = vaccination_data[['week', 'age', 'fips', 'daily_incidence']]
            # average out + sort
            vaccination_data = vaccination_data.groupby(by=['week', 'age', 'fips']).mean('daily_incidence').sort_index().reset_index()
            # remove negative entries (there may be some in the first week(s) of the season)
            vaccination_data['daily_incidence'] = np.where(vaccination_data['daily_incidence'] < 0, 0, vaccination_data['daily_incidence'])

        # assign to object
        self.vaccination_data = vaccination_data

        # compute state sizes
        self.n_age = len(self.vaccination_data['age'].unique())
        self.n_loc = len(self.vaccination_data['fips'].unique())

    @lru_cache() # avoid heavy IO while simulating
    def get_vaccination_incidence(self, t):
        """ Returns the daily vaccination incidence as an np.ndarray of shape (n_age, n_loc)
        """
        week_number = t.isocalendar().week
        try:
            return np.array(self.vaccination_data[self.vaccination_data['week'] == week_number]['daily_incidence'].values, np.float64).reshape(self.n_age, self.n_loc) 
        except:
            return np.zeros([self.n_age, self.n_loc], np.float64)
    
    def vaccination_function(self, t, states, param, vaccine_incidence_modifier, vaccine_incidence_timedelta):
        """ pySODM compatible wrapper
        """
        return vaccine_incidence_modifier * self.get_vaccination_incidence(t - timedelta(days=vaccine_incidence_timedelta))

################################
## Initial condition function ##
################################

from influenza_USA.shared.utils import construct_initial_susceptible
from influenza_USA.SVI2RHD.utils import get_cumulative_vaccinated, get_vaccination_data, convert_vaccination_data, get_spatial_mappings
class make_initial_condition_function():

    def __init__(self, spatial_resolution, age_resolution, spatial_coordinates, start_sim, season):
        # retrieve the vaccination data
        vaccination_data = get_vaccination_data()
        # convert the vaccination data to the right age and spatial resolution
        vaccination_data = convert_vaccination_data(vaccination_data, spatial_resolution, age_resolution)
        # retrieve the demography (susceptible pool)
        self.demography = construct_initial_susceptible(spatial_resolution, age_resolution, spatial_coordinates)
        # retrieve the cumulative vaccinated individuals at `start_sim` in `season`
        self.vaccinated = get_cumulative_vaccinated(start_sim, season, vaccination_data)
        # retrieve region/state --> state/county parameter  mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)

    def initial_condition_function(self, f_I, f_R, delta_f_R_states, delta_f_R_regions):
        """
        A function setting the model's initial condition. Uses a hierarchal structure for the initial immunity.
        
        input
        -----

        f_I: float / np.ndarray
            fraction of the unvaccinated population infected at simulation start

        f_R: float
            initial immunity of US (parent distribution)
        
        delta_f_R_regions: np.ndarray (len: 9)
            initial immunity modifier of US regions (child distributions; level 1)
        
        delta_f_R_states: np.ndarray (len: 52)
            initial immunity modifier of US regions (child distributions; level 2)

        output
        ------

        initial_condition: dict
            Keys: 'S', 'V', 'I', 'R'. Values: np.ndarray (n_age x n_loc).
        """

        # convert all delta_f_R from region/state to state/county level
        delta_f_R_regions = delta_f_R_regions[self.region_mapping]
        delta_f_R_states = delta_f_R_states[self.state_mapping]

        # construct hierarchal initial immunity
        f_R = f_R * (1 + delta_f_R_regions) * (1 + delta_f_R_states) 

        return {'S':  self.demography - (1 - f_I - f_R) * self.vaccinated - (f_I + f_R) * self.demography,
                'V': (1 - f_I - f_R) * self.vaccinated,
                'I': f_I * self.demography,
                'R': f_R * self.demography}