"""
This script contains the time-dependent parameter functions associated with the age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import lru_cache
from dateutil.easter import easter
from datetime import datetime, timedelta

##################################
## Hierarchal transmission rate ##
##################################

class hierarchal_waning_natural_immunity():

    def __init__(self, spatial_resolution):
        # retrieve region/state --> state/county parameter  mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)
        pass

    def __call__(self, t, states, param, T_r_US, delta_T_r_regions):
        return T_r_US * (1 + delta_T_r_regions[self.region_mapping]) 

class hierarchal_transmission_rate_function():

    def __init__(self, spatial_resolution):
        # retrieve region/state --> state/county parameter  mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)
        pass
    
    def get_smoothed_modifier(self, modifiers, simulation_date, half_life_days=5, window_size=30, freq='biweekly'):
        """
        Calculate the smoothed temporal modifier for all US states based on the current simulation date.
        Supports 'biweekly' or 'monthly' frequencies for the modifiers.

        Parameters:
        - modifiers: numpy array of shape (10, 52) or (5, 52), depending on freq ('biweekly' or 'monthly').
        - simulation_date: datetime object representing the current simulation date.
        - half_life_days: Half-life in days for the exponential smoothing (default: 5 days).
        - window_size: The number of days used in the smoothing window (default: 30 days).
        - freq: 'biweekly' (default) or 'monthly'. Determines the structure of the modifiers.

        Returns:
        - smoothed_modifier: numpy array of shape (1, 52), representing the smoothed temporal modifier.
        """
        # Exponential smoothing factor (alpha)
        alpha = 1 - np.exp(np.log(0.5) / half_life_days)

        # Define mappings based on frequency
        if freq == 'biweekly':
            period_mapping = {
                (11, 1): 0, (11, 2): 1,
                (12, 1): 2, (12, 2): 3,
                (1, 1): 4, (1, 2): 5,
                (2, 1): 6, (2, 2): 7,
                (3, 1): 8, (3, 2): 9
            }
        elif freq == 'monthly':
            period_mapping = {
                11: 0,  # Nov
                12: 1,  # Dec
                1: 2,   # Jan
                2: 3,   # Feb
                3: 4    # Mar
            }
        else:
            raise ValueError("Invalid frequency. Use 'biweekly' or 'monthly'.")

        # Initialize arrays for the modifier window and smoothing weights
        modifier_window = []
        smoothing_weights = []

        # Traverse backward to fill the window
        days_to_collect = window_size
        current_date = simulation_date

        while days_to_collect > 0:
            month = current_date.month
            day = current_date.day

            # Determine the period row based on frequency
            if freq == 'biweekly':
                biweekly_period = 1 if day <= 15 else 2
                row = period_mapping.get((month, biweekly_period), None)
            elif freq == 'monthly':
                row = period_mapping.get(month, None)

            # Skip if the month is outside Nov-Mar
            if row is not None:
                # Calculate the number of days in the current period
                if freq == 'biweekly' and biweekly_period == 1:
                    days_in_this_period = min(day, days_to_collect)
                elif freq == 'biweekly' and biweekly_period == 2:
                    days_in_this_period = min(day - 15, days_to_collect)
                else:  # Monthly case
                    days_in_this_period = min(day, days_to_collect)

                # Append the modifiers and weights for each day
                for _ in range(days_in_this_period):
                    modifier_window.append(modifiers[row, :])
                    smoothing_weights.append((1 - alpha) ** (window_size - days_to_collect))
                    days_to_collect -= 1
                    current_date -= timedelta(days=1)

            # Move to the previous day
            else:
                current_date -= timedelta(days=1)

        # Convert lists to NumPy arrays
        modifier_window = np.array(modifier_window)
        smoothing_weights = np.array(smoothing_weights)

        # Normalize weights
        smoothing_weights /= np.sum(smoothing_weights)

        # Compute the smoothed modifier
        smoothed_modifier = np.average(modifier_window, axis=0, weights=smoothing_weights)

        return smoothed_modifier


    def __call__(self, t, states, param, beta_US, delta_beta_regions, delta_beta_states, delta_beta_temporal,
                 delta_beta_regions_Nov1, delta_beta_regions_Nov2, delta_beta_regions_Dec1,  delta_beta_regions_Dec2,
                 delta_beta_regions_Jan1, delta_beta_regions_Jan2, delta_beta_regions_Feb1, delta_beta_regions_Feb2,
                 delta_beta_regions_Mar1, delta_beta_regions_Mar2):
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

        delta_beta_regions_Nov1: np.ndarray (len: 9)
            a spatio-temporal modifier for every US region in 1-15 Nov. hierarchal level 2.

        output
        ------

        beta: np.ndarray
            transmission rate per US state
        """

        # state parameter mapping
        delta_beta_states = delta_beta_states[self.state_mapping]
        # regional parameter mapping
        delta_beta_regions = delta_beta_regions[self.region_mapping]
        delta_beta_regions_Nov1 = delta_beta_regions_Nov1[self.region_mapping]
        delta_beta_regions_Nov2 = delta_beta_regions_Nov2[self.region_mapping]
        delta_beta_regions_Dec1 = delta_beta_regions_Dec1[self.region_mapping]
        delta_beta_regions_Dec2 = delta_beta_regions_Dec2[self.region_mapping]
        delta_beta_regions_Jan1 = delta_beta_regions_Jan1[self.region_mapping]
        delta_beta_regions_Jan2 = delta_beta_regions_Jan2[self.region_mapping]
        delta_beta_regions_Feb1 = delta_beta_regions_Feb1[self.region_mapping]
        delta_beta_regions_Feb2 = delta_beta_regions_Feb2[self.region_mapping]
        delta_beta_regions_Mar1 = delta_beta_regions_Mar1[self.region_mapping]
        delta_beta_regions_Mar2 = delta_beta_regions_Mar2[self.region_mapping]

        # construct modifiers (time x US state)
        modifiers = beta_US * np.array([
                        (delta_beta_temporal[0] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Nov1 + 1),
                        (delta_beta_temporal[1] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Nov2 + 1),

                        (delta_beta_temporal[2] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Dec1 + 1),
                        (delta_beta_temporal[3] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Dec2 + 1),

                        (delta_beta_temporal[4] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Jan1 + 1),
                        (delta_beta_temporal[5] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Jan2 + 1),

                        (delta_beta_temporal[6] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Feb1 + 1),
                        (delta_beta_temporal[7] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Feb2 + 1),

                        (delta_beta_temporal[8] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Mar1 + 1),
                        (delta_beta_temporal[9] + 1) * (delta_beta_regions + 1) * (delta_beta_states + 1) * (delta_beta_regions_Mar2 + 1),

                        (delta_beta_regions + 1) * (delta_beta_states + 1)
                    ])

        # compute smoothed modifier
        return self.get_smoothed_modifier(modifiers, t, half_life_days=5, window_size=30, freq='biweekly')
    
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

#####################
## Social contacts ##
#####################

class make_contact_function():

    def __init__(self, contact_matrix_week_noholiday, contact_matrix_week_holiday, contact_matrix_weekend):
        """ Load the contact matrices and stores them in class
        """
        self.contact_matrix_week_noholiday = contact_matrix_week_noholiday
        self.contact_matrix_week_holiday = contact_matrix_week_holiday
        self.contact_matrix_weekend = contact_matrix_weekend

        pass

    @lru_cache()
    def __call__(self, t):
        """ Returns the right contact matrix depending on daytype; cached to avoid heavy IO referencing during simulation
        """
        if t.weekday() >= 5:
            return self.contact_matrix_weekend
        else:
            if self.is_school_holiday(t):
                return self.contact_matrix_week_holiday
            else:
                return self.contact_matrix_week_noholiday
            
    def contact_function(self, t, states, param):
        """ pySODM compatible wrapper
        """
        return tf.convert_to_tensor(self.__call__(t), dtype=float)

    @staticmethod
    def is_school_holiday(d):
        """
        A function returning 'True' if a given date is a school holiday for primary and secundary schools in the USA.
        Tertiary education is not considered in this work.

        Input
        =====

        d: datetime.datetime
            current date

        Returns
        =======

        is_school_holiday: bool
            True: date `d` is a school holiday for primary and secundary schools
        """

        #########################
        ## Fixed date holidays ##
        #########################

        # summer vacation
        if d.month in [6, 7, 8]:  
            return True

        # fixed-date holidays
        if (d.month, d.day) in {(1, 1): "New Year's Day", (7, 4): "Independence Day", (11, 11): "Veterans Day", (12, 25): "Christmas", (6,19): "Juneteenth"}:
            return True

        #############################
        ## winter and spring break ##
        #############################

        holiday_weeks = []
        # Winter break
        # Typically two week break covering Christmas and NY --> Similar to Belgium
        # If Christmas falls on Saturday or Sunday, winter break starts week after
        w_christmas_current = datetime(
            year=d.year, month=12, day=25).isocalendar().week
        if datetime(year=d.year, month=12, day=25).isoweekday() in [6, 7]:
            w_christmas_current += 1
        w_christmas_previous = datetime(
            year=d.year-1, month=12, day=25).isocalendar().week
        if datetime(year=d.year-1, month=12, day=25).isoweekday() in [6, 7]:
            w_christmas_previous += 1
        # Christmas "logic"
        if w_christmas_previous == 52:
            if datetime(year=d.year-1, month=12, day=31).isocalendar().week != 53:
                holiday_weeks.append(1)
        if w_christmas_current == 51:
            holiday_weeks.append(w_christmas_current)
            holiday_weeks.append(w_christmas_current+1)
        if w_christmas_current == 52:
            holiday_weeks.append(w_christmas_current)
            if datetime(year=d.year, month=12, day=31).isocalendar().week == 53:
                holiday_weeks.append(w_christmas_current+1)
        holiday_weeks = [1,] # two weeks might be a huge overestimation of the impact of this holiday --> first week of year is most consistent with data

        # Spring break
        # Extract date of easter
        d_easter = easter(d.year)
        # Convert from datetime.date to datetime.datetime
        d_easter = datetime(d_easter.year, d_easter.month, d_easter.day)
        # Get week of easter
        w_easter = d_easter.isocalendar().week
        # Default logic: Easter holiday starts first monday of April
        # Unless: Easter falls after 04-15: Easter holiday ends with Easter
        # Unless: Easter falls in March: Easter holiday starts with Easter
        if d_easter >= datetime(year=d.year, month=4, day=15):
            w_easter_holiday = w_easter - 1
        elif d_easter.month == 3:
            w_easter_holiday = w_easter + 1
        else:
            w_easter_holiday = datetime(
                d.year, 4, (8 - datetime(d.year, 4, 1).weekday()) % 7).isocalendar().week
        holiday_weeks.append(w_easter_holiday)
        holiday_weeks.append(w_easter_holiday+1)

        # Check winter and spring break
        if d.isocalendar().week in holiday_weeks:
            return True

        #######################
        ## Variable holidays ##
        #######################

        ## MLKJ day: Third monday of January
        jan_first = datetime(d.year, 1, 1)                                              # start with january first
        first_monday = jan_first + timedelta(days=(7 - jan_first.weekday()) % 7)        # find first monday
        date = first_monday + timedelta(weeks=2)                                        # add to weeks to it
        if d == date:
            return True

        ## Presidents day: Third monday of February
        feb_first = datetime(d.year, 2, 1)  
        first_monday = feb_first + timedelta(days=(7 - feb_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=2)
        if d == date:
            return True

        ## Memorial day: Last monday of May
        may_first = datetime(d.year, 5, 1)  
        first_monday = may_first + timedelta(days=(7 - may_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=3)
        if d == date:
            return True

        ## Labor day: First monday of September
        sept_first = datetime(d.year, 9, 1)                                 
        first_monday = sept_first + timedelta(days=(7 - sept_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=0)
        if d == date:
            return True

        ## Columbus day: Second monday of October
        oct_first = datetime(d.year, 10, 1)                                 
        first_monday = oct_first + timedelta(days=(7 - oct_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=0)
        if d == date:
            return True

        ## Thanksgiving day: Fourth Thursday of November
        nov_first = datetime(d.year, 11, 1)  
        first_thursday = nov_first + timedelta(days=(3 - nov_first.weekday() + 7) % 7)
        date = first_thursday + timedelta(weeks=3)
        if d == date:
            return True
        
        return False

################################
## Initial condition function ##
################################

from influenza_USA.SVI2RHD.utils import construct_initial_susceptible, get_cumulative_vaccinated, get_vaccination_data, convert_vaccination_data, get_spatial_mappings
class make_initial_condition_function():

    def __init__(self, spatial_resolution, age_resolution, start_sim, season):
        # retrieve the vaccination data
        vaccination_data = get_vaccination_data()
        # convert the vaccination data to the right age and spatial resolution
        vaccination_data = convert_vaccination_data(vaccination_data, spatial_resolution, age_resolution)
        # retrieve the demography (susceptible pool)
        self.demography = construct_initial_susceptible(spatial_resolution, age_resolution)
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

################################################################
## PARKING: seasonality & exponential vaccine efficacy waning ##
################################################################

def seasonality_function(t, states, param, amplitude, peak_shift):
    """
    Default output function. Returns a sinusoid with average value 1.
    
    t : datetime.datetime
        simulation time
    amplitude : float
        maximum deviation of output with respect to the average (1)
    peak_shift : float
        phase. Number of days after January 1st after which the maximum value of the seasonality rescaling is reached 
    """
    ref_date = datetime(2021,1,1)
    # If peak_shift = 0, the max is on the first of January
    maxdate = ref_date + timedelta(days=float(peak_shift))
    # One period is one year long (seasonality)
    t = (t - maxdate)/timedelta(days=1)/365
    rescaling = 1 + amplitude*np.cos( 2*np.pi*(t))
    return rescaling

def exponential_waning_function(t, states, param, waning_start, T_s):
    if t < waning_start:
        return 1.0
    else:
        return np.exp(-1/T_s * float((t - waning_start)/timedelta(days=1)))
