"""
This script contains the time-dependent parameter functions associated with the age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
import tensorflow as tf
from functools import lru_cache
from dateutil.easter import easter
from datetime import datetime, timedelta

##################################
## Hierarchal transmission rate ##
##################################

class hierarchal_transmission_rate_function():

    def __init__(self, spatial_resolution):
        # retrieve region/state --> state/county parameter mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)
        pass
    
    def get_smoothed_modifier(self, modifiers, simulation_date, half_life_days=5, window_size=30, freq='biweekly'):
        """
        Calculate the smoothed temporal modifier for all US states based on the current simulation date.
        Supports 'biweekly' or 'monthly' frequencies for the modifiers.

        Parameters:
        - modifiers: numpy array of shape (10, 9) or (5, 9), depending on freq ('biweekly' or 'monthly').
        - simulation_date: datetime object representing the current simulation date.
        - half_life_days: Half-life in days for the exponential smoothing (default: 5 days).
        - window_size: The number of days used in the smoothing window (default: 30 days).
        - freq: 'biweekly' (default) or 'monthly'. Determines the structure of the modifiers.

        Returns:
        - smoothed_modifier: numpy array of shape (1, 9), representing the smoothed temporal modifier.
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
            
            # Determine the day and month
            month = current_date.month
            day = current_date.day

            # Determine the period row; if outside Nov-Mar --> assume modifier is none
            if freq == 'biweekly':
                biweekly_period = 1 if day <= 15 else 2
                row = period_mapping.get((month, biweekly_period), None)
            elif freq == 'monthly':
                row = period_mapping.get(month, None)
            
            # Get current date modifier values
            if row == None:
                modifier_values = np.ones(modifiers.shape[1])
            else:
                modifier_values = modifiers[row, :]

            # Calculate the number of days in the current period
            if freq == 'biweekly' and biweekly_period == 1:
                days_in_this_period = min(day, days_to_collect)
            elif freq == 'biweekly' and biweekly_period == 2:
                days_in_this_period = min(day - 15, days_to_collect)
            else:  # Monthly case
                days_in_this_period = min(day, days_to_collect)

            # Append the modifiers and weights for each day
            for _ in range(days_in_this_period):
                modifier_window.append(modifier_values)
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

    def strain1_function(self, t, states, param, beta1_US, delta_beta1_regions, delta_beta1_states, delta_beta_temporal):
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

        output
        ------

        beta: np.ndarray
            transmission rate per US state
        """

        # state parameter mapping
        delta_beta1_states = 1 + delta_beta1_states[self.state_mapping]
        # regional parameter mapping
        delta_beta1_regions = 1 + delta_beta1_regions[self.region_mapping]
        # temporal betas
        delta_beta_temporal = 1 + delta_beta_temporal
        # get smoothed temporal components
        temporal_modifiers_smooth = self.get_smoothed_modifier(delta_beta_temporal[:, np.newaxis], t, half_life_days=7, window_size=30, freq='biweekly')
        # construct modifiers
        return beta1_US * temporal_modifiers_smooth * delta_beta1_regions * delta_beta1_states

    def strain2_function(self, t, states, param, beta2_US, delta_beta2_regions, delta_beta2_states, delta_beta_temporal):
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

        output
        ------

        beta: np.ndarray
            transmission rate per US state
        """

        # state parameter mapping
        delta_beta2_states = 1 + delta_beta2_states[self.state_mapping]
        # regional parameter mapping
        delta_beta2_regions = 1 + delta_beta2_regions[self.region_mapping]
        # temporal betas
        delta_beta_temporal = 1 + delta_beta_temporal
        # get smoothed temporal components
        temporal_modifiers_smooth = self.get_smoothed_modifier(delta_beta_temporal[:, np.newaxis], t, half_life_days=5, window_size=30, freq='biweekly')
        # construct modifiers
        return beta2_US * temporal_modifiers_smooth * delta_beta2_regions * delta_beta2_states

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

from influenza_USA.SIR_SequentialTwoStrain.utils import construct_initial_susceptible, get_spatial_mappings
class make_initial_condition_function():

    def __init__(self, spatial_resolution, age_resolution):
        # retrieve the demography (susceptible pool)
        self.demography = construct_initial_susceptible(spatial_resolution, age_resolution)
        # retrieve region/state --> state/county parameter  mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)

    def initial_condition_function(self, f_I1, f_I2, f_R1_R2, f_R1, delta_f_I1_regions, delta_f_I2_regions, delta_f_R1_regions, delta_f_R2_regions):
        """
        A function setting the model's initial condition. Uses a hierarchal structure for the initial immunity.
        
        input
        -----

        output
        ------

        initial_condition: dict
            Keys: 'S', ... . Values: np.ndarray (n_age x n_loc).
        """

        # compute initial fractions in R1 and R2
        # --> modeled in this way so we can constraint f_R1_R2 between 0 and 1.
        f_R2 = (1-f_R1) * f_R1_R2
        f_R1 = f_R1 * f_R1_R2
        
        # convert all delta_f_R/delta_f_I from region/state to state/county level
        delta_f_I1_regions = delta_f_I1_regions[self.region_mapping]
        delta_f_I2_regions = delta_f_I2_regions[self.region_mapping]
        delta_f_R1_regions = delta_f_R1_regions[self.region_mapping]
        delta_f_R2_regions = delta_f_R2_regions[self.region_mapping]

        # construct hierarchal initial immunity
        f_I1 = f_I1 * (1 + delta_f_I1_regions) 
        f_I2 = f_I2 * (1 + delta_f_I2_regions) 
        f_R1 = f_R1 * (1 + delta_f_R1_regions)  
        f_R2 = f_R2 * (1 + delta_f_R2_regions)

        return {'S':  (1 - f_I1 - f_I2 - f_R1 - f_R2) * self.demography,
                'I1': 0.5 * f_I1 * self.demography,     # assumption 50/50 I1 versus I21
                'I2': 0.5 * f_I2 * self.demography,
                'R1': f_R1 * self.demography,
                'R2': f_R2 * self.demography,
                'I12': 0.5 * f_I2 * self.demography,
                'I21': 0.5 * f_I1 * self.demography,
                }