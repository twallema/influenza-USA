"""
This script contains the time-dependent parameter functions associated with the age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
import pandas as pd
import tensorflow as tf
from functools import lru_cache
from dateutil.easter import easter
from datetime import datetime, timedelta

##################################
## Hierarchal transmission rate ##
##################################

from scipy.ndimage import gaussian_filter1d

class transmission_rate_function():

    def __init__(self, sigma):
        self.sigma = sigma
        pass
    
    @staticmethod
    def smooth_modifier(modifiers, simulation_date, sigma):
        """ A function to smooth a vector of temporal modifiers with a gaussian filter
        """
        # Step 1: Expand the vector to daily values; assume 15 days per entry
        expanded_vector = np.repeat(modifiers, 15)
    
        # Step 2: Prepend and append ones
        padding = np.ones(31)
        padded_vector = np.concatenate([padding, expanded_vector, padding])
    
        # Step 3: Apply a gaussian 1D smoother
        smoothed_series = pd.Series(gaussian_filter1d(padded_vector, sigma=sigma))

        # Step 4: Remove the prepended padding, retain the appended padding
        trimmed_smoothed_vector = smoothed_series.iloc[31:].values

        # Step 5: Compute the number of days since the last October 1
        year = simulation_date.year
        # Compute the last October 1
        oct1_this_year = datetime(year, 10, 1)
        if simulation_date >= oct1_this_year:
            last_oct1 = oct1_this_year
        else:
            last_oct1 = datetime(year - 1, 10, 1)
        # Calculate the difference in days
        days_difference = (simulation_date - last_oct1).days

        # Step 6: Get the right smoothed value
        try:
            return trimmed_smoothed_vector[days_difference]
        except:
            return 1

    def __call__(self, t, states, param, delta_beta_temporal):
        """
        A function constructing a temporal transmission rate 'beta'

        input
        -----

        t: datetime.datetime
            time in simulation

        states: dict
            model states on time `t`
        
        param: dict
            model parameters

        delta_beta_temporal: np.ndarray
            multiplicative piecewise-continuous modifier of transmission rate between Nov-Apr
            no effect: modifier = 0
            biweekly: length 10, monthly: length 5

        output
        ------

        beta(t): np.ndarray
            time-varying transmission rate
        """

        # smooth modifier
        temporal_modifiers_smooth = self.smooth_modifier(1+delta_beta_temporal[:, np.newaxis], t, sigma=self.sigma)

        # apply modifier
        return param * temporal_modifiers_smooth

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

from influenza_USA.SIR_SequentialTwoStrain_stateSlice.utils import construct_initial_susceptible, get_spatial_mappings
class make_initial_condition_function():

    def __init__(self, spatial_resolution, age_resolution, spatial_coordinates):
        # retrieve the demography (susceptible pool)
        self.demography = construct_initial_susceptible(spatial_resolution, age_resolution, spatial_coordinates)
        # retrieve region/state --> state/county parameter  mapping
        self.region_mapping, self.state_mapping = get_spatial_mappings(spatial_resolution)

    def initial_condition_function(self, f_I1, f_I2, f_R1_R2, f_R1):
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

        return {'S':  (1 - f_I1 - f_I2 - f_R1 - f_R2) * self.demography,
                'I1': 0.5 * f_I1 * self.demography,     # assumption 50/50 I1 versus I21
                'I2': 0.5 * f_I2 * self.demography,
                'R1': f_R1 * self.demography,
                'R2': f_R2 * self.demography,
                'I12': 0.5 * f_I2 * self.demography,
                'I21': 0.5 * f_I1 * self.demography,
                }