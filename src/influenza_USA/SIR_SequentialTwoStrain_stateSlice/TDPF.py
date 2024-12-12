"""
This script contains the time-dependent parameter functions associated with the age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
from datetime import datetime

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
        """
        A function to smooth a vector of temporal modifiers with a gaussian filter
        """

        # Define number of days between Nov 1 and Apr 1
        num_days = 152

        # Step 1: Project the input vector on the right knots 
        ## Calculate the positions for each interval
        interval_size = num_days / len(modifiers)
        expanded_vector = np.zeros(num_days)
        ## Project the input values onto the output
        for i in range(len(modifiers)):
            start = int(i * interval_size)
            end = int((i + 1) * interval_size) if i != len(modifiers) - 1 else num_days  # Ensure last interval includes all remaining days
            expanded_vector[start:end] = modifiers[i]

        # Step 2: Prepend and append 31 days of ones
        padded_vector = np.concatenate([np.ones(31), expanded_vector, np.ones(31)])
    
        # Step 3: Apply a gaussian 1D smoother
        smoothed_series = gaussian_filter1d(padded_vector, sigma=sigma)

        # Step 4: Remove the prepended padding, retain the appended padding
        trimmed_smoothed_vector = smoothed_series

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
        temporal_modifiers_smooth = self.smooth_modifier(1+np.array(delta_beta_temporal), t, sigma=self.sigma)

        # apply modifier
        return param * temporal_modifiers_smooth

################################
## Initial condition function ##
################################

from influenza_USA.shared.utils import construct_initial_susceptible
class make_initial_condition_function():

    def __init__(self, spatial_resolution, age_resolution, spatial_coordinates):
        # retrieve the demography (susceptible pool)
        self.demography = construct_initial_susceptible(spatial_resolution, age_resolution, spatial_coordinates)

    def initial_condition_function(self, f_I1, f_I2, f_R1_R2, f_R1):
        """
        A function setting the model's initial condition.
        
        input
        -----

        f_I1: float
            Fraction of the population infected with strain 1.
        
        f_I2: float
            Fraction of the population infected with strain 2.

        f_R1_R2: float
            Fraction of the population with immunity to either strain 1 and strain 2 (= f_R1 + f_R2).
        
        f_R1: float
            Fraction of f_R1_R2 with immunity to strain 1.

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
                'I1': 0.5 * f_I1 * self.demography,     # assumption: split f_I1 50/50 between states I1/I21; idem strain 2
                'I2': 0.5 * f_I2 * self.demography,
                'R1': f_R1 * self.demography,
                'R2': f_R2 * self.demography,
                'I12': 0.5 * f_I2 * self.demography,
                'I21': 0.5 * f_I1 * self.demography,
                }