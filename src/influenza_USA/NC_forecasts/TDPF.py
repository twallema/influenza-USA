"""
This script contains the time-dependent parameter functions associated with the North Carolina influenza forecasting models
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
from influenza_USA.shared.utils import get_smooth_temporal_modifier

##################################
## Hierarchal transmission rate ##
##################################

class transmission_rate_function():

    def __init__(self, sigma):
        self.sigma = sigma
        pass

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
        temporal_modifiers_smooth = get_smooth_temporal_modifier(1+np.array(delta_beta_temporal), t, sigma=self.sigma)

        # apply modifier
        return param * temporal_modifiers_smooth

################################
## Initial condition function ##
################################

from influenza_USA.shared.utils import construct_initial_susceptible
class make_initial_condition_function():

    def __init__(self, spatial_resolution, age_resolution, spatial_coordinates):
        self.demography = construct_initial_susceptible(spatial_resolution, age_resolution, spatial_coordinates)
        pass

    def initial_condition_function_oneStrain(self, f_I, f_R):
        """
        A function setting the model's initial condition.
        
        input
        -----

        f_I: float
            Fraction of the population initially infected
        
        f_R1: float
            Fraction of the population initially immune

        output
        ------

        initial_condition: dict
            Keys: 'S', ... . Values: np.ndarray (n_age x n_loc).
        """

        return {'S':  (1 - f_I - f_R) * self.demography,
                'I': f_I * self.demography,   
                'R': f_R * self.demography,
                }

    def initial_condition_function_twoStrain(self, f_I1, f_I2, f_R1_R2, f_R1):
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