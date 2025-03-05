"""
This script contains the integration function of a single strain and a two-strain sequential infection SIR model, used to forecast influenza in North Carolina.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

# import packages
import numpy as np
from pySODM.models.base import ODE

# define integration function
class SIR_strains(ODE):
    """
    SIR model with one strain, observed ILI and hospitalisation incidences
    """
    
    states = ['S','I','R',                      # states
              'I_inc', 'H_inc_0', 'H_inc',      # outcomes
              ]
    parameters = ['delta_beta_t', 'T_r', 'T_h', 'rho_i']
    stratified_parameters = ['beta', 'rho_h']
    dimensions = ['strain',]

    @staticmethod
    def integrate(t, S, I, R, I_inc, H_inc_0, H_inc, beta, delta_beta_t, T_r, T_h, rho_i, rho_h):

        # compute total population
        T = S+I+R

        # compute new infected 
        I_new = delta_beta_t * beta * S * I/T

        # calculate state differentials
        dS = - I_new
        dI = I_new - (1/T_r) * I
        dR = (1/T_r) * I

        # calculate outcome differentials 
        ## ED visits
        dI_inc = I_new*rho_i - I_inc
        ## hospitalisations
        dH_inc_0 = I_new*rho_h - (1/T_h)*H_inc_0
        dH_inc = (1/T_h)*H_inc_0 - H_inc

        return dS, dI, dR, dI_inc, dH_inc_0, dH_inc