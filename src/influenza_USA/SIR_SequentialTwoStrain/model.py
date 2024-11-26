"""
This script contains an age-stratified spatially-explicit SIR model with a vaccine state for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."


import numpy as np
from pySODM.models.base import ODE

###################
## Deterministic ##
###################

class ODE_SIR_SequentialTwoStrain(ODE):
    """
    Influenza model with vaccines and age/spatial stratification
    """
    
    states = ['S','I1','I2','R1','R2','I12','I21','R',       # states
              'H1_inc', 'H2_inc', 'H_inc',                   # outcomes
              ]
    parameters = ['beta1', 'beta2', 'N', 'T_r', 'rho_h1', 'rho_h2', 'CHR']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, I1, I2, R1, R2, I12, I21, R, H1_inc, H2_inc, H_inc, beta1, beta2, N, T_r, rho_h1, rho_h2, CHR):

        # compute total population
        T = S+I1+I2+R1+R2+I12+I21+R

        # compute infectious populations
        infpop_1 = (I1 + I21) / T
        infpop_2 = (I2 + I12) / T

        # new infected with strain 1
        I1_inc = beta1 * S * np.matmul(N, infpop_1)
        I21_inc = beta1 * R2 * np.matmul(N, infpop_1)

        # new infected with strain 2
        I2_inc = beta2 * S * np.matmul(N, infpop_2)
        I12_inc = beta2 * R1 * np.matmul(N, infpop_2)

        # u-shaped severity curve
        rho_h1 = (rho_h1 * CHR)[:, np.newaxis]
        rho_h2 = (rho_h2 * CHR)[:, np.newaxis]

        # calculate state differentials
        dS = - (I1_inc + I2_inc)
        dI1 = I1_inc - (1/T_r) * I1
        dI2 = I2_inc - (1/T_r) * I2
        dR1 = (1/T_r) * I1 - I12_inc
        dR2 = (1/T_r) * I2 - I21_inc
        dI12 = I12_inc - (1/T_r) * I12
        dI21 = I21_inc - (1/T_r) * I21
        dR = (1/T_r) * (I12 + I21)

        # calculate outcome differentials
        dH1_inc = (I1_inc + I21_inc) * rho_h1 - H1_inc
        dH2_inc = (I2_inc + I12_inc) * rho_h2 - H2_inc
        dH_inc = (I1_inc + I21_inc) * rho_h1 + (I2_inc + I12_inc) * rho_h2 - H_inc

        return dS, dI1, dI2, dR1, dR2, dI12, dI21, dR, dH1_inc, dH2_inc, dH_inc
