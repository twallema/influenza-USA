"""
This script contains the integration function of an age-stratified spatially-explicit two-strain sequential infection SIR model.
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
    
    states = ['S','I1','I2','R1','R2','I12','I21','R',      # states
              'I1_inc', 'I2_inc',                           # needed for outcomes
              ]
    parameters = ['beta1', 'beta2', 'N', 'T_r', 'T_h', 'rho_i', 'rho_h1', 'rho_h2', 'CHR']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, I1, I2, R1, R2, I12, I21, R, I1_inc, I2_inc, beta1, beta2, N, T_r, T_h, rho_i, rho_h1, rho_h2, CHR):

        # compute total population
        T = S+I1+I2+R1+R2+I12+I21+R

        # compute infectious populations
        infpop_1 = (I1 + I21) / T
        infpop_2 = (I2 + I12) / T

        # new infected with strain 1
        I1_new = beta1 * S * np.matmul(N, infpop_1)
        I21_new = beta1 * R2 * np.matmul(N, infpop_1)

        # new infected with strain 2
        I2_new = beta2 * S * np.matmul(N, infpop_2)
        I12_new = beta2 * R1 * np.matmul(N, infpop_2)

        # calculate state differentials
        dS = - (I1_new + I2_new)
        dI1 = I1_new - (1/T_r) * I1
        dI2 = I2_new - (1/T_r) * I2
        dR1 = (1/T_r) * I1 - I12_new
        dR2 = (1/T_r) * I2 - I21_new
        dI12 = I12_new - (1/T_r) * I12
        dI21 = I21_new - (1/T_r) * I21
        dR = (1/T_r) * (I12 + I21)

        # calculate outcome differentials
        dI1_inc = (I1_new + I21_new) - I1_inc
        dI2_inc = (I2_new + I12_new) - I2_inc

        return dS, dI1, dI2, dR1, dR2, dI12, dI21, dR, dI1_inc, dI2_inc


    # TODO: add to model states + checks
    outcomes = ['I_inc_obs', 'H1_inc', 'H2_inc']

    @staticmethod
    def compute_outcomes(simout, beta1, beta2, N, T_r, T_h, rho_i, rho_h1, rho_h2, CHR):
        
        # U-shaped severity curve
        rho_i = (rho_i * CHR)[:, np.newaxis]
        rho_h1 = (rho_h1 * CHR)[:, np.newaxis]
        rho_h2 = (rho_h2 * CHR)[:, np.newaxis]

        # Infectious (I) observed incidence
        simout['I_inc_obs'] = (simout['I1_inc'] + simout['I2_inc'])*rho_i
        # Hospitalised (H) observed incidence
        simout['H1_inc'] = (simout['I1_inc']*rho_h1).shift(date=T_h, fill_value=0)
        simout['H2_inc'] = (simout['I2_inc']*rho_h2).shift(date=T_h, fill_value=0)

        return simout