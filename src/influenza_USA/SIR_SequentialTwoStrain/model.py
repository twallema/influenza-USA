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
    
    states = ['S','I1','I2','R1','R2','I12','I21','R',                  # states
              'I_inc', 'H1_inc', 'H2_inc', 'H1_inc_0', 'H2_inc_0',      # outcomes
              ]
    parameters = ['beta1', 'beta2', 'N', 'T_r', 'T_h', 'rho_i', 'rho_h1', 'rho_h2', 'CHR']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, I1, I2, R1, R2, I12, I21, R, I_inc, H1_inc, H2_inc, H1_inc_0, H2_inc_0, beta1, beta2, N, T_r, T_h, rho_i, rho_h1, rho_h2, CHR):

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

        # U-shaped severity curve
        rho_i = (rho_i * CHR)[:, np.newaxis]
        rho_h1 = (rho_h1 * CHR)[:, np.newaxis]
        rho_h2 = (rho_h2 * CHR)[:, np.newaxis]

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
        dI_inc = (I1_new + I21_new)*rho_i + (I2_new + I12_new)*rho_i - I_inc

        # delayed hospitalisation -- exponentially distributed delay
        ## strain I
        dH1_inc_0 = (I1_new + I21_new)*rho_h1 - (1/T_h)*H1_inc_0
        dH1_inc = (1/T_h)*H1_inc_0 - H1_inc
        # strain II
        dH2_inc_0 = (I2_new + I12_new)*rho_h2 - (1/T_h)*H2_inc_0
        dH2_inc = (1/T_h)*H2_inc_0 - H2_inc
 
        return dS, dI1, dI2, dR1, dR2, dI12, dI21, dR, dI_inc, dH1_inc, dH2_inc, dH1_inc_0, dH2_inc_0