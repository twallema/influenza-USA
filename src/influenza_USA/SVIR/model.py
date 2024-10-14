"""
This script contains an age-stratified spatially-explicit SIR model with a vaccine state for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."


import numpy as np
import tensorflow as tf
from pySODM.models.base import ODE

###################
## Deterministic ##
###################

class ODE_SVI2RHD(ODE):
    """
    Influenza model with vaccines and age/spatial stratification
    """
    
    states = ['S','V','I','Iv','R','H','D',     # states
              'I_inc', 'H_inc', 'D_inc'         # outcomes
              ]
    parameters = ['beta', 'f_v', 'N', 'M', 'n_vacc', 'e_i', 'e_h', 'T_r', 'T_v', 'rho_h', 'CHR', 'T_h', 'rho_d', 'T_d', 'asc_case']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, V, I, Iv, R, H, D, I_inc, H_inc, D_inc, beta, f_v, N, M, n_vacc, e_i, e_h, T_r, T_v, rho_h, CHR, T_h, rho_d, T_d, asc_case):

        # compute total population
        T = S+V+I+Iv+R+H

        # compute visiting populations
        I_v = np.matmul((I+Iv), M)
        T_v = np.matmul(T, M)

        # compute force of infection
        l = np.atleast_1d(beta)[np.newaxis, :] * (tf.einsum ('lj, il -> ij', (I+Iv)/T, (1-f_v)*N) + tf.einsum ('jk, lk, il -> ij', M, I_v/T_v, f_v*N))

        # u-shaped severity curve
        rho_h = (rho_h * CHR)[:, np.newaxis]

        # non-targeted vaccine campaign: only S receive vaccine
        n_vacc = n_vacc * (S/(T-V-I-Iv))

        # calculate state differentials
        dS = - n_vacc - l*S + (1/T_r)*R + (1/T_v)*V
        dV = n_vacc - (1-e_i)*l*V  - (1/T_v)*V
        dI = l*S - (1/T_h)*I
        dIv = (1-e_i)*l*V - (1/T_h)*Iv
        dR = (1-rho_h)*(1/T_h)*I + 1 - ((1-e_h)*rho_h)*(1/T_h)*Iv + (1-rho_d)*(1/T_d)*H - (1/T_r)*R
        dH = rho_h*(1/T_h)*I + rho_h*(1-e_h)*(1/T_h)*Iv - (1/T_d)*H
        dD = rho_d*(1/T_d)*H

        # calculate outcome differentials
        dI_inc = asc_case*(l*S + (1-e_i)*l*V) - I_inc
        dH_inc = rho_h*(1/T_h)*I + rho_h*(1-e_h)*(1/T_h)*Iv - H_inc
        dD_inc = dD - D_inc

        return dS, dV, dI, dIv, dR, dH, dD, dI_inc, dH_inc, dD_inc
