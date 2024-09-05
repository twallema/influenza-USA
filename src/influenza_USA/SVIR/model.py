"""
This script contains an age-stratified spatially-explicit SIR model with a vaccine state for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."


import numpy as np
import tensorflow as tf
from pySODM.models.base import ODE, JumpProcess

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
    parameters = ['beta', 'f_v', 'N', 'M', 'r_vacc', 'e_i', 'e_h', 'T_s', 'rho_h', 'T_h', 'rho_d', 'T_d', 'f_waning', 'f_seasonality', 'asc_case']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, V, I, Iv, R, H, D, I_inc, H_inc, D_inc, beta, f_v, N, M, r_vacc, e_i, e_h, T_s, rho_h, T_h, rho_d, T_d, f_waning, f_seasonality, asc_case):

        # compute contact tensor with different home vs. visited contacts
        C =  ((1 - f_v) * tf.einsum('ab,cd->abcd', N, tf.eye(M.shape[0])) + f_v * tf.einsum('ab,cd->abcd', N, M))

        # compute force of infection
        l = f_seasonality * beta * tf.einsum ('abcd,bd->ac', C, (I+Iv)/(S+V+I+Iv+R+H))

        # calculate state differentials
        dS = - r_vacc*S - l*S + (1/T_s)*(R + V)
        dV = r_vacc*S - (1-f_waning*e_i)*l*V  - (1/T_s)*V
        dI = l*S - (1/T_h)*I
        dIv = (1-f_waning*e_i)*l*V - (1/T_h)*Iv
        dR = (1-rho_h)*(1/T_h)*I + 1 - ((1-f_waning*e_h)*rho_h)*(1/T_h)*Iv + (1-rho_d)*(1/T_d)*H - (1/T_s)*R
        dH = rho_h*(1/T_h)*I + rho_h*(1-f_waning*e_h)*(1/T_h)*Iv - (1/T_d)*H
        dD = rho_d*(1/T_d)*H

        # calculate outcome differentials
        dI_inc = asc_case*(l*S + (1-f_waning*e_i)*l*V) - I_inc
        dH_inc = rho_h*(1/T_h)*I + rho_h*(1-f_waning*e_h)*(1/T_h)*Iv - H_inc
        dD_inc = dD - D_inc

        return dS, dV, dI, dIv, dR, dH, dD, dI_inc, dH_inc, dD_inc
    
################
## Stochastic ##
################

class TL_SVI2RHD(JumpProcess):
    """
    Stochastic influenza model with vaccines and age/spatial stratification
    """

    states = ['S','V','I','Iv','R','H','D',     # states
              'I_inc', 'H_inc', 'D_inc'         # outcomes
              ]
    parameters = ['beta', 'f_v', 'N', 'M', 'r_vacc', 'e_i', 'e_h', 'T_s', 'rho_h', 'T_h', 'rho_d', 'T_d', 'f_waning', 'asc_case']
    dimensions = ['age_group', 'location']

    @staticmethod
    def compute_rates(t, S, V, I, Iv, R, H, D, I_inc, H_inc, D_inc, beta, f_v, N, M, r_vacc, e_i, e_h, T_s, rho_h, T_h, rho_d, T_d, f_waning, asc_case):

        # compute contact tensor with different home vs. visited contacts
        C =  ((1 - f_v) * tf.einsum('ab,cd->abcd', N, tf.eye(M.shape[0])) + f_v * tf.einsum('ab,cd->abcd', N, M))

        # compute force of infection
        l = beta * tf.einsum ('abcd,bd->ac', C, (I+Iv)/(S+V+I+Iv+R+H))

        # compute rates of transitionings
        size_dummy = np.ones(S.shape, np.float64)
        rates = {
            'S': [l.numpy(), r_vacc], 
            'V': [(1-f_waning*e_i)*l.numpy(), size_dummy*(1/T_s)],
            'I': [size_dummy*(1-rho_h)*(1/T_h), size_dummy*rho_h*(1/T_h)],
            'Iv': [size_dummy*(1-((1-f_waning*e_h)*rho_h)*(1/T_h)), size_dummy*rho_h*(1-f_waning*e_h)*(1/T_h)],
            'H': [size_dummy*(1-rho_d)*(1/T_d), size_dummy*rho_d*(1/T_d)],
            'R': [size_dummy*(1/T_s),]
            }
        
        return rates

    @ staticmethod
    def apply_transitionings(t, tau, transitionings, S, V, I, Iv, R, H, D, I_inc, H_inc, D_inc, beta, f_v, N, M, r_vacc, e_i, e_h, T_s, rho_h, T_h, rho_d, T_d, f_waning, asc_case):
        
        # states
        S_new = S - transitionings['S'][0] - tau*transitionings['S'][1] + transitionings['V'][1] + transitionings['R'][0]
        V_new = V - transitionings['V'][0] - transitionings['V'][1] + tau*transitionings['S'][1]
        I_new = I + transitionings['S'][0] - transitionings['I'][0] - transitionings['I'][1]
        Iv_new = Iv + transitionings['V'][0] - transitionings['Iv'][0] - transitionings['Iv'][1]
        R_new = R + transitionings['I'][0] + transitionings['Iv'][0] - transitionings['R'][0] + transitionings['H'][0]
        H_new = H + transitionings['I'][1] + transitionings['Iv'][1] - transitionings['H'][0] - transitionings['H'][1] 
        D_new = D + transitionings['H'][1] 
        
        # incidences
        I_inc_new = tau*asc_case*(transitionings['S'][0] + transitionings['V'][0])
        H_inc_new = tau*(transitionings['I'][1] + transitionings['Iv'][1])
        D_inc_new = tau*transitionings['H'][1]

        return(S_new, V_new, I_new, Iv_new, R_new, H_new, D_new, I_inc_new, H_inc_new, D_inc_new)