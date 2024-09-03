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

class ODE_SVIR(ODE):
    """
    SIR model with vaccines and age/spatial stratification
    """
    
    states = ['S','V','I','R']
    parameters = ['beta', 'gamma', 'f_v', 'N', 'M', 'r_vacc', 'e_vacc']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, V, I, R, beta, gamma, f_v, N, M, r_vacc, e_vacc):

        # compute contact tensor with different home vs. visited contacts
        C =  ((1 - f_v) * tf.einsum('ab,cd->abcd', N, tf.eye(M.shape[0])) + f_v * tf.einsum('ab,cd->abcd', N, M))

        # compute force of infection
        l = beta * tf.einsum ('abcd,bd->ac', C, I/(S+V+I+R))

        # calculate differentials
        dS = - r_vacc * S - l * S
        dV = r_vacc  * S - (1-e_vacc) * l * V
        dI = l * S + (1-e_vacc) * l * V - 1/gamma*I
        dR = 1/gamma*I

        return dS, dV, dI, dR
    
################
## Stochastic ##
################

class TL_SVIR(JumpProcess):
    """
    Stochastic SIR model with vaccines and age/spatial stratification
    """
    states = ['S','V','I','R']
    parameters = ['beta','gamma', 'f_v', 'N', 'M', 'r_vacc', 'e_vacc']
    dimensions = ['age_group', 'location']


    @staticmethod
    def compute_rates(t, S, V, I, R, beta, gamma, f_v, N, M, r_vacc, e_vacc):

        # compute contact tensor with different home vs. visited contacts
        C =  ((1 - f_v) * tf.einsum('ab,cd->abcd', N, tf.eye(M.shape[0])) + f_v * tf.einsum('ab,cd->abcd', N, M))

        # compute force of infection
        l = beta * tf.einsum ('abcd,bd->ac', C, I/(S+V+I+R))

        rates = {
            'S': [l.numpy(), r_vacc], 
            'V': [(1-e_vacc) * l.numpy(), ],
            'I': [np.ones(S.shape, np.float64)*(1/gamma)],
            }
        
        return rates

    @ staticmethod
    def apply_transitionings(t, tau, transitionings, S, V, I, R, beta, f_v, gamma, N, M, r_vacc, e_vacc):
        
        S_new = S - transitionings['S'][1] - transitionings['S'][0]
        V_new = V + transitionings['S'][1] - transitionings['V'][0]
        I_new = I + transitionings['S'][0] + transitionings['V'][0] - transitionings['I'][0]
        R_new = R + transitionings['I'][0]
        
        return(S_new, V_new, I_new, R_new)