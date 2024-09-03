"""
This script contains an age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."


import numpy as np
import tensorflow as tf
from pySODM.models.base import ODE, JumpProcess

###################
## Deterministic ##
###################

class ODE_SIR(ODE):
    """
    SIR model with age and spatial stratification; tensorflow einstein summation
    """
    
    states = ['S','I','R']
    parameters = ['beta','gamma', 'f_v', 'N', 'M']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma, f_v, N, M):

        # compute contact tensor with different home vs. visited contacts
        C =  ((1 - f_v) * tf.einsum('ab,cd->abcd', N, tf.eye(M.shape[0])) + f_v * tf.einsum('ab,cd->abcd', N, M))

        # compute force of infection
        l = beta * tf.einsum ('abcd,bd->ac', C, I/(S+I+R))

        # calculate differentials
        dS = - l * S
        dI = l * S - 1/gamma*I
        dR = 1/gamma*I

        return dS, dI, dR
    
################
## Stochastic ##
################

class TL_SIR(JumpProcess):
    """
    Stochastic SIR model with age and spatial stratification; tensorflow einstein summation
    """
    states = ['S', 'I','R']
    parameters = ['beta','gamma', 'f_v', 'N', 'M']
    dimensions = ['age_group', 'location']


    @staticmethod
    def compute_rates(t, S, I, R, beta, gamma, f_v, N, M):

        # compute contact tensor with different home vs. visited contacts
        C =  ((1 - f_v) * tf.einsum('ab,cd->abcd', N, tf.eye(M.shape[0])) + f_v * tf.einsum('ab,cd->abcd', N, M))

        # compute force of infection
        l = beta * tf.einsum ('abcd,bd->ac', C, I/(S+I+R))

        rates = {
            'S': [l.numpy()], 
            'I': [np.ones(S.shape, np.float64)*(1/gamma)], 
            }
        
        return rates

    @ staticmethod
    def apply_transitionings(t, tau, transitionings, S, I, R, beta, f_v, gamma, N, M):
        
        S_new = S - transitionings['S'][0]
        I_new = I + transitionings['S'][0] - transitionings['I'][0]
        R_new = R + transitionings['I'][0]
        
        return(S_new, I_new, R_new)