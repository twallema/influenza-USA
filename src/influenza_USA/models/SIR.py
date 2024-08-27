"""
This script contains an age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
from pySODM.models.base import ODE

###################
## Deterministic ##
###################

class ODE_SIR(ODE):
    """
    SIR model with age and spatial stratification
    """
    
    states = ['S','I','R']
    parameters = ['beta','gamma', 'f_v', 'N', 'M']
    dimensions = ['age_group', 'location']

    @staticmethod
    def integrate(t, S, I, R, beta, gamma, f_v, N, M):

        # compute contact tensor with different home vs. visited contacts
        C =  ((1 - f_v) * np.einsum('ab,cd->abcd', N, np.eye(M.shape[0])) + f_v * np.einsum('ab,cd->abcd', N, M))

        # compute force of infection
        l = beta * np.einsum ('abcd,bd->ac', C, I/(S+I+R))

        # calculate differentials
        dS = - l * S
        dI = l * S - 1/gamma*I
        dR = 1/gamma*I

        return dS, dI, dR