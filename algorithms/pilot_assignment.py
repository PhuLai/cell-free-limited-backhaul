# -*- coding: utf-8 -*-
"""
Pilot assignment methods defined here

@author: Phu Lai
"""
import numpy as np
from utilities import utils


def assign_pilots_bjornson(tau_p, gain_over_noise_dB):
    """ Assign pilots to UEs
    Pilot assignment method proposed in Section V.A of Björnson, E. and Sanguinetti, L., 2020.
    Scalable cell-free massive MIMO systems. IEEE Transactions on Communications, 68(7), pp.4247-4261.
    Code translated from the original MATLAB code at: https://github.com/emilbjornson/scalable-cell-free/ to Python with
    some minor changes
    :param tau_p:               number of orthogonal pilots
    :param gain_over_noise_dB:  channel gain over noise in dB
    :return:                    list of pilots for each UE
    """
    K = gain_over_noise_dB.shape[1]  # number of UEs
    # pilot index for each UE. Populate with K Nones
    pilot_index = np.full(K, None)
    for k in range(K):
        master_AP_idx = np.argmax(gain_over_noise_dB[:, k])
        if k < tau_p:  # assign orthogonal pilots to the first tau_p UEs
            pilot_index[k] = k
        else:  # Assign pilot for remaining UEs
            # Compute received power at the master AP for each pilot
            pilot_interference = np.zeros(tau_p)
            for t in range(tau_p):
                pilot_UEs = [i for i, x in enumerate(pilot_index) if x == t]  # UEs on pilot t
                pilot_interference[t] = sum(utils.db2pow(gain_over_noise_dB[master_AP_idx, pilot_UEs]))
            # Find the pilot with the least receiver power
            pilot_index[k] = np.argmin(pilot_interference)
    return pilot_index