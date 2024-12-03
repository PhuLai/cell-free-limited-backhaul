# -*- coding: utf-8 -*-
"""
Solution for determining UE - AP association
Nearest CPU
Assumption/constraint: 1 UE - 1 CPU, UE allocated to all APs associated with that CPU
Note: Nearest CPU might not have sufficient resource to serve the UE. This approach allocates UEs to their nearest APs/CPUs
regardless of the resource availability.

@author: Phu Lai
"""
import time
import numpy as np
from algorithms import pilot_assignment
from utilities import constants, utils


def allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs,
                         locations_CPUs, locations_UEs, max_power_AP, upsilon, kappa, is_print_summary):
    start = time.perf_counter()
    algo = constants.ALGO_NEAREST_NAIVE
    # number of UEs, APs, CPUs
    K, L, U = gain_over_noise_dB.shape[1], gain_over_noise_dB.shape[0], len(np.unique(AP_CPU_association))
    N = R.shape[0]  # number of antennas
    pilot_index = pilot_assignment.assign_pilots_bjornson(tau_p, gain_over_noise_dB)  # assign pilots to UEs
    D = np.zeros((L, K), dtype=int)
    # calculate distances between CPUs and UEs
    distances_CPUs_UEs = utils.calculate_distances_CPUs_UEs(locations_CPUs, locations_UEs)

    for k in range(K):
        # find nearest CPU to UE k
        nearest_CPU = np.argmin(distances_CPUs_UEs[:, k])
        APs_nearest_CPU = utils.get_APs_associated_with_CPU(nearest_CPU, AP_CPU_association)
        D[APs_nearest_CPU, k] = 1

    execution_time = time.perf_counter() - start
    from utilities import algo_result
    result = algo_result.AlgorithmResult(algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs,
                                         execution_time, AP_CPU_association, N,
                                         max_power_AP, gain_over_noise_dB, upsilon, kappa)
    if is_print_summary:
        utils.print_result_summary(result)
    return result
