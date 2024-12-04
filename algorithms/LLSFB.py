# -*- coding: utf-8 -*-
"""
LLSFB (Largest-Large-Scale-Fading-Based): A network-centric variant of the LLSFB user association scheme proposed in [20]
that we came up with. For each UE, we first identify which network-centric cluster has the greatest sum LSFC, then select
APs that collectively contribute at least δ% of the total LSFC of all APs in this cluster to serve the UE.

@author: Phu Lai
"""
import time
import numpy as np
from algorithms import pilot_assignment
from utilities import constants, utils, utils_comms


def allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs,
                         max_UEs_per_AP, threshold_LSF, max_power_AP, upsilon, kappa, is_print_summary):
    algo = constants.ALGO_LLSFB
    start = time.perf_counter()
    # number of UEs, APs, CPUs
    K, L, U = gain_over_noise_dB.shape[1], gain_over_noise_dB.shape[0], len(np.unique(AP_CPU_association))
    N = R.shape[0]  # number of antennas
    pilot_index = pilot_assignment.assign_pilots_bjornson(tau_p, gain_over_noise_dB)
    D = np.zeros((L, K), dtype=int)

    for k in range(K):
        best_CPU, best_APs = find_APs_for_UE_k(k, AP_CPU_association, gain_over_noise_dB, False, D,
                                               max_UEs_per_AP, threshold_LSF)
        D[best_APs, k] = 1

    execution_time = time.perf_counter() - start
    from utilities import algo_result
    result = algo_result.AlgorithmResult(algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs, execution_time,
                                         AP_CPU_association, N, max_power_AP, gain_over_noise_dB, upsilon, kappa)
    if is_print_summary:
        utils.print_result_summary(result)
    return result


def find_APs_for_UE_k(k, AP_CPU_association, gain_over_noise_dB, is_LSF_averaged, D, max_UEs_per_AP, threshold_LSF):
    CPUs_gains = get_CPUs_gains_k(k, AP_CPU_association, gain_over_noise_dB, is_LSF_averaged, D, max_UEs_per_AP)
    # sort CPUs_gains in descending order of values
    CPUs_gains = dict(sorted(CPUs_gains.items(), key=lambda item: item[1], reverse=True))
    best_CPU = list(CPUs_gains.keys())[0]  # get CPU with highest gain
    if threshold_LSF == 1:  # no threshold, get all APs associated with the best CPU
        best_APs = utils.get_APs_associated_with_CPU(best_CPU, AP_CPU_association)
    else:
        APs_best_CPU = utils.get_APs_associated_with_CPU(best_CPU, AP_CPU_association)
        best_APs = utils_comms.get_top_APs_contribute_threshold_LSF(k, APs_best_CPU, gain_over_noise_dB, threshold_LSF)
    return best_CPU, best_APs


def get_CPUs_gains_k(k, AP_CPU_association, gain_over_noise_dB, is_LSF_averaged, D, max_UEs_per_AP):
    """
    Get the gain of each UE to each CPU:
    - For each UE and CPU:
        - get all APs associated with that CPU
        - sum the gain of all APs, get average (if is_LSF_averaged = True)
    """
    U = len(np.unique(AP_CPU_association))
    CPUs_gains = {i: 0 for i in range(U)}
    for u in range(U):
        APs_u = utils.get_APs_associated_with_CPU(u, AP_CPU_association)
        APs_u_within_UE_limit = utils.get_APs_within_nb_UEs_limit(D, max_UEs_per_AP, APs_u)
        CPUs_gains[u] = np.sum(utils.db2pow(gain_over_noise_dB[APs_u_within_UE_limit, k]))
        if is_LSF_averaged:
            CPUs_gains[u] /= len(APs_u_within_UE_limit)
    CPUs_gains = {k: v for k, v in CPUs_gains.items() if v != 0}  # remove CPU with 0 gain
    return CPUs_gains
