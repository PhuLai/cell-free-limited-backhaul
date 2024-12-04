# -*- coding: utf-8 -*-
"""
Approach in "Scalability Aspects of Cell-Free Massive MIMO"
ICC 2019 - 2019 IEEE International Conference on Communications (ICC)

@author: Phu Lai
"""
import time
import numpy as np

from algorithms import pilot_assignment
from utilities import constants, utils


def allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs,
                         max_power_AP, upsilon, kappa, top_N_CPUs, is_print_summary):
    start = time.perf_counter()
    algo = constants.ALGO_SCF2
    # number of UEs, APs, CPUs
    K, L, U = gain_over_noise_dB.shape[1], gain_over_noise_dB.shape[0], len(np.unique(AP_CPU_association))
    N = R.shape[0]  # number of antennas
    D = np.zeros((L, K), dtype=int)  # AP-UE association matrix. This is the decision variable D used in our paper
    pilot_index = pilot_assignment.assign_pilots_bjornson(tau_p, gain_over_noise_dB)

    for k in range(K):
        CPUs_gains_k = get_CPUs_gains(k, AP_CPU_association, gain_over_noise_dB)
        top_CPUs = dict(sorted(CPUs_gains_k.items(), key=lambda item: item[1], reverse=True)[:top_N_CPUs])
        for u in top_CPUs.keys():
            APs_u = utils.get_APs_associated_with_CPU(u, AP_CPU_association)
            D[APs_u, k] = 1

    execution_time = time.perf_counter() - start
    from utilities import algo_result
    result = algo_result.AlgorithmResult(algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs, execution_time,
                                         AP_CPU_association, N, max_power_AP, gain_over_noise_dB, upsilon, kappa)
    if is_print_summary:
        utils.print_result_summary(result)

    return result


def get_CPUs_gains(k, AP_CPU_association, gain_over_noise_dB):
    U = len(np.unique(AP_CPU_association))
    CPUs_gains_k = {i: 0 for i in range(U)}  # the sum gain of all APs associated with each CPU
    for u in range(U):
        APs_u = utils.get_APs_associated_with_CPU(u, AP_CPU_association)  # all APs associated with CPU u
        CPUs_gains_k[u] = np.sum(utils.db2pow(gain_over_noise_dB[APs_u, k]))
    return CPUs_gains_k

