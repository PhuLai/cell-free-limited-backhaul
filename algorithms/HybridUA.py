# -*- coding: utf-8 -*-
"""
Proposed hybrid network- and user-centric solution for determining UE - AP association.
UEs are classified into two categories: cell-center and cell-edge UEs.
Cell-center UEs are associated with one single CPU.
Cell-edge UEs can be associated with multiple CPUs.

@author: Phu Lai
"""
import numpy as np
from algorithms import pilot_assignment
from utilities import constants, utils, utils_comms


def allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs, threshold_z, threshold_LSF,
                         max_power_AP, upsilon, kappa, top_N_CPUs, is_print_summary):

    algo = constants.ALGO_HEURISTIC_HYBRID
    # number of UEs, APs, CPUs
    K, L, U = gain_over_noise_dB.shape[1], gain_over_noise_dB.shape[0], len(np.unique(AP_CPU_association))
    N = R.shape[0]  # number of antennas
    D = np.zeros((L, K), dtype=int)  # AP-UE association matrix. This is the decision variable D used in our paper
    pilot_index = pilot_assignment.assign_pilots_bjornson(tau_p, gain_over_noise_dB)
    UEs_cell_edge = set()
    for k in range(K):
        CPUs_gains_k = get_CPUs_gains(k, AP_CPU_association, gain_over_noise_dB)  # Line 2
        CPUs_gains_k_list = list(CPUs_gains_k.values())  # convert dict to list
        z_scores = calculate_z_scores(CPUs_gains_k_list)  # Line 3
        is_k_cell_center_ = is_k_cell_center(z_scores, threshold_z, CPUs_gains_k_list)  # Line 4
        if is_k_cell_center_:
            top_CPUs = [max(CPUs_gains_k, key=CPUs_gains_k.get)]  # Line 5
        else:
            top_CPUs = dict(sorted(CPUs_gains_k.items(), key=lambda item: item[1], reverse=True)[:top_N_CPUs])  # Line 7
            UEs_cell_edge.add(k)
        allocate_UE_to_APs(k, top_CPUs, D, AP_CPU_association, gain_over_noise_dB, threshold_LSF)  # Line 9

    from utilities import algo_result
    result = algo_result.AlgorithmResult(algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs, 0,
                                         AP_CPU_association, N, max_power_AP, gain_over_noise_dB, upsilon, kappa)
    if is_print_summary:
        utils.print_result_summary(result)

    return result


def allocate_UE_to_APs(k, top_CPUs, D, AP_CPU_association, gain_over_noise_dB, threshold_LSF):
    APs_top_CPUs = utils.get_APs_associated_with_CPUs(top_CPUs, AP_CPU_association)
    top_APs = utils_comms.get_top_APs_contribute_threshold_LSF(k, APs_top_CPUs, gain_over_noise_dB, threshold_LSF)
    D[top_APs, k] = 1


def calculate_z_scores(CPUs_gains_k_list):
    # Calculate the Z-scores for each number
    z_scores = np.abs((CPUs_gains_k_list - np.mean(CPUs_gains_k_list)) / np.std(CPUs_gains_k_list))
    return z_scores


def is_k_cell_center(z_scores, threshold_z, CPUs_gains_k_list):
    # Identify outliers based on the threshold
    outliers = [num for num, z_score in zip(CPUs_gains_k_list, z_scores) if z_score > threshold_z]
    # Determine if there is a single outlier that is greater than the rest
    if len(outliers) == 1 and outliers[0] == max(CPUs_gains_k_list):
        return True


def get_CPUs_gains(k, AP_CPU_association, gain_over_noise_dB):
    U = len(np.unique(AP_CPU_association))
    CPUs_gains_k = {i: 0 for i in range(U)}  # the sum gain of all APs associated with each CPU
    for u in range(U):
        APs_u = utils.get_APs_associated_with_CPU(u, AP_CPU_association)  # all APs associated with CPU u
        CPUs_gains_k[u] = np.sum(utils.db2pow(gain_over_noise_dB[APs_u, k]))
    return CPUs_gains_k

