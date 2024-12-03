# -*- coding: utf-8 -*-
"""
UE-AP association method proposed in Cell-Free mMIMO Support in the O-RAN Architecture: A PHY Layer Perspective for 5G and Beyond Networks

@author: Phu Lai
"""
import time
import numpy as np
from algorithms import pilot_assignment
from utilities import constants, utils, utils_comms


def allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs,
                         max_power_AP, upsilon, kappa, top_N_CPUs, locations_UEs, locations_CPUs, dist_to_border, is_print_summary):
    start = time.perf_counter()
    algo = constants.ALGO_VIDA
    threshold_LSF = 1
    # number of UEs, APs, CPUs
    K, L, U = gain_over_noise_dB.shape[1], gain_over_noise_dB.shape[0], len(np.unique(AP_CPU_association))
    N = R.shape[0]  # number of antennas
    D = np.zeros((L, K), dtype=int)  # AP-UE association matrix. This is the decision variable D used in our paper
    pilot_index = pilot_assignment.assign_pilots_bjornson(tau_p, gain_over_noise_dB)
    distances_CPUs_UEs = utils.calculate_distances_CPUs_UEs(locations_CPUs, locations_UEs)
    cell_edge_UEs = utils_comms.classify_UEs_dist(locations_UEs, locations_CPUs, dist_to_border)
    for k in range(K):
        if k in cell_edge_UEs:
            allocate_cell_edge_UE(k, D, top_N_CPUs, AP_CPU_association, gain_over_noise_dB, threshold_LSF, distances_CPUs_UEs)
        else:
            allocate_cell_centre_UE(k, D, AP_CPU_association, threshold_LSF, gain_over_noise_dB, distances_CPUs_UEs)

    execution_time = time.perf_counter() - start
    from utilities import algo_result
    result = algo_result.AlgorithmResult(algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs, execution_time,
                                         AP_CPU_association, N, max_power_AP, gain_over_noise_dB, upsilon, kappa)
    if is_print_summary:
        utils.print_result_summary(result)

    return result


def allocate_cell_centre_UE(k, D, AP_CPU_association, threshold_LSF, gain_over_noise_dB, distances_CPUs_UEs):
    """
    Allocate cell-centre UE k
    """
    nearest_CPU = np.argmin(distances_CPUs_UEs[:, k])
    APs_best_CPU = utils.get_APs_associated_with_CPU(nearest_CPU, AP_CPU_association)
    if threshold_LSF < 1:  # select top APs that contributes at least threshold_LSF% total gain
        APs_best_CPU = utils_comms.get_top_APs_contribute_threshold_LSF(k, APs_best_CPU, gain_over_noise_dB,
                                                                        threshold_LSF)
    D[APs_best_CPU, k] = 1


def allocate_cell_edge_UE(k, D, top_N_CPUs, AP_CPU_association, gain_over_noise_dB, threshold_LSF, distances_CPUs_UEs):
    """
    Allocate cell-edge UE k
    """
    # get top top_N_CPUs CPUs that are closest to UE k
    top_CPUs = dict(sorted(enumerate(distances_CPUs_UEs[:, k]), key=lambda item: item[1])[:top_N_CPUs])
    if threshold_LSF < 1:
        APs_top_CPUs = utils.get_APs_associated_with_CPUs(top_CPUs, AP_CPU_association)
        top_APs = utils_comms.get_top_APs_contribute_threshold_LSF(k, APs_top_CPUs, gain_over_noise_dB, threshold_LSF)
        utils.get_CPUs_associated_with_APs(top_APs, AP_CPU_association)
        D[top_APs, k] = 1
    else:
        for u in top_CPUs.keys():
            APs_u = utils.get_APs_associated_with_CPU(u, AP_CPU_association)
            D[APs_u, k] = 1
