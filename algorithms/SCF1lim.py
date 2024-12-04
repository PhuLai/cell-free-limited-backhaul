# -*- coding: utf-8 -*-
"""
Solution for determining UE - AP association
Freitas, M.M., Souza, D.D., da Costa, D.B., Cavalcante, A.M., Valcarenghi, L., Borges, G.S., Rodrigues, R. and Costa, J.C., 2023.
Reducing Inter-CPU Coordination in User-Centric Distributed Massive MIMO Networks. IEEE Wireless Communications Letters.
Source code provided by the author Marx Freitas and translated to Python.

@author: Phu Lai
"""
import copy
import time
import numpy as np

from algorithms import pilot_assignment
from utilities import constants, utils
from algorithms import SCF1


def allocate_UEs_to_APs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, max_inter_UEs_per_CPU,
                        p_UEs, max_power_AP, upsilon, kappa, is_print_summary):
    start = time.perf_counter()
    algo = constants.ALGO_SCF1LIM
    U = len(np.unique(AP_CPU_association))  # number of CPUs
    N = R.shape[0]  # number of antennas
    gain_over_noise_mW = utils.db2pow(gain_over_noise_dB)

    result, master_CPUs, _ = SCF1.allocate_UEs_to_APs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c,
                                                             max_power_AP, upsilon, kappa, p_UEs, False)
    pilot_index = pilot_assignment.assign_pilots_bjornson(tau_p, gain_over_noise_dB)
    D = copy.deepcopy(result.D)

    for u in range(U):
        drop_inter_coordinated_UEs(u, D, AP_CPU_association, gain_over_noise_mW, max_inter_UEs_per_CPU, master_CPUs)

    execution_time = time.perf_counter() - start
    from utilities import algo_result
    result = algo_result.AlgorithmResult(algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs, execution_time,
                                         AP_CPU_association, N, max_power_AP, gain_over_noise_dB, upsilon, kappa)
    if is_print_summary:
        utils.print_result_summary(result)
    return result


def drop_inter_coordinated_UEs(u, D, AP_CPU_association, gain_over_noise_mW, max_inter_UEs_per_CPU, master_CPUs):
    """
    Drop inter-coordinated UEs that CPU u is serving not as primary CPU
    """
    # find UEs that CPU u is serving as primary CPU - UEs_u_primary
    UEs_u_primary = np.where(master_CPUs == u)[0]
    # find UEs that CPU u is serving regardless of whether it is primary or secondary CPU - UEs_u
    UEs_u = utils.get_UEs_associated_with_CPU(u, D, AP_CPU_association)
    # UEs_u - UEs_u_primary: UEs that CPU u is serving not as primary CPU
    UEs_u_non_primary = list(set(UEs_u) - set(UEs_u_primary))
    channelGainUEsIC_CPU = {}
    APs_u_UEs_u_non_primary = {}
    for k in UEs_u_non_primary:
        # find APs linked to CPU u that is serving UE k - APs_k_u
        APs_k_u = utils.get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
        APs_u_UEs_u_non_primary[k] = APs_k_u
        # initially, drop the connection of the UE k with all APs linked to CPU u
        D[APs_k_u, k] = 0
        # Compute the partial channel sum gain of the inter-coordinated UE. That is, considering only the APs that serve
        # the UE and are also linked to CPU u
        channelGainUEsIC_CPU[k] = sum(gain_over_noise_mW[APs_k_u, k])
    # Sort the channel gains of all inter-coordinated UEs that the CPU is "serving" in descending order
    channelGainUEsIC_CPU_sorted = dict(sorted(channelGainUEsIC_CPU.items(), key=lambda item: item[1], reverse=True))
    # Impose that the maximum numer of inter-coordinated UEs per CPU cannot exceed the maximum limit max_inter_UEs_per_CPU
    max_inter_UEs_per_CPU = min(max_inter_UEs_per_CPU, len(UEs_u_non_primary))
    # The CPU selects top max_inter_UEs_per_CPU UEs with strongest channel gains (large-scale fading coefficients)
    retained_IC_UEs = list(channelGainUEsIC_CPU_sorted.keys())[:max_inter_UEs_per_CPU]
    # print(f"CPU {u} retains {len(retained_IC_UEs)} inter-coordinated UEs")
    for k in retained_IC_UEs:
        D[APs_u_UEs_u_non_primary[k], k] = 1

