# -*- coding: utf-8 -*-
"""
Solution for determining UE - AP association
Dynamic Cooperation Cluster (DCC): Björnson, E. and Sanguinetti, L., 2020. Scalable cell-free massive MIMO systems.
IEEE Transactions on Communications, 68(7), pp.4247-4261. (Fig. 3)
Original source code: https://github.com/emilbjornson/scalable-cell-free (MATLAB), translated to Python and adapted to
our setup for comparison with our proposed method.

@author: Phu Lai
"""
import time
import numpy as np
from utilities import constants, utils
from algorithms import pilot_assignment


def allocate_UEs_to_APs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, max_power_AP, upsilon, kappa,
                        p_UEs, is_print_summary):
    """
    Run the algorithm
    :param gain_over_noise_dB:      channel gain over noise in dB
    :param R:                       normalized spatial correlation matrix (with dimension N x N x L x K) using the local
                                    scattering model
    :param H:                       Matrix of dimension (L x N) x nb_realisations x K with the true channel realisations
    :param Np:                      Matrix of dimension N x nb_realisations x L x tau_p with the normalized noise
    :param AP_CPU_association:      list of CPU indices that each AP is associated with
    :param tau_p:                   number of pilots
    :param tau_c:                   length of coherence block
    :param max_power_AP:            maximum power of each AP (mW)
    :param upsilon:                 parameter for the centralised fractional power allocation
    :param kappa:                   parameter for the centralised fractional power allocation
    :param p_UEs:                   list of UEs' transmit power (mW)
    :param is_print_summary:        whether to print the result summary
    :return result:                 AlgorithmResult object
    """
    start = time.perf_counter()
    algo = constants.ALGO_SCF1
    pilot_index = pilot_assignment.assign_pilots_bjornson(tau_p, gain_over_noise_dB)
    threshold = constants.THRESHOLD_NON_MASTER_AP_SERVE_UE  # threshold for when a non-master AP decides to serve a UE
    K, L = gain_over_noise_dB.shape[1], gain_over_noise_dB.shape[0]  # number of UEs, APs
    N = R.shape[0]  # number of antennas
    D = np.zeros((L, K), dtype=int)  # AP-UE association matrix. This is the decision variable D used in our paper
    master_CPUs, master_APs = np.zeros(K, dtype=int), np.zeros(K, dtype=int)  # list of master CPUs/APs of each UE

    # Determine the master AP/CPU for UE k by looking for AP with best channel condition
    for k in range(K):
        master_AP_idx = np.argmax(gain_over_noise_dB[:, k])
        D[master_AP_idx, k] = 1
        master_APs[k] = master_AP_idx
        master_CPUs[k] = AP_CPU_association[master_AP_idx]
    # Each AP serves the UE with the strongest channel condition on each of the pilots where the AP isn't the master AP,
    # but only if its channel is not too weak compared to the master AP
    for l in range(L):
        for t in range(tau_p):
            pilot_UEs = [i for i, x in enumerate(pilot_index) if x == t]  # UEs on pilot t
            if sum(D[l, pilot_UEs]) == 0 and len(pilot_UEs) > 0:  # if AP l is not the master AP of any UEs on pilot t
                # Find the UE with pilot t that has the best channel
                gain_value, UE_index = max([(gain_over_noise_dB[l, i], i) for i in
                                            pilot_UEs])  # as K->inf, this doesn't also go to inf as each AP serves only tau_p UEs
                # Serve this UE if the channel is at most "threshold" weaker than the master AP's channel
                if gain_value - gain_over_noise_dB[master_APs[UE_index], UE_index] >= threshold:
                    D[l, UE_index] = 1

    execution_time = time.perf_counter() - start
    from utilities import algo_result
    result = algo_result.AlgorithmResult(algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs, execution_time,
                                         AP_CPU_association, N, max_power_AP, gain_over_noise_dB, upsilon, kappa)
    if is_print_summary:
        utils.print_result_summary(result)
    return result, master_CPUs, master_APs
