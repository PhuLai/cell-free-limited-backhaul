# -*- coding: utf-8 -*-
"""
Supporting functions related to communication

@author: Phu Lai
"""
import math
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.integrate import quad
from sklearn.cluster import KMeans
from utilities import utils, constants


def generate_setup(L, N, K, U, area, local_scattering_model, nb_realisations, tau_p, sigma_sf, constant_term,
                   alpha, noise_variance_dBm, ASD_deg, antenna_spacing, existing_data, is_print_summary, save_to_pickle):
    """
    generate setup of APs, CPUs, and UEs
    is_hotspot_enabled, hotspot_area_pct, pct_UEs_moved_into_hotspot,
    :param L:                       number of APs
    :param N:                       number of antennas per AP
    :param K:                       number of UEs
    :param U:                       number of CPUs
    :param area:                    size of the coverage area (km2)
    :param local_scattering_model:  method of generating local scattering matrix.
                                    See helpers.generate_local_scattering_matrix_R for more details
    :param nb_realisations:         number of channel realisations
    :param tau_p:                   number of orthogonal pilots
    :param sigma_sf:                standard deviation of shadow fading
    :param constant_term:           average channel gain in dB at a reference distance of 1 meter. Note that -35.3 dB
                                    corresponds to -148.1 dB at 1 km, using pathloss exponent 3.76
    :param alpha:                   path loss exponent
    :param noise_variance_dBm:      noise variance in dBm
    :param ASD_deg:                 angular standard deviation around the nominal angle (measured in degrees)
    :param antenna_spacing:         antenna spacing (in number of wavelengths)
    :param existing_data:           None to use new data that are randomised.
                                    MATLAB data for comparison between results produced by equivalent MATLAB code.
                                    PYTHON data so that results can be reproduced exactly between runs for easier debugging
    :param is_print_summary:        whether to print the summary of the generated setup
    :param save_to_pickle:          whether to save the generated data to pickle files for reproducibility
    :return locations_APs:      locations of APs
    :return locations_UEs:      locations of UEs
    :return distances_APs_UEs:  distances between APs and UEs
    :return AP_CPU_association: list of CPU indices that each AP is associated with
    :return C_CPUs:             list of CPUs' resource capacity (list of tuples of integers)
    :return gain_over_noise_dB: channel gain over noise in dB
    :return R:                  normalized spatial correlation matrix (with dimension (N x N x L x K) using the local
                                scattering model
    :return CPUs_locations:     locations of CPUs
    :return H:                  Matrix of dimension (L x N) x nb_realisations x K with the true channel realisations
    :return Np:                 Matrix of dimension N x nb_realisations x L x tau_p with the normalized noise
    """
    if existing_data == constants.USE_EXISTING_DATA_PYTHON:
        locations_APs = utils.load_pickle(constants.PICKLE_LOCATION_APS)
        locations_UEs = utils.load_pickle(constants.PICKLE_LOCATION_UES)
        locations_CPUs = utils.load_pickle(constants.PICKLE_LOCATIONS_CPUS)
        AP_CPU_association = utils.load_pickle(constants.PICKLE_AP_CPU_ASSOCIATION)
        gain_over_noise_dB = utils.load_pickle(constants.PICKLE_GAIN_OVER_NOISE_dB)
        R = utils.load_pickle(constants.PICKLE_R)
        H = utils.load_pickle(constants.PICKLE_H)
        Np = utils.load_pickle(constants.PICKLE_NP)
    else:
        square_length = round(math.sqrt(area) * 1e3)  # meters

        # Random AP and UE locations with uniform distribution
        locations_APs = (np.random.rand(L) + 1j * np.random.rand(L)) * square_length
        locations_UEs = (np.random.rand(K) + 1j * np.random.rand(K)) * square_length
        locations_CPUs, AP_CPU_association = generate_CPU_setup(locations_APs, U)

        # compute Euclidean distance assuming APs are constants.AP_HEIGHT meters above UEs
        data_APs = pd.DataFrame({'x': locations_APs.real, 'y': locations_APs.imag})
        data_UEs = pd.DataFrame({'x': locations_UEs.real, 'y': locations_UEs.imag})
        distances_APs_UEs = np.sqrt(constants.AP_HEIGHT ** 2 +
                                    np.sum((data_APs[['x', 'y']].to_numpy()[:, np.newaxis] - data_UEs[
                                        ['x', 'y']].to_numpy()) ** 2, axis=2))
        # compute channel gain
        gain_over_noise_dB = np.zeros(distances_APs_UEs.shape)
        for k in range(K):
            sf = np.random.randn(distances_APs_UEs[:, k].size)
            # 3GPP LTE model Further Advancements for E-UTRA Physical Layer Aspects (Release 9), Standard 3GPP TS 36.814, 2010.
            gain_over_noise_dB[:, k] = constant_term - alpha * 10 * np.log10(
                distances_APs_UEs[:, k]) + sigma_sf * sf - noise_variance_dBm

        # generate normalized spatial correlation matrix. VERY slow...
        R = generate_normalised_scattering_matrix(locations_APs, locations_UEs, N, gain_over_noise_dB,
                                                  local_scattering_model, ASD_deg, antenna_spacing)

        # Generate uncorrelated Rayleigh fading channel realizations
        # H: Matrix of dimension (L x N) x nb_realisations x K with the true channel realisations
        H = (np.random.randn(L * N, nb_realisations, K) + 1j * np.random.randn(L * N, nb_realisations, K))
        # Go through all channels and apply the spatial correlation matrices to the uncorrelated channel realizations
        for l in range(L):
            for k in range(K):
                R_sqrt = linalg.sqrtm(R[:, :, l, k])
                H[l * N:(l + 1) * N, :, k] = np.sqrt(0.5) * R_sqrt @ H[l * N:(l + 1) * N, :, k]

        # Generate realizations of normalized noise
        Np = np.sqrt(0.5) * (np.random.randn(N, nb_realisations, L, tau_p) + 1j * np.random.randn(N, nb_realisations, L, tau_p))

        # save data to pickle files to create reproducible results
        if save_to_pickle:
            utils.to_pickle(locations_APs, constants.PICKLE_LOCATION_APS)
            utils.to_pickle(locations_UEs, constants.PICKLE_LOCATION_UES)
            utils.to_pickle(locations_CPUs, constants.PICKLE_LOCATIONS_CPUS)
            utils.to_pickle(AP_CPU_association, constants.PICKLE_AP_CPU_ASSOCIATION)
            utils.to_pickle(gain_over_noise_dB, constants.PICKLE_GAIN_OVER_NOISE_dB)
            utils.to_pickle(R, constants.PICKLE_R)
            utils.to_pickle(H, constants.PICKLE_H)
            utils.to_pickle(Np, constants.PICKLE_NP)
    if is_print_summary:
        print(f"=========================================\n"
              f"Summary of generated setup:\n"
              f"{len(locations_UEs)} UEs, {len(locations_APs)} APs ({int(H.shape[0] / len(locations_APs))} antennas per AP)"
              f"associated with {len(locations_CPUs)} CPUs \n"
              f"Square area: {area} squared meters\n"
              f"Number of channel realisations: {H.shape[1]}\n")
    return locations_APs, locations_UEs, AP_CPU_association, gain_over_noise_dB, R, locations_CPUs, H, Np


def generate_CPU_setup(locations_APs, U):
    """
    Generate CPU setup
    :param locations_APs:   locations of APs
    :param U:               number of CPUs
    :return CPUs_locations:     locations of CPUs
    :return AP_CPU_association: list of CPU indices that each AP is associated with
    :return C_CPUs:             list of CPUs' resource capacity (tuple of integers)
    """
    # get AP-CPU association matrix
    data_APs = pd.DataFrame({'x': locations_APs.real, 'y': locations_APs.imag})
    kmeans = KMeans(n_clusters=U, random_state=0, n_init="auto").fit(data_APs[['x', 'y']].values)
    locations_CPUs = kmeans.cluster_centers_
    AP_CPU_association = kmeans.labels_
    data_APs['CPU'] = AP_CPU_association
    return locations_CPUs, AP_CPU_association


def generate_normalised_scattering_matrix(locations_APs, locations_UEs, N, gain_over_noise_dB, local_scattering_model,
                                          ASD_deg, antenna_spacing):
    """
    Generate normalized spatial correlation matrix as part of generate_setup()
    :param locations_APs:           locations of APs (array of complex numbers)
    :param locations_UEs:           locations of UEs (array of complex numbers)
    :param N:                       number of antennas per AP
    :param gain_over_noise_dB:      channel gain over noise in dB
    :param local_scattering_model:  method of generating local scattering matrix.
                                    See generate_local_scattering_matrix() func for more details
    :param ASD_deg:                 angular standard deviation around the nominal angle (measured in degrees)
    :param antenna_spacing:         antenna spacing (in number of wavelengths)
    :return:                        normalized spatial correlation matrix (with dimension N x N x L x K) using the local
                                    scattering model
    """
    K = locations_UEs.shape[0]
    L = locations_APs.shape[0]
    # normalized spatial correlation matrix
    R = np.zeros((N, N, L, K), dtype=np.complex128)
    for k in range(K):
        # Go through all APs to generate normalized spatial correlation matrix using the local scattering model
        for l in range(L):
            theta = np.angle(locations_UEs[k] - locations_APs[l])  # nominal angle between UE k and AP l
            R[:, :, l, k] = (utils.db2pow(gain_over_noise_dB[l, k])
                             * generate_local_scattering_matrix(N, theta, ASD_deg, antenna_spacing,
                                                                local_scattering_model))
    return R


def generate_local_scattering_matrix(N, theta, ASD_deg, antenna_spacing, model):
    """
    Generate the spatial correlation matrix for the local scattering model as part of generate_setup().
    Gaussian angular distribution is assumed.
    Some logics translated from the original MATLAB code at https://github.com/emilbjornson/scalable-cell-free/blob/master/functionRlocalscattering.m
    to Python with some minor changes.

    :param N:       number of antennas
    :param theta:   nominal angle
    :param ASD_deg:  Angular standard deviation around the nominal angle (unit: degree)
    :param antenna_spacing: Antenna spacing (in number of wavelengths)
    :param model:   model for generating R, either '2.23' or '2.24', corresponding to E.q 2.23 and 2.24 in Massive MIMO
                    Networks: Spectral, Energy, and Hardware Efficiency (https://doi.org/10.1561/2000000093)
    :return:    N x N spatial correlation matrix
    """
    ASD = ASD_deg * math.pi / 180
    distances = antenna_spacing * np.arange(N)  # Distance from the first antenna to the column-th antenna
    first_row = np.zeros((N, 1), dtype=np.complex128)

    if model == constants.LOCAL_SCATTERING_MODEL_2_23:
        # OPT 1: https://github.com/emilbjornson/scalable-cell-free/blob/master/functionRlocalscattering.m
        # E.q 2.23 in Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency
        # Very high computational complexity in simulations
        for i in range(N):
            integrable_func = lambda Delta: np.exp(1j * 2 * np.pi * distances[i] * np.sin(theta + Delta)) \
                                            * np.exp(-Delta ** 2 / (2 * ASD ** 2)) / (np.sqrt(2 * np.pi) * ASD)
            real_part = lambda Delta: np.real(integrable_func(Delta))
            imag_part = lambda Delta: np.imag(integrable_func(Delta))
            real_result, _ = quad(real_part, -20 * ASD, 20 * ASD)
            imag_result, _ = quad(imag_part, -20 * ASD, 20 * ASD)
            first_row[i] = real_result - 1j * imag_result

    elif model == constants.LOCAL_SCATTERING_MODEL_2_24:
        # OPT 2: https://github.com/emilbjornson/power-allocation-cell-free/blob/main/functionRlocalscattering.py
        # E.q 2.24 in Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency
        # Lower computational complexity
        first_row = np.exp(1j * 2 * math.pi * distances * np.sin(theta)) \
                    * np.exp(-ASD ** 2 / 2 * (2 * math.pi * distances * np.cos(theta)) ** 2)

    lsm = linalg.toeplitz(first_row)
    return lsm


def generate_channel_estimate(R, H, Np, tau_p, pilot_index, p_UEs):
    """
    Generate channel estimate
    Equation numbers below correspond to equations in Scalable Cell-Free Massive MIMO Systems, Björnson, E. and
    Sanguinetti, L., 2020.
    Code translated from the original MATLAB code at: https://github.com/emilbjornson/scalable-cell-free/blob/master/functionChannelEstimates.m
    to Python with some changes (heterogeneous UE transmit power, etc.)

    :param R:                   Matrix with dimension N x N x L x K where (:,:,l,k) is the spatial correlation matrix
                                between AP l and UE k in setup n, normalized by noise
    :param H:                   Matrix of dimension (L x N) x nb_realisations x K with the true channel realisations
    :param Np:                  Matrix of dimension N x nb_realisations x L x tau_p with the normalized noise
    :param tau_p:               number of orthogonal pilots
    :param pilot_index:         list of pilots for each UE
    :param p_UEs                transmit power of all UEs
    :return H_hat:              Matrix with dimension (L x N) x nb_realisations x K where (:,n,k) is the estimated
                                collective channel to UE k in channel realisation n
    :return R_cee:              Matrix with dimension (N x N) x L x K where (:,:,l,k) is the spatial correlation matrix
                                of the channel estimation error between AP l and UE k in setup n, normalized by noise
    """
    L, N, K = R.shape[2], R.shape[0], R.shape[3]  # number of APs, number of antennas per AP, number of UEs

    # identity matrix of size N x N
    eyeN = np.eye(N)
    H_hat = np.zeros(H.shape[:3], dtype=np.complex128)
    # Matrix with dimension (N x N) x L x K where (:,:,l,k) is the spatial correlation matrix of the estimate between
    # AP l and UE k in setup n, normalized by noise
    R_e = np.zeros(R.shape, dtype=np.complex128)
    R_cee = np.zeros(R.shape, dtype=np.complex128)

    for l in range(L):
        start_idx = l * N
        end_idx = (l + 1) * N
        for t in range(tau_p):
            # Compute processed pilot signal for all UEs that use pilot t
            H_sub = H[start_idx:end_idx, :, t == pilot_index]
            UEs_pilot_t_idx = [i for i, x in enumerate(pilot_index) if x == t]  # get index of UEs that use pilot t
            # E.q (2)
            yp = tau_p * np.sum(H_sub * np.sqrt(p_UEs[UEs_pilot_t_idx]), axis=2) + np.sqrt(tau_p) * Np[:, :, l, t]
            # E.q (4)
            Psi = tau_p * np.sum(R[:, :, l, t == pilot_index] * p_UEs[UEs_pilot_t_idx], axis=2) + eyeN
            # Go through all UEs that use pilot t
            for UE_pilot_t_idx in UEs_pilot_t_idx:
                # Compute the MMSE estimate
                # R_Psi = linalg.lu_solve(linalg.lu_factor(Psi), R[:, :, l, UE_pilot_t_idx])  # Psi^{-1}R
                R_Psi = np.dot(R[:, :, l, UE_pilot_t_idx], np.linalg.inv(Psi))  # identical to matlab, not as precise
                H_hat[start_idx:end_idx, :, UE_pilot_t_idx] = np.sqrt(p_UEs[UE_pilot_t_idx]) * R_Psi @ yp  # E.q (3)
                # Compute the spatial correlation matrix of the estimate
                R_e[:, :, l, UE_pilot_t_idx] = p_UEs[UE_pilot_t_idx] * tau_p * R_Psi @ R[:, :, l, UE_pilot_t_idx]
                # Compute the spatial correlation matrix of the estimation error
                R_cee[:, :, l, UE_pilot_t_idx] = R[:, :, l, UE_pilot_t_idx] - R_e[:, :, l, UE_pilot_t_idx]
    return H_hat, R_cee


def calculate_SEs_downlink_uplink(D, R, H, Np, tau_c, tau_p, p_UEs, max_power_AP, pilot_index, gain_over_noise_dB,
                                  upsilon, kappa):
    """
    Calculate downlink and uplink spectral efficiency.
    Combining scheme: partial MMSE (P-MMSE) proposed in Björnson, E. and Sanguinetti, L., 2020.
    Scalable cell-free massive MIMO systems. IEEE Transactions on Communications, 68(7), pp.4247-4261.
    Reference book: Foundations of User-Centric Cell-Free Massive MIMO
    Code translated from the original MATLAB code at https://github.com/emilbjornson/cell-free-book
        to Python with some changes
    """
    H_hat, R_cee = generate_channel_estimate(R, H, Np, tau_p, pilot_index, p_UEs)

    K, L, N, nb_realisations = R.shape[3], D.shape[0], R_cee.shape[0], H_hat.shape[1]
    prelog_factor = 1 - tau_p / tau_c  # Compute the prelog factor assuming only uplink/downlink data transmission
    # Obtain the diagonal matrix with UE transmit powers as its diagonal entries
    power_matrix = np.diag(p_UEs)

    # Scale R_cee by power coefficients
    R_cee_p = np.zeros(R_cee.shape, dtype=np.complex128)
    for k in range(K):
        R_cee_p[:, :, :, k] = p_UEs[k] * R_cee[:, :, :, k]

    # simulation results
    signal_P_MMSE = np.zeros((K, K), dtype=np.complex128)
    signal2_P_MMSE = np.zeros((K, K))
    scaling_P_MMSE = np.zeros((L, K))
    SEs_P_MMSE_UL = np.zeros((K, 1))

    for k in range(K):
        # getting the set of APs serving this UE k
        serving_APs = utils.get_APs_serving_UEs(D, [k])
        nb_serving_APs = len(serving_APs)
        # getting UEs being served by partially the same set of APs as UE k - interfering UEs
        if nb_serving_APs > 0:
            interfering_UEs_mask = sum(D[serving_APs, :], 0) >= 1
        else:  # generate an array of K x False - UE k is not allocated yet
            interfering_UEs_mask = np.zeros(K, dtype=bool)

        # Extract channel realisations and estimation error correlation matrices for the APs that involved in the service of UE k
        H_all_j_active = np.zeros((N * nb_serving_APs, K), dtype=np.complex128)
        H_hat_all_j_active = np.zeros((N * nb_serving_APs, K), dtype=np.complex128)
        R_cee_tot_blk = np.zeros((N * nb_serving_APs, N * nb_serving_APs), dtype=np.complex128)
        R_cee_tot_blk_partial = np.zeros((N * nb_serving_APs, N * nb_serving_APs), dtype=np.complex128)

        for channel_realisation in range(nb_realisations):
            for l, AP in enumerate(serving_APs):
                start_idx, end_idx = l * N, (l + 1) * N
                H_all_j_active[start_idx:end_idx, :] = np.reshape(H[AP * N:(AP + 1) * N, channel_realisation, :],
                                                                  [N, K])
                H_hat_all_j_active[start_idx:end_idx, :] = np.reshape(
                    H_hat[AP * N:(AP + 1) * N, channel_realisation, :], [N, K])
                R_cee_tot_blk[start_idx:end_idx, start_idx:end_idx] = np.sum(R_cee_p[:, :, AP, :], axis=2)
                R_cee_tot_blk_partial[start_idx:end_idx, start_idx:end_idx] = np.sum(
                    R_cee_p[:, :, AP, interfering_UEs_mask], axis=2)
            # ===================== start calculation for UL =====================
            matrix1ul = p_UEs[k] * (p_UEs[k] * (H_hat_all_j_active[:, interfering_UEs_mask] @ H_hat_all_j_active[:,
                                                                                              interfering_UEs_mask].conj().T + R_cee_tot_blk_partial /
                                                p_UEs[k])
                                    + np.eye(nb_serving_APs * N))
            matrix2ul = H_hat_all_j_active[:, k]
            v_ul = linalg.lstsq(matrix1ul, matrix2ul, lapack_driver='gelsy', check_finite=False)[0]
            numerator = p_UEs[k] * abs(v_ul.conj().T @ H_hat_all_j_active[:, k]) ** 2
            denominator = (p_UEs[k] * linalg.norm(v_ul.conj().T @ H_hat_all_j_active) ** 2
                           + v_ul.conj().T @ (
                                       p_UEs[k] * R_cee_tot_blk / p_UEs[k] + np.eye(nb_serving_APs * N)) @ v_ul
                           - numerator)
            if denominator == 0j:
                denominator = 1  # just a hack to avoid division by zero below
            SEs_P_MMSE_UL[k] = SEs_P_MMSE_UL[k] + prelog_factor * np.real(
                np.log2(1 + numerator / denominator)) / nb_realisations
            # ===================== end calculation for UL =====================

            # =====================  start calculation for DL =====================
            H_p_hat_all_j_active = H_hat_all_j_active @ np.sqrt(power_matrix)
            # Compute P-MMSE combining/precoding
            matrix1dl = (H_p_hat_all_j_active[:, interfering_UEs_mask] @ H_p_hat_all_j_active[:,
                                                                         interfering_UEs_mask].conj().T) + R_cee_tot_blk_partial + np.eye(
                nb_serving_APs * N)
            matrix2dl = H_p_hat_all_j_active[:, k]
            v_dl = linalg.lstsq(matrix1dl, matrix2dl, lapack_driver='gelsy', check_finite=False)[0] * np.sqrt(p_UEs[k])

            # Compute realizations of the terms inside the expectations of the signal and interference terms in the SE
            # expressions and update Monte-Carlo estimates
            tempor = H_all_j_active.conj().T @ v_dl
            signal_P_MMSE[:, k] = signal_P_MMSE[:, k] + (H_all_j_active[:, k].conj().T @ v_dl) / nb_realisations
            signal2_P_MMSE[:, k] = signal2_P_MMSE[:, k] + np.abs(tempor) ** 2 / nb_realisations

            for l, AP in enumerate(serving_APs):
                start_idx, end_idx = l * N, (l + 1) * N
                v2 = v_dl[start_idx:end_idx]
                scaling_P_MMSE[AP, k] = scaling_P_MMSE[AP, k] + np.sum(abs(v2) ** 2, axis=0) / nb_realisations

    # check inteference
    ck_total_1, ck_total_2 = 0, 0
    # Compute the terms in (7.13)-(7.15)
    bk_PMMSE = np.zeros((K, 1))
    ck_PMMSE = signal2_P_MMSE.T
    for k in range(K):
        bk_PMMSE[k] = np.abs(signal_P_MMSE[k, k]) ** 2
        ck_PMMSE[k, k] = ck_PMMSE[k, k] - bk_PMMSE[k][0]
        ck_total_1 += ck_PMMSE[k, k]
    # Obtain the scaling factors for the precoding vectors and scale the terms in (7.13)-(7.15) accordingly
    sigma2_PMMSE = np.sum(scaling_P_MMSE, axis=0).T
    for k in range(K):
        bk_PMMSE[k] = bk_PMMSE[k] / sigma2_PMMSE[k]
        ck_PMMSE[k, :] = ck_PMMSE[k, :] / sigma2_PMMSE[k]
        ck_total_2 += ck_PMMSE[k, :]

    # Scale the expected values of the norm squares of the portions of the centralized precoding in accordance with the
    # normalized precoding vectors in (7.16)
    portion_scaling_P_MMSE = scaling_P_MMSE / np.tile(np.sum(scaling_P_MMSE, axis=0), (L, 1))  # should sum to K
    # Compute the fractional power allocation for centralized precoding according to (7.43) with different \upsilon and \kappa parameters
    p_UEs_FPA = allocate_power_centralised_DL(gain_over_noise_dB, D, max_power_AP, portion_scaling_P_MMSE, upsilon,
                                              kappa)
    # Compute SEs according to Theorem 6.1
    SEs_P_MMSE_DL = prelog_factor * np.log2(1 + np.multiply(bk_PMMSE, p_UEs_FPA) / (np.dot(ck_PMMSE.T, p_UEs_FPA) + 1))
    SEs_P_MMSE_DL = [SE[0] for SE in SEs_P_MMSE_DL]
    SEs_P_MMSE_UL = [SE[0] for SE in SEs_P_MMSE_UL]

    return SEs_P_MMSE_DL, SEs_P_MMSE_UL


def get_top_APs_contribute_threshold_LSF(k, APs, gain_over_noise_dB, threshold_LSF):
    """
    select top APs that contributes at least threshold_LSF% total gain
    """
    APs_gain = utils.db2pow(gain_over_noise_dB[APs, k])
    APs_gain_sorted = np.sort(APs_gain)[::-1]  # Sort in descending order to prioritize higher gains
    sum_gain = np.sum(APs_gain)  # sum gain
    threshold = threshold_LSF * sum_gain  # Set the threshold for selection (80%,... of the total value)
    cumulative_sum = 0
    selected_items = []
    for item in APs_gain_sorted:  # Greedy selection process
        cumulative_sum += item
        selected_items.append(item)
        if cumulative_sum >= threshold:
            break
    return [APs[i] for i in np.where(np.isin(APs_gain, selected_items))[0]]


def allocate_power_centralised_DL(gain_over_noise_dB, D, max_power_AP, portion_scaling_P_MMSE, upsilon, kappa):
    """
    :param portionScaling_P_MMSE:  Matrix with dimension L x K where (l,k) is the expected value of the norm square of the portion
                                    of the normalized centralized transmit precoder of UE k
    """
    K = gain_over_noise_dB.shape[1]
    gain_over_noise_mW = utils.db2pow(gain_over_noise_dB)
    poww = np.zeros((K, 1))
    maxPow = np.zeros((K, 1))
    # Prepare to store the normalization factor in (7.43)
    normalization_factor = np.zeros((K, 1))

    for k in range(K):
        APs_k = utils.get_APs_serving_UEs(D, [k])
        # Compute the numerator of (7.43)
        poww[k] = np.sum(gain_over_noise_mW[APs_k, k]) ** upsilon
        poww[k] = poww[k] / max(portion_scaling_P_MMSE[APs_k, k]) ** kappa
        # Compute \omega_k in (7.41)
        maxPow[k] = np.max(portion_scaling_P_MMSE[APs_k, k])

    for k in range(K):
        APs_k = utils.get_APs_serving_UEs(D, [k])
        for l in APs_k:
            UEs_l = utils.get_UEs_served_by_APs([l], D)
            # Compute the normalization factor in (7.43)
            tempor_scalar = maxPow[UEs_l].T @ poww[UEs_l] / max_power_AP
            normalization_factor[k] = max(normalization_factor[k], tempor_scalar[0])
    # Normalize the numerator terms in (7.43) to obtain \rho_k
    p_coefficients = poww / normalization_factor
    validate_AP_power_constraint_DL(D, p_coefficients, max_power_AP, portion_scaling_P_MMSE)
    return p_coefficients


def validate_AP_power_constraint_DL(D, p_UEs_FPA, max_power_AP, portion_scaling_P_MMSE):
    """
    Validate the power constraint for each AP
    :param D:
    :param p_UEs_FPA:
    :param max_power_AP:
    :param portion_scaling_P_MMSE:
    :return:
    """
    L = D.shape[0]
    for l in range(L):
        UEs_l = utils.get_UEs_served_by_APs([l], D)
        total_power = sum([p_UEs_FPA[k] * portion_scaling_P_MMSE[l,k] for k in UEs_l])
        if total_power > max_power_AP and not math.isclose(total_power, max_power_AP, rel_tol=1e-9):
            print(f"AP {l} ({total_power}mW) exceeds the maximum power constraint ({max_power_AP}mW)!")
