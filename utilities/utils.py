# -*- coding: utf-8 -*-
"""
Supporting functions defined here

@author: Phu Lai
"""
import itertools
import math
import numpy as np
import pandas as pd
from utilities import algo_result, constants
import os
import pickle
import scipy.io


# @jit(nopython=True)
def db2pow(xdb):
    """Convert decibels (dB) to power (mW)
    :param xdb: decibels (dB)
    :return: power (mW)
    """
    return 10. ** (xdb / 10.)


def convert_SE_to_data_rate(SEs, bandwidth):
    """Convert spectral efficiency (bit/s/Hz) to data rate (MB/s)
    :param SEs:         list of spectral efficiency (SE)
    :param bandwidth:   communication bandwidth (Hz)
    :return:            data rate
    """
    return SEs * bandwidth / 8e6  # convert from bit/s/Hz to MB/s


def load_mat(mat_file):
    """Load variable from MATLAB mat file
    :param mat_file: .mat file name
    :return: variable
    """
    mat_path = os.path.join(constants.PATH_DATASET, mat_file)
    return scipy.io.loadmat(mat_path)


def load_pickle(pickle_file):
    """Load variable from pickle file
    :param pickle_file: pickle file name
    :return: variable
    """
    pickle_path = os.path.join(constants.PATH_DATASET, pickle_file)
    with open(pickle_path, 'rb') as handle:
        return pickle.load(handle)


def to_pickle(data, pickle_file):
    """Save variable to pickle file
    :param data:        variable to save
    :param pickle_file: pickle file name
    :return: None
    """
    pickle_path = os.path.join(constants.PATH_DATASET, pickle_file)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_result_to_csv(result: algo_result.AlgorithmResult, area, L, N, U, threshold_LSF, pilot_allocation_method,
                       upsilon, kappa, threshold_z, top_N_CPUs, path_results):
    """
    Add result to CSV file
    """
    if path_results is not None:
        data = {constants.HEADER_AREA:                      area,
                constants.HEADER_THRESHOLD_LSF:             threshold_LSF,
                # constants.HEADER_PILOT_ALLOCATION_METHOD:   pilot_allocation_method,
                constants.HEADER_NB_AP:                     L,
                constants.HEADER_NB_ANTENNA:                N,
                constants.HEADER_NB_CPU:                    U,
                constants.HEADER_UPSILON:                   upsilon,
                constants.HEADER_KAPPA:                     kappa,
                constants.HEADER_THRESHOLD_Z:               threshold_z,
                constants.HEADER_TOPNCPUs:                  top_N_CPUs,
                constants.HEADER_ALGO:                      result.algo,
                constants.HEADER_NB_UE:                     f"{result.D.shape[1]}",
                # constants.HEADER_SUM_SE_UL:                 f"{sum(result.SEs_UL): .2f}",
                # constants.HEADER_SUM_SE_DL:                 f"{sum(result.SEs_DL): .2f}",
                constants.HEADER_AVG_SE_UL:                 f"{sum(result.SEs_UL)/result.D.shape[1]: .2f}",
                constants.HEADER_AVG_SE_DL:                 f"{sum(result.SEs_DL)/result.D.shape[1]: .2f}",
                constants.HEADER_FAIRNESS_SE_UL:            f"{result.fairness_SE_UL: .2f}",
                constants.HEADER_FAIRNESS_SE_DL:            f"{result.fairness_SE_DL: .2f}",
                # constants.HEADER_CPU_USAGE:                 f"{result.times_CPUs_used}",
                constants.HEADER_MIN_UE_PER_AP:             f"{result.min_UEs_per_AP}",
                constants.HEADER_MAX_UE_PER_AP:             f"{result.max_UEs_per_AP}",
                constants.HEADER_AVG_UE_PER_AP:             f"{result.avg_UEs_per_AP}",
                constants.HEADER_MIN_AP_PER_UE:             f"{result.min_APs_per_UE}",
                constants.HEADER_MAX_AP_PER_UE:             f"{result.max_APs_per_UE}",
                constants.HEADER_AVG_AP_PER_UE:             f"{result.avg_APs_per_UE}",
                constants.HEADER_MIN_CPU_PER_UE:            f"{result.min_CPUs_per_UE}",
                constants.HEADER_MAX_CPU_PER_UE:            f"{result.max_CPUs_per_UE}",
                constants.HEADER_AVG_CPU_PER_UE:            f"{result.avg_CPUs_per_UE}",
                constants.HEADER_MIN_UE_PER_CPU:            f"{result.min_UEs_per_CPU}",
                constants.HEADER_MAX_UE_PER_CPU:            f"{result.max_UEs_per_CPU}",
                constants.HEADER_AVG_UE_PER_CPU:            f"{result.avg_UEs_per_CPU}",
                constants.HEADER_NB_COOR_UE:                f"{result.nb_coordinated_UEs}",
                constants.HEADER_BACKHAUL_LOAD_UL:          f"{result.backhaul_load_UL}",
                constants.HEADER_BACKHAUL_LOAD_DL:          f"{result.backhaul_load_DL}",
                constants.HEADER_TIME:                      f"{result.execution_time: .2f}",
                constants.HEADER_SEs_UL:                    f"{result.SEs_UL}",
                constants.HEADER_SEs_DL:                    f"{result.SEs_DL}"}

        df = pd.DataFrame(data=data, index=[0])
        df.to_csv(path_results, mode='a', header=not os.path.exists(path_results), index=False)


def allocate_UE_to_CPU_APs(k, k_requirement, u, D, APs, C_CPUs_live):
    """
    Allocate a UE to APs
    NOTE: 1 UE - 1 CPU.
    This func will unallocate the UE from its current CPU/APs, if any

    - update UE-APs association matrix D
    - update CPU capacity C_CPUs
    - update UEs_resource_requirements_satisfied

    :param k:                   UE index
    :param k_requirement:        UE k's resource requirement
    :param u:                   CPU index
    :param D:                   UE - AP association matrix D
    :param APs:                 list of APs to allocate UE k to
    :param C_CPUs_live:              list of CPUs' resource capacity (list of tuples of integers)
    :param UEs_resource_requirements_satisfied: list of K booleans indicating if UEs' resource requirements are satisfied
    :return D:                  UE - AP association matrix D
    """
    D[:, k] = 0  # unallocate UE k from its current CPU/APs, if any (just to be safe)
    D[APs, k] = 1
    # check if the selected CPU u has enough resource to allocate to UE k
    is_k_requirement_satisfied = all(
        CPU_resource >= k_resource for CPU_resource, k_resource in zip(C_CPUs_live[u], k_requirement))
    if is_k_requirement_satisfied:
        C_CPUs_live[u] = (C_CPUs_live[u][0] - k_requirement[0], C_CPUs_live[u][1] - k_requirement[1], C_CPUs_live[u][2] - k_requirement[2])

    return D, is_k_requirement_satisfied, C_CPUs_live


def get_UEs_associated_with_CPU(u, D, AP_CPU_association):
    """
    Get all UEs associated with a CPU, i.e., all UEs associated with all APs associated with the CPU
    :param u:                   CPU index
    :param D:                   UE - AP association matrix D
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :return UEs_u:              list of UEs associated with CPU u
    """
    APs_u = np.where(AP_CPU_association == u)[0]  # get all APs associated with CPU u
    UEs_u = set()  # a set of UEs associated with CPU u
    for l in APs_u:
        UEs_u.update(np.where(D[l, :] == 1)[0])  # get all UEs associated with AP_u
    return UEs_u


def get_UEs_served_by_APs(APs, D):
    """
    Get all UEs associated with a list of APs
    :param APs: list of APs
    :param D:   UE - AP association matrix D
    :return:    list of UEs associated with APs
    """
    UEs = set()  # a set of UEs associated with APs
    for AP in APs:
        UEs.update(np.where(D[AP, :] == 1)[0])  # get all UEs associated with AP
    return list(UEs)


def get_UEs_partially_associated_with_APs(APs, D):
    """
    Get UEs partially associated with a list of APs. i.e., at least 2 APs in common, unlike get_UEs_associated_with_APs()
    :param APs: list of APs
    :param D:   UE - AP association matrix D
    :return:    list of UEs associated with APs
    """
    return np.sum(D[APs, :], axis=0) >= 1


def get_UEs_inter_coordinated(u, D, AP_CPU_association):
    UEs_u = get_UEs_associated_with_CPU(u, D, AP_CPU_association)
    inter_coordinated_UEs = set()
    for k in UEs_u:
        CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)
        if len(CPUs_k) > 1:
            inter_coordinated_UEs.add(k)
    return list(inter_coordinated_UEs)


def count_UEs_unallocated(D):
    """
    Count the number of UEs that are not allocated to any AP
    :param D:   UE - AP association matrix D
    :return:    number of UEs that are not allocated to any AP
    """
    nb_users_unallocated = np.count_nonzero(np.count_nonzero(D, axis=0) == 0)
    return nb_users_unallocated


def count_UEs_allocated(D):
    """
    Count the number of UEs that are allocated to at least one AP
    :param D:   UE - AP association matrix D
    :return:    number of UEs that are allocated to at least one AP
    """
    nb_users_allocated = np.count_nonzero(np.count_nonzero(D, axis=0) > 0)
    return nb_users_allocated


def get_copilot_UEs_given_UE(k, pilot_index):
    """
    Get all UEs that are assigned the same pilot as UE k
    :param k:           UE index
    :param pilot_index: list of pilots for each UE
    :return:            list of UEs that are assigned the same pilot as UE k
    """
    return [i for i, x in enumerate(pilot_index) if x == pilot_index[k]]


def get_copilot_UEs_given_pilot(t, pilot_index):
    """
    Get all UEs that are assigned pilot t
    :param t:           pilot t
    :param pilot_index: list of pilots for each UE
    :return:            list of UEs that are assigned pilot t
    """
    return [i for i, x in enumerate(pilot_index) if x == t]


def get_APs_associated_with_CPU(u, AP_CPU_association):
    """
    Get all APs associated with a CPU
    :param u:                   CPU index
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :return:                    list of APs associated with CPU u
    """
    return np.where(AP_CPU_association == u)[0]


def get_APs_associated_with_CPUs(CPUs, AP_CPU_association):
    """
    Get all APs associated with a list of CPUs
    :param CPUs:                CPU list
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :return:                    list of APs associated with CPU u
    """
    APs = set()
    for u in CPUs:
        APs.update(np.where(AP_CPU_association == u)[0])
    return list(APs)


def get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D):
    """
    Get all APs associated with a CPU u and serving UE k
    :param k:                   UE index
    :param u:                   CPU index
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :param D:                   UE - AP association matrix D
    :return:                    list of APs associated with CPU u
    """
    APs_k = get_APs_serving_UEs(D, [k])  # APs serving UE k
    APs_u = get_APs_associated_with_CPU(u, AP_CPU_association)  # APs associated with CPU u
    return [l for l in APs_k if l in APs_u]


def get_APs_serving_UEs(D, UEs):
    """
    Get all APs serving given UEs
    :param D:       APs-UEs association matrix D
    :param UEs:     list of UEs
    :return:        list of APs serving UEs
    """
    return np.unique(np.where(D[:, UEs] == 1)[0])


def get_CPUs_associated_with_APs(APs, AP_CPU_association):
    """
    Get all CPUs associated with a list of APs
    :param APs:                 list of APs
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :return:                    list of CPUs associated with APs
    """
    return np.unique(AP_CPU_association[APs])


def get_CPUs_serving_UE(k, D, AP_CPU_association):
    """
    Get the CPUs serving a UE
    NOTE: use this when a UE can be served by multiple CPUs
    :param k:                   UE index
    :param D:                   APs-UEs association matrix D
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    """
    APs_k = get_APs_serving_UEs(D, [k])
    # create a new set to hold the CPUs serving UE k
    CPUs_k = set()
    if len(APs_k) == 0:
        return None
    else:
        for l in APs_k:
            CPUs_k.add(AP_CPU_association[l])
        return list(CPUs_k)


def is_1UE_1CPU_constraint_satisfied(D, AP_CPU_association):
    """
    Check if the 1UE-1CPU constraint is satisfied
    :param D:                   UE - AP association matrix D
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :return:                    True if the 1UE-1CPU constraint is satisfied, False otherwise
    """
    U = len(np.unique(AP_CPU_association))  # number of CPUs
    UEs_CPUs = [[] for _ in range(U)]  # list of UEs associated with each CPU
    for u in range(U):
        UEs_CPUs[u] = get_UEs_associated_with_CPU(u, D, AP_CPU_association)
    # get the list of UEs that are associated with more than one CPU
    sets = [set(UEs_u) for UEs_u in UEs_CPUs]
    common_UEs = [set1.intersection(set2) for set1, set2 in itertools.combinations(sets, 2)]
    common_UEs = set(itertools.chain.from_iterable(common_UEs))
    if len(common_UEs) > 0:
        return False
    return True


def is_CPU_capacity_constraint_satisfied(C_CPUs, D, AP_CPU_association, UEs_resource_requirements, UEs_resource_requirements_satisfied):
    """
    Check if the CPU capacity constraint is satisfied
    :param C_CPUs:              list of CPUs' resource capacity
    :param D:                   UE - AP association matrix D
    :return:                    True if the CPU capacity constraint is satisfied, False otherwise
    """
    U = len(C_CPUs)  # number of CPUs
    for u in range(U):
        C_u = C_CPUs[u]
        UEs_u = get_UEs_associated_with_CPU(u, D, AP_CPU_association)
        if len(UEs_u) > 0:
            # get UE in UEs_u whose UEs_resource_requirements_satisfied is true (some UEs may still be allocated to have connectivity but no compute)
            UEs_u_satisfied = [UE for UE in UEs_u if UEs_resource_requirements_satisfied[UE]]
            if len(UEs_u_satisfied) > 0:
                C_UEs = [UEs_resource_requirements[k] for k in UEs_u_satisfied]
                C_UEs_sum = tuple(map(sum, zip(*C_UEs)))
                if any(C_u[i] < C_UEs_sum[i] for i in range(len(C_u))):
                    return False
    return True


def calculate_SE_fairness(SEs):
    """
    Calculate SE fairness using Jain's fairness index
    :param SEs: list of SEs
    :return fairness: Jain's fairness index
    """
    K = len(SEs)  # number of UEs
    SE_sum_square = sum([SE ** 2 for SE in SEs])
    fairness = sum(SEs) ** 2 / (K * SE_sum_square)
    return fairness


def calculate_rate_fairness(data_rates):
    """
    Calculate rate fairness using Jain's fairness index
    :param data_rates: list of data rates
    :return fairness: Jain's fairness index
    """
    K = len(data_rates)  # number of UEs
    data_rate_sum = sum(data_rates)
    data_rate_sum_square = sum([data_rate ** 2 for data_rate in data_rates])
    fairness = data_rate_sum ** 2 / (K * data_rate_sum_square)
    return fairness


def calculate_resource_fairness(D, AP_CPU_association, C):
    """
    Calculate resource fairness using Jain's fairness index
    NOTE: unallocated UEs considered to have 0 resource allocated to them
    :param D:                   UE - AP association matrix D
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :param C:                   list of CPUs' resource capacity
    :return fairness:           Jain's fairness index
    """
    K = D.shape[1]  # number of UEs
    C_per_UE_sum = 0  # for numerator of Jain's fairness index
    C_per_UE_sum_square = 0  # for denominator of Jain's fairness index
    # go through each CPU and find average compute resource allocated to each UE
    for u, C_u in enumerate(C):
        # find UEs allocated to CPU u
        nb_UEs_u = len(get_UEs_associated_with_CPU(u, D, AP_CPU_association))
        if nb_UEs_u > 0:
            # find compute resource allocated to each UE
            C_per_UE = C_u / nb_UEs_u
            # print(f"CPU {u} has {len(UEs)} UEs, C_per_UE = {C_per_UE}")
            C_per_UE_sum += C_per_UE * nb_UEs_u
            C_per_UE_sum_square += (C_per_UE ** 2) * nb_UEs_u
    fairness = C_per_UE_sum ** 2 / (K * C_per_UE_sum_square)
    return fairness


def calculate_utility_weighted_metric(weight_1, value_1, weight_2, value_2, weighted_metric):
    """
    Calculate utility value using weight metric method (the smaller the better). See more at
    https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf
    https://pdfs.semanticscholar.org/0bd0/8c5bb8ee851d5ad0ca8d4a8eb820d007409f.pdf
    Modeling, Assessment, and Optimization of Energy Systems (Chapter 6)
    https://www.sciencedirect.com/science/article/pii/B9780128166567000063

    weighted_metric is in [1, inf]. When weighted_metric = 1, the utility value is the same as the weighted sum method.
    """
    utility_value = (weight_1 * ((1 - value_1) ** weighted_metric)
                     + weight_2 * ((1 - value_2) ** weighted_metric)) ** (1 / weighted_metric)
    return utility_value


def get_APs_exceed_nb_UEs_limit(D, max_UEs_per_AP, APs=None):
    """
    Get a list of APs that are associated with more than max_UEs_per_AP UEs
    :param D:                   APs-UEs association matrix D
    :param max_UEs_per_AP:      maximum number of UEs per AP
    :param APs:                 list of APs to consider. None will return APs in D. Otherwise, return APs in APs
    :return: list of APs that are associated with more than max_UEs_per_AP UEs
    """
    APs_exceed_nb_UEs_limit = np.where(np.count_nonzero(D, axis=1) > max_UEs_per_AP)[0]
    if APs is None:
        return APs_exceed_nb_UEs_limit
    else:
        return np.intersect1d(APs_exceed_nb_UEs_limit, APs)


def get_APs_within_nb_UEs_limit(D, max_UEs_per_AP, APs=None):
    """
    Get a list of APs that are associated with more than max_UEs_per_AP UEs
    :param D:                   APs-UEs association matrix D
    :param max_UEs_per_AP:      maximum number of UEs per AP
    :param APs:                 list of APs to consider. None will return APs in D. Otherwise, return APs in APs
    :return: list of APs that are associated with more than tau_p UEs
    """
    APs_within_nb_UEs_limit = np.where(np.count_nonzero(D, axis=1) < max_UEs_per_AP)[0]
    if APs is None:
        return APs_within_nb_UEs_limit
    else:
        return np.intersect1d(APs_within_nb_UEs_limit, APs)


def get_CPUs_sufficient_resource(C_CPUs, k_requirement):
    """
    Get a list of CPUs that have enough resource to serve UE k
    :param C_CPUs:              list of CPUs' resource capacity
    :param k_requirement:        UE k's resource requirement
    :return: list of CPUs that have enough resource to allocate to UE k
    """
    CPUs_sufficient_resource = [u for u, C_u in enumerate(C_CPUs) if
                                all(CPU_resource >= k_resource for CPU_resource, k_resource in zip(C_u, k_requirement))]
    return CPUs_sufficient_resource


def calculate_distances_CPUs_UEs(locations_CPUs, locations_UEs):
    """
    Calculate distances between UEs and CPUs
    :param locations_CPUs:         list of CPUs' locations
    :param locations_UEs:          list of UEs' locations
    """
    data_CPUs = pd.DataFrame(locations_CPUs)
    data_CPUs = data_CPUs.rename(columns={0: 'x', 1: 'y'})
    data_UEs = pd.DataFrame({'x': locations_UEs.real, 'y': locations_UEs.imag})
    distances_CPUs_UEs = np.sqrt(constants.AP_HEIGHT ** 2 +
                                 np.sum((data_CPUs[['x', 'y']].to_numpy()[:, np.newaxis] -
                                         data_UEs[['x', 'y']].to_numpy()) ** 2, axis=2))  # dimension U x K
    return distances_CPUs_UEs


def normalise_dict(dict):
    """
    normalise the values in a dict. If min value is equal to max value, set to 1
    :param dict:         a dictionary
    :return:     the normalised dictionary
    """
    min_val = min(dict.values())
    max_val = max(dict.values())
    dict_normalised = {k: 1 if max_val == min_val else (v - min_val) / (max_val - min_val) for k, v in dict.items()}
    return dict_normalised


def normalise_tuple_list_by_column(tuple_list):
    """
    Min-max normalise a list of tuples by column. If min value is equal to max value, set to 1
    Example: [(1, 2, 3), (4, 6, 6), (2, 3, 6)] -> [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.33, 0.25, 1.0)]
    """
    columns = list(zip(*tuple_list))
    normalised_data = [
        tuple(
            (value - min(column)) / (max(column) - min(column)) if (max(column) - min(column)) != 0 else 1 for value in
            column)
        for column in columns
    ]
    normalised_data = list(zip(*normalised_data))
    return normalised_data


def calculate_standard_deviation_tuple_list(normalised_tuple_list):
    """
    Calculate standard deviation of every tuple in a list of tuples (which should be normalised)
    Example: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.33, 0.25, 1.0)] -> {0:0, 1:0, 2:0.34}
    """
    standard_deviations = {idx: np.std(tuple) for idx, tuple in enumerate(normalised_tuple_list)}
    return standard_deviations


def calculate_mean_tuple_list(normalised_tuple_list):
    """
    Calculate mean of every tuple in a list of tuples (which should be normalised)
    Example: [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.33, 0.25, 1.0)] -> [0, 1, 0.53]
    """
    means = {idx: np.mean(tuple) for idx, tuple in enumerate(normalised_tuple_list)}
    return means


def get_nb_APs_serving_more_than_N_UEs(D, nb_UEs_per_AP):
    return np.count_nonzero(np.count_nonzero(D, axis=1) > nb_UEs_per_AP)


def get_times_CPUs_used(D, AP_CPU_association):
    """
    Count the number of times CPUs are used to serve all UEs
    :param D:                   UE - AP association matrix D
    :param AP_CPU_association:  list of CPU indices that each AP is associated with
    :return:                    list of times each CPU is used
    """
    times_CPUs_used = 0
    for k in range(D.shape[1]):
        CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)
        times_CPUs_used += len(CPUs_k)
    return times_CPUs_used


def get_min_max_avg_UEs_per_AP(D):
    L = D.shape[0]
    min_UEs_per_AP, max_UEs_per_AP, avg_UEs_per_AP = math.inf, 0, 0
    nb_UEs_APs = np.zeros(L, dtype=int)
    for l in range(L):
        nb_UEs_l = len(get_UEs_served_by_APs([l], D))
        nb_UEs_APs[l] = nb_UEs_l
        if nb_UEs_l > max_UEs_per_AP:
            max_UEs_per_AP = nb_UEs_l
        if nb_UEs_l < min_UEs_per_AP:
            min_UEs_per_AP = nb_UEs_l
    avg_UEs_per_AP = np.mean(nb_UEs_APs)
    return min_UEs_per_AP, max_UEs_per_AP, avg_UEs_per_AP


def get_min_max_avg_APs_per_UE(D):
    K = D.shape[1]
    min_APs_per_UE, max_APs_per_UE, avg_APs_per_UE = math.inf, 0, 0
    nb_APs_UEs = np.zeros(K, dtype=int)
    for k in range(K):
        nb_APs_k = len(get_APs_serving_UEs(D, [k]))
        nb_APs_UEs[k] = nb_APs_k
        if nb_APs_k > max_APs_per_UE:
            max_APs_per_UE = nb_APs_k
        if nb_APs_k < min_APs_per_UE:
            min_APs_per_UE = nb_APs_k
    avg_APs_per_UE = np.mean(nb_APs_UEs)
    return min_APs_per_UE, max_APs_per_UE, avg_APs_per_UE


def get_min_max_avg_CPUs_per_UE(D, AP_CPU_association):
    K = D.shape[1]
    min_CPUs_per_UE, max_CPUs_per_UE, avg_CPUs_per_UE = math.inf, 0, 0
    nb_CPUs_UEs = np.zeros(K, dtype=int)
    for k in range(K):
        nb_CPUs_k = len(get_CPUs_serving_UE(k, D, AP_CPU_association))
        nb_CPUs_UEs[k] = nb_CPUs_k
        if nb_CPUs_k > max_CPUs_per_UE:
            max_CPUs_per_UE = nb_CPUs_k
        if nb_CPUs_k < min_CPUs_per_UE:
            min_CPUs_per_UE = nb_CPUs_k
    avg_CPUs_per_UE = np.mean(nb_CPUs_UEs)
    return min_CPUs_per_UE, max_CPUs_per_UE, avg_CPUs_per_UE


def get_min_max_avg_UEs_per_CPU(D, AP_CPU_association):
    U = len(np.unique(AP_CPU_association))
    min_CPUs_per_UE, max_CPUs_per_UE, avg_CPUs_per_UE = math.inf, 0, 0
    nb_UEs_CPUs = np.zeros(U, dtype=int)
    for u in range(U):
        nb_UEs_u = len(get_UEs_associated_with_CPU(u, D, AP_CPU_association))
        nb_UEs_CPUs[u] = nb_UEs_u
        if nb_UEs_u > max_CPUs_per_UE:
            max_CPUs_per_UE = nb_UEs_u
        if nb_UEs_u < min_CPUs_per_UE:
            min_CPUs_per_UE = nb_UEs_u
    avg_CPUs_per_UE = np.mean(nb_UEs_CPUs)
    return min_CPUs_per_UE, max_CPUs_per_UE, avg_CPUs_per_UE


def get_UEs_more_than_1_CPU(D, AP_CPU_association):
    K = D.shape[1]
    coordinated_UEs = []
    for k in range(K):
        CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)
        if len(CPUs_k) > 1:
            coordinated_UEs.append(k)
    return coordinated_UEs


def get_frontbackhaul_load(D, AP_CPU_association, N, tau_c, tau_p, pilot_index):
    """
    Calculate backhaul load for pilot and UL data transmission
    In each coherence block, for each UE, an AP sends N complex scalars (pilot) and N complex scalars (UL data) to a CPU,
    which may forward them to another CPU if it's not the designated serving CPU

    Serving CPU: CPU responsible for channel estimation, receive combining, data detection. It is the CPU that has the
    most number of APs serving the UE to lower the backhaul load
    """
    K = D.shape[1]
    U = len(np.unique(AP_CPU_association))  # number of CPUs

    # fronthaul
    total_fronthaul_load = 0
    for u in range(U):
        APs_u = get_APs_associated_with_CPU(u, AP_CPU_association)
        total_fronthaul_load += tau_c * N * len(APs_u)

    # backhaul
    CPUs_APs_pilots = set()  # holds tuples (u, l, t) to check if CPU u has received pilot t from AP l
    CPUs_APs_UL_signals = set()  # holds tuples (u, l) to check if CPU u has received UL data signal from AP l
    CPUs_APs_pilots_backhaul = set()  # holds tuples (u, l, t) to check if CPU u has received pilot t from AP l through backhaul
    CPUs_APs_UL_signals_backhaul = set()  # holds tuples (u, l) to check if CPU u has received UL data signal from AP l through backhaul
    for k in range(K):  # go through each UE
        CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)  # get CPUs serving UE k
        if len(CPUs_k) == 1:  # UE k is served by 1 CPU, no backhaul signalling is required
            APs_k = get_APs_serving_UEs(D, [k])
            for l in APs_k:  # for each AP l serving UE k
                # CPU u received pilot t from AP l via fronthaul
                CPUs_APs_pilots.add((CPUs_k[0], l, pilot_index[k]))
                # CPU u received UL data signal from AP l via fronthaul
                CPUs_APs_UL_signals.add((CPUs_k[0], l))
        else:  # UE k is coordinated by more than 1 CPU, backhaul signalling is required
            # find serving CPU (one has the most number of APs serving UE k)
            highest_nb_APs_serving_UEs_in_u, serving_CPU = 0, None
            for u in CPUs_k:
                APs_k_u = get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
                if len(APs_k_u) > highest_nb_APs_serving_UEs_in_u:
                    highest_nb_APs_serving_UEs_in_u = len(APs_k_u)
                    serving_CPU = u
            # start calculating backhaul load
            APs_k_serving_CPU = get_APs_serving_UE_associated_with_CPU(k, serving_CPU, AP_CPU_association, D)
            for l in APs_k_serving_CPU:  # for each AP l serving UE k and associated with the serving CPU
                # Serving CPU received pilot t from its associated APs via fronthaul
                CPUs_APs_pilots.add((serving_CPU, l, pilot_index[k]))
                # Serving CPU received UL data signal from its associated APs via fronthaul
                CPUs_APs_UL_signals.add((serving_CPU, l))
            # remove serving CPU from the list of CPUs serving UE k
            CPUs_k.remove(serving_CPU)  # only consider non-serving CPUs, which forward signals to the serving CPU
            for u in CPUs_k:  # for each non-serving CPU, which forwards pilots and UL signals to the serving CPU
                APs_k_u = get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
                for l in APs_k_u:  # for each AP l serving UE k and associated with the non-serving CPU
                    # non-serving CPU u received pilot t from the APs that are serving UE k via fronthaul
                    CPUs_APs_pilots.add((u, l, pilot_index[k]))
                    # then forward the pilot to the serving CPU via backhaul
                    CPUs_APs_pilots.add((serving_CPU, l, pilot_index[k]))
                    CPUs_APs_pilots_backhaul.add((serving_CPU, l, pilot_index[k]))
                    # non-serving CPU u received UL data signal from the APs that are serving UE k via fronthaul
                    CPUs_APs_UL_signals.add((u, l))
                    # then forward the UL signal to the serving CPU via backhaul
                    CPUs_APs_UL_signals.add((serving_CPU, l))
                    CPUs_APs_UL_signals_backhaul.add((serving_CPU, l))

    total_backhaul_load_UL = len(CPUs_APs_pilots_backhaul) * N + len(CPUs_APs_UL_signals_backhaul) * N * (tau_c - tau_p)
    # It's len(CPUs_APs_pilots_backhaul) * N instead of len(CPUs_APs_pilots_backhaul) * N * tau_p because
    # CPUs_APs_pilots_backhaul already contains pilot t
    total_backhaul_load_DL = len(CPUs_APs_UL_signals_backhaul) * N * (tau_c - tau_p)
    validate_backhaul_load(U, N, AP_CPU_association, tau_c, tau_p, CPUs_APs_pilots_backhaul, CPUs_APs_UL_signals_backhaul)
    return total_fronthaul_load, total_backhaul_load_UL, total_backhaul_load_DL


def get_frontbackhaul_load_no_pilot_spreading(D, AP_CPU_association, N, tau_c, tau_p):
    """
    Calculate backhaul load for pilot and UL data transmission
    In each coherence block, for each UE, an AP sends N complex scalars (pilot) and N complex scalars (UL data) to a CPU,
    which may forward them to another CPU if it's not the designated serving CPU

    Serving CPU: CPU responsible for channel estimation, receive combining, data detection. It is the CPU that has the
    most number of APs serving the UE to lower the backhaul load
    """
    K = D.shape[1]
    U = len(np.unique(AP_CPU_association))  # number of CPUs

    # fronthaul
    total_fronthaul_load = 0
    for u in range(U):
        APs_u = get_APs_associated_with_CPU(u, AP_CPU_association)
        total_fronthaul_load += tau_c * N * len(APs_u)

    # backhaul
    CPUs_APs_pilots_backhaul = set()  # holds tuples (u, l) to check if CPU u has received pilot signal from AP l through backhaul
    CPUs_APs_UL_signals_backhaul = set()  # holds tuples (u, l) to check if CPU u has received UL data signal from AP l through backhaul
    for k in range(K):  # go through each UE
        CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)  # get CPUs serving UE k
        if len(CPUs_k) > 1:  # UE k is coordinated by more than 1 CPU, backhaul signalling is required
            # find serving CPU (one has the most number of APs serving UE k)
            highest_nb_APs_serving_UEs_in_u, serving_CPU = 0, None
            for u in CPUs_k:
                APs_k_u = get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
                if len(APs_k_u) > highest_nb_APs_serving_UEs_in_u:
                    highest_nb_APs_serving_UEs_in_u = len(APs_k_u)
                    serving_CPU = u
            # start calculating backhaul load
            # remove serving CPU from the list of CPUs serving UE k
            CPUs_k.remove(serving_CPU)  # only consider non-serving CPUs, which forward signals to the serving CPU
            for u in CPUs_k:  # for each non-serving CPU, which forwards pilots and UL signals to the serving CPU
                APs_k_u = get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
                for l in APs_k_u:  # for each AP l serving UE k and associated with the non-serving CPU
                    # non-serving CPU forward the pilot to the serving CPU via backhaul
                    CPUs_APs_pilots_backhaul.add((serving_CPU, l))
                    # non-serving CPU forward the UL signal to the serving CPU via backhaul
                    CPUs_APs_UL_signals_backhaul.add((serving_CPU, l))

    total_backhaul_load_UL = len(CPUs_APs_pilots_backhaul) * N * tau_p + len(CPUs_APs_UL_signals_backhaul) * N * (tau_c - tau_p)
    total_backhaul_load_DL = len(CPUs_APs_UL_signals_backhaul) * N * (tau_c - tau_p)
    total_backhaul_load_DL_model = validate_backhaul_load_model(D, AP_CPU_association, N, tau_c, tau_p)
    validate_backhaul_load(U, N, AP_CPU_association, tau_c, tau_p, CPUs_APs_pilots_backhaul, CPUs_APs_UL_signals_backhaul)
    return total_fronthaul_load, total_backhaul_load_UL, total_backhaul_load_DL


def find_master_CPUs_of_UEs(D, AP_CPU_association):
    K = D.shape[1]
    # create an empty list of size K
    master_CPUs = [None] * K
    for k in range(K):  # go through each UE
        CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)  # get CPUs serving UE k
        # find serving CPU (one has the most number of APs serving UE k)
        highest_nb_APs_serving_UEs_in_u = 0
        for u in CPUs_k:
            APs_k_u = get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
            if len(APs_k_u) > highest_nb_APs_serving_UEs_in_u:
                highest_nb_APs_serving_UEs_in_u = len(APs_k_u)
                master_CPUs[k] = u
    return master_CPUs


def validate_backhaul_load_model(D, AP_CPU_association, N, tau_c, tau_p):
    U = len(np.unique(AP_CPU_association))  # number of CPUs
    master_CPUs = find_master_CPUs_of_UEs(D, AP_CPU_association)
    total_backhaul_load_DL = 0
    for u in range(U):
        # UEs whose master CPU is u
        UEs_u_master = [k for k, v in enumerate(master_CPUs) if v == u]
        # create a set to hold APs serving UEs whose master CPU is u
        APs_of_UEs_master_u = set()
        for k in UEs_u_master:
            CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)
            if u in CPUs_k:
                CPUs_k.remove(u)  # remove u from the set
            for CPU_k in CPUs_k:
                APs_of_UEs_master_u.update(get_APs_serving_UE_associated_with_CPU(k, CPU_k, AP_CPU_association, D))
        total_backhaul_load_DL += len(APs_of_UEs_master_u) * N * (tau_c - tau_p)
    return total_backhaul_load_DL


def validate_backhaul_load(U, N, AP_CPU_association, tau_c, tau_p, CPUs_APs_pilots_backhaul, CPUs_APs_UL_signals_backhaul):
    APs_all = set(range(len(AP_CPU_association)))
    for u in range(U):
        APs_u = get_APs_associated_with_CPU(u, AP_CPU_association)
        # see if any CPU received pilot or UL data signal from associated APs through backhaul
        for l in APs_u:
            if (u, l) in CPUs_APs_UL_signals_backhaul:
                print(f"CPU {u} ain't supposed to receive UL data signal from AP {l} via backhaul!")
            for t in range(tau_p):
                if (u, l, t) in CPUs_APs_pilots_backhaul:
                    print(f"CPU {u} ain't supposed to receive pilot {t} from AP {l} via backhaul!")
        # see if any CPU received pilot or UL data signal more than theoretical limit
        APs_not_u = APs_all - set(APs_u)
        # each AP sends tau_c * N complex scalars in the UL in each coherence block
        CPU_u_APs_UL_signals_backhaul = {(a, b) for (a, b) in CPUs_APs_UL_signals_backhaul if a == u}
        CPU_u_APs_pilots_backhaul = {(a, b) for (a, b) in CPUs_APs_pilots_backhaul if a == u}
        if tau_p * N * len(CPU_u_APs_pilots_backhaul) > tau_p * N * len(APs_not_u):
            print(f"CPU {u} received more complex scalars than the limit in the UL pilot training in a coherence block!")
        if (tau_c - tau_p) * N * len(CPU_u_APs_UL_signals_backhaul) > (tau_c - tau_p) * N * len(APs_not_u):
            print(f"CPU {u} received more complex scalars than the limit in the UL pilot training in a coherence block!")


def find_dist_and_projection(point_ref, point1, point2):
    """
    Find the distance between a point point_ref and a line that passes through two points point1 and point2
    also return the coordinates of the projection of point_ref on the line that passes through point1 and point2
    point_ref, point1, point2 = [1873.7810891260942, 1975.3288819514462],[2426.35235802, 2568.35357404], [2601.82708297, 2747.31201747]
    """
    point_ref, point1, point2 = np.array(point_ref), np.array(point1), np.array(point2)
    # point_ref, point1, point2 are 2D points (e.g., [0,1]). Find distance between point_ref and the line that passes
    # through point1 and point2
    distance = (np.linalg.norm(np.cross(point2 - point1, point1 - point_ref)) / np.linalg.norm(point2 - point1))
    # find the projection of point_ref on the line that passes through point1 and point2
    point1_point_ref = point_ref - point1
    point1_point2 = point2 - point1
    projection = point1 + np.dot(point1_point_ref, point1_point2) / np.dot(point1_point2, point1_point2) * point1_point2
    # check if the projection is within the line segment
    distance_projection_point1 = np.linalg.norm(projection - point1)
    distance_projection_point2 = np.linalg.norm(projection - point2)
    distance_point1_point2 = np.linalg.norm(point1 - point2)
    is_projection_within_line_segment = math.isclose(distance_projection_point1 + distance_projection_point2, distance_point1_point2)
    return distance, projection, is_projection_within_line_segment


def get_voronoi_segments(vor):
    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)
    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])
    return finite_segments, infinite_segments


def print_result_summary(result: algo_result.AlgorithmResult):
    """
    Print the result summary
    """
    nb_UEs_connectivity = count_UEs_allocated(result.D)
    print('------')
    print(f"Algorithm: {result.algo}")
    print(f"Number of UEs with connectivity/total: {nb_UEs_connectivity}/{result.D.shape[1]}")
    print(f"Sum SE UL/DL: {sum(result.SEs_UL): .2f}/{sum(result.SEs_DL): .2f} bits/s/Hz")
    print(f"Avg SE UL/DL: {sum(result.SEs_UL)/nb_UEs_connectivity: .2f}/{sum(result.SEs_DL)/nb_UEs_connectivity: .2f} bits/s/Hz")
    print(f"Fairness SE UL/DL: {result.fairness_SE_UL: .2f}/{result.fairness_SE_DL: .2f} (/1)")
    print(f"Execution time: {result.execution_time: .2f} seconds")
    print(f"CPU usage: {result.times_CPUs_used}")
    # print(f"Fronthaul load UL: {result.fronthaul_load_UL} complex scalars")
    print(f"Backhaul load UL: {result.backhaul_load_UL} complex scalars")
    print(f"Backhaul load DL: {result.backhaul_load_DL} complex scalars")
    print(f"Min/max UEs per AP: {result.min_UEs_per_AP}/{result.max_UEs_per_AP}")
    print(f"Min/max APs per UE: {result.min_APs_per_UE}/{result.max_APs_per_UE}")
    print(f"Min/max CPUs per UE: {result.min_CPUs_per_UE}/{result.max_CPUs_per_UE}")
    print(f"Nb coordinated UEs: {result.nb_coordinated_UEs}")