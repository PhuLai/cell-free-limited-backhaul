# -*- coding: utf-8 -*-
"""
Supporting functions defined here

@author: Phu Lai
"""
import math
import numpy as np
import pandas as pd
from utilities import algo_result, constants
import os
import pickle


def db2pow(xdb):
    """Convert decibels (dB) to power (mW)
    :param xdb: decibels (dB)
    :return: power (mW)
    """
    return 10. ** (xdb / 10.)


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


def count_UEs_allocated(D):
    """
    Count the number of UEs that are allocated to at least one AP
    :param D:   UE - AP association matrix D
    :return:    number of UEs that are allocated to at least one AP
    """
    nb_users_allocated = np.count_nonzero(np.count_nonzero(D, axis=0) > 0)
    return nb_users_allocated


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


def get_fronthaul_load(D, AP_CPU_association, N, tau_c, tau_p):
    """
    Calculate fronthaul load for DL data transmission, no pilot spreading
    In each coherence block, for each UE, an AP sends N complex scalars (pilot) and N complex scalars (UL data) to a CPU,
    which may forward them to another CPU if it's not the designated serving CPU

    Serving CPU: CPU responsible for channel estimation, receive combining, data detection. It is the CPU that has the
    most number of APs serving the UE to lower the fronthaul load
    """
    K = D.shape[1]
    U = len(np.unique(AP_CPU_association))  # number of CPUs

    # calculate fronthaul load
    CPUs_APs_pilots_fronthaul = set()  # holds tuples (u, l) to check if CPU u has received pilot signal from AP l through fronthaul
    CPUs_APs_UL_signals_fronthaul = set()  # holds tuples (u, l) to check if CPU u has received UL data signal from AP l through fronthaul
    for k in range(K):  # go through each UE
        CPUs_k = get_CPUs_serving_UE(k, D, AP_CPU_association)  # get CPUs serving UE k
        if len(CPUs_k) > 1:  # UE k is coordinated by more than 1 CPU, fronthaul signalling is required
            # find serving CPU (one has the most number of APs serving UE k)
            highest_nb_APs_serving_UEs_in_u, serving_CPU = 0, None
            for u in CPUs_k:
                APs_k_u = get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
                if len(APs_k_u) > highest_nb_APs_serving_UEs_in_u:
                    highest_nb_APs_serving_UEs_in_u = len(APs_k_u)
                    serving_CPU = u
            # start calculating fronthaul load
            # remove serving CPU from the list of CPUs serving UE k
            CPUs_k.remove(serving_CPU)  # only consider non-serving CPUs, which forward signals to the serving CPU
            for u in CPUs_k:  # for each non-serving CPU, which forwards pilots and UL signals to the serving CPU
                APs_k_u = get_APs_serving_UE_associated_with_CPU(k, u, AP_CPU_association, D)
                for l in APs_k_u:  # for each AP l serving UE k and associated with the non-serving CPU
                    # non-serving CPU forward the pilot to the serving CPU
                    CPUs_APs_pilots_fronthaul.add((serving_CPU, l))
                    # non-serving CPU forward the UL signal to the serving CPU
                    CPUs_APs_UL_signals_fronthaul.add((serving_CPU, l))

    total_fronthaul_load_UL = len(CPUs_APs_pilots_fronthaul) * N * tau_p + len(CPUs_APs_UL_signals_fronthaul) * N * (tau_c - tau_p)
    total_fronthaul_load_DL = len(CPUs_APs_UL_signals_fronthaul) * N * (tau_c - tau_p)
    validate_fronthaul_load(U, N, AP_CPU_association, tau_c, tau_p, CPUs_APs_pilots_fronthaul, CPUs_APs_UL_signals_fronthaul)
    return total_fronthaul_load_UL, total_fronthaul_load_DL


def validate_fronthaul_load(U, N, AP_CPU_association, tau_c, tau_p, CPUs_APs_pilots_fronthaul, CPUs_APs_UL_signals_fronthaul):
    APs_all = set(range(len(AP_CPU_association)))
    for u in range(U):
        APs_u = get_APs_associated_with_CPU(u, AP_CPU_association)
        # see if any CPU received pilot or UL data signal from associated APs through fronthaul
        for l in APs_u:
            if (u, l) in CPUs_APs_UL_signals_fronthaul:
                print(f"CPU {u} ain't supposed to receive UL data signal from AP {l} via fronthaul!")
            for t in range(tau_p):
                if (u, l, t) in CPUs_APs_pilots_fronthaul:
                    print(f"CPU {u} ain't supposed to receive pilot {t} from AP {l} via fronthaul!")
        # see if any CPU received pilot or UL data signal more than theoretical limit
        APs_not_u = APs_all - set(APs_u)
        # each AP sends tau_c * N complex scalars in the UL in each coherence block
        CPU_u_APs_UL_signals_fronthaul = {(a, b) for (a, b) in CPUs_APs_UL_signals_fronthaul if a == u}
        CPU_u_APs_pilots_fronthaul = {(a, b) for (a, b) in CPUs_APs_pilots_fronthaul if a == u}
        if tau_p * N * len(CPU_u_APs_pilots_fronthaul) > tau_p * N * len(APs_not_u):
            print(f"CPU {u} received more complex scalars than the limit in the UL pilot training in a coherence block!")
        if (tau_c - tau_p) * N * len(CPU_u_APs_UL_signals_fronthaul) > (tau_c - tau_p) * N * len(APs_not_u):
            print(f"CPU {u} received more complex scalars than the limit in the DL transmission in a coherence block!")


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


def print_result_summary(result: algo_result.AlgorithmResult):
    """
    Print the result summary
    """
    nb_UEs_connectivity = count_UEs_allocated(result.D)
    print('------')
    print(f"Algorithm: {result.algo}")
    print(f"Number of UEs with connectivity: {nb_UEs_connectivity}/{result.D.shape[1]}")
    print(f"Sum SE UL/DL: {sum(result.SEs_UL): .2f}/{sum(result.SEs_DL): .2f} bits/s/Hz")
    print(f"Avg SE UL/DL: {sum(result.SEs_UL)/nb_UEs_connectivity: .2f}/{sum(result.SEs_DL)/nb_UEs_connectivity: .2f} bits/s/Hz")
    print(f"Fairness SE UL/DL: {result.fairness_SE_UL: .2f}/{result.fairness_SE_DL: .2f} (/1)")
    print(f"Execution time: {result.execution_time: .2f} seconds")
    print(f"Fronthaul load UL: {result.fronthaul_load_UL} complex scalars")
    print(f"Fronthaul load DL: {result.fronthaul_load_DL} complex scalars")
    print(f"Min/max UEs per AP: {result.min_UEs_per_AP}/{result.max_UEs_per_AP}")
    print(f"Min/max APs per UE: {result.min_APs_per_UE}/{result.max_APs_per_UE}")
    print(f"Min/max CPUs per UE: {result.min_CPUs_per_UE}/{result.max_CPUs_per_UE}")
    print(f"Nb coordinated UEs: {result.nb_coordinated_UEs}")