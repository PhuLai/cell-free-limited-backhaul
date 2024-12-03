# -*- coding: utf-8 -*-
"""
Class that contains result pack

@author: Phu Lai
"""
from utilities import utils_comms, utils


class AlgorithmResult:
    def __init__(self, algo, D, pilot_index, R, H, Np, tau_c, tau_p, p_UEs,
                 execution_time, AP_CPU_association, N,
                 max_power_AP, gain_over_noise_dB, upsilon, kappa):
        self.algo = algo
        self.D = D
        self.pilot_index = pilot_index
        self.SEs_DL, self.SEs_UL = utils_comms.calculate_SEs_downlink_uplink(D, R, H, Np, tau_c, tau_p, p_UEs,
                                                                             max_power_AP, pilot_index,
                                                                             gain_over_noise_dB, upsilon, kappa)
        self.fairness_SE_DL, self.fairness_SE_UL = utils.calculate_SE_fairness(self.SEs_DL), utils.calculate_SE_fairness(self.SEs_UL)
        # K = D.shape[1]
        # self.SEs_DL, self.SEs_UL = [SE[0] for SE in np.zeros((K, 1))], [SE[0] for SE in np.zeros((K, 1))]
        # self.fairness_SE_DL, self.fairness_SE_UL = 0, 0

        self.execution_time = execution_time

        # total number of times CPUs involved
        self.times_CPUs_used = utils.get_times_CPUs_used(D, AP_CPU_association)

        # Min/max UEs per AP
        self.min_UEs_per_AP, self.max_UEs_per_AP, self.avg_UEs_per_AP = utils.get_min_max_avg_UEs_per_AP(D)

        # Min/max APs per UE
        self.min_APs_per_UE, self.max_APs_per_UE, self.avg_APs_per_UE = utils.get_min_max_avg_APs_per_UE(D)

        # Min/max CPUs per UE
        self.min_CPUs_per_UE, self.max_CPUs_per_UE, self.avg_CPUs_per_UE = utils.get_min_max_avg_CPUs_per_UE(D, AP_CPU_association)

        # Min/max UEs per CPU
        self.min_UEs_per_CPU, self.max_UEs_per_CPU, self.avg_UEs_per_CPU = utils.get_min_max_avg_UEs_per_CPU(D, AP_CPU_association)

        # Nb coordinated UEs (those served by more than 1 CPU)
        self.coordinated_UEs = utils.get_UEs_more_than_1_CPU(D, AP_CPU_association)
        self.nb_coordinated_UEs = len(self.coordinated_UEs)

        # Backhaul load UL, including pilot and UL data transmission
        self.fronthaul_load_UL, self.backhaul_load_UL, self.backhaul_load_DL = utils.get_frontbackhaul_load_no_pilot_spreading(D, AP_CPU_association, N, tau_c, tau_p)
        # self.fronthaul_load_UL, self.backhaul_load_UL, self.backhaul_load_DL = 0, 0, 0





