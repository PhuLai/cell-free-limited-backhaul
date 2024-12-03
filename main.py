# -*- coding: utf-8 -*-
import math
import time
import numpy as np
from algorithms import SCF1, SCF2, SCF1lim, Border, HybridUA, Nearest, LLSFB
from utilities import utils_comms, constants

# parameters that are fixed for simulation
sigma_sf = constants.SIGMA_SF  # standard deviation of shadow fading
constant_term = constants.CONSTANT_TERM  # average channel gain in dB at a reference distance of 1 meter.
alpha = constants.ALPHA  # path loss exponent
noise_variance_dBm = constants.NOISE_VARIANCE  # noise power
nb_realisations = constants.NB_REALISATIONS  # number of channel realisations
tau_p = constants.TAU_P  # number of pilots
tau_c = constants.TAU_C  # length of coherence block
ASD_deg = constants.ASD_DEG  # angular standard deviation around the nominal angle (measured in degrees)
antenna_spacing = constants.ANTENNA_SPACING  # antenna spacing (in number of wavelengths)
local_scattering_model = constants.LOCAL_SCATTERING_MODEL_2_24  # local scattering model, 2.24 much faster than 2.23
bandwidth = constants.B  # communication bandwidth (Hz)

# parameters that might be adjusted for simulation
existing_data, is_print_summary, save_to_pickle = constants.USE_EXISTING_DATA_PYTHON, True, True
UE_power = 100  # UE UL power (mW)
area = 8  # km2
max_power_AP = 1000  # AP maximum power (mW)
upsilon, kappa = -0.5, 0.5  # parameters for centralised DL fractional power allocation
L = 200  # number of APs
N = 4  # number of antennas per AP
K = 50  # number of UEs.
U = 40  # number of CPUs
threshold_LSF = 0.95  # threshold for LSF-based algorithms (largest LSF, heuristic hybrid).
p_UEs = np.ones(K) * UE_power  # mW
top_N_CPUs = 2

print(f"Generating simulation setup...")
(locations_APs, locations_UEs, AP_CPU_association, gain_over_noise_dB, R,
 locations_CPUs, H, Np) = utils_comms.generate_setup(L, N, K, U, area, local_scattering_model, nb_realisations,
                                                     tau_p, sigma_sf, constant_term, alpha, noise_variance_dBm,
                                                     ASD_deg, antenna_spacing, existing_data, is_print_summary,
                                                     save_to_pickle)
print(f"Running algorithms...")
# SCF1 [1]
_ = SCF1.allocate_UEs_to_APs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, max_power_AP, upsilon,
                            kappa, p_UEs, is_print_summary)
# SCF2 [10]
_ = SCF2.allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs, max_power_AP,
                              upsilon, kappa, top_N_CPUs, is_print_summary)
# SCF1lim [12]
max_inter_UEs_per_CPU = tau_p
_ = SCF1lim.allocate_UEs_to_APs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, max_inter_UEs_per_CPU,
                                p_UEs, max_power_AP, upsilon, kappa, None, is_print_summary)
# Border [11]
dist_to_border = 100  # meters
_ = Border.allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs, max_power_AP,
                              upsilon, kappa, top_N_CPUs, locations_UEs, locations_CPUs, dist_to_border, is_print_summary)
# HybridUA (Ours)
threshold_z = 0.4  # threshold to determine cell-center and cell-edge UEs
_ = HybridUA.allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs,
                                          threshold_z, threshold_LSF, max_power_AP, upsilon, kappa, top_N_CPUs,
                                          is_print_summary)
# Nearest
_ = Nearest.allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs,
                                       locations_CPUs, locations_UEs, max_power_AP, upsilon, kappa, is_print_summary)
# LLSFB
is_LSF_averaged, max_UEs_per_AP = False, math.inf
_ = LLSFB.allocate_UEs_to_CPUs(gain_over_noise_dB, R, H, Np, AP_CPU_association, tau_p, tau_c, p_UEs, is_LSF_averaged,
                               max_UEs_per_AP, threshold_LSF, max_power_AP, upsilon, kappa, is_print_summary)