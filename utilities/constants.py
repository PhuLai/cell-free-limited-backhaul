# -*- coding: utf-8 -*-
"""
Constants

@author: Phu Lai
"""
import numpy as np

# Length of coherence block
TAU_C = 200
# Length of pilot sequences
TAU_P = 10
# Number of channel realisations per setup
NB_REALISATIONS = 10
# Antenna spacing (in number of wavelengths)
ANTENNA_SPACING = 0.5  # Half wavelength distance
# Communication bandwidth
B = 20e6  # Hz
# AP height
AP_HEIGHT = 10  # meters
# Standard deviation of shadow fading
SIGMA_SF = 10
# Average channel gain in dB at a reference distance of 1 meter. Note that -35.3 dB corresponds to -148.1 dB at 1 km, using pathloss exponent 3.76
CONSTANT_TERM = -35.3
# Path loss exponent
ALPHA = 3.76
# Noise figure( in dB)
NOISE_FIGURE = 7
# Compute noise power
NOISE_VARIANCE = -174 + 10 * np.log10(B) + NOISE_FIGURE  # dBm
# Angular standard deviation around the nominal angle (measured in degrees)
ASD_DEG = 20
# Set threshold for when a non-master AP decides to serve a UE
THRESHOLD_NON_MASTER_AP_SERVE_UE = -40  # dB
# List of possible UE transmit power (mW)
UE_POWERs = list(range(10, 110, 10))  # list(range(10, 110, 20))


"""
PROGRAM CONSTANTS
"""
AP_SELECTION_BY_SUM_LSF_THRESHOLD = 'sum_LSF_threshold'
AP_SELECTION_BY_NB_AP = 'nb_AP'
LOCAL_SCATTERING_MODEL_2_23 = '2.23'  # E.q 2.23 in Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency (https://doi.org/10.1561/2000000093)
LOCAL_SCATTERING_MODEL_2_24 = '2.24'  # E.q 2.24 in Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency (https://doi.org/10.1561/2000000093)
NB_UES_PER_CPU_ROUNDING_CEIL = 'CEILING'
NB_UES_PER_CPU_ROUNDING_FLOOR = 'FLOOR'
NB_UES_PER_CPU_ROUNDING_NEAR = 'NEAREST'
USE_EXISTING_DATA_MATLAB = 'matlab'
USE_EXISTING_DATA_PYTHON = 'python'

PATH_DATASET = 'datasets'
PICKLE_LOCATION_APS = 'locations_APs.pickle'
PICKLE_LOCATION_UES = 'locations_UEs.pickle'
PICKLE_LOCATIONS_CPUS = 'locations_CPUs.pickle'
PICKLE_C_CPUs = 'C_CPUs.pickle'
PICKLE_UE_REQUIREMENTS = 'UE_requirements.pickle'
PICKLE_C_APs = 'C_APs.pickle'
PICKLE_AP_CPU_ASSOCIATION = 'AP_CPU_association.pickle'
PICKLE_GAIN_OVER_NOISE_dB = 'gain_over_noise_dB.pickle'
PICKLE_R = 'R.pickle'
PICKLE_H = 'H.pickle'
PICKLE_NP = 'Np.pickle'

ALGO_BORDER = 'Border'
ALGO_HYBRID_UA = 'HybridUA'
ALGO_LLSFB = 'LLSFB'
ALGO_NEAREST = 'Nearest'
ALGO_SCF1 = 'SCF1'
ALGO_SCF1LIM = 'SCF1lim'
ALGO_SCF2 = 'SCF2'