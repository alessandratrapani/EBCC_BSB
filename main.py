"""
Script required to run cerebellar scaffold EBCC simulations
"""

import json
import numpy as np
from bsb.config import JSONConfig


print("Simulation results dispaly")
network_file = "scaffolds/standard_300x_200z.hdf5"
nest_config = JSONConfig("configurations/mouse_cerebellum_config_healthy.json")

#Extracting simulation parameters from network configuration (.json) file
with open('mouse_cerebellum_config_healthy.json') as fl:
    all_data = json.load(fl)

selected_trials = np.linspace(1,100,100).astype(int) #Can specify trials to be analyzed

maf_step = 100 #selected step for moving average filter when computing motor output from DCN SDF

threshold = 3.9 #6th trial of DeOude2020 - 70% CRs, value based on sdf_maf_max_dcn output

# All cell names:
# 'basket', 'dcn', 'gaba', 'glomerulus', 'gly', 'golgi', 'grc', 'io', 'mossy', 'pc', 'stellate'

