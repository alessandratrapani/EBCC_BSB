"""
Script required to run cerebellar scaffold EBCC simulations
"""

import json
import numpy as np
from bsb.config import JSONConfig
import sdf_calculation as sc
import plot_sdf_f as ps

print("Simulation results dispaly")
network_file = "scaffolds/standard_300x_200z.hdf5"
nest_config = JSONConfig("configurations/mouse_cerebellum_config_healthy.json")


# All cell names:
# 'basket', 'dcn', 'gaba', 'glomerulus', 'gly', 'golgi', 'grc', 'io', 'mossy', 'pc', 'stellate'

ps.plot_sdf('results/results_DCN_update_16719039526625657936074583371', 'pc')
