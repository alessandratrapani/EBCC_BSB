"""
Script required to run cerebellar scaffold EBCC simulations
"""

import json
import sys
import numpy as np
from bsb.core import Scaffold, from_hdf5
from bsb.config import JSONConfig
from bsb.output import HDF5Formatter
from bsb.reporting import set_verbosity
import fnmatch
import os
import mf_cylinder_targeting

set_verbosity(3)
print("Simulating network")
network_file = "scaffolds/standard_300x_200z.hdf5" #Change network filE "balanced_DCN_IO.hdf5"
nest_config = JSONConfig("configurations/mouse_cerebellum_config_healthy.json") #Change configuration file (e. g. for different impairments)
HDF5Formatter.reconfigure(network_file, nest_config)
network_scaffold = from_hdf5(network_file)

network_scaffold.run_simulation("DCN_update")
