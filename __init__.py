import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import math

#Extracting simulation parameters from network configuration (.json) file
with open('configurations/mouse_cerebellum_config_healthy.json') as fl:
    all_data = json.load(fl)

first = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['start_first']
n_trials = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['n_trials']
between_start = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['between_start']
last = first + between_start*(n_trials-1)
burst_dur = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['burst_dur']
burst_dur_us = all_data['simulations']['DCN_update']['devices']['US']['parameters']['burst_dur']
burst_dur_cs = burst_dur- burst_dur_us
trials_start = np.arange(first, last+between_start, between_start)

selected_trials = np.linspace(1,100,100).astype(int) #Can specify trials to be analyzed

maf_step = 100 #selected step for moving average filter when computing motor output from DCN SDF

threshold = 3.9 #6th trial of DeOude2020 - 70% CRs, value based on sdf_maf_max_dcn output