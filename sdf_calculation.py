import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from __init__ import *

def read_file(file):
    fname = file + '.hdf5'
    f = h5py.File(fname)
    return f

#Compute SDF per one selected trial. Returns SDF firing rate of each cell at each time instant
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - integer indicating the trial.
def sdf_comput(file, cell, trial):
    f = read_file(file)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])

    if cell == 'dcn':
        g_size = 10
    else:
        g_size = 20

    neurons = np.unique(spk[:,0])

    spk_first = spk[(spk[:,1]>=trials_start[trial]-50) & (spk[:,1]<trials_start[trial]+burst_dur+50)]
    spk_first[:,1] -= trials_start[trial]-50
    dur = burst_dur+100

    sdf_full = np.empty([len(neurons),int(dur)])
    sdf = []
    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(dur)):
            tau_first = t-spike_times_first
            sdf_full[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)

        sdf.append(sdf_full[neu][50:330])

    return(sdf)


#Compute mean SDF for each trial. SDF values are averaged across cells, returns mean firing rate at each time instant
#of a trial
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - integer indicating the trial.
def sdf_mean(file, cell, trial):
    sdf = sdf_comput(file, cell, trial)
    sdf_mean = np.mean(sdf, axis=0)

    return(sdf_mean)


#Compute mean SDF during baseline (outside the CS time window)
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def sdf_baseline(file, cell):
    f = read_file(file)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])

    if cell == 'dcn':
        g_size = 10
    else:
        g_size = 20

    neurons = np.unique(spk[:,0])

    spk_first = spk[(spk[:,1]>trials_start[0]+burst_dur) & (spk[:,1]<=trials_start[1])]
    spk_first[:,1] -= trials_start[0]+burst_dur

    sdf = np.empty([len(neurons),int(between_start-burst_dur)])

    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(between_start-burst_dur)):
            tau_first = t-spike_times_first
            sdf[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)
    sdf = np.mean(sdf, axis=1)

    return(sdf)


#Compute mean SDF change during the last 10 trials of the simulation. Mean change is computed by subtracting
#mean firing rate during the first 100 ms of a trial (for PCs) or during baseline (for DCN) from the firing rate
#of each cell during the LTD window (150-200 ms of a trial).
##file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def sdf_change(file, cell):
    sdf_change = []
    if cell == 'dcn':
        base_sdf = np.mean(sdf_baseline(file, cell))
    for i in range(91,100):
        sdf = sdf_comput(file, cell, i)
        sdf_change_trial = []
        for neuron in range(len(sdf)):
            if cell == 'pc':
                baseline_sdf = sdf[neuron][:100]
                avg_baseline_sdf = np.mean(baseline_sdf)
            elif cell == 'dcn':
                avg_baseline_sdf = base_sdf
            current_sdf_change = np.sum(sdf[neuron][150:200]-avg_baseline_sdf)/50
            sdf_change_trial.append(current_sdf_change)
        sdf_change.append(np.array(sdf_change_trial))

    return(sdf_change)


#Compute SDF with moving average filter.
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - trial - integer indicating the trial,
#step - time step for convolution in ms
def sdf_maf(file, cell, trial, step):
    sdf_maf = np.convolve(sdf_mean(file, cell, trial), np.ones(step), 'valid') / step
    return(sdf_maf)


#Compute coefficient of variation of the inter spike interval (ISI CV)
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def cv(file, cell):
    f = read_file(file)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])
    spk = spk[(spk[:,1]>trials_start[0]+burst_dur) & (spk[:,1]<=trials_start[1])]
    neurons = np.unique(spk[:,0])

    cvs = []
    for i in range(len(neurons)):
        single_spikes = []
        for j in range(spk.shape[0]):
            if spk[j][0] == neurons[i]:
                single_spikes.append(spk[j][1])
        isi = np.diff(single_spikes)
        mu, std = isi.mean(), isi.std()
        cv = std / mu
        cvs.append(cv)

    return(cvs)


#Extract maximum values of SDF during each trial, and split the array into 10 blocks of 10 trials.
#Used for CR threshold selection.
#file - file name (without '.hdf5')
def sdf_maf_max_dcn(file):
    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio)
    isi_end = int(burst_dur_cs*sdf_maf_ratio-1)

    baseline = np.mean(sdf_baseline(file, 'dcn'))
    sdf_maf_max_all = []
    for j in range(1,n_trials):
        sdf_maf = sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline
        sdf_maf = sdf_maf[isi_start:isi_end]
        sdf_maf_max = np.max(sdf_maf)
        sdf_maf_max_all.append(sdf_maf_max)
    sdf_maf_max_all = np.split(np.asarray(sdf_maf_max_all), 10)

    return(sdf_maf_max_all)


#Calculate conditioned responses for each block of 10 trials. Returns 0 if no CR, 1 in presence of CR.
#Criteria for CR: 1) CR threshold is reached no earlier than after the first 100 ms of a trial; 2) after crossing
#the CR threshold, the motor output has to stay above the CR threshold for 75% of the remaining time until US.
#file - file name (without '.hdf5')
def cr(file):

    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio)
    isi_end = int(burst_dur_cs*sdf_maf_ratio)
    baseline = np.mean(sdf_baseline(file, 'dcn'))
    over_threshold = []
    for j in selected_trials:
        sdf_maf = sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline

        sdf_maf_pre_cs = sdf_maf[:isi_start]
        sdf_maf_cs = sdf_maf[isi_start:isi_end]

        sdf_maf_pre_cs_over = sdf_maf_pre_cs[sdf_maf_pre_cs >= threshold]
        if len(sdf_maf_pre_cs_over) > 0:
            over_threshold.append(0)
        elif len(sdf_maf_pre_cs_over) == 0:
            sdf_maf_win_over = sdf_maf_cs[sdf_maf_cs >= threshold]
            if len(sdf_maf_win_over) == 0:
                over_threshold.append(0)
            elif len(sdf_maf_win_over) > 0:
                for i in range(len(sdf_maf_cs)):
                    if sdf_maf_cs[i] >= threshold:
                        onset_index = i
                        break
                sdf_maf_cs_onset = sdf_maf_cs[onset_index:]
                if len(sdf_maf_win_over) >= len(sdf_maf_cs_onset)*0.75:
                    over_threshold.append(1)
                else:
                    over_threshold.append(0)

    over_threshold = np.split(np.asarray(over_threshold), 10)
    return(over_threshold)


#CR onset latency. Returns the time points from which the motor output begins to consistently rise until reaching
#the CR threshold. Trials in which no CR was produced are indicated as 0.
#file - file name (without '.hdf5')
def onset_latency(file):
    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio)
    isi_end = int(burst_dur_cs*sdf_maf_ratio)

    baseline = np.mean(sdf_baseline(file, 'dcn'))
    ol_all = []
    for j in range(1,n_trials):
        sdf_maf = sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline
        sdf_maf_cs = sdf_maf[isi_start:isi_end]
        sdf_maf_pre = sdf_maf[:isi_start]
        sdf_maf_pre_over = sdf_maf_pre[sdf_maf_pre>=threshold]

        if len(sdf_maf_pre_over) > 0:
            ol_all.append(0)
        elif len(sdf_maf_pre_over) == 0:
            sdf_maf_cs_over = sdf_maf_cs[sdf_maf_cs>=threshold]
            if len(sdf_maf_cs_over) == 0:
                ol_all.append(0)
            elif len(sdf_maf_cs_over) > 0:
                for i in range(len(sdf_maf_cs)):
                    if sdf_maf_cs[i] >= threshold:
                        onset_index = i
                        break
                sdf_maf_cs_onset = sdf_maf_cs[onset_index:]
                if len(sdf_maf_cs_over) < len(sdf_maf_cs_onset)*0.75:
                    ol_all.append(0)
                elif len(sdf_maf_cs_over) >= len(sdf_maf_cs_onset)*0.75:
                    for i in range(len(sdf_maf)):
                        if sdf_maf[i] >= threshold:
                            thr_index = i
                            break
                    sdf_to_thr = sdf_maf[:thr_index]
                    sdf_to_thr_diff = np.diff(sdf_to_thr)
                    for k in range(len(sdf_to_thr_diff)):
                        if sdf_to_thr_diff[k] > 0:
                            sdf_to_thr_diff_k = sdf_to_thr_diff[k:-1]
                            sdf_to_thr_diff_k_positive = sdf_to_thr_diff_k[sdf_to_thr_diff_k>0]
                            if len(sdf_to_thr_diff_k) == len(sdf_to_thr_diff_k_positive):
                                ol_time = k+1
                                ol_all.append(np.round((isi_end-ol_time) / sdf_maf_ratio))
                                break
    ol_all = np.array(ol_all)
    return(ol_all)