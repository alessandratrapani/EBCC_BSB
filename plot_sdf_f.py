import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from __init__ import *
import sdf_calculation as sc

""" Plot simulation output: SDF, motor output, SDF change, SDF baseline, ISI CV,
    percentages of conditioned responses, raster plots"""


#Graph colors for main cells to be plotted
color_pc = all_data["cell_types"]["purkinje_cell"]["plotting"]["color"]
color_dcn = all_data["cell_types"]["dcn_cell_glut_large"]["plotting"]["color"]
color_io = all_data["cell_types"]["io_cell"]["plotting"]["color"]


#Plot SDF curves in all trials
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def plot_sdf(file, cell):
    if cell == 'pc':
        clr = color_pc
    elif cell == 'dcn':
        clr = color_dcn
    else:
        clr = 'blue'

    for j in selected_trials:
        sdf = sc.sdf_mean(file, cell, j)
        plt.figure(file+' {} SDF'.format(cell.upper()))
        plt.title('{} SDF'.format(cell.upper()))
        plt.ylim([20,190])
        sdf_plot = plt.plot(sdf)
        cc = 0.75-j/(max(selected_trials)*4/3)
        rgb_range = [[0,230/255], [100/255,240/255], [0/255,230/255]]
        rc=rgb_range[0][1]-(j/(max(selected_trials)*4/3))*(rgb_range[0][1]-rgb_range[0][0])
        gc=rgb_range[1][1]-(j/(max(selected_trials)*4/3))*(rgb_range[1][1]-rgb_range[1][0])
        bc=rgb_range[2][1]-(j/(max(selected_trials)*4/3))*(rgb_range[2][1]-rgb_range[2][0])
        plt.setp(sdf_plot, color=[rc,gc,bc])
        #plt.setp(sdf_plot, color=clr, alpha=0.1+0.007*j)
        if j == selected_trials[-1]:
            sdf_baseline = np.mean(sc.sdf_baseline(file, cell))
            sdf_baseline = [[0, burst_dur], [sdf_baseline, sdf_baseline]]
            plt.plot(sdf_baseline[0], sdf_baseline[1], color="black", linestyle = "dashed")
            plt.axvline(x=burst_dur_cs, color="red")
            plt.xlabel("Time [ms]")
            plt.ylabel("SDF [Hz]")
        #plt.xlim([50,280])
        plt.savefig(file+"_"+cell+"_SDF.svg")
        plt.show()


#Plot motor output curves in all trials
#file - file name (without '.hdf5')
def plot_motor_output(file):
    clr = color_dcn

    baseline = np.mean(sc.sdf_baseline(file, 'dcn'))
    for j in selected_trials:
        sdf_maf = sc.sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline

        plt.figure(file+" {} SDF + moving average filter".format('dcn'.upper()))
        plt.title("Motor output")
        plt.xlabel("Time [ms]")
        plt.ylabel("Motor output")
        plt.ylim([-11,19])

        sdf_maf_plot = plt.plot(sdf_maf)

        #plt.setp(sdf_maf_plot, color=clr, alpha=0.1+0.007*j)
        cc = 0.75-j/(max(selected_trials)*4/3)
        plt.setp(sdf_maf_plot, color=[cc,cc,cc])
        if j == selected_trials[-1]:
            axis = [[0, burst_dur-maf_step], [0, 0]]
            plt.plot(axis[0], axis[1], color="black", linestyle = "dashed")
            us_start = (burst_dur-burst_dur_us)*((burst_dur-maf_step)/burst_dur)
            plt.axvline(x=us_start, color="red")
            cr_threshold = [[0, burst_dur-maf_step], [threshold, threshold]]
            plt.plot(cr_threshold[0], cr_threshold[1], color="cyan")
        #plt.xlim([33,181])
        plt.savefig(file+"_motor_output_full.svg")
        plt.show()


#Plot CR incidence per 10-trial block.
#file - file name (without '.hdf5'), params - integer from 0 to 3, which indicates CR curve parameters (higher number -
#darker gray color)
def plot_cr(file, params):
    if params == 0:
        curve_params = ['black', "o", "white", "black"]
    elif params == 1:
        curve_params = ['black', "^", "lightgray", "black"]
    elif params == 2:
        curve_params = ['black', "s", "darkgray", "black"]
    elif params == 3:
        curve_params = ['black', "p", "dimgray", "black"]
    plt.figure("CR incidence")
    plt.title("CR incidence")
    plt.ylim([-5,105])
    plt.xlabel("Block")
    plt.ylabel("% CR")
    x = range(1,11)
    crs = sc.cr(file)
    y = []
    for i in range(len(crs)):
        crs_over = crs[i][crs[i]>0]
        crs_trial = len(crs_over)*10
        y.append(crs_trial)
    plt.plot(x, y, color=curve_params[0], marker = curve_params[1], mfc=curve_params[2], mec=curve_params[3])
    plt.xticks(x)
    plt.show()


#Colors used for onset latency, SDF baseline, SDF change and ISI CV plots.
colorh = 'white'
colorp1 = 'lightgray'
colorp2 = 'darkgray'
colorp3 = 'dimgray'


#Plot onset latency barplot
#file - file name (without '.hdf5'), clr - bar color, name - x axis label
def plot_onset_latency(file, clr, label):
    plt.figure("CR onset latency")
    plt.title("CR onset latency")
    ol_raw = sc.onset_latency(file)
    ol_raw = ol_raw[ol_raw > 0]
    x = label
    y = np.mean(ol_raw)
    err = np.std(ol_raw)
    plt.ylim([-5, 280])
    plt.bar(x, y, yerr = err, color=clr, edgecolor = 'black', width=0.5, capsize = 5)
    plt.ylabel("Time before US [ms]")
    plt.show()


#Plot SDF baseline as boxplots (up to 4 files). Colors specified above are used by default.
#files - file name(s) (without '.hdf5'), labels - x axis label(s), cell - cell name (e. g., 'pc')
def plot_sdf_baseline(files, labels, cell):
    plt.figure(files[1]+"_"+cell+"_baseline")
    plt.title("{} baseline firing rate".format(cell.upper()))
    y = []
    for i in range(len(files)):
        baseline = sc.sdf_baseline(files[i], cell)
        mean_baseline = baseline
        y.append(mean_baseline)
    medianprops = dict(linewidth = 2, color='firebrick')
    meanprops = dict(linewidth = 2, color='#00aeef', linestyle='-')
    bplot = plt.boxplot(y, labels = labels, patch_artist = True, showmeans=True, meanline = True, medianprops = medianprops, meanprops = meanprops)
    if len(files) == 1:
        colors = sc.colorh
    elif len(files) == 2:
        colors = [sc.colorh, sc.colorp1]
    elif len(files) == 3:
        colors = [sc.colorh, sc.colorp1, sc.colorp2]
    elif len(files) == 4:
        colors = [sc.colorh, sc.colorp1, sc.colorp2, sc.colorp3]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    #if cell == 'pc':
        #plt.ylim([60,110])
    #elif cell == 'dcn':
        #plt.ylim([50,100])
    plt.ylabel("Mean baseline firing rate")
    plt.savefig(files[1]+"_"+cell+"_baseline.svg")
    plt.show()
#E. g., sc.plot_sdf_baseline(['healthy', 'pathology1', 'pathology2'], ['Healthy', 'Pathology1', 'Pathology2'], 'pc')


#Plot ISI CV as boxplots (up to 4 files). Colors specified above are used by default.
#files - file name(s) (without '.hdf5'), labels - x axis label(s), cell - cell name (e. g., 'pc')
def plot_cv(files, labels, cell):
    plt.figure(files[1]+"_"+cell+"_cv")
    plt.title("{} ISI CV".format(cell.upper()))
    y = []
    for i in range(len(files)):
        cv = sc.cv(files[i], cell)
        mean_cv = cv
        y.append(mean_cv)
    medianprops = dict(linewidth = 2, color='firebrick')
    meanprops = dict(linewidth = 2, color='#00aeef', linestyle='-')
    bplot = plt.boxplot(y, labels = labels, patch_artist = True, showmeans=True, meanline = True, medianprops = medianprops, meanprops = meanprops)
    if len(files) == 1:
        colors = sc.colorh
    elif len(files) == 2:
        colors = [sc.colorh, sc.colorp1]
    elif len(files) == 3:
        colors = [sc.colorh, sc.colorp1, sc.colorp2]
    elif len(files) == 4:
        colors = [sc.colorh, sc.colorp1, sc.colorp2, sc.colorp3]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    #if cell == 'pc':
        #plt.ylim([0.1,1.5])
    #elif cell == 'dcn':
        #plt.ylim([0.1,0.7])
    plt.ylabel("Mean ISI CV")
    plt.savefig(files[1]+"_"+cell+"_cv.svg")
    plt.show()
#E. g., sc.plot_cv(['healthy', 'pathology1'], ['Healthy', 'Pathology1'], 'pc')


#Plot SDF change as boxplots (up to 4 files). Colors specified above are used by default.
#files - file name(s) (without '.hdf5'), labels - x axis label(s), cell - cell name (e. g., 'pc')
def plot_sdf_change(files, labels, cell):
    plt.figure(files[1]+"_"+cell+"_sdf_change")
    plt.title("{} SDF Change".format(cell.upper()))
    y = []
    for i in range(len(files)):
        sdf_change = sc.sdf_change(files[i], cell)
        mean_change = np.mean(sdf_change, axis=1)
        y.append(mean_change)
    medianprops = dict(linewidth = 2, color='firebrick')
    meanprops = dict(linewidth = 2, color='#00aeef', linestyle='-')
    bplot = plt.boxplot(y, labels = labels, patch_artist = True, showmeans=True, meanline = True, medianprops = medianprops, meanprops = meanprops)
    if len(files) == 1:
        colors = sc.colorh
    elif len(files) == 2:
        colors = [sc.colorh, sc.colorp1]
    elif len(files) == 3:
        colors = [sc.colorh, sc.colorp1, sc.colorp2]
    elif len(files) == 4:
        colors = [sc.colorh, sc.colorp1, sc.colorp2, sc.colorp3]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    #if cell == 'pc':
        #plt.ylim([-35,15])
    #elif cell == 'dcn':
        #plt.ylim([-5,15])
    plt.ylabel("Mean SDF change")
    plt.savefig(files[1]+"_"+cell+"_sdf_change.svg")
    plt.show()
#E. g., sc.plot_sdf_change('healthy', 'Healthy', 'dcn')


#Raster plot for one trial.
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - trial - integer indicating the trial,
#window: 0 - only the CS time window, 1 - CS time window + pause (until the next trial)
def plot_spikes(file, cell, trial, window):
    if cell == 'pc':
        clr = color_pc
    elif cell == 'dcn':
        clr = color_dcn
    elif cell == 'io':
        clr = color_io
    elif cell == 'glomerulus':
        clr = 'gray'
    elif cell == 'mossy':
        clr = 'gray'

    f = sc.read_file(file)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])
    if window == 0:
        spk = spk[(spk[:,1]>trials_start[trial]-100) & (spk[:,1]<=trials_start[trial]+burst_dur+100)]
    elif window == 1:
        spk = spk[(spk[:,1]>trials_start[trial]-100) & (spk[:,1]<=trials_start[trial+1])]

    plt.figure(file +' '+ cell.upper() + ' Raster trial no. ' + str(trial), figsize = (12,6))
    plt.title(cell.upper() + ' Spikes')
    plt.scatter(spk[:,1], spk[:,0], s=5, color=clr)
    plt.axvline(x=trials_start[trial], color="red")
    plt.axvline(x=trials_start[trial]+burst_dur_cs, color="red")
    plt.axvline(x=trials_start[trial]+burst_dur, color="red")
    if window == 0:
        plt.xlim([1000,1480])
    elif window == 1:
        plt.xlim([1000,2100])
    plt.show()