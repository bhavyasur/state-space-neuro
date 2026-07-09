"""
DATA DESCRIPTION:
Rbp-Cre data from Shulan. Once again working session-by-session. Each session contains data from Arduino, behavior data, and calcium imaging data.
I am ignoring the arduino data since there is a serial communication delay and working exclusively with behavior and Ca imaging data.

Rbp-Cre dataset is for L5 pyramidal cells, the retrograde virus injected ensures that it's only those neurons projecting to a specific region

Animal folder -> "behavior_output" -> "behavior_Intan" and "Ca_imaging_data".
behavior_Intan loads a mat struct called "behavior", Ca_imaging_data loads a mat struct called "Ca_data"  """

import scipy.io
import os
import numpy as np
import sys
from pathlib import Path
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import glob
import zipfile
from typing import Literal

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict


def load_dfoverf_rbp(folder_path, path_type: Literal["suite2p", "manual", None] = None, roi: int = None):
    """returns a list of arrays. each item in list is a trial, each array is (num_neurons x num_timesteps)"""

    outer = Path(folder_path)
    inner = Path(outer / "behavior_output") 

    if path_type == "suite2p":
        calcium = inner / "Ca_suite2p_data.mat"
        c = scipy.io.loadmat(calcium, simplify_cells=True)
        if roi: # for tuft v trunk. roi=1 is tuft, roi=2 is trunk
            pre = c['Ca_data']['ROI'][int(roi-1)]['DeltaFoverF']
        else:
            pre = c['Ca_data']['ROI']['DeltaFoverF']

    elif path_type == "manual":
        c = scipy.io.loadmat(folder_path, simplify_cells=True) # it's not a folder, it's a file, but variable name is folder_path
        pre = c['Ca_data']['ROI']['DeltaFoverF']

    else:
        calcium = inner / "Ca_imaging_data.mat"
        c = scipy.io.loadmat(calcium, simplify_cells=True)
        pre = c['Ca_data']['ROI'][int(roi-1)]['DeltaFoverF']

    dfoverf = list(np.asarray(pre[i]) for i in range(len(pre)))

    return dfoverf

def full_session_rbp(dfoverf):
    """
    INPUT: dfoverf is a list, each item represents trial and is a numpy array of (num_neurons, num_timebins)
    OUTPUT: full_sess is a numpy array of (num neurons, n`um_trials * num_timebins). flattens the data so all trials are 
            represented in one row for each neuron.
    """
    num_trials = len(dfoverf)
    num_neurons = np.shape(dfoverf[0])[0]
    
    sum_timebins = sum(np.shape(dfoverf[i])[1] for i in range(num_trials))

    full_sess = np.zeros((num_neurons, sum_timebins))

    for i in range(num_neurons):
        time_now = 0
        for j in range(num_trials):
            x = dfoverf[j][i, :]
            num_timebins_this_trial = np.shape(dfoverf[j])[1]
            full_sess[i, time_now : num_timebins_this_trial+time_now] = x
            time_now += num_timebins_this_trial
        
    return full_sess

def full_session_trialsliced_rbp(dfoverf):
    """
    INPUT: dfoverf is a list, each item represents trial and is a numpy array of (num_neurons, num_timebins)
    OUTPUT: full_sess is a numpy array of (num neurons, num_trials * num_timebins). flattens the data so all trials are 
            represented in one row for each neuron.
    """
    num_trials = len(dfoverf)
    num_neurons = np.shape(dfoverf[0])[0]
    
    # 1st cut up each trial and turn into a list

    all_neurons = []
    trial_break_sliced = np.zeros(num_trials)
    for i in range(num_neurons):
        sliced_list = []
        time_now = 0
        for j in range(num_trials):
            trial_row_single_neuron = dfoverf[j][i, :]
            # print("trial row shape", np.shape(trial_row_single_neuron))
            length = np.shape(trial_row_single_neuron)[0]
            fiveper = int(0.05 * length)
            # print("fiveper", fiveper)
            sliced_trial = trial_row_single_neuron[fiveper:(length-fiveper),]
            # print("sliced trial shape", np.shape(sliced_trial))
            sliced_list.append(sliced_trial)
            slice_length = np.shape(sliced_trial)[0]
            trial_break_sliced[j] = slice_length
            time_now += slice_length

        single_neuron_concat = np.zeros(time_now)
        # at this point you have an empty array for each neuron that is the length of all the concatenated trials combined.
        # you also have the sliced_list which is each trial.

        time_iterator = 0
        for j in range(num_trials):
            trial = sliced_list[j]
            slice_trial_length = np.shape(trial)[0]
            single_neuron_concat[time_iterator:(time_iterator+slice_trial_length),] = trial
            time_iterator += slice_trial_length
        
        all_neurons.append(single_neuron_concat)
        # now for each neuron you have single_neuron_concat which is the row array of all trials for that neuron

    #all_neurons at this point should be a list of 1D arrays, of shape (num_neurons, length of all trials)
    full_length = np.shape(all_neurons)[1]
    full_sess = np.zeros((num_neurons, full_length))
    for i in range(num_neurons):
        full_sess[i, :] = all_neurons[i]

    return full_sess, trial_break_sliced

def load_trialtype_idx_rbp(data_path, path_type: Literal["suite2p", "manual", None] = None, roi: int = None):

    outer = Path(data_path)
    inner = Path(outer / "behavior_output") 

    if path_type == "suite2p":
        calcium = inner / "Ca_suite2p_data.mat"
        c = scipy.io.loadmat(calcium, simplify_cells=True)
        if roi:
            go = c['Ca_data']['ROI'][int(roi-1)]['GO_trial']
            nogo = c['Ca_data']['ROI'][int(roi-1)]['NOGO_trial']
        else:
            go = c['Ca_data']['ROI']['GO_trial']
            nogo = c['Ca_data']['ROI']['NOGO_trial']

    elif path_type == "manual":
        c = scipy.io.loadmat(folder_path, simplify_cells=True) # it's not a folder, it's a file, but variable name is folder_path
        go = c['Ca_data']['ROI'][int(roi-1)]['GO_trial']
        nogo = c['Ca_data']['ROI'][int(roi-1)]['NOGO_trial']
    else:
        calcium = inner / "Ca_imaging_data.mat"
        c = scipy.io.loadmat(calcium, simplify_cells=True)
        go = c['Ca_data']['ROI']['GO_trial']
        nogo = c['Ca_data']['ROI']['NOGO_trial']

    return go, nogo



def load_trialbreak_rbp(data_path, path_type: Literal["suite2p", "manual", None] = None, roi: int = None):

    outer = Path(data_path)
    inner = Path(outer / "behavior_output") 

    if path_type == "suite2p":
        calcium = inner / "Ca_suite2p_data.mat"
        c = scipy.io.loadmat(calcium, simplify_cells=True)
        if roi:
            pre = c['Ca_data']['ROI'][int(roi-1)]['trial_break']
        else:
            pre = c['Ca_data']['ROI']['trial_break']

    elif path_type == "manual":
        c = scipy.io.loadmat(folder_path, simplify_cells=True) # it's not a folder, it's a file, but variable name is folder_path
        pre = c['Ca_data']['ROI']['trial_break']

    else:
        calcium = inner / "Ca_imaging_data.mat"
        c = scipy.io.loadmat(calcium, simplify_cells=True)
        pre = c['Ca_data']['ROI'][int(roi-1)]['trial_break']

    return np.asarray(pre)


def gonogotrials_sliced_rbp(dfoverf, gonogo):
    """
    INPUT: dfoverf is a list, each item represents trial and is a numpy array of (num_neurons, num_timebins)
            gonogo is a 1d array of indices that represent which trials are go trials or nogo trials.
    OUTPUT: full_sess is a numpy array of (num neurons, num_trials * num_timebins). flattens the data so all trials are 
            represented in one row for each neuron.
    """
    num_neurons = np.shape(dfoverf[0])[0]

    # use go trial indices to only select go trials
    gonogo_trials = []
    for i in range(len(gonogo)):
        idx = gonogo[i]
        trial = dfoverf[idx-1]
        gonogo_trials.append(trial)

    num_trials = len(gonogo)

    # 1st cut up each trial and turn into a list

    all_neurons = []
    for i in range(num_neurons):
        sliced_list = []
        time_now = 0
        for j in range(num_trials):
            trial_row_single_neuron = gonogo_trials[j][i, :]
            # print("trial row shape", np.shape(trial_row_single_neuron))
            length = np.shape(trial_row_single_neuron)[0]
            fiveper = int(0.05 * length)
            # print("fiveper", fiveper)
            sliced_trial = trial_row_single_neuron[fiveper:(length-fiveper),]
            # print("sliced trial shape", np.shape(sliced_trial))
            sliced_list.append(sliced_trial)
            slice_length = np.shape(sliced_trial)[0]
            time_now += slice_length

        single_neuron_concat = np.zeros(time_now)
        # at this point you have an empty array for each neuron that is the length of all the concatenated trials combined.
        # you also have the sliced_list which is each trial.

        time_iterator = 0
        for j in range(num_trials):
            trial = sliced_list[j]
            slice_trial_length = np.shape(trial)[0]
            single_neuron_concat[time_iterator:(time_iterator+slice_trial_length),] = trial
            time_iterator += slice_trial_length
        
        all_neurons.append(single_neuron_concat)
        # now for each neuron you have single_neuron_concat which is the row array of all trials for that neuron

    #all_neurons at this point should be a list of 1D arrays, of shape (num_neurons, length of all trials)
    gonogo_length = np.shape(all_neurons)[1]
    gonogo_sess = np.zeros((num_neurons, gonogo_length))
    for i in range(num_neurons):
        gonogo_sess[i, :] = all_neurons[i]

    return gonogo_sess



def visualize_trace(full_sess):
    num_neurons = np.shape(full_sess)[0]

    fig, axes = plt.subplots(nrows=num_neurons, ncols=1, figsize=(10, 6), sharex=True)

    for i in range(num_neurons):
        axes[i].plot(full_sess[i, :], lw=1.5)
        axes[i].set_ylabel(f"Neuron {i+1}", fontsize=8)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Index", fontsize = 10)

    return fig, axes

def trace_sanity_check(full_sess, random_seed=None):
    """this function visualizes a trace of all the neurons for a random set of 100 time steps so you can sanity check that the neurons activity is correct."""
    num_neurons = np.shape(full_sess)[0]
    
    if random_seed:
        rng = np.random.default_rng(seed=random_seed)
        randint = rng.integers(0,1000)
    else:
        rng = np.random.default_rng()
        randint = rng.integers(0,1000)
    
    len_slice = 15
    sliced = full_sess[0:len_slice, randint:randint+1000]

    fig, axes = plt.subplots(nrows=len_slice, ncols=1, figsize=(8, 8), sharex=True)

    for i in range(len_slice):
        axes[i].plot(sliced[i, :], lw=1.5)
        axes[i].set_ylabel(f"Neuron {i+1}", fontsize=7, rotation=90)
        axes[i].grid(True, alpha=0.3)

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    axes[-1].set_xlabel("Time Index", fontsize = 10)

    return fig, axes

    

# ---------- running things but ignore for now

if __name__ == "__main__":
    folder_path = "data/shulan/091024_61601"
    dfoverf = load_dfoverf_rbp(folder_path, 1)
    full = full_session_rbp(dfoverf)

    mdic = {"full": full, "label": "experiment"}
    scipy.io.savemat("output/091024_61601/tester_dfoverf.mat", mdic)
