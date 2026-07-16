"""
DATA DESCRIPTION:
both ETL and Bessel data from shivam. ETL data follows the same structures as the rbp-cre data and thus we can use the same functions.
many of the rbp-cre load functions are copied for ETL. bessel is also similar, with some adaptations.
"""

import scipy.io
import os
import numpy as np
import sys
from pathlib import Path
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import zipfile
from typing import Literal

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict

def load_trialtype_idx_l23(data_path, specific_loadtype: Literal["session_concat", None]=None):
    """can take either a single path or a list of paths to concatenate, for which you need to set specific_loadtype=='session_concat'"""
    if specific_loadtype == "session_concat":
        go_list = []
        nogo_list = []
        for session in data_path:
            outer = Path(session)
            calcium = outer / "Ca_imaging_data.mat"
            c = scipy.io.loadmat(calcium, simplify_cells=True)
            go_x = c['Ca_data']['ROI']['GO_trial']
            go_list.append(go_x)
            nogo_x = c['Ca_data']['ROI']['NOGO_trial']
            nogo_list.append(nogo_x) 
        go = np.concatenate(go_list)
        nogo = np.concatenate(nogo_list)
        print("Loaded go and nogo indices for concatenated sessions.\n")
    
    else:   
        outer = Path(data_path)
        calcium = outer / "Ca_imaging_data.mat"
        c = scipy.io.loadmat(calcium, simplify_cells=True)
        go = c['Ca_data']['ROI']['GO_trial']
        nogo = c['Ca_data']['ROI']['NOGO_trial']

    return go, nogo

def load_dfoverf_l23(data_path, layer: Literal["L2", "L3", None] = None):
    """DATA_PATH IS NAME OF FOLDER that contains Ca_imaging_data.mat !!!!"""
    """returns a list of arrays. each item in list is a trial, each array is (num_neurons x num_timesteps)"""
    """ note that layer is only for ETL data """

    outer = Path(data_path)
    calcium = outer / "Ca_imaging_data.mat"
    c = scipy.io.loadmat(calcium, simplify_cells=True)
    pre = c['Ca_data']['ROI']['DeltaFoverF']

    if layer == "L2" or layer == "L3":
        ROI_centroid = c['Ca_data']['ROI']['ROIcentroid']

        l2pre = []
        l3pre = []
        for i in range(len(pre)): # num trials
            l2_pertrial = []
            l3_pertrial = []
            for j in range(np.shape(ROI_centroid)[0]): # num neurons
                if ROI_centroid[j, 2] == 2:
                    l2_pertrial.append(pre[i][j, :])
                elif ROI_centroid[j, 2] == 3:
                    l3_pertrial.append(pre[i][j, :])
                else:
                    if ROI_centroid[j, 2] == -1:
                        pass
                    else:
                        raise ValueError("Layer should be 2, 3, or -1. Another value was found, please check your data.")
            l2pre.append(np.vstack(l2_pertrial))
            l3pre.append(np.vstack(l3_pertrial))

    # now l2pre and l3pre should be lists of arrays, each array is (num_neurons x num_timesteps) for that trial, but only for neurons in that layer.

    if layer == "L2":
        dfoverf = list(np.asarray(l2pre[i]) for i in range(len(l2pre)))
    elif layer == "L3":
        dfoverf = list(np.asarray(l3pre[i]) for i in range(len(l3pre)))
    else:
        dfoverf = list(np.asarray(pre[i]) for i in range(len(pre)))

    # print("dfoverf num trials", len(dfoverf))
    # print("dfoverf shape of first trial (num_neurons x num_frames)", np.shape(dfoverf[0]))
   
    return dfoverf

def full_session_l23(dfoverf):
    """
    INPUT: dfoverf is a list, each item represents trial and is a numpy array of (num_neurons, num_timebins)
    OUTPUT: full_sess is a numpy array of (num neurons, num_trials * num_timebins). flattens the data so all trials are 
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

def full_session_trialsliced_l23(dfoverf):
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

def gonogotrials_sliced_l23(dfoverf, gonogo):
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



def trace_sanity_check_l23(full_sess, random_seed=None):
    """this function visualizes a trace of all the neurons for a random set of 100 time steps so you can sanity check that the neurons activity is correct."""
    num_neurons = np.shape(full_sess)[0]
    
    if random_seed:
        rng = np.random.default_rng(seed=random_seed)
        randint = rng.integers(0,1000)
    else:
        rng = np.random.default_rng()
        randint = rng.integers(0,1000)
    
    len_slice = min(num_neurons, 15)
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

def load_trialbreak_l23(raw_data):
    outer = Path(raw_data)
    calcium = outer / "Ca_imaging_data.mat"
    c = scipy.io.loadmat(calcium, simplify_cells=True)
    trial_break = c['Ca_data']['ROI']['trial_break']

    return np.asarray(trial_break)
    


# take the non-trial-sliced version for this
def behavioral_plot_l23(trial_break, l23_type, ax=None, trial_structure: Literal["single_trial", "full_sess", None] = None, trial_idx=None):
    """ FOR L23: 
        Exact time of piston 1.67 seconds until 2.8 seconds after piston start: So your online frames will be frame number (1.67 until 2.78)*frame rate
        Offline is anything beyond 8 seconds after beginning of trial.
    """

    if l23_type == "bessel":
        frame_rate = 30.08 # frames per second
    elif l23_type == "etl":
        frame_rate = 15.01 # frames per second
    else:
        raise ValueError("type must be 'bessel' or 'etl'")
    
    online_start = 1.67 * frame_rate
    online_end = 2.78 * frame_rate
    offline_start = 8 * frame_rate

    if trial_structure == "single_trial":
        if not trial_idx:
            raise ValueError("trial_idx must be set to use trial_structure='single_trial'")
        my_trial_length = trial_break[trial_idx]

    elif trial_structure == "full_sess":
        my_trial_length = int(np.sum(trial_break))

    else:
        min = np.min(trial_break)
        my_trial_length = min        

    on_s = [online_start]
    on_e = [online_end]
    off_s = [offline_start]
    off_e = [my_trial_length - 1]

    online_duration = np.subtract(on_e, on_s)
    offline_duration = np.subtract(off_e, off_s)
    
    # make gantt chart of online offline for full session

    categories = ["offline", "online"]
    ax.set_yticks([0.075, 0.225], categories)
    ax.barh(0.225, online_duration, left=on_s, height=0.15, color='green', label="online", alpha=0.5)
    ax.barh(0.075, offline_duration, left=off_s, height=0.15, color='red', label="offline", alpha=0.5)
    ax.set_xlim(0, my_trial_length-1)

    return ax


def concat_sessions_l23(list_fulls):
    """INPUT: takes in a list of full sessions (each is (num_neurons, num_timesteps)) and concatenates them based on condition type. Returns a concatenated full session that represents multiple sessions concatenated, 2D array."""

    # NOTE: this is used BEFORE pipeline and output is used as input to the pipeline if type=='session_concat' in pipeline instantiation.

    num_sessions = len(list_fulls)
    num_neurons = np.shape(list_fulls[0])[0]
    
    sum_timebins = sum(np.shape(list_fulls[i])[1] for i in range(num_sessions))

    full_sess = np.zeros((num_neurons, sum_timebins))

    for i in range(num_neurons):
        time_now = 0
        for j in range(num_sessions):
            x = list_fulls[j][i, :]
            num_timebins_this_session = np.shape(list_fulls[j])[1]
            full_sess[i, time_now : num_timebins_this_session+time_now] = x
            time_now += num_timebins_this_session
        
    return full_sess

def session_concat_pipeline_l23(list_of_folder_paths, trial_selection: Literal["go", "nogo", None] = None, layer: Literal["L2", "L3", None] = None):
    """pipeline to concatenate sessions from raw data. takes in list of date strings, returns concatenated session.
    condition_dates will be the input to 'dates' in run_rslds_pipeline"""

    num_sessions = len(list_of_folder_paths)
    list_full = []
    list_go = []
    list_nogo = []
    list_breaksliced = []
    for session in list_of_folder_paths:
        dfoverf = load_dfoverf_l23(session, layer=layer)
        go_idx, nogo_idx = load_trialtype_idx_l23(session)
        single_session, tb_sliced = full_session_trialsliced_l23(dfoverf)
        single_session_go = gonogotrials_sliced_l23(dfoverf, go_idx)
        single_session_nogo = gonogotrials_sliced_l23(dfoverf, nogo_idx)
        list_breaksliced.append(tb_sliced)
        list_full.append(single_session)
        list_go.append(single_session_go)  
        list_nogo.append(single_session_nogo)  

    trial_break_sliced_concat = np.concatenate(list_breaksliced, axis=0)

    if trial_selection == "go":
        concat_condition = concat_sessions_l23(list_go)
        print(f"Concatenating go trials from {num_sessions} sessions.\n")
    elif trial_selection == "nogo":
        concat_condition = concat_sessions_l23(list_nogo)
        print(f"Concatenating nogo trials from {num_sessions} sessions.\n")
    else:
        concat_condition = concat_sessions_l23(list_full)
        print(f"Concatenating all trials from {num_sessions} sessions.\n")    

    return concat_condition, trial_break_sliced_concat # this is the equivalent of full



def keep_untracked(path1, path2):
    """returns a numpy array of the cells that are in path1 but are not in path2. this is specifically to check the significance of the cells tracked vs untracked cells in shivam's dataset"""

    dfoverf1 = load_dfoverf_l23(path1) # LARGER ARRAY
    dfoverf2 = load_dfoverf_l23(path2) # TRACKED ARRAY
    trial_break1 = load_trialbreak_l23(path1)
    mid = int(trial_break1[0]) / 2
    start = mid - 5
    end = mid + 5

    full1 = full_session_l23(dfoverf1)
    print("full1 shape", np.shape(full1))
    full2 = full_session_l23(dfoverf2)
    print("full2 shape", np.shape(full2))

    all_idx = np.arange(np.shape(full1)[0])
    print("all idx", np.shape(all_idx))
    aligned_idx_list = []

    for i in range(np.shape(full2)[0]):
        row2 = full2[i,:]
        for j in range(np.shape(full1)[0]):
            row1 = full1[j,:]
            if np.allclose(row1, row2):
                aligned_idx_list.append(j)
                break

    aligned_idx = np.array(aligned_idx_list)

    print("aligned shape", np.shape(aligned_idx))
    
    not_aligned_idx_list = np.setdiff1d(all_idx, aligned_idx)

    print("not aligned", np.shape(not_aligned_idx_list))

    save = []
    for idx in not_aligned_idx_list:
        row = full1[idx,:]
        save.append(row)

    full_untracked = np.vstack(save)

    return full_untracked

    
# ---------- running things but ignore for now

if __name__ == "__main__":

    p1 = "data/shivam/Bessel_140_250/1348DR/Expert/GO"

    p2 = "data/shivam/Bessel_140_250/1348DR/Naive_to_expert/Operant/In"

    full_unt = keep_untracked(p1, p2)

    print(np.shape(full_unt))