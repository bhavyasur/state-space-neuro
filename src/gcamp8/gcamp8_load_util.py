"""
DATA DESCRIPTION:
both ETL and Bessel data from shivam. ETL data follows the same structures as the rbp-cre data and thus we can use the same functions.
many of the rbp-cre load functions are copied for ETL. bessel is also similar, with some adaptations.
"""

import scipy.io
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import glob
import scipy.ndimage as spnd
import warnings
from typing import Literal
import scipy.stats as stats

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict

def load_dfoverf_dendrite(data_path, session_date, path_type: Literal["sliceTCA", None] = None):
    """DATA_PATH IS NAME OF FOLDER that contains Ca_imaging_data.mat !!!!"""
    """returns a list of arrays. each item in list is a trial, each array is (num_neurons x num_timesteps)"""
    
    outer = Path(data_path)
    inner = Path(outer / "Ca_data-same_cells") 

    if path_type == "sliceTCA":
        matching_files = glob.glob(f"{inner}/*-Ca_sliceTCA_reconstructed_data.mat")
    else:
        matching_files = glob.glob(f"{inner}/*-Ca_manual_curate_data.mat")

    sessions = [file for file in matching_files if session_date in file]
    my_session = sessions[0]

    c = scipy.io.loadmat(my_session, simplify_cells=True)

    if path_type == "sliceTCA":
        pre = c['Ca_data']['ROI']['F']
        print("Using raw F data rather than DeltaFoverF, since path_type was set to sliceTCA.\n")
    else:
        pre = c['Ca_data']['ROI']['DeltaFoverF']
  
    dfoverf = list(np.asarray(pre[i]) for i in range(len(pre)))

    return dfoverf

def load_trialbreak_dendrite(data_path, session_date, path_type: Literal["sliceTCA", None] = None):
    outer = Path(data_path)
    inner = Path(outer / "Ca_data-same_cells") 
    if path_type == "sliceTCA":
        matching_files = glob.glob(f"{inner}/*-Ca_sliceTCA_reconstructed_data.mat")
    else:
        matching_files = glob.glob(f"{inner}/*-Ca_manual_curate_data.mat")

    sessions = [file for file in matching_files if session_date in file]
    my_session = sessions[0]

    c = scipy.io.loadmat(my_session, simplify_cells=True)

    pre = c['Ca_data']['ROI']['trial_break']

    return np.asarray(pre)

def load_trialtype_idx_dendrite(data_path, session_date, path_type: Literal["sliceTCA", None] = None):
    outer = Path(data_path)
    inner = Path(outer / "Ca_data-same_cells") 
    if path_type == "sliceTCA":
        matching_files = glob.glob(f"{inner}/*-Ca_sliceTCA_reconstructed_data.mat")
    else:
        matching_files = glob.glob(f"{inner}/*-Ca_manual_curate_data.mat")

    sessions = [file for file in matching_files if session_date in file]
    my_session = sessions[0]

    c = scipy.io.loadmat(my_session, simplify_cells=True)

    go = c['Ca_data']['ROI']['GO_trial']
    nogo = c['Ca_data']['ROI']['NOGO_trial']

    return go, nogo



def load_dfoverf_problemtest(data_path, path_type: Literal["binarized", "reconstructed", None] = None):
    """returns a list of arrays. each item in list is a trial, each array is (num_neurons x num_timesteps)"""

    c = scipy.io.loadmat(data_path, simplify_cells=True)

    if path_type == "binarized":
        pre = c['Ca_data']['ROI']['DeltaFoverF']

    elif path_type == "reconstructed":
        pre = c['Ca_data']['ROI']['DeltaFoverF_r']
    else:
        raise ValueError("path_type must be either 'binarized' or 'reconstructed' for this test.")

    dfoverf = list(np.asarray(pre[i]) for i in range(len(pre)))

    return dfoverf


def full_session_dendrite(dfoverf):
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


def full_session_trialsliced_dendrite(dfoverf):
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


def gonogotrials_sliced_dendrite(dfoverf, gonogo):
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


def full_session_trialsliced_thresholded_dendrite(dfoverf):
    """
    INPUT: dfoverf is a list, each item represents trial and is a numpy array of (num_neurons, num_timebins)
    OUTPUT: full_sess is a numpy array of (num neurons, num_trials * num_timebins). flattens the data so all trials are 
            represented in one row for each neuron.
    """
    num_trials = len(dfoverf)
    num_neurons = np.shape(dfoverf[0])[0]
    
    # 1st cut up each trial and turn into a list

    all_neurons = []
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

    # thresholding
    threshold = 0.5
    for row in range(np.shape(full_sess)[0]):
        if full_sess[row, :].max(axis=-1) < threshold:
            full_sess[row, :] = 0
        else:
            full_sess[row, :] = full_sess[row, :]

    full_sess_thresholded = full_sess[full_sess.any(axis=1)]
            
    return full_sess_thresholded


def behavioral_plot_dendrite(raw_data, session_date, trial_break, path_type: Literal["sliceTCA", None] = None, ax=None, trial_structure: Literal["single_trial", "trial_averaged", None] = None, trial_idx=None):
    """ should load in trial_break_sliced since the trials have 5% cut off beginning and end"""
    
    outer = Path(raw_data)
    inner = Path(outer / "Ca_data-same_cells") 

    if path_type == "sliceTCA":
        matching_files = glob.glob(f"{inner}/*-Ca_sliceTCA_reconstructed_data.mat")
    else:
        matching_files = glob.glob(f"{inner}/*-Ca_manual_curate_data.mat")

    sessions = [file for file in matching_files if session_date in file]
    my_session = sessions[0]

    c = scipy.io.loadmat(my_session, simplify_cells=True)

    Fs = c['Ca_data']['ROI']['Fs'] # float value
    piston_firing = c['Ca_data']['ROI']['piston_time'] # (1x2) array of floats

    piston_1 = piston_firing[0] * Fs
    piston_2 = piston_firing[1] * Fs

    if trial_structure == "single_trial":
        if not trial_idx:
            raise ValueError("trial_idx must be set to use trial_structure='single_trial'")
        my_trial_length = trial_break[trial_idx]

    elif trial_structure == "trial_averaged":
        min = np.min(trial_break)
        my_trial_length = min        

    else:
        raise ValueError("should be 'single_trial' or 'trial_averaged'")

    one_s = [0]
    one_e = [piston_1]
    two_s = [piston_1]
    two_e = [piston_2]
    three_s = [piston_2]
    three_e = [my_trial_length - 1]

    one_dur = np.subtract(one_e, one_s)
    two_dur = np.subtract(two_e, two_s)
    three_dur = np.subtract(three_e, three_s)
    
    # make gantt chart of online offline for full session

    categories = ["baseline", "sensory sampling", "reward"]

    ax.set_yticks([0.075, 0.225, 0.375], categories)
    ax.barh(0.075, one_dur, left=one_s, height=0.15, color='red', label="baseline", alpha=0.5)
    ax.barh(0.225, two_dur, left=two_s, height=0.15, color='blue', label="sensory sampling", alpha=0.5)
    ax.barh(0.375, three_dur, left=three_s, height=0.15, color='green', label="reward", alpha=0.5)

    ax.set_xlim(0, my_trial_length-1)

    return ax


# should load trial_break_sliced for this since the run function uses 
# def lick_and_reward_time(raw_data, session_date, path_type: Literal["sliceTCA", None] = None):
#     outer = Path(raw_data)
#     inner = Path(outer / "Ca_data-same_cells") 
#     if path_type == "sliceTCA":
#         matching_files = glob.glob(f"{inner}/*-Ca_sliceTCA_reconstructed_data.mat")
#     else:
#         matching_files = glob.glob(f"{inner}/*-Ca_manual_curate_data.mat")

#     sessions = [file for file in matching_files if session_date in file]
#     my_session = sessions[0]

#     c = scipy.io.loadmat(my_session, simplify_cells=True)

#     lick_time = c['Ca_data']['ROI']['lick_time']
#     reward_time = c['Ca_data']['ROI']['reward_time']
#     return 


# def behavioral_plot_dendrite(lick_time, reward_time, trial_break, ax=None, trial_selection: Literal["go", "nogo", None] = None, 
#                              trial_structure: Literal["single_trial", "trial_averaged", None] = None, trial_idx=None, gonogo=None):
#     """should load trial_break and NOT trial_break_sliced, since the lick_time and reward_time are before slicing."""

#     og_lick_trial_list = []
#     og_reward_trial_list = []
#     time_now = 0
#     for i in range(len(trial_break)):
#         len_trial = trial_break[i]
#         this_lick = lick_time[time_now:(time_now + len_trial)]
#         this_reward = reward_time[time_now:(time_now + len_trial)]
#         og_lick_trial_list.append(this_lick)
#         og_reward_trial_list.append(this_reward)
    
#     sliced_lick_trial_list = []
#     sliced_reward_trial_list = []
#     for trial in og_lick_trial_list:
#         length = len(og_lick_trial_list[trial]) # length of trial in terms of timesteps
#         fiveper = int(0.05 * length)
#         sliced_lick = trial[fiveper:(length-fiveper)]
#         sliced_lick_trial_list.append(sliced_lick)
#     for trial in og_reward_trial_list:
#         length = len(og_reward_trial_list[trial]) # length of trial in terms of timesteps
#         fiveper = int(0.05 * length)
#         sliced_reward = trial[fiveper:(length-fiveper)]
#         sliced_reward_trial_list.append(sliced_lick)

#     min_lick = np.min(len(arr) for arr in sliced_lick_trial_list)
#     min_reward = np.min(len(arr) for arr in sliced_reward_trial_list)

#     if min_lick != min_reward:
#         raise ValueError("Minimum length of trial for the lick and reward sets should be the same. Something is wrong in the code if this error has arisen.")
    
#     min = min_lick

#     if trial_structure == "single_trial":
#         warnings.warn("Since you set trial_structure to 'single_trial', if you are running on go or nogo trials only, the behavioral plot does not take this into account. This is because"
#         "you have set a trial index, that trial will be displayed regardless of the trial type.")

#     elif trial_structure == "trial_averaged":
#         if trial_selection == "go" or trial_selection == "nogo":

#             sliced_trunc_lick_list_gonogo = [] # list of the actual lick trials data for go or nogo 
#             sliced_trunc_reward_list_gonogo = [] # list of the actual reward trials data for go or nogo
#             for i in range(len(gonogo)):
#                 idx = gonogo[i] # matlab indexing, idx starts at 1
#                 gonogo_lick = sliced_lick_trial_list[idx]
#                 gonogo_reward = sliced_reward_trial_list[idx]
#                 trunc_gonogo_lick = gonogo_lick[0:min]
#                 trunc_gonogo_reward = gonogo_reward[0:min]
#                 sliced_trunc_lick_list_gonogo.append(trunc_gonogo_lick)
#                 sliced_trunc_reward_list_gonogo.append(trunc_gonogo_reward)
            
#             # now find average position of a lick occurrence and reward occurrence
#             stack = np.stack(truncated_list, axis=0)
#             result = stats.mode(stack, axis=0)
#             squeeze = result.mode.squeeze()
#             print("Successfully found mean zhat for full trial set.")
#             return squeeze # should be a numpy array of the averaged trial
        
#         else:
#             zhat_trial_list = []

#             time_now = int(0)
#             for i in range(len(trial_break)):
#                 len_trial = int(trial_break[i])
#                 trial_zhat = zhat_lem[time_now:(time_now + len_trial)]
#                 zhat_trial_list.append(trial_zhat)
#                 time_now += len_trial

#             if len(zhat_trial_list) != len(trial_break):
#                 raise ValueError("trial list does not contain the correct number of trials")

#             truncated_list = []
#             for trial in zhat_trial_list:
#                 retain = trial[0:min]
#                 truncated_list.append(retain)

#             if len(truncated_list) != len(trial_break):
#                 raise ValueError("truncated trial list does not contain the correct number of trials")
            
#             stack = np.stack(truncated_list, axis=0)
#             result = stats.mode(stack, axis=0)
#             squeeze = result.mode.squeeze()
#             print("Successfully found mean zhat for full trial set.")
#             return squeeze # should be a numpy array of the averaged trial
#     else:
#         raise ValueError("Your trial_structure should be set to either 'trial_averaged' or 'single_trial' for this plot.")



""""NOTE: WITH CONCATENATION, YOU CAN ONLY CONCATENATE ON NON THRESHOLDED SETS.
    to concatenate thresholded sets, must incorporate thresholding AFTER concatenating sessions. need to add to pipeline.
"""

def concatenate_sessions(list_fulls):
    """INPUT: takes in a list of full sessions (each is (num_neurons, num_timesteps)) and concatenates them based on condition type. Returns a concatenated full session that represents multiple sessions concatenated, 2D array."""

    # NOTE: this is used BEFORE pipeline and output is used as input to the pipeline if type=='session_concat' in pipeline instantiation.

    num_sessions = len(list_fulls)
    print("num sessions", num_sessions)
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

def session_concat_pipeline(folder_path, condition_dates: list[str], trial_selection: Literal["go", "nogo", None] = None, path_type: Literal["sliceTCA", None] = None):
    """pipeline to concatenate sessions from raw data. takes in list of date strings, returns concatenated session.
    condition_dates will be the input to 'dates' in run_rslds_pipeline"""

    num_sessions = len(condition_dates)
    list_full = []
    list_go = []
    list_nogo = []
    for date in condition_dates:
        dfoverf = load_dfoverf_dendrite(folder_path, date, path_type=path_type)
        go_idx, nogo_idx = load_trialtype_idx_dendrite(folder_path, session_date=date, path_type=path_type)
        single_session, _ = full_session_trialsliced_dendrite(dfoverf)
        single_session_go = gonogotrials_sliced_dendrite(dfoverf, go_idx)
        single_session_nogo = gonogotrials_sliced_dendrite(dfoverf, nogo_idx)
        list_full.append(single_session)
        list_go.append(single_session_go)  
        list_nogo.append(single_session_nogo)  

    if trial_selection == "go":
        concat_condition = concatenate_sessions(list_go)
        print(f"\nConcatenating go trials from {num_sessions} sessions.\n")
    elif trial_selection == "nogo":
        concat_condition = concatenate_sessions(list_nogo)
        print(f"\nConcatenating nogo trials from {num_sessions} sessions.\n")
    else:
        concat_condition = concatenate_sessions(list_full)
        print(f"\nConcatenating all trials from {num_sessions} sessions.\n")    

    return concat_condition # this is the equivalent of full
    

def trace_sanity_check_dendrite(full_sess, random_seed=None):
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
    folder_path = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2"
    
    # date sets per condition:
    naive = ["050425_23819"]
    naive_plot_key = '#23819M_GCaMP8m_rg_PoM-FOV2/naive_concat'

    disc_states = 4 
    latent_dims = 8

    # naive concat
    list_naive = []
    for date in naive:
        dfoverf = load_dfoverf_dendrite(folder_path, date)
        single_session = full_session_trialsliced_dendrite(dfoverf)
        list_naive.append(single_session)  
    concat_naive = concatenate_sessions(list_naive)
   
    mdic = {"concat_naive": concat_naive, "label": "full_data"}
    scipy.io.savemat("output/naive_concat_23819.mat", mdic)


    dfoverf_tester = load_dfoverf_dendrite(folder_path, "050425_23819")
    test = full_session_trialsliced_dendrite(dfoverf_tester)

    mdic2 = {"full_trialsliced_0504_23819": test, "label": "test single session"}
    scipy.io.savemat("output/full_trialsliced_0504_23819.mat", mdic2)


def spikes_smooth(data_path, session_date):

    outer = Path(data_path)
    inner = Path(outer / "Ca_data-same_cells") 
    matching_files = glob.glob(f"{inner}/*-Ca_manual_curate_data.mat")

    sessions = [file for file in matching_files if session_date in file]
    my_session = sessions[0]

    c = scipy.io.loadmat(my_session, simplify_cells=True)

    pre = c['Ca_data']['ROI']['Spikes']

    spikes = list(np.asarray(pre[i]) for i in range(len(pre)))

    full = full_session_dendrite(spikes)

    neural_data = full.T

    neural_data = spnd.gaussian_filter1d(neural_data, sigma=2, axis=-1) # (num_timesteps, num_neurons)

    return neural_data


if __name__ == "__main__":
    path = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2"
    date = "050425"

    trial_break = load_trialbreak_dendrite(path, date)
    print(np.shape(trial_break))
