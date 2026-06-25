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
import zipfile

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict


def load_behavior(folder_path):
    """
    INPUT: data is a folder for the session from a specific mouse. titled with session number and animal number, contains behavior_output folder which contains mat files
    OUTPUT: spikes is a list, each item represents a neuron and is a numpy array of (num trials x timesteps)
    """
    outer = Path(folder_path)
    inner = Path(outer / "behavior_output") 
    behavior = inner / "behavior_Intan.mat"

    b = scipy.io.loadmat(behavior)

    print("mat type: " + str(type(b))) # datatype debugging
    print("mat keys: " + str(b.keys())) # check keys in mat file

    behavior_dict = {
        "session_break_time": b['behavior']['session_break_time'], # length of session in ms?
        "Fs": b['behavior']['Fs'], # sampling rate for intan
        "GO_timeidx": b['behavior']['GO_timeidx'], # use to get trial times
        "GOend_timeidx": b['behavior']['GOend_timeidx'], # use to get trial times
        "NOGO_timeidx": b['behavior']['NOGO_timeidx'], # use to get trial times
        "NOGOend_timeidx": b['behavior']['NOGOend_timeidx'], # use to get trial times
        "triallength": b['behavior']['triallength'], # ??
        "GOidx": b['behavior']['GOidx'], # ?? use to correspond trial data from time_idx to GO or NOGO
        "NOGOidx": b['behavior']['NOGOidx'], # ?? use to correspond trial data from time_idx to GO or NOGO
        "hit_idx": b['behavior']['hit_idx'], # identifies which trials were hits
        "fa_idx": b['behavior']['fa_idx'], # identifies which trials were false alarms
        "teach_idx": b['behavior']['teach_idx'], # identifies which trials were teach trials
        "missed_teach_idx": b['behavior']['missed_teach_idx'],
        "hit_rate": b['behavior']['hit_rate'],
        "fa_rate": b['behavior']['fa_rate'],
        "GO_lick": b['behavior']['GO_lick'],
        "NOGO_lick": b['behavior']['NOGO_lick'],
        "trial_map": b['behavior']['trial_map']
    }

    return behavior_dict



def load_calcium(folder_path):
    """
    INPUT: data is a folder for the session from a specific mouse. titled with session number and animal number, contains behavior_output folder which contains mat files
    OUTPUT: spikes is a list, each item represents a neuron and is a numpy array of (num trials x timesteps)
    """
    outer = Path(folder_path)
    inner = Path(outer / "behavior_output") 
    calcium = inner / "Ca_imaging_data.mat"

    c = scipy.io.loadmat(calcium)

    roi1 = c['Ca_data']['ROI'][:, 0]
    roi2 = c['Ca_data']['ROI'][:, 1]
    roi3 = c['Ca_data']['ROI'][:, 2]

    calcium_dict_roi1 = {
        "trial_break": roi1["trial_break"],
        "Fs": roi1["Fs"],
        "ROIcentroid": roi1["ROIcentroid"],
        "DeltaFoverF": roi1["DeltaFoverF"], # (num trials, num_neurons, num_timebins). data in each cell of array is indicative of activity.
        "F": roi1["F"],
        "trial_start_time": roi1["trial_start_time"],
        "trial_end_time": roi1["trial_end_time"],
    }



def load_dfoverf(folder_path, roi: int):
    """returns a list of arrays. each item in list is a trial, each array is (num_neurons x num_timesteps)"""

    outer = Path(folder_path)
    inner = Path(outer / "behavior_output") 
    calcium = inner / "Ca_imaging_data.mat"

    c = scipy.io.loadmat(calcium, simplify_cells=True)

    pre = c['Ca_data']['ROI'][int(roi-1)]['DeltaFoverF']
    print(len(pre))
    print(np.shape(pre[0]))
    print(pre[0].dtype)
    dfoverf = list(np.asarray(pre[i]) for i in range(len(pre)))
    print("len of dfoverf", len(dfoverf))
    print("shape of dfoverf[0]", np.shape(dfoverf[0]))
    print("type dfoverf", type(dfoverf))
    print("type dfoverf[0]", type(dfoverf[0]))

    return dfoverf



def full_session_rbp(dfoverf):
    """
    INPUT: dfoverf is a list, each item represents trial and is a numpy array of (num_neurons, num_timebins)
    OUTPUT: full_sess is a numpy array of (num neurons, num_trials * num_timebins). flattens the data so all trials are 
            represented in one row for each neuron.
    """
    num_trials = len(dfoverf)
    num_neurons = np.shape(dfoverf[0])[0]
    
    sum_timebins = sum(np.shape(dfoverf[i])[1] for i in range(num_trials))
    print(sum_timebins) # full length of the trial-concatenation of timesteps

    full_sess = np.zeros((num_neurons, sum_timebins))

    for i in range(num_neurons):
        time_now = 0
        for j in range(num_trials):
            x = dfoverf[j][i, :]
            num_timebins_this_trial = np.shape(dfoverf[j])[1]
            full_sess[i, time_now : num_timebins_this_trial+time_now] = x
            time_now += num_timebins_this_trial
        
    return full_sess


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
    dfoverf = load_dfoverf(folder_path, 1)
    full = full_session_rbp(dfoverf)

    mdic = {"full": full, "label": "experiment"}
    scipy.io.savemat("output/091024_61601/tester_dfoverf.mat", mdic)
