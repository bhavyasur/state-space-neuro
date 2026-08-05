"""
DATA DESCRIPTION:
M2 data from DJ. put main .mat file into 'data' folder. then set that to be the path. from there, variable is 'NeuronByDay', then 'Dxxxx', 
where xxxx is the date as MMDD. within each day there is 'SigD', which is the dfoverf activity from individual neurons.
"""

import scipy.io
import os
import numpy as np
import sys
from pathlib import Path
import mat73
import logging
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import zipfile
from typing import Literal
import pandas as pd
import matlab.engine

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

sys.path.append(str(Path(__file__).parent.parent.parent))


from utils.utils import mat_to_dict

# def load_trialtype_idx_l23(data_path, specific_loadtype: Literal["session_concat", None]=None):
#     """can take either a single path or a list of paths to concatenate, for which you need to set specific_loadtype=='session_concat'"""
#     if specific_loadtype == "session_concat":
#         go_list = []
#         nogo_list = []
#         for session in data_path:
#             outer = Path(session)
#             calcium = outer / "Ca_imaging_data.mat"
#             c = scipy.io.loadmat(calcium, simplify_cells=True)
#             go_x = c['Ca_data']['ROI']['GO_trial']
#             go_list.append(go_x)
#             nogo_x = c['Ca_data']['ROI']['NOGO_trial']
#             nogo_list.append(nogo_x) 
#         go = np.concatenate(go_list)
#         nogo = np.concatenate(nogo_list)
#         print("Loaded go and nogo indices for concatenated sessions.\n")
    
#     else:   
#         outer = Path(data_path)
#         calcium = outer / "Ca_imaging_data.mat"
#         c = scipy.io.loadmat(calcium, simplify_cells=True)
#         go = c['Ca_data']['ROI']['GO_trial']
#         nogo = c['Ca_data']['ROI']['NOGO_trial']

#     return go, nogo


def bin_sigd_m2(sigd, bin_size: int):
    """INPUT: sigd is a 2D array of (num_neurons, num_timesteps). bin_size is the number of timebins to average over. OUTPUT: binned_sigd is a 2D array of (num_neurons, num_timesteps/bin_size)"""
    num_neurons = np.shape(sigd)[0]
    num_timesteps = np.shape(sigd)[1]
    num_bins = int(num_timesteps/bin_size)
    binned_sigd = np.zeros((num_neurons, num_bins))
    for i in range(num_neurons):
        for j in range(num_bins):
            start = j*bin_size
            end = (j+1)*bin_size
            binned_sigd[i,j] = np.mean(sigd[i,start:end])
    
    return binned_sigd

def load_sigd_m2(data_path, date, trial_selection: Literal["right", "left", None] = None, m2_correct_only=False):
    """
    INPUT: dfoverf is a list, each item represents trial and is a numpy array of (num_neurons, num_timebins)
            gonogo is a 1d array of indices that represent which trials are go trials or nogo trials.
    OUTPUT: full_sess is a numpy array of (num neurons, num_trials * num_timebins). flattens the data so all trials are 
            represented in one row for each neuron.
    """
    # LOAD SIGD
    calcium = Path(data_path)
    
    logging.getLogger().setLevel(logging.CRITICAL)

    c = mat73.loadmat(calcium)

    logging.getLogger().setLevel(logging.WARNING)

    pre = c['NeuronByDay'][f'D{date}']['SigD']

    sigd = pre

    # GETTING NECESSARY VARIABLES FROM MAT FILE
    eng = matlab.engine.start_matlab()
    
    eng.load(data_path, nargout=0)

    eng.eval(f"T2 = NeuronByDay.D{date}.S_WM1.T2;", nargout=0)
    eng.eval("T2_struct = table2struct(T2, 'ToScalar', true);", nargout=0)

    S = eng.workspace['T2_struct']

    instructed_turn = list(S['InstructedTurn'])
    correct = list(S['Correct'])

    eng.eval(f"WM1 = NeuronByDay.D{date}.S_WM1.eventFrameIdx;", nargout=0)
    eng.eval(f"Return = NeuronByDay.D{date}.S_RETURN.eventFrameIdx;", nargout=0)
    
    wm1 = list(eng.workspace['WM1'])
    ret = list(eng.workspace['Return'])

    # IF: right or left trials only
    if trial_selection:
        keep_trial_idx = []
        if trial_selection == "right":
            if m2_correct_only:
                keep_trial_idx = [i for i in range(len(instructed_turn)) if instructed_turn[i] == 1 and correct[i] == 1]
            else:
                keep_trial_idx = [i for i in range(len(instructed_turn)) if instructed_turn[i] == 1]
        elif trial_selection == "left":
            if m2_correct_only:
                keep_trial_idx = [i for i in range(len(instructed_turn)) if instructed_turn[i] == 0 and correct[i] == 1]
            else:
                keep_trial_idx = [i for i in range(len(instructed_turn)) if instructed_turn[i] == 0]

        retain = []
        for i in keep_trial_idx:
            start = int(wm1[i])
            end = int(ret[i])
            retain.append(sigd[:, start:end])

        sliced_sigd = np.hstack(retain, axis=1) # stack the trials along the time axis
        print('Shape of sliced_sigd:', np.shape(sliced_sigd))
        
        return sliced_sigd, keep_trial_idx

    # if: FULL TRIAL SET
    else:
        if m2_correct_only:
            correct_trial_idx = [i for i in range(len(correct)) if correct[i] == 1]
        
            retain = []
            for i in correct_trial_idx:
                start = int(wm1[i])
                end = int(ret[i])
                retain.append(sigd[:, start:end])
        
            sliced_sigd = np.hstack(retain, axis=1) # stack the trials along the time axis
            print('Shape of sliced_sigd, correct_only for all trials:', np.shape(sigd))
            return sliced_sigd, correct_trial_idx
        else:
            return sigd, list(range(len(instructed_turn))) # second return value is just a list of all trial indices, since we are not slicing the data


def trace_sanity_check_m2(binned):
    """this function visualizes a trace of all the neurons for a random set of 100 time steps so you can sanity check that the neurons activity is correct."""
    num_neurons = np.shape(binned)[0]
  
    rng = np.random.default_rng()
    randint = rng.integers(0,1000)
    
    len_slice = min(num_neurons, 15)
    sliced = binned[0:len_slice, randint:randint+1000]

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

def load_trialbreak_m2(data_path, date, sliced=False, idx_list: list = None):
    # you'd want to remove the first two trials, so the first value in trial_break is the value that you should retain the data from, delete everything before that
    eng = matlab.engine.start_matlab()

    eng.load(data_path, nargout=0)

    # Assuming the MAT file contains a variable named NeuronByDay
    eng.eval(f"WM1 = NeuronByDay.D{date}.S_WM1.eventFrameIdx;", nargout=0)
    eng.eval(f"Gate = NeuronByDay.D{date}.S_GATE.eventFrameIdx;", nargout=0)
    eng.eval(f"WM2 = NeuronByDay.D{date}.S_WM2.eventFrameIdx;", nargout=0)
    eng.eval(f"Cue = NeuronByDay.D{date}.S_CUE.eventFrameIdx;", nargout=0)
    eng.eval(f"Lick = NeuronByDay.D{date}.S_LICK.eventFrameIdx;", nargout=0)
    eng.eval(f"Return = NeuronByDay.D{date}.S_RETURN.eventFrameIdx;", nargout=0)

    eng.eval(f"T2 = NeuronByDay.D{date}.S_WM1.T2;", nargout=0)
    eng.eval("T2_struct = table2struct(T2, 'ToScalar', true);", nargout=0)

    S = eng.workspace['T2_struct']

    print("shape of T2_struct", np.shape(S))
    print("keys of T2_struct", S.keys())

    if sliced:
        trialidx = idx_list
    else:
        num_trials = len(S['TrialIdx'])
        trialidx = np.arange(num_trials)

    # store trial times as a list of dicts, each dict has times for Wm1, gate, wm2, cue played, lick, return
    all_trials = []
    for i in trialidx:
        trial_dict = {}
        trial_dict["WM1"] = eng.workspace['WM1'][i]
        trial_dict["Gate"] = eng.workspace['Gate'][i]
        trial_dict["WM2"] = eng.workspace['WM2'][i]
        trial_dict["CuePlayed"] = eng.workspace['Cue'][i]
        trial_dict["Lick"] = eng.workspace['Lick'][i]
        trial_dict["Return"] = eng.workspace['Return'][i]

        all_trials.append(trial_dict)

    return all_trials

def find_session_accuracy(data_path, date):
    calcium = Path(data_path)

    logging.getLogger().setLevel(logging.CRITICAL)

    c = mat73.loadmat(calcium)

    logging.getLogger().setLevel(logging.WARNING)

    T2 = c['NeuronByDay'][f'D{date}']['S_WM1']['T2']
    correct = T2["Correct"]
    accuracy = np.mean(correct)
    return accuracy

def zhat_lem_sliced_plot(zhat_slice, ax, disc_states):
    diff = np.diff(zhat_slice)
    timesteps = len(zhat_slice)
    trial_len = len(zhat_slice)
    rising_draft = np.where(diff != 0)[0] + 1 # the first index where the new term exists
    len_r = len(rising_draft)
    length_bar_draft = np.diff(rising_draft) 

    if rising_draft.size == 0:
        print("zhat_lem does not contain any state changes. this may be correct, but could indicate an error in your data. Please check!")
        duration = len(zhat_slice)
        ax.barh(0.075, [duration], left=[0], height=0.15, color=colors[0 % len(colors)], alpha=0.8)
        ax.set_yticks([0.075], [f"state {zhat_slice[0]}"])

    else:
        rising = np.concatenate(([0], rising_draft))
        length_bar = np.concatenate(([rising_draft[0]], length_bar_draft, [timesteps - rising_draft[len_r-1]]))

        # to length_bar, prepend the first index of rising
        tick_list = []
        states = [f"state {i}" for i in range(disc_states)]

        for i in range(disc_states):
            bar_list_per_state = []
            for j in range(len(rising)):
                # rising[j] is the time index where the state rises, length_bar[j] is how long it stays high
                if zhat_slice[rising[j]] == i:
                    bar_list_per_state.append((rising[j], length_bar[j]))
            tick = ((i*2)+1)*0.075
            tick_list.append(tick)
            ax.barh(tick, [length for _, length in bar_list_per_state], left=[start for start, _ in bar_list_per_state], height=0.15, color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_yticks(tick_list, states)

    return ax

def plot_zhatlem_indivtrials(trial_break, data_path, date, zhat_lem, disc_states, bin_size):
    
    all_trials = trial_break

    num_trials = len(all_trials)
    early = num_trials // 3
    middle = (num_trials * 2) // 3
    late = num_trials - 1
    trial_list = [early, middle, late]

    fig, axes = plt.subplots(len(trial_list), 1, figsize=(10, 4*len(trial_list)))
    
    for trial in trial_list:
        trial_dict = all_trials[trial-1]
        wm1 = float(trial_dict["WM1"][0])
        gate = float(trial_dict["Gate"][0])
        wm2 = float(trial_dict["WM2"][0])
        cue = float(trial_dict["CuePlayed"][0])
        lick = float(trial_dict["Lick"][0])
        ret = float(trial_dict["Return"][0])

        trial_start = int(wm1) // bin_size
        trial_end = int(ret) // bin_size

        print("trial_start", trial_start)
        print("trial_end", trial_end)

        ax = axes[trial_list.index(trial)]

        print("zhat_lem shape", len(zhat_lem))
        zhat_slice = zhat_lem[trial_start:trial_end]
        print(f"zhat_slice shape: {len(zhat_slice)}")

        zhat_lem_sliced_plot(zhat_slice, ax, disc_states)
        padding = 2
        ax.set_xlim(-padding, len(zhat_slice) - 1 + padding)

        print(f"Trial Index {trial}: WM1={wm1}, Gate={gate}, WM2={wm2}, CuePlayed={cue}, Lick={lick}, Return={ret}")

        ax.axvline(x=(wm1 // bin_size) - trial_start, lw=1.25, color='r', linestyle='--', label='WM1')
        ax.axvline(x=(gate // bin_size) - trial_start, lw=1.25, color='g', linestyle='--', label='Gate')
        ax.axvline(x=(wm2 // bin_size) - trial_start, lw=1.25, color='b', linestyle='--', label='WM2')
        ax.axvline(x=(cue // bin_size) - trial_start, lw=1.25, color='c', linestyle='--', label='Cue Played')
        ax.axvline(x=(lick // bin_size) - trial_start, lw=1.25, color='m', linestyle='--', label='Lick')
        ax.axvline(x=(ret // bin_size) - trial_start, lw=1.25, color='y', linestyle='--', label='Return')

        ax.set_title(f'Trial Index {trial}')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Most Likely State')
        ax.legend()

    return fig, axes


if __name__ == "__main__":

    p1 = "data/shivam/Bessel_140_250/1348DR/Expert/GO"

    p2 = "data/shivam/Bessel_140_250/1348DR/Naive_to_expert/Operant/In"
