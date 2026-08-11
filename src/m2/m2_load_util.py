"""
DATA DESCRIPTION:
M2 data from DJ. put main .mat file into 'data' folder. then set that to be the path. from there, variable is 'NeuronByDay', then 'Dxxxx', 
where xxxx is the date as MMDD. within each day there is 'SigD', which is the dfoverf activity from individual neurons.
"""

import scipy.io
import os
import numpy as np
import warnings
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
from scipy import stats

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['font.size'] = 8
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 10  # For X and Y axis titles
plt.rcParams['xtick.labelsize'] = 9 # For X-axis tick numbers
plt.rcParams['ytick.labelsize'] = 9 # For Y-axis tick numbers
plt.rcParams['legend.fontsize'] = 11

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
    print('Shape of sigd:', np.shape(sigd))

    # GETTING NECESSARY VARIABLES FROM MAT FILE
    eng = matlab.engine.start_matlab()
    
    eng.load(data_path, nargout=0)

    eng.eval(f"T2 = NeuronByDay.D{date}.S_WM1.T2;", nargout=0)
    eng.eval("T2_struct = table2struct(T2, 'ToScalar', true);", nargout=0)

    S = eng.workspace['T2_struct']

    instructed_turn = np.int32(list(S['InstructedTurn']))
    correct = np.int32(list(S['Correct']))

    eng.eval(f"WM1 = NeuronByDay.D{date}.S_WM1.eventFrameIdx;", nargout=0)
    eng.eval(f"frameTime = NeuronByDay.D{date}.S_WM1.frameTime;", nargout=0)
    eng.eval(f"Return = NeuronByDay.D{date}.S_RETURN.eventFrameIdx;", nargout=0)
    
    wm1 = list(np.int32((eng.workspace['WM1'])))
    ret = list(np.int32((eng.workspace['Return'])))
    frame_time = np.array(list((eng.workspace['frameTime'])))

    if len(ret) != len(wm1):
        ret.append(frame_time[-1])
        warnings.warn("Length of Return and WM1 are not equal. Appending last frame time to Return to make them equal.")

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
            if i == keep_trial_idx[-1]: # if this value is the last value in keep_trial_idx
                end = int(frame_time[-1]) # use the last frame time as the end
            else:
                end = int(wm1[i+1]) # otherwise use the beginning of the next trial 
            retain.append(sigd[:, start:end])

        sliced_sigd = np.hstack(retain) # stack the trials along the time axis
        print('Shape of sliced_sigd:', np.shape(sliced_sigd))
        
        return sliced_sigd, keep_trial_idx

    # if: FULL TRIAL SET
    else:
        if m2_correct_only:
            correct_trial_idx = [i for i in range(len(correct)) if correct[i] == 1]

            print("len correct_trial_idx", len(correct_trial_idx))
            print("len wm1", len(wm1))
            print("len ret", len(ret))

            retain = []
            for i in correct_trial_idx:
                start = int(wm1[i])
                if i == correct_trial_idx[-1]: # if this value is the last value in keep_trial_idx
                    end = int(frame_time[-1]) # use the last frame time as the end
                else:
                    end = int(wm1[i+1]) # otherwise use the beginning of the next trial 
                retain.append(sigd[:, start:end])
        
            sliced_sigd = np.hstack(retain) # stack the trials along the time axis
            print('Shape of sliced_sigd, correct_only for all trials:', np.shape(sliced_sigd))
            return sliced_sigd, correct_trial_idx
        else:
            return sigd, list(range(len(instructed_turn))) # second return value is just a list of all trial indices, since we are not slicing the data


def trace_sanity_check_m2(binned):
    """this function visualizes a trace of all the neurons for a random set of 100 time steps so you can sanity check that the neurons activity is correct."""
    num_neurons = np.shape(binned)[0]
  
    # rng = np.random.default_rng()
    # randint = rng.integers(0,1000)
    
    len_slice = min(num_neurons, 15)
    sliced = binned[0:len_slice, 100:1100]

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

def load_trialbreak_m2(data_path, date):
    # you'd want to remove the first two trials, so the first value in trial_break is the value that you should retain the data from, delete everything before that
    eng = matlab.engine.start_matlab()

    eng.load(data_path, nargout=0)

    eng.eval(f"WM1 = NeuronByDay.D{date}.S_WM1.eventFrameIdx;", nargout=0)
    eng.eval(f"Gate = NeuronByDay.D{date}.S_GATE.eventFrameIdx;", nargout=0)
    eng.eval(f"WM2 = NeuronByDay.D{date}.S_WM2.eventFrameIdx;", nargout=0)
    eng.eval(f"Cue = NeuronByDay.D{date}.S_CUE.eventFrameIdx;", nargout=0)
    eng.eval(f"Lick = NeuronByDay.D{date}.S_LICK.eventFrameIdx;", nargout=0)
    eng.eval(f"Return = NeuronByDay.D{date}.S_RETURN.eventFrameIdx;", nargout=0)
    eng.eval(f"frameTime = NeuronByDay.D{date}.S_WM1.frameTime;", nargout=0)
    eng.eval(f"Fs = NeuronByDay.D{date}.S_WM1.fs;", nargout=0)

    wm1 = list(np.int32((eng.workspace['WM1'])))
    gate = list(np.int32((eng.workspace['Gate'])))
    wm2 = list(np.int32((eng.workspace['WM2'])))
    cue = list(np.int32((eng.workspace['Cue'])))
    lick = list(np.int32((eng.workspace['Lick'])))
    ret = list(np.int32((eng.workspace['Return'])))
    frame_time = np.array(list((eng.workspace['frameTime'])))
    Fs = float(eng.workspace['Fs'])

    if len(ret) != len(wm1):
        ret.append(frame_time[-1])
        warnings.warn("Length of Return and WM1 are not equal. Appending last frame time to Return to make them equal.")

    eng.eval(f"T2 = NeuronByDay.D{date}.S_WM1.T2;", nargout=0)
    eng.eval("T2_struct = table2struct(T2, 'ToScalar', true);", nargout=0)

    S = eng.workspace['T2_struct']

    num_trials = len(S['TrialIdx'])
    trialidx = np.arange(num_trials)

    # store trial times as a list of dicts, each dict has times for Wm1, gate, wm2, cue played, lick, return
    all_trials = []
    for i in trialidx:
        trial_dict = {}
        trial_dict["WM1"] = wm1[i]
        trial_dict["Gate"] = gate[i]
        trial_dict["WM2"] = wm2[i]
        trial_dict["CuePlayed"] = cue[i]
        trial_dict["Lick"] = lick[i]
        trial_dict["Return"] = ret[i]

        all_trials.append(trial_dict)

    return all_trials, Fs

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


def plot_zhatlem_indivtrials(trial_break, zhat_lem, retained_trial_idx, disc_states, bin_size):

    # all original trials
    all_trials = trial_break
    retained_trials = []
    compressed_start = 0

    for idx in retained_trial_idx:
        # cannot compute duration for the last original trial
        if idx >= len(all_trials) - 1:
            continue

        trial_dict = all_trials[idx]

        wm1 = int(trial_dict["WM1"][0])
        gate = int(trial_dict["Gate"][0])
        wm2 = int(trial_dict["WM2"][0])
        cue = int(trial_dict["CuePlayed"][0])
        lick = int(trial_dict["Lick"][0])
        ret = int(trial_dict["Return"][0])

        next_wm1 = int(all_trials[idx + 1]["WM1"][0])
        duration = next_wm1 - wm1

        retained_trials.append({
            "original_idx": idx,
            "trial_start": compressed_start,
            "trial_end": compressed_start + duration,
            "WM1": compressed_start,
            "Gate": compressed_start + (gate - wm1),
            "WM2": compressed_start + (wm2 - wm1),
            "CuePlayed": compressed_start + (cue - wm1),
            "Lick": compressed_start + (lick - wm1),
            "Return": compressed_start + (ret - wm1),
        })
        compressed_start += duration

    num_trials = len(retained_trials)

    trial_list = [ num_trials // 3, (2 * num_trials) // 3, num_trials - 1, ]

    fig, axes = plt.subplots(len(trial_list), 1, figsize=(10, 4 * len(trial_list)))

    if len(trial_list) == 1:
        axes = [axes]

    for ax, plot_idx in zip(axes, trial_list):
        trial = retained_trials[plot_idx]

        trial_start = trial["trial_start"] // bin_size
        trial_end = trial["trial_end"] // bin_size

        print("trial_start", trial_start)
        print("trial_end", trial_end)

        zhat_slice = zhat_lem[trial_start:trial_end]

        print("zhat_lem shape", len(zhat_lem))
        print("zhat_slice shape", len(zhat_slice))

        zhat_lem_sliced_plot(zhat_slice, ax, disc_states)

        padding = 2
        ax.set_xlim(-padding, len(zhat_slice) - 1 + padding)

        print(
            f"Original Trial {trial['original_idx']}: "
            f"WM1={trial['WM1'] // bin_size}, "
            f"Gate={trial['Gate'] // bin_size}, "
            f"WM2={trial['WM2'] // bin_size}, "
            f"Cue={trial['CuePlayed'] // bin_size}, "
            f"Lick={trial['Lick'] // bin_size}, "
            f"Return={trial['Return'] // bin_size}"
        )

        ax.axvline((trial["WM1"] // bin_size) - trial_start,
                   lw=1.25, color='r', linestyle='--', label='WM1')

        ax.axvline((trial["Gate"] // bin_size) - trial_start,
                   lw=1.25, color='g', linestyle='--', label='Gate')

        ax.axvline((trial["WM2"] // bin_size) - trial_start,
                   lw=1.25, color='b', linestyle='--', label='WM2')

        ax.axvline((trial["CuePlayed"] // bin_size) - trial_start,
                   lw=1.25, color='c', linestyle='--', label='Cue Played')

        ax.axvline((trial["Lick"] // bin_size) - trial_start,
                   lw=1.25, color='m', linestyle='--', label='Lick')

        ax.axvline((trial["Return"] // bin_size) - trial_start,
                   lw=1.25, color='y', linestyle='--', label='Return')

        ax.set_title(f"Original Trial {trial['original_idx']}")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Most Likely State")
        ax.legend()

    return fig, axes

def plot_zhatlem_lick(trial_break, zhat_lem, Fs, retained_trial_idx, disc_states, bin_size):

    # all original trials
    all_trials = trial_break
    retained_trials = []
    compressed_start = 0

    for idx in retained_trial_idx:
        # cannot compute duration for the last original trial
        if idx >= len(all_trials) - 1:
            continue

        trial_dict = all_trials[idx]

        wm1 = int(trial_dict["WM1"][0])
        gate = int(trial_dict["Gate"][0])
        wm2 = int(trial_dict["WM2"][0])
        cue = int(trial_dict["CuePlayed"][0])
        lick = int(trial_dict["Lick"][0])
        ret = int(trial_dict["Return"][0])

        next_wm1 = int(all_trials[idx + 1]["WM1"][0])
        duration = next_wm1 - wm1

        retained_trials.append({
            "original_idx": idx,
            "trial_start": compressed_start,
            "trial_end": compressed_start + duration,
            "WM1": compressed_start,
            "Gate": compressed_start + (gate - wm1),
            "WM2": compressed_start + (wm2 - wm1),
            "CuePlayed": compressed_start + (cue - wm1),
            "Lick": compressed_start + (lick - wm1),
            "Return": compressed_start + (ret - wm1),
        })
        compressed_start += duration

    fig, ax = plt.subplots(figsize=(10, 4))

    retain_zhat_list = [] # will contain sublists of the sliced zhat_lem you keep from each trial (1s before lick, 2s after lick)
    for i in range(len(retained_trials)):
        trial = retained_trials[i]
        trial_lick = trial["Lick"] // bin_size
        before = int(trial_lick - ((3 * Fs) // bin_size))
        after = int(trial_lick + ((3 * Fs) // bin_size))
        zhat_retain = zhat_lem[before:after]
        retain_zhat_list.append(zhat_retain)

    print("min len", min(len(i) for i in retain_zhat_list))
    print("max len", max(len(i) for i in retain_zhat_list))

    stack = np.stack(retain_zhat_list, axis=0)
    result = stats.mode(stack, axis=0)
    zhat_mode = result.mode.squeeze()

    print("Successfully found mean zhat for full trial set.\n")

    zhat_lem_sliced_plot(zhat_mode, ax, disc_states)

    ax.axvline(((3*Fs) // bin_size), lw=1.25, color='r', linestyle='--', label='Lick')

    return fig, ax


def plot_zhatlem_probability(trial_break, zhat_lem, Fs, retained_trial_idx, disc_states, bin_size):

    # all original trials
    all_trials = trial_break
    retained_trials = []
    compressed_start = 0

    for idx in retained_trial_idx:
        # cannot compute duration for the last original trial
        if idx >= len(all_trials) - 1:
            continue

        trial_dict = all_trials[idx]

        wm1 = int(trial_dict["WM1"][0])
        gate = int(trial_dict["Gate"][0])
        wm2 = int(trial_dict["WM2"][0])
        cue = int(trial_dict["CuePlayed"][0])
        lick = int(trial_dict["Lick"][0])
        ret = int(trial_dict["Return"][0])

        next_wm1 = int(all_trials[idx + 1]["WM1"][0])
        duration = next_wm1 - wm1

        retained_trials.append({
            "original_idx": idx,
            "trial_start": compressed_start,
            "trial_end": compressed_start + duration,
            "WM1": compressed_start,
            "Gate": compressed_start + (gate - wm1),
            "WM2": compressed_start + (wm2 - wm1),
            "CuePlayed": compressed_start + (cue - wm1),
            "Lick": compressed_start + (lick - wm1),
            "Return": compressed_start + (ret - wm1),
        })
        compressed_start += duration

    fig, ax = plt.subplots(figsize=(10, 4))

    retain_zhat_list = [] # will contain sublists of the sliced zhat_lem you keep from each trial (1s before lick, 2s after lick)
    for i in range(len(retained_trials)):
        trial = retained_trials[i]
        trial_lick = trial["Lick"] // bin_size
        before = int(trial_lick - ((3 * Fs) // bin_size))
        after = int(trial_lick + ((3 * Fs) // bin_size))
        zhat_retain = zhat_lem[before:after]
        retain_zhat_list.append(zhat_retain)

    print("min len", min(len(i) for i in retain_zhat_list))
    print("max len", max(len(i) for i in retain_zhat_list))

    stack = np.stack(retain_zhat_list, axis=0)
    
    print("Number of trials:", stack.shape[0])
    print("Frames per trial:", stack.shape[1])

    # probability of each state at each frame

    new_arr = np.zeros(
        (disc_states, stack.shape[1]),
        dtype=float
    )

    for state in range(disc_states):

        new_arr[state, :] = np.mean(
            stack == state,
            axis=0
        )

    for state in range(disc_states):

        ax.plot(
            np.arange(stack.shape[1]),
            new_arr[state, :],
            linewidth=2,
            label=f"State {state}",
            color=colors[state % len(colors)]
        )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, stack.shape[1])
    
    ax.axvline(((3*Fs) // bin_size), lw=1.25, color='r', linestyle='--', label='Lick')

    ax.legend()

    return fig, ax
    


if __name__ == "__main__":

    path_1357 = "data/dj/NeuronByDay_1357_qc_rescued.mat"
    plot_1357 = "M2_1357"

    naive_1357 = "0520"
    interm_1357 = "0605"
    expert_1357 = "0626"

    path_073723 = "data/dj/NeuronByDay_073723_qc_rescued.mat"
    plot_073723 = "M2_073723"

    naive_073723 = "0417" # 55%
    interm_073723 = "0424" 
    expert_073723 = "0501" # 83%


    # ----------------------------

    for date in [naive_1357, interm_1357, expert_1357]:
        for trial_selection in [None, "right", "left"]:
            for m2_correct_only in [False, True]:
                raw_data = path_1357
                key = plot_1357
                output_folder_1 = f"output/{key}/{date}"
                if trial_selection:
                    output_folder_2 = f"{output_folder_1}/{trial_selection}"
                else:
                    output_folder_2 = f"{output_folder_1}/full"
                if m2_correct_only:
                    output_folder_3 = f"{output_folder_2}/correct_only"
                else:
                    output_folder_3 = output_folder_2

                new_key = output_folder_3

                output_folder = Path(f"{output_folder_3}/4states_8dims")
                output_folder.mkdir(parents=True, exist_ok=True)

                sigd, retained_trial_idx = load_sigd_m2(raw_data, date=date, trial_selection=trial_selection, m2_correct_only=m2_correct_only)
                binned = bin_sigd_m2(sigd, bin_size=3)
                data = binned.T.astype(int)
                for_trace = binned

                fig0, axes0 = trace_sanity_check_m2(for_trace)
                fig0.suptitle(f"Calcium Trace of Neurons: {new_key}")  
                fig0.savefig(output_folder / "calcium_trace.png")
                print(f"saved at {output_folder}")


    for date in [naive_073723, interm_073723, expert_073723]:
        for trial_selection in [None, "right", "left"]:
            for m2_correct_only in [False, True]:
                raw_data = path_073723
                key = plot_073723
                output_folder_1 = f"output/{key}/{date}"
                if trial_selection:
                    output_folder_2 = f"{output_folder_1}/{trial_selection}"
                else:
                    output_folder_2 = f"{output_folder_1}/full"
                if m2_correct_only:
                    output_folder_3 = f"{output_folder_2}/correct_only"
                else:
                    output_folder_3 = output_folder_2

                new_key = output_folder_3

                output_folder = Path(f"{output_folder_3}/4states_8dims")
                output_folder.mkdir(parents=True, exist_ok=True)

                sigd, retained_trial_idx = load_sigd_m2(raw_data, date=date, trial_selection=trial_selection, m2_correct_only=m2_correct_only)
                binned = bin_sigd_m2(sigd, bin_size=3)
                data = binned.T.astype(int)
                for_trace = binned

                fig0, axes0 = trace_sanity_check_m2(for_trace)
                fig0.suptitle(f"Calcium Trace of Neurons: {new_key}")  
                fig0.savefig(output_folder / "calcium_trace.png")
                print(f"saved at {output_folder}")