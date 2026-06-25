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

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict

def load_dfoverf_l23(data_path):
    """DATA_PATH IS NAME OF FOLDER that contains Ca_imaging_data.mat !!!!"""
    """returns a list of arrays. each item in list is a trial, each array is (num_neurons x num_timesteps)"""

    outer = Path(data_path)
    calcium = outer / "Ca_imaging_data.mat"

    c = scipy.io.loadmat(calcium, simplify_cells=True)

    pre = c['Ca_data']['ROI']['DeltaFoverF']
    print(len(pre)) # = number of trials
    print(np.shape(pre[0]))
    print(pre[0].dtype)
    dfoverf = list(np.asarray(pre[i]) for i in range(len(pre)))
    print("len of dfoverf", len(dfoverf))
    print("shape of dfoverf[0]", np.shape(dfoverf[0]))
    print("type dfoverf", type(dfoverf))
    print("type dfoverf[0]", type(dfoverf[0]))

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

def trace_sanity_check_l23(full_sess, random_seed=None):
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

# if __name__ == "__main__":
#     folder_path = "data/shulan/090824_61601"
#     dfoverf = load_dfoverf(folder_path, 1)
#     full = full_session_rbp(dfoverf)

#     mdic = {"full": full, "label": "experiment"}
#     scipy.io.savemat("output/090824_61601/tester_dfoverf.mat", mdic)
