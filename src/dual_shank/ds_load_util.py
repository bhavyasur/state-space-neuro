"""
DATA DESCRIPTION FROM OM:
it all has M1 and M2 spikes. We can just focus on M1 spikes first and then see if we can incorporate M2 spikes as input to the dynamical system.
M1Spikes.PSTH.hit is the cell array that has the spiking data for the hit Trial.

It is a cell array with each cell corresponding to individual neuron detected.
each cell is a trial x time matrix with the spiking acitivity of that particular neuron across all the trials. 
you can start with any mat file
"""

import scipy.io
import numpy as np
import sys
from pathlib import Path
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict

data = "data/om/07538_M1_Day1_CCA_data.mat"

def load_spikes(data):
    """
    INPUT: data is a .mat file from the DualShank dataset in Box. this file contains keys 'fpath', 'IntanBehavior', 'M1Spikes', 'M2Spikes'
    OUTPUT: spikes is a list, each item represents a neuron and is a numpy array of (num trials x timesteps)
    """
    mat = mat73.loadmat(data)
    # print("mat type: " + str(type(mat))) # datatype debugging
    # print("mat keys: " + str(mat.keys())) # check keys in mat file

    spikes = mat['M1Spikes']['PSTH']['hit']['spks']
    print(np.shape(spikes)) # check shape of spikes array  
    print("dtype spikes[0]", type(spikes[0]))

    return spikes

def load_spikes_all_days(data_zipped):
    """loads the spikes for all days that sessions were taken. essentially, concatenates sessions
    INPUT: zip file of all sessions to put together. this is specifically for chronic data."""

def psth_firing(spikes, PSTH_length): 
    """calculates firing rate from raw spiking data using Peristimulus Time Histogram
    INPUT: spikes is list, each item represents trial and is a numpy array of (num neurons x timesteps)
           PSTH_length is the length of each trial in seconds
    OUTPUT: rates is a numpy array of (num neurons x timesteps), and is trial-averaged. gives the firing rate of each neuron at each timestep.
    """

    rates = np.zeros((len(spikes), np.shape(spikes)[2])) # initialize array to hold firing rates for each neuron, shape is num_neurons x num_time_bins

    for i in range(len(spikes)): # for each neuron
        trial = spikes[i] # (64 x 3001) array of trials x time steps for each neuron
        num_trials = np.shape(trial)[0]
        num_bins = int(np.shape(trial)[1]) 
        time_step = float(PSTH_length * pq.s/ (num_bins - 1)) * pq.s
        collapsed = np.sum(trial, axis=0)
        """sum across trials to get total spike count at each time step, then 
        divide by number of trials and time step to get firing rate"""
        rho = (1 / time_step) * (1 / num_trials) * collapsed * pq.Hz
        rates[i] = rho
    
    return rates 

# EXAMPLE PARAMETERS

def select_trial(spikes, trial_idx):
    """selects a single trial to pull data from across all neurons.
    INPUT: spikes is list, each item represents trial and is a numpy array of (num neurons x timesteps)
           trial_idx is the index of the trial you want to select for your dataset
    OUTPUT: trial_spikelist is a numpy array of (num neurons x timestep), represents the spiking data of one single trial
    """
    trial_spikelist = []
    for i in range(len(spikes)):
        trial = spikes[i][trial_idx] # check shape of trial array
        # print("trial shape:", np.shape(trial)) # check shape of trial array 
        trial_spikelist.append(trial)  

    return trial_spikelist

def full_session(spikes):
    """
    INPUT: spikes is a list, each item represents trial and is a numpy array of (num neurons x timesteps)
    OUTPUT: full_sess is a numpy array of (num neurons x num_trials*num_timesteps_per_trial). flattens the data so all trials are 
            represented in one row for each neuron.
    """
    num_neurons = np.shape(spikes)[0]
    num_trials = np.shape(spikes)[1]
    num_timesteps = np.shape(spikes)[2]
    full_sess = np.zeros((num_neurons, num_trials * num_timesteps))
    for i in range(num_neurons):
        trialsbytime = spikes[i]
        df = pd.DataFrame(trialsbytime)
        df_flat = pd.DataFrame(df.to_numpy().reshape(1,-1))
        # print(df_flat.shape)
        full_sess[i] = df_flat
    return full_sess

def visualize_trial(trial_spikelist, trial_rates, trial_idx):
    plt.figure(figsize=(6, 4))
    plt.imshow(trial_spikelist, aspect="auto", cmap="inferno")
    plt.colorbar()
    plt.title("Spike counts: Trial " + str(trial_idx))

    plt.figure(figsize=(6, 4))
    plt.imshow(trial_rates, aspect="auto", cmap="inferno")
    plt.colorbar()
    plt.title("Firing Rates: Trial-Averaged, All Neurons")

    plt.show()
    return 

def visualize_session(full_spikes, trial_rates):
    plt.figure(figsize=(6, 4))
    plt.imshow(full_spikes, aspect="auto", cmap="inferno")
    plt.colorbar()
    plt.title("Spike counts: Trials Concatenated")

    plt.figure(figsize=(6, 4))
    plt.imshow(trial_rates, aspect="auto", cmap="inferno")
    plt.colorbar()
    plt.title("Firing Rates: Trial-Averaged, All Neurons")

    plt.show()

    return 

# ---------- running things but ignore for now

if __name__ == "__main__":
    spikes = load_spikes(data)

    flat = full_session(spikes)

    print(flat.shape)

    scipy.io.savemat('output/dual_shank/full_sess_07538_day1.mat', {'matrix': flat})

    trial0_spikelist = select_trial(spikes, 0)
    print("trial0_spikelist shape:", np.shape(trial0_spikelist))

    rates =psth_firing(spikes, 3)
    print("rates shape:", np.shape(rates))  

    visualize_session(flat, rates)

    visualize_trial(trial0_spikelist, rates, 0)
