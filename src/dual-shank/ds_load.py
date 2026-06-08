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

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict

data = "data/om/07538_M1_Day1_CCA_data.mat"

def load_spikes(data):
    mat = mat73.loadmat(data)
    print("mat type: " + str(type(mat))) # datatype debugging
    print("mat keys: " + str(mat.keys())) # check keys in mat file

    spikes = mat['M1Spikes']['PSTH']['hit']['spks']
    firing = mat['M1Spikes']['PSTH']['hit']['spkRates'] # 89 x 3001
    print(np.shape(spikes)) # check shape of spikes array  

    return spikes, firing

def psth_firing(spikes):
    """calculates firing rate from raw spiking data using Peristimulus Time Histogram with bin size of 10 ms"""


def select_trial(spikes, firing, trial_idx):
    trial_spikelist = []
    for i in range(len(spikes)):
        trial_spikelist.append(spikes[trial_idx][0]) # check shape of trial1 array

    return trial_spikelist

def visualize_trial(trial_spikelist, trial_idx):
    plt.figure(figsize=(6, 4))
    plt.imshow(trial_spikelist, aspect="auto")
    plt.colorbar()
    plt.title("Spike counts: Trial " + str(trial_idx))
    plt.show()

# ----------

spikes, firing = load_spikes(data)
trial0_spikelist = select_trial(spikes, firing, 0)
visualize_trial(trial0_spikelist, 0)