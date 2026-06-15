import sys
from pathlib import Path
import numpy as np

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, visualize_session, full_session

data = "data/om/07538_M1_Day1_CCA_data.mat" # selected data

spikes = load_spikes(data)
full = full_session(spikes)
trial0_spikelist = select_trial(spikes, 0)
rates = psth_firing(spikes, 3)
num_neurons = np.shape(full)[0]