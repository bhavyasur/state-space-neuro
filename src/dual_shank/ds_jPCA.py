import numpy as np
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.conversion import BinnedSpikeTrain
from kneed import KneeLocator

# from elephant.gpfa import GPFA
import matplotlib.pyplot as plt

import sys
from pathlib import Path
import os

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../external/jPCA'))


from jPCA.jPCA import JPCA

from utils.elephant_util import random_projection, generate_spiketrains, integrated_lorenz
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial

# ------------- MAIN FUNCTION DEFINITION --------------

data = "data/om/07538_M1_Day1_CCA_data.mat"

def run_jPCA(data):

    # ----------- DATA LOADING --------------
    spiketrains = np.array(load_spikes(data)) # list of numpy arrays (num_trials, num_neurons, time_steps)

    num_trials = np.shape(spiketrains)[1]
    trial_length = 3 * pq.s
    num_spiketrains = np.shape(spiketrains)[0]
    num_timesteps = np.shape(spiketrains)[2]

    # calculate instantaneous rate
    instantaneous_rates = psth_firing(spiketrains, trial_length)

    # -------------- PLOTTING SPIKE TRAINS AND TRAJECTORY --------------
    from mpl_toolkits.mplot3d import Axes3D

    f = plt.figure(figsize=(10, 7))
    ax1 = f.add_subplot(2,1,1)
    ax2 = f.add_subplot(2,1,2)

    trial_to_plot = 0
    trial_set = select_trial(spiketrains, trial_to_plot)
    ax1.set_title(f'Raster plot of trial {trial_to_plot}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Neuron id')

    im1 = ax1.imshow(
            trial_set, 
            aspect="auto"
            )
    cbar1 = f.colorbar(im1, ax=ax1, label='Spike Trains')

    time_bins = np.linspace(0, int(trial_length), num_timesteps)
    ax2.set_title('PSTH plot of firing rates')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Firing Rate (Hz)')

    im2 = ax2.imshow(
        instantaneous_rates,
        aspect='auto',
        extent=[time_bins[0], time_bins[-1], 0, num_spiketrains],
        origin='lower'
    )
    cbar2 = f.colorbar(im2, ax=ax2, label='Firing Rate (Hz)')

    plt.tight_layout()
    plt.show()

    # ----------------- APPLY jPCA -------------------
    # specify fitting parameters
    latent_dimensionality = 4

    jpca_4dim = JPCA(num_jpcs=latent_dimensionality)

    # # note: for fit, datas should be "a list containing trials. each element of the list should be (length of trial x number of neurons)"

    # (n_trials, n_neurons, length of trial)
    # binned_trials is (num_trials, num_neurons, n_bins)
    binned_trials = spiketrains
    print("shape of binned_trials", np.shape(binned_trials))

    # per-trial arrays
    trial_list = [binned_trials[i].T.astype(float) for i in range(num_trials)] 
    bin_size = 1 * pq.ms
    n_bins = binned_trials.shape[2]
    times_binned = list(np.arange(n_bins) * bin_size)

    projected, full_var, pca_var, jpca_var = jpca_4dim.fit(trial_list, times=times_binned, tstart=times_binned[0], tend=times_binned[-1])

    # (n_trials, n_bins, 3)
    trajectories = np.array(projected)

    f = plt.figure(figsize=(10, 5))
    ax2 = f.add_subplot(1, 1, 1, projection='3d')

    linewidth_single_trial = 0.5
    color_single_trial = 'C0'
    alpha_single_trial = 0.5

    linewidth_trial_average = 2
    color_trial_average = 'C1'

    ax2.set_title('Latent dynamics extracted by jPCA')
    ax2.set_xlabel('jPC 1')
    ax2.set_ylabel('jPC 2')
    ax2.set_zlabel('jPC 3')
    # single trial trajectories
    for single_trial_trajectory in trajectories:
        ax2.plot(single_trial_trajectory[:, 0], single_trial_trajectory[:, 1],
                lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)

    average_trajectory = np.mean(trajectories, axis=0)
    ax2.plot(average_trajectory[:, 0], average_trajectory[:, 1],
            lw=linewidth_trial_average, c=color_trial_average, label='Trial averaged trajectory')
    ax2.legend()
    ax2.view_init(azim=-5, elev=60)  # an optimal viewing angle for the trajectory extracted from our fixed spike trains

    plt.tight_layout()
    plt.show()

    # --------------- CROSS-VALIDATION --------------
    x_dims = [4, 6, 8, 10, 16, 22, 28]
    explained_variances = []

    # per-trial arrays
    trial_list = [binned_trials[i].T.astype(float) for i in range(num_trials)] 

    for x_dim in x_dims:
        jpca_cv = JPCA(num_jpcs=x_dim)
        # note: for fit, datas should be "a list containing trials. each element of the list should be (length of trial x number of neurons)"
        _, full_var, _, jpca_var_capt = jpca_cv.fit(trial_list, times=times_binned, tstart=times_binned[0], tend=times_binned[-1], num_pcs=x_dim)
        explained_variances.append(np.sum(jpca_var_capt / full_var))

    f = plt.figure(figsize=(7, 5))
    plt.xlabel('Dimensionality of latent variables')
    plt.ylabel('Explained variance ratio')
    plt.plot(x_dims, explained_variances, '.-')

    kneedle = KneeLocator(x_dims, explained_variances, S=1.0, curve='convex', direction='increasing')
    plat_x = kneedle.knee
    plat_y = kneedle.knee_y

    if plat_x is not None:
        plt.scatter(plat_x, plat_y, color='red', label=f'knee = {plat_x}')
        plt.axvline(x=plat_x, color='gray', linestyle='--', label=f'x = {plat_x}')
    else:
        print("Warning: KneeLocator could not find a knee — try adjusting S or inspect the curve")

    plt.legend()
    plt.tight_layout()
    plt.show()

    return int(plat_x)