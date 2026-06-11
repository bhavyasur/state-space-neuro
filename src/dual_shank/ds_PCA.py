import numpy as np
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from src.dual_shank.ds_load_util import psth_firing, load_spikes, select_trial

# ---------- MAIN FUNCTION DEFINITION -----------
data = "data/om/07538_M1_Day1_CCA_data.mat"

def run_PCA(data):
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

    # ----------------- APPLY PCA -------------------
    # specify fitting parameters
    latent_dimensionality = 3

    pca_3dim = PCA(n_components=latent_dimensionality)

    # binned_trials is (num_trials, num_neurons, n_bins)
    binned_trials = spiketrains
    print("shape of binned_trials", np.shape(binned_trials))

    # PCA expects (samples, features) = (n_bins, n_neurons), fit on all trials concatenated
    X_all = binned_trials.transpose(0, 2, 1).reshape(-1, num_spiketrains)  # (n_trials*n_bins, n_neurons)
    print("shape xall", np.shape(X_all))
    # NOTE: the reshape fixes the number of columns to be the number of neurons, adjusts number of rows to fit that based on the data
    pca_3dim.fit(X_all)

    # (n_trials, n_bins, 3)
    trajectories = np.array([
        pca_3dim.transform(trial.T)  # trial shape: (n_neurons, n_bins) -> .T -> (n_bins, n_neurons)
        for trial in binned_trials
    ])  # shape: (n_trials, n_bins, 3)

    f = plt.figure(figsize=(15, 5))
    ax2 = f.add_subplot(1, 1, 1, projection='3d')

    linewidth_single_trial = 0.5
    color_single_trial = 'C0'
    alpha_single_trial = 0.5

    linewidth_trial_average = 2
    color_trial_average = 'C1'

    ax2.set_title('Latent dynamics extracted by PCA')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_zlabel('Dim 3')

    # single trial trajectories
    for single_trial_trajectory in trajectories:
        ax2.plot(single_trial_trajectory[:, 0], single_trial_trajectory[:, 1], single_trial_trajectory[:, 2],
                lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)
        
    # trial averaged trajectory
    average_trajectory = np.mean(trajectories, axis=0)
    ax2.plot(average_trajectory[:, 0], average_trajectory[:, 1], average_trajectory[:, 2], lw=linewidth_trial_average, c=color_trial_average, label='Trial averaged trajectory')
    ax2.legend()
    ax2.view_init(azim=-5, elev=60)  # an optimal viewing angle for the trajectory extracted from our fixed spike trains

    plt.tight_layout()
    plt.show()

    # --------------- CROSS-VALIDATION --------------
    x_dims = [1, 2, 3, 4, 5]
    explained_variances = []

    X_all = binned_trials.transpose(0, 2, 1).reshape(-1, num_spiketrains)

    for x_dim in x_dims:
        pca_cv = PCA(n_components=x_dim)
        pca_cv.fit(X_all)
        explained_variances.append(np.sum(pca_cv.explained_variance_ratio_))

    diffs = np.diff(explained_variances)
    elbow = np.argmax(np.diff(diffs)) + 1

    f = plt.figure(figsize=(7, 5))
    plt.xlabel('Dimensionality of latent variables')
    plt.ylabel('Explained variance ratio')
    plt.plot(x_dims, explained_variances, '.-')
    plt.plot(x_dims[elbow], explained_variances[elbow], 'x', markersize=10, color='r', label=f'Elbow at dim={x_dims[elbow]}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return int(x_dims[elbow])