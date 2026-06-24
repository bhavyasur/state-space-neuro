import numpy as np
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.gpfa import GPFA
import matplotlib.pyplot as plt
from kneed import KneeLocator
from enum import Enum
from mpl_toolkits.mplot3d import Axes3D

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from src.dual_shank.ds_load_util import psth_firing, load_spikes, select_trial, full_session

# -------------- MAIN FUNCTION DEFINITION --------------
class DataType(Enum):
    EPhys = "ephys"
    Calcium = "calcium"
    Bessel = "bessel"

def run_GPFA(data, plot_key, latent_dims=3, type=DataType):

    # --------------- DATA LOADING ---------------

    if type is DataType.EPhys:
        """ Your 'data' should be a .mat file, where each cell is an individual neuron and each cell contains an array 
            of (num_trials * num_timesteps). load_spikes converts this to a list of numpy arrays, where 
            each object in the list is a neuron, the numpy array is (num_trials, num_timesteps) """
        
        spiketrains = np.array(load_spikes(data)) # now: list of numpy arrays (num_neurons, num_trials, time_steps)
        full = full_session(spiketrains) # now: trial-concatenated (num_neurons, num_timesteps)

        num_trials = np.shape(spiketrains)[1]
        trial_length = 3 * pq.s

        num_neurons = np.shape(spiketrains)[0] # same as number of neurons
        num_timesteps = np.shape(spiketrains)[2]

        # calculate instantaneous rate
        instantaneous_rates = psth_firing(spiketrains, trial_length)
    
    elif type is DataType.Calcium:
        1
    

    # --------------- PLOTTING DATA ITSELF ---------------

    if type is DataType.EPhys:
        f = plt.figure(figsize=(10, 7))
        ax1 = f.add_subplot(2,1,1)
        ax2 = f.add_subplot(2,1,2)

        ax1.set_title(f'Raster plot of {plot_key}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Neuron id')

        im1 = ax1.imshow(
                full, 
                aspect="auto"
                )

        time_bins = np.linspace(0, int(trial_length), num_timesteps)
        ax2.set_title(f'PSTH firing rates of {plot_key}')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Firing Rate (Hz)')

        im2 = ax2.imshow(
            instantaneous_rates,
            aspect='auto',
            extent=[time_bins[0], time_bins[-1], 0, num_neurons],
            origin='lower'
        )

        plt.tight_layout()
        plt.show()

    elif type is DataType.Calcium:
        1

    # --------------- APPLY GPFA ---------------
    # specify fitting parameters
    bin_size = 10 * pq.ms
    latent_dimensionality = latent_dims
    times = list(np.nonzero(spiketrains[i:]) for i in range(np.shape(spiketrains)[0]))

    spiketrain_list = []
    for trial_idx in range(num_trials):
        trial_trains = []
        for neuron_idx in range(num_neurons):
            times = np.where(spiketrains[neuron_idx][trial_idx] > 0)[0]

            st = neo.SpikeTrain(
                times=times,
                units=pq.ms,
                t_stop=spiketrains[neuron_idx].shape[1]
            )
            trial_trains.append(st)
        spiketrain_list.append(trial_trains)

    print("spiketrain list shape", np.shape(spiketrain_list))

    gpfa_3dim = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)
    trajectories = gpfa_3dim.fit_transform(spiketrain_list)

    f = plt.figure(figsize=(15, 5))
    ax2 = f.add_subplot(1, 1, 1, projection='3d')

    linewidth_single_trial = 0.5
    color_single_trial = 'C0'
    alpha_single_trial = 0.5

    linewidth_trial_average = 2
    color_trial_average = 'C1'

    ax2.set_title(f'GPFA Latent dynamics: {plot_key}')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_zlabel('Dim 3')
    # single trial trajectories
    for single_trial_trajectory in trajectories:
        ax2.plot(single_trial_trajectory[0], single_trial_trajectory[1], single_trial_trajectory[2],
                lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)
    # trial averaged trajectory
    average_trajectory = np.mean(trajectories, axis=0)
    ax2.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2], lw=linewidth_trial_average, c=color_trial_average, label='Trial averaged trajectory')
    ax2.legend()
    ax2.view_init(azim=-5, elev=60)  # an optimal viewing angle for the trajectory extracted from our fixed spike trains

    plt.tight_layout()
    plt.show()

    # ---------------- CROSS VALIDATION --------------------
    from sklearn.model_selection import cross_val_score

    x_dims = [2, 5, 7, 9, 11, 13]
    log_likelihoods = []
    for x_dim in x_dims:
        gpfa_cv = GPFA(x_dim=x_dim)
        # estimate the log-likelihood for the given dimensionality as the mean of the log-likelihoods from 3 cross-vailidation folds
        cv_log_likelihoods = cross_val_score(gpfa_cv, spiketrain_list, cv=3, n_jobs=3, verbose=True)
        log_likelihoods.append(np.mean(cv_log_likelihoods))

    f = plt.figure(figsize=(7, 5))
    plt.xlabel('Dimensionality of latent variables')
    plt.ylabel('Log-likelihood')
    plt.plot(x_dims, log_likelihoods, '.-')

    kneedle = KneeLocator(x_dims, log_likelihoods, S=1.0, curve='concave', direction='increasing')
    plat_x = kneedle.knee
    plat_y = kneedle.knee_y
    plt.scatter(plat_x, plat_y, color='red', label='plateau start')
    plt.axvline(x=plat_x, color='gray', linestyle='--', label=f'x = {plat_x:.2f}')

    plt.tight_layout()
    plt.show()

    return int(plat_x)

if __name__ == "__main__":
    data = "data/om/07538_M1_Day1_CCA_data.mat"
    plot_key = "07538_M1_Day1_CCA"
    run_GPFA(data, plot_key, type=DataType.EPhys)