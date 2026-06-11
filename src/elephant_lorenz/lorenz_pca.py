import numpy as np
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.conversion import BinnedSpikeTrain
from sklearn.decomposition import PCA
# from elephant.gpfa import GPFA
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from utils.elephant_util import random_projection, generate_spiketrains, integrated_lorenz

# ----------- LORENZ DATA GENERATION --------------

# set parameters for the integration of the lorenz attractor
timestep = 1 * pq.ms
transient_duration = 10 * pq.s
trial_duration = 30 * pq.s
num_steps_transient = int((transient_duration.rescale('ms')/timestep).magnitude)
num_steps = int((trial_duration.rescale('ms')/timestep).magnitude)

# set parameters for spike train generation
max_rate = 70 * pq.Hz
np.random.seed(42)  # for visualization purposes, we want to get identical spike trains at any run

# specify data
num_trials = 20
num_spiketrains = 50

# calculate the oscillator
times, lorenz_trajectory_3dim = integrated_lorenz(
    timestep, num_steps=num_steps_transient+num_steps, x0=0, y0=1, z0=1.25)
times = (times - transient_duration).rescale('s').magnitude
times_trial = times[num_steps_transient:]

# random projection
lorenz_trajectory_Ndim = random_projection(
    lorenz_trajectory_3dim[:, num_steps_transient:], embedding_dimension=num_spiketrains)

# calculate instantaneous rate
normed_traj = lorenz_trajectory_Ndim / lorenz_trajectory_Ndim.max()
instantaneous_rates_lorenz = np.power(max_rate.magnitude, normed_traj)
print("shape rates", np.shape(instantaneous_rates_lorenz))

# generate spiketrains
spiketrains_lorenz = generate_spiketrains(
    instantaneous_rates_lorenz, num_trials, timestep)
# NOTE: spiketrains_lorenz is a list of spiketrains (each trial is an index in the list)

# -------------- PLOTTING LORENZ SPIKE TRAINS AND TRAJECTORY --------------
from mpl_toolkits.mplot3d import Axes3D

f = plt.figure(figsize=(15, 10))
ax1 = f.add_subplot(2, 2, 1)
ax2 = f.add_subplot(2, 2, 2, projection='3d')
ax3 = f.add_subplot(2, 2, 3)
ax4 = f.add_subplot(2, 2, 4)

ax1.set_title('Lorenz system')
ax1.set_xlabel('Time [s]')
labels = ['x', 'y', 'z']
for i, x in enumerate(lorenz_trajectory_3dim):
    ax1.plot(times, x, label=labels[i])
ax1.axvspan(-transient_duration.rescale('s').magnitude, 0, color='gray', alpha=0.1)
ax1.text(-5, -20, 'Initial transient', ha='center')
ax1.legend()

ax2.set_title(f'Trajectory in 3-dim space')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_ylabel('z')
ax2.plot(lorenz_trajectory_3dim[0, :num_steps_transient],
         lorenz_trajectory_3dim[1, :num_steps_transient],
         lorenz_trajectory_3dim[2, :num_steps_transient], c='C0', alpha=0.3)
ax2.plot(lorenz_trajectory_3dim[0, num_steps_transient:],
         lorenz_trajectory_3dim[1, num_steps_transient:],
         lorenz_trajectory_3dim[2, num_steps_transient:], c='C0')

ax3.set_title(f'Projection to {num_spiketrains}-dim space')
ax3.set_xlabel('Time [s]')
y_offset = lorenz_trajectory_Ndim.std() * 3
for i, y in enumerate(lorenz_trajectory_Ndim):
    ax3.plot(times_trial, y + i*y_offset)

trial_to_plot = 0
ax4.set_title(f'Raster plot of trial {trial_to_plot}')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Neuron id')
for i, spiketrain in enumerate(spiketrains_lorenz[trial_to_plot]):
    ax4.plot(spiketrain, np.ones(len(spiketrain)) * i, ls='', marker='|')

plt.tight_layout()
plt.show()

# ----------------- APPLY PCA -------------------
# specify fitting parameters
bin_size = 20 * pq.ms
latent_dimensionality = 3

pca_3dim = PCA(n_components=latent_dimensionality)

# binned_trials is (num_trials, num_neurons, n_bins)
binned_trials = np.array([
    BinnedSpikeTrain(list(trial), bin_size=bin_size).to_array()
    for trial in spiketrains_lorenz
])  
print("shape of binned_trials", np.shape(binned_trials))
print("type of binned trials:", binned_trials.dtype)

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
ax1 = f.add_subplot(1, 2, 1, projection='3d')
ax2 = f.add_subplot(1, 2, 2, projection='3d')

linewidth_single_trial = 0.5
color_single_trial = 'C0'
alpha_single_trial = 0.5

linewidth_trial_average = 2
color_trial_average = 'C1'

ax1.set_title('Original latent dynamics')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot(lorenz_trajectory_3dim[0, num_steps_transient:],
         lorenz_trajectory_3dim[1, num_steps_transient:],
         lorenz_trajectory_3dim[2, num_steps_transient:])

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

binned_trials = np.array([
    BinnedSpikeTrain(list(trial), bin_size=bin_size).to_array()
    for trial in spiketrains_lorenz
])

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