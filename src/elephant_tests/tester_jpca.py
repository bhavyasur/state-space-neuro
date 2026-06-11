import numpy as np
from scipy.integrate import odeint
import quantities as pq
import neo
from elephant.spike_train_generation import inhomogeneous_poisson_process
from elephant.conversion import BinnedSpikeTrain

# from elephant.gpfa import GPFA
import matplotlib.pyplot as plt

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from external.jPCA.jPCA import JPCA

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

# generate spiketrains
spiketrains_lorenz = generate_spiketrains(
    instantaneous_rates_lorenz, num_trials, timestep)

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

# ----------------- APPLY jPCA -------------------
# specify fitting parameters
latent_dimensionality = 4

jpca_4dim = JPCA(num_jpcs=latent_dimensionality)

# # note: for fit, datas should be "a list containing trials. each element of the list should be (length of trial x number of neurons)"

# (n_trials, n_neurons, length of trial)
binned_trials = np.array([
    BinnedSpikeTrain(list(trial), bin_size=1 * pq.ms).to_array()
    for trial in spiketrains_lorenz
])  
# per-trial arrays
trial_list = [binned_trials[i].T.astype(float) for i in range(num_trials)] 
bin_size = 1 * pq.ms
n_bins = binned_trials.shape[2]
times_binned = list(np.arange(n_bins) * bin_size)

projected, full_var, pca_var, jpca_var = jpca_4dim.fit(trial_list, times=times_binned, tstart=times_binned[0], tend=times_binned[-1])

# (n_trials, n_bins, 3)
trajectories = np.array(projected)

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
x_dims = [4, 6, 8]
explained_variances = []

# per-trial arrays
trial_list = [binned_trials[i].T.astype(float) for i in range(num_trials)] 

for x_dim in x_dims:
    jpca_cv = JPCA(num_jpcs=x_dim)
    # note: for fit, datas should be "a list containing trials. each element of the list should be (length of trial x number of neurons)"
    _, full_var, _, jpca_var_capt = jpca_cv.fit(trial_list, times=times_binned, tstart=times_binned[0], tend=times_binned[-1], num_pcs=max(x_dim, 6))
    explained_variances.append(np.sum(jpca_var_capt / full_var))

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