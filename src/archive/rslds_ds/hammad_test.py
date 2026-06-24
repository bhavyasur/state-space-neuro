from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ssm import SLDS
import numpy as np
from ds_load_util import full_session, visualize_session, psth_firing, load_spikes
from scipy.ndimage import gaussian_filter1d
import ssm
import matplotlib.pyplot as plt
import copy

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from rslds.rslds_util import plot_trajectory, plot_most_likely_dynamics

def compute_binned_spike_data(spike_data, sigma = 5, bin_size_ms = 10):
    """
    Compute continuous firing rates from binned spike data using Gaussian smoothing.
    INPUT: spike_data is 2D array, concatenated spiking data from trials
            sigma is gaussian kernel, int set to 5
            bin_size is int, set to 10
    OUTPUT: 2d array, (num_trials * num_timesteps_per_trial, num_neurons)
    """
    # Check input dimensions
    if len(spike_data.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {spike_data.shape}")

    # 2. Bin the data into 20ms bins
    n_neurons = spike_data.shape[1]
    n_timebins = spike_data.shape[0]
    bin_size = bin_size_ms
    n_bins = n_timebins // bin_size
    binned_spike_data = np.zeros((n_bins, n_neurons))
    for i in range(n_bins):
        binned_spike_data[i] = spike_data[i * bin_size:(i + 1) * bin_size].sum(axis=0)
    print("Binned data shape:", binned_spike_data.shape)

    # Convert to Hz (spikes/second) by scaling
    scale_factor = bin_size_ms  # Convert to Hz
    smoothed_spike_data = np.zeros_like(binned_spike_data)
    # Apply Gaussian smoothing to each neuron individually
    for i in range(n_neurons):
        # Explicitly use array indexing
        current_neuron = binned_spike_data[:, i].copy()  # Get copy of this neuron's data
        # Scale first, then smooth
        smoothed_spike_data[:,i] = gaussian_filter1d(current_neuron * scale_factor, sigma=sigma)

    return smoothed_spike_data

data = "data/om/07538_M1_Day1_CCA_data.mat"

spikes = load_spikes(data)
full = full_session(spikes)
insert = full.T

binned_spike_data = compute_binned_spike_data(insert)

print("np.shape binned", np.shape(binned_spike_data))
firing = psth_firing(spikes, 3)

visualize_session(binned_spike_data, firing)

# Choose number of components for latent space (2-3 is good for visualization)
n_components = 10

# Fit PCA to the spike count data
scaler = StandardScaler(with_std=False)
smoothed_spikes_standardized = scaler.fit_transform(binned_spike_data)

pca = PCA(n_components=n_components)
latent_dynamics = pca.fit_transform(smoothed_spikes_standardized)

# 3. rSlds initialization
num_states = 3
obs_dim = binned_spike_data.shape[1] # Get 3 from PCA components
latent_dim = 2

# Create the model and initialize its parameters
slds = SLDS(obs_dim, num_states, latent_dim, emissions="poisson_orthog", transitions="recurrent_only",emission_kwargs=dict(link="softplus"))
binned_spike_data = binned_spike_data.astype(int)
assert binned_spike_data.dtype == int, f"binned_spike_data is instead dtype {binned_spike_data.dtype}"
slds.initialize(binned_spike_data,verbose=1)

# Fit the model using Laplace-EM with a structured variational posterior
q_lem_elbos, q_lem = slds.fit(binned_spike_data, method="laplace_em", 
                              variational_posterior="structured_meanfield",
                              num_iters=50,initialize=False)

# Get the posterior mean of the continuous states
q_lem_x = q_lem.mean_continuous_states[0]

# Find the permutation that matches the true and inferred states
rslds_states = slds.most_likely_states(q_lem_x, binned_spike_data)

# Smooth the data under the variational posterior
q_lem_y = slds.smooth(q_lem_x, binned_spike_data)

slds_lem = copy.deepcopy(slds)

# Print ELBO of the model
print('Elbos model performance:',q_lem_elbos[-1:])


# rslds = ssm.SLDS(obs_dim, num_states, latent_dim,
#                  transitions="recurrent_only",
#                  emissions="poisson_orthog",
#                  emission_kwargs=dict(link="softplus"))
#                  #single_subspace=True)

# rslds.initialize(y, inputs=None)
# q_elbos_lem, q_lem = rslds.fit(y, method="laplace_em",
#                                variational_posterior="structured_meanfield",
#                                initialize=False, num_iters=100, alpha=0.0)
# xhat_lem = q_lem.mean_continuous_states[0]
# zhat_lem = rslds.most_likely_states(xhat_lem, y)
# rslds_lem = copy.deepcopy(rslds)

# Plot some results
plt.figure()
plt.plot(q_lem_elbos[1:], label="Laplace-EM")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.show()

ax3 = plt.subplot(111)
plot_trajectory(rslds_states, q_lem_x, ax=ax3)
plt.title("Inferred Trajectory, Laplace-EM")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
ax = plt.subplot(111)
lim = abs(q_lem_x).max(axis=0) + 1
plot_most_likely_dynamics(slds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("Inferred Dynamics, Laplace-EM")
plt.show()