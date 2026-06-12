""" run RSLDS on dualshank spiking data"""

# ------------ IMPORTS ------------ #
import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import sklearn as skl
import ssm
import copy
from ssm.util import random_rotation, find_permutation
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, full_session
from utils.rslds_util import plot_most_likely_dynamics, plot_observations, plot_trajectory
from src.dual_shank.ds_PCA import run_PCA
from src.dual_shank.ds_jPCA import run_jPCA
from src.dual_shank.ds_GPFA import run_GPFA

color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")
npr.seed(42)

# ------------ LOAD AND VISUALIZE DATA ------------ #

data = "data/om/07538_M1_Day1_CCA_data.mat" # selected data

spikes = load_spikes(data)
full = full_session(spikes)
trial0_spikelist = select_trial(spikes, 0)
rates =psth_firing(spikes, 3)
num_neurons = np.shape(spikes)[0]

D_obs = num_neurons

# ------------ PCA / jPCA / GPFA TO DETERMINE D_LATENT (CONT STATES) ----------- #

# cont_latent_determination_type = "PCA"
# cont_latent_determination_type = "jPCA"
# cont_latent_determination_type = "GPFA"

# if cont_latent_determination_type == "PCA":
#     print("Launching subprocess: running PCA on input data")
#     D_latent = run_PCA(data)
#     print(f"PCA process finished. Continuous latent dimension is {D_latent}. continuing rSLDS process.")
# elif cont_latent_determination_type == "jPCA":
#     print("Launching subprocess: running jPCA on input data")
#     D_latent = run_jPCA(data)
#     print(f"jPCA process finished. Continuous latent dimension is {D_latent}. continuing rSLDS process.")
# elif cont_latent_determination_type == "GPFA":
#     print("Launching subprocess: running GPFA on input data")
#     D_latent = run_GPFA(data)
#     print(f"GPFA process finished. Continuous latent dimension is {D_latent}. continuing rSLDS process.")
# else:
#     print("Dimensionality reduction input is not available. Please use 'PCA', 'jPCA', or 'GPFA'.")

#

# ------------ INSTANTIATE RSLDS OBJECT ------------ #
K = 4 # should depend on held-out cross validation
y = full.T
D_latent = 2
print(y.shape)

rslds = ssm.SLDS(D_obs, K, D_latent,
                 transitions="recurrent_only",
                 dynamics="diagonal_gaussian",
                 emissions="gaussian_orthog",
                 single_subspace=True)

rslds.initialize(y, inputs=None)
q_elbos_lem, q_lem = rslds.fit(y, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               initialize=False, num_iters=100, alpha=0.0)
xhat_lem = q_lem.mean_continuous_states[0]
zhat_lem = rslds.most_likely_states(xhat_lem, y)
rslds_lem = copy.deepcopy(rslds)

# Plot some results
plt.figure()
plt.plot(q_elbos_lem[1:], label="Laplace-EM")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.show()

ax3 = plt.subplot(111)
plot_trajectory(zhat_lem, xhat_lem, ax=ax3)
plt.title("Inferred, Laplace-EM")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
ax = plt.subplot(111)
lim = abs(xhat_lem).max(axis=0) + 1
plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("Inferred Dynamics, Laplace-EM")
plt.show()