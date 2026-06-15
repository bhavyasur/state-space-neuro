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
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, full_session
from utils.rslds_util import plot_most_likely_dynamics, plot_observations, plot_trajectory, bin_only, bin_smooth

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
num_neurons = np.shape(full)[0]

D_obs = num_neurons


# ------------ INSTANTIATE RSLDS OBJECT ------------ #
K = 3 # should depend on held-out cross validation
data = full.T
y = bin_smooth(data).astype(int)
D_latent = 2

rslds = ssm.SLDS(D_obs, K, D_latent,
                 transitions="recurrent_only",
                 emissions="poisson_orthog",
                 emission_kwargs=dict(link="softplus"))

rslds.initialize(y, verbose=1)
q_elbos_lem, q_lem = rslds.fit(y, method="laplace_em",
                               variational_posterior="structured_meanfield",
                               initialize=False, num_iters=50)
xhat_lem = q_lem.mean_continuous_states[0]
zhat_lem = rslds.most_likely_states(xhat_lem, y)
yhat_lem = rslds.smooth(xhat_lem, y)

rslds_lem = copy.deepcopy(rslds)

# Plot some results
plt.figure()
plt.plot(q_elbos_lem[1:], label="Laplace-EM")
plt.legend()
plt.title("ELBO Plot")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.show()

ax3 = plt.subplot(111)
plot_trajectory(zhat_lem, xhat_lem, ax=ax3)
plt.title("Inferred Trajectory, Laplace-EM")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
ax = plt.subplot(111)
lim = abs(xhat_lem).max(axis=0) + 1
plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
plt.title("Inferred Dynamics, Laplace-EM")
plt.show()



   
