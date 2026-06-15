import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
npr.seed(42)
from matplotlib import gridspec
import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")
from sklearn.model_selection import TimeSeriesSplit

import ssm
from ssm.util import random_rotation, find_permutation
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, full_session
from utils.rslds_util import plot_most_likely_dynamics, plot_observations, plot_trajectory, bin_smooth


K_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # search space for K hyperparameter

data = "data/om/07538_M1_Day1_CCA_data.mat"

spks = load_spikes(data)
full = full_session(spks)
data = bin_smooth(spks).T

def cross_val(data, D_obs, D_latent):


    # scores = []
    # n_obs = data[]
    
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


def param_tune():
    return




if __name__ == "__main__":
    result = cross_val(spks)
    print("result of cross validation is:", str(result))