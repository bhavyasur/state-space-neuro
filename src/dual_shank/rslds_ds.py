""" run RSLDS on dualshank spiking data"""

# ------------ IMPORTS ------------ #
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
import sklearn as skl

from external.ssm.ssm.util import random_rotation, find_permutation
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial
from utils.rslds_util import plot_most_likely_dynamics, plot_observations, plot_trajectory
from src.dual_shank.ds_PCA import run_PCA
from src.dual_shank.ds_jPCA import run_jPCA
from src.dual_shank.ds_GPFA import run_GPFA

# ------------ LOAD AND VISUALIZE DATA ------------ #

data = "data/om/07538_M1_Day1_CCA_data.mat" # selected data

spikes = load_spikes(data)
trial0_spikelist = select_trial(spikes, 0)
rates =psth_firing(spikes, 3)
num_neurons = np.shape(spikes)[1]

# ------------ PCA / jPCA / GPFA TO DETERMINE HOW MANY CONT STATES ----------- #

# cont_latent_determination_type = "PCA"
# cont_latent_determination_type = "jPCA"
cont_latent_determination_type = "GPFA"

if cont_latent_determination_type == "PCA":
    print("Launching subprocess: running PCA on input data")
    D_latent = run_PCA(data)
    print(f"PCA process finished. Continuous latent dimension is {D_latent}. continuing rSLDS process.")
elif cont_latent_determination_type == "jPCA":
    print("Launching subprocess: running jPCA on input data")
    D_latent = run_jPCA(data)
    print(f"jPCA process finished. Continuous latent dimension is {D_latent}. continuing rSLDS process.")
elif cont_latent_determination_type == "GPFA":
    print("Launching subprocess: running GPFA on input data")
    D_latent = run_GPFA(data)
    print(f"GPFA process finished. Continuous latent dimension is {D_latent}. continuing rSLDS process.")
else:
    print("Dimensionality reduction input is not available. Please use 'PCA', 'jPCA', or 'GPFA'.")

# ------------ INSTANTIATE RSLDS OBJECT ------------ #
# D_obs = num_neurons
# K = 1

# rslds = ssm.SLDS(D_obs, K, D_latent,
#                  transitions="recurrent_only",
#                  dynamics="diagonal_gaussian",
#                  emissions="gaussian_orthog",
#                  single_subspace=True)