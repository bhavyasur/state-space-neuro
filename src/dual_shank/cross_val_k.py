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

import ssm
from ssm.util import random_rotation, find_permutation
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial
from utils.rslds_util import plot_most_likely_dynamics, plot_observations, plot_trajectory
from src.dual_shank.ds_PCA import run_PCA
from src.dual_shank.ds_jPCA import run_jPCA
from src.dual_shank.ds_GPFA import run_GPFA

K_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

data = "data/om/07538_M1_Day1_CCA_data.mat"

spks = load_spikes(data)

def cross_val(spikes, D_obs, D_latent):
    
    rslds = ssm.SLDS(D_obs, K, D_latent,
                    transitions="recurrent",
                    dynamics="diagonal_gaussian",
                    emissions="gaussian_orthog",
                    single_subspace=True)
    rslds.initialize(y)




if __name__ == "__main__":
    result = cross_val(spks)
    print("result of cross validation is:", str(result))