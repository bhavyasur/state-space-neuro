## Helper Functions

import sys
from pathlib import Path

ext =  Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.dual_shank.clean_rslds_ds import run_rslds

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]

colors = sns.xkcd_palette(color_names)

import ssm
from ssm.util import random_rotation
from ssm.plots import plot_dynamics_2d

# Specify whether or not to save figures

##  Set the parameters of the LDS
time_bins = 200   # number of time bins
state_dim = 2     # number of latent dimensions
obs_dim = 10      # number of observed dimensions

lds_point = ssm.LDS(obs_dim, state_dim, transitions="recurrent_only", emissions="poisson_orthog", emission_kwargs=dict(link="softplus"))

s = (2,2)
A1 = np.zeros(s)
A1[1][1] = -1
A1[0][0] = -1
b = np.zeros(state_dim)

# Set the dynamics matrix (A)  
lds_point.dynamics.A = A1

print(np.shape(A1))
print(A1)
lds_point.dynamics.b = b

# Sample from the LDS object: Point Attractor
A1[1][1] = -1
A1[0][0] = -1

states, emissions = lds_point.sample(time_bins)
plt.figure(figsize=(5,5))

# Plot the dynamics vector field
q = plot_dynamics_2d(A1, 
                     bias_vector=b,
                     mins=np.array([-20,-20]),
                     maxs=np.array([20,20]),
                     npts=16,
                     color=colors[0])

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Simulated State Space")
plt.tight_layout()
plt.show()


