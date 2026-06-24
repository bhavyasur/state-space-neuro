# --------------- IMPORTS ---------------
import sys
from pathlib import Path
ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import autograd.numpy as np
import autograd.numpy.random as npr
import seaborn as sns
from collections import namedtuple
from sklearn.decomposition import PCA
import ssm
import copy
from src.dual_shank.ds_load_util import load_spikes, full_session
from src.rslds.rslds_util import plot_trajectory, bin_smooth, plot_pca_flowfield, eigs_timeconstants

color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")
npr.seed(42)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 10  # For X and Y axis titles
plt.rcParams['xtick.labelsize'] = 9 # For X-axis tick numbers
plt.rcParams['ytick.labelsize'] = 9 # For Y-axis tick numbers
plt.rcParams['legend.fontsize'] = 11

# --------------- MAIN FUNCTION ---------------

def run_rslds_pca_flowfield(data, disc_states, latent_dims, num_obs, plot_key, plot: bool = True,
                             nxpts=20, nypts=20, alpha=0.8, num_iters=50, margin=1.0):
    """
    Run rSLDS on a full session of data, then PCA-project the resulting latent trajectory
    to 2D and plot the flow field of each discrete state's dynamics in PC space.

    INPUT: data is a full session transposed to be (num_timesteps * num_trials, num_neurons). you can do this with 
    full_session(spikes) and then transpose with .T

    OUTPUT: plots if set to True, plus rslds_lem, xhat_lem, x_pc, zhat_lem, q_elbos_lem, q_lem, pca. Most useful is rslds_lem,
    which returns the actual model item. You can extract the dynamics matrix with rslds_lem.dynamics.As. check ssm library on github
    for more assistance (specifically observations.py, the specifics depend on what type of dynamics you run the model with. 
    defaults to 'gaussian'.
    """

    # data = full.T, data is a full session transposed to be (num_timesteps * num_trials, num_neurons)

    y = bin_smooth(data).astype(int)

    print("y shape", np.shape(y))

    rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                      transitions="recurrent_only",
                      emissions="poisson_orthog",
                      emission_kwargs=dict(link="softplus"))

    rslds.initialize(y, verbose=1)
    q_elbos_lem, q_lem = rslds.fit(y, method="laplace_em",
                                    variational_posterior="structured_meanfield",
                                    initialize=False, num_iters=num_iters)
    xhat_lem = q_lem.mean_continuous_states[0]          # (T, latent_dims)
    zhat_lem = rslds.most_likely_states(xhat_lem, y)    # (T,)
    yhat_lem = rslds.smooth(xhat_lem, y)

    rslds_lem = copy.deepcopy(rslds)

    # ---- PCA on the latent trajectory ----
    pca = PCA(n_components=2)
    x_pc = pca.fit_transform(xhat_lem)   # (T, 2)
    print("x_pc shape", np.shape(x_pc))
    W = pca.components_.T                # (latent_dims, 2) loading matrix
    mu = pca.mean_                       # (latent_dims,)

    # ---- Eigenvalues Calculations -----
    eigs, vecs, tc = eigs_timeconstants(rslds_lem, 1) # select state you want to extract dynamics matrix from. each state has different dynamics matrix.

    # ---- PLOTS ----

    output_folder = Path(f"output/{plot_key}/{disc_states}states_{latent_dims}dims")
    output_folder.mkdir(parents=True, exist_ok=True)

    # ELBO
    fig1, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(q_elbos_lem[1:], label="Laplace-EM")
    ax1.legend()
    ax1.set_title(f"ELBO Plot: {plot_key}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ELBO")
    fig1.tight_layout(pad=2)

    fig1.savefig(output_folder / "elbo.png")

    # INFERRED TRAJECTORY
    fig2, ax2 = plt.subplots(figsize=(7,5))
    plot_trajectory(zhat_lem, x_pc, ax=ax2)
    ax2.set_title(f"Inferred Trajectory in PC Space (Laplace-EM): {plot_key}")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    fig2.tight_layout(pad=2)

    fig2.savefig(output_folder / "trajectory.png")

    # FLOW FIELD
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    lim = abs(x_pc).max(axis=0) + margin
    plot_pca_flowfield(rslds_lem, W, mu, plot_key,
                        xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]),
                        nxpts=nxpts, nypts=nypts, alpha=alpha, ax=ax3)
    fig3.tight_layout(pad=2)
    
    fig3.savefig(output_folder /" flowfield.png")

    # EIGENSPECTRUM
    real = eigs.real
    imag = eigs.imag
    fig4, ax4 = plt.subplots(figsize=(6,6))
    ax4.scatter(real, imag)
    ax4.grid()
    ax4.set_title(f"Eigenvalues from Dynamics Matrix of State: {plot_key}")
    ax4.set_xlabel("Real Axis")
    ax4.set_ylabel("Imaginary Axis")
    fig4.tight_layout(pad=2)

    fig4.savefig(output_folder / "eigenspectrum.png")

    # TIMECONSTANTS
    labels = eigs.astype(str)
    fig5, ax5 = plt.subplots(figsize=(6,6))
    bars = ax5.bar(labels, tc)
    ax5.set_xticks([])
    ax5.bar_label(bars, padding=3)
    ax5.set_title(f'Bar Plot of Time Constants: {plot_key}')
    ax5.set_ylabel('Milliseconds') # need to check this
    fig5.tight_layout(pad=2)

    fig5.savefig(output_folder / "timeconstants.png")

    # dynamic_velocity(rslds_lem)

    if plot:
        plt.show()

    return rslds_lem, xhat_lem, x_pc, zhat_lem, q_elbos_lem, q_lem, pca

if __name__=="__main__":
    # -- SET YOUR DATA PATH HERE AS my_data. should be list of arrays, (num_neurons, num_trials, num_timesteps) --

    my_data = "data/om/07538_M1_Day1_CCA_data.mat" # actual data path
    plot_key = "07538_M1_Day1" # key for plot titles & plot save path

    raw1 = "data/om/07538_M1_Day1_CCA_data.mat"
    day2 = "data/om/075356Mouse2_M1_Day2_CCA_data.mat"
    day5 = "data/om/075356_M1_Day5_CCA_data.mat"
    day6 = "data/om/075356_M1_Day6_CCA_data_74c141t.mat" # 74 neurons, 141 trials
    raw3 = "data/om/87564_M1R_Day12DualShank_41c324t.mat" # 41 cells, 324 trials

    # -- LOADS DATA --
    full = full_session(load_spikes(my_data))
    num_neurons = np.shape(full)[0]
    data = full.T # -> this is input into the function

    # -- SET DESIRED HYPERPARAMETERS (determine with cross validation in rslds_crossval.py) --
    disc_states = 4 
    latent_dims = 8

    # -- RUN --
    run_rslds_pca_flowfield(data, disc_states, latent_dims, num_neurons, plot_key, num_iters=2, plot=False)
