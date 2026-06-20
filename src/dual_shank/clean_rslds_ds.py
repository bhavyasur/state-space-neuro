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
from ssm.preprocessing import pca_with_imputation

color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")
npr.seed(42)

def plot_eigenvalues():
    return

def run_rslds_binned(binned, disc_states, latent_dims, plot: bool = False):
    # binned is one numpy array (num_timesteps, num_neurons)
    a = [binned]
    y = np.asarray(a)
    #y is (1, num_timesteps, num_neurons), dtype list of numpy arrays
    
    print("y shape", np.shape(y))
    print("y[0] shape", np.shape(np.asarray(y[0])))
    print("y[0] type", np.asarray(y[0]).dtype)

    num_obs = np.shape(y)[2]
    num_neurons = num_obs

    rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                    transitions="recurrent_only",
                    emissions="poisson_orthog",
                    emission_kwargs=dict(link="softplus"),
                    M=0)
    
    T = np.shape(y)[1]
    inputs = [np.zeros((num_obs,num_obs))]

    rslds.initialize(y[0], inputs=None, verbose=1)
    q_elbos_lem, q_lem = rslds.fit(y[0], method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False, num_iters=50)
    xhat_lem = q_lem.mean_continuous_states[0]
    zhat_lem = rslds.most_likely_states(xhat_lem, y[0])
    yhat_lem = rslds.smooth(xhat_lem, y[0])

    rslds_lem = copy.deepcopy(rslds)

    if plot == True:
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

    return rslds_lem, xhat_lem, zhat_lem, q_elbos_lem, q_lem


def run_rslds(raw_data, disc_states, latent_dims, plot: bool = False):

    spikes = load_spikes(raw_data)
    full = full_session(spikes)
    num_neurons = np.shape(full)[0]
    num_obs = num_neurons
    data = full.T
    y = bin_smooth(data).astype(int)
    
    print("y shape", np.shape(y))

    rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
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

    if plot == True:
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

    return rslds_lem, xhat_lem, zhat_lem, q_elbos_lem, q_lem

def run_rslds_pca_flowfield(data, disc_states, latent_dims, plot: bool = True,
                             nxpts=20, nypts=20, alpha=0.8, margin=1.0):
    """
    Run rSLDS on a full session of data, then PCA-project the resulting latent trajectory
    to 2D and plot the flow field of each discrete state's dynamics in PC space.

    Mirrors run_rslds(), but instead of plotting raw latent dims 1 vs 2
    (which only makes sense if latent_dims == 2), this runs PCA on xhat_lem
    (T x latent_dims) to get a 2D embedding, then projects each state's
    affine dynamics (A_k, b_k) into that same PC space so the flow field
    arrows are consistent with the plotted trajectory.

    Math: if p = W^T (x - mu) is the PC-space point (W: D x 2 PCA loadings,
    mu: D-dim mean of xhat_lem), and the latent dynamics are x' = A x + b,
    then substituting x = W p + mu:
        x' = A(W p + mu) + b
        p' = W^T (x' - mu) = W^T A W p + W^T (A mu + b - mu)
    So in PC space: p' = A_pc p + b_pc, where
        A_pc = W^T A W            (2x2)
        b_pc = W^T (A @ mu + b - mu)   (2,)
    This is the standard linear-projection result for affine dynamics.
    """
    from sklearn.decomposition import PCA

    # data = full.T, data is a full sessiont transposed to be (num_timesteps * num_trials, num_neurons)
    y = bin_smooth(data).astype(int)

    print("y shape", np.shape(y))

    rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                      transitions="recurrent_only",
                      emissions="poisson_orthog",
                      emission_kwargs=dict(link="softplus"))

    rslds.initialize(y, verbose=1)
    q_elbos_lem, q_lem = rslds.fit(y, method="laplace_em",
                                    variational_posterior="structured_meanfield",
                                    initialize=False, num_iters=50)
    xhat_lem = q_lem.mean_continuous_states[0]          # (T, latent_dims)
    zhat_lem = rslds.most_likely_states(xhat_lem, y)    # (T,)
    yhat_lem = rslds.smooth(xhat_lem, y)

    rslds_lem = copy.deepcopy(rslds)

    # ---- PCA on the latent trajectory ----
    pca = PCA(n_components=2)
    x_pc = pca.fit_transform(xhat_lem)   # (T, 2)
    W = pca.components_.T                # (latent_dims, 2) loading matrix
    mu = pca.mean_                       # (latent_dims,)

    if plot:
        plt.figure()
        plt.plot(q_elbos_lem[1:], label="Laplace-EM")
        plt.legend()
        plt.title("ELBO Plot")
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.show()

        ax3 = plt.subplot(111)
        plot_trajectory(zhat_lem, x_pc, ax=ax3)
        plt.title("Inferred Trajectory in PC Space, Laplace-EM")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        lim = abs(x_pc).max(axis=0) + margin
        plot_pca_flowfield(rslds_lem, W, mu,
                            xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]),
                            nxpts=nxpts, nypts=nypts, alpha=alpha, ax=ax)
        plt.title("Inferred Dynamics (PCA Flow Field), Laplace-EM")
        plt.show()

    return rslds_lem, xhat_lem, x_pc, zhat_lem, q_elbos_lem, q_lem, pca


def plot_pca_flowfield(model, W, mu,
                        xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
                        alpha=0.8, ax=None, figsize=(4, 4)):
    """
    Plot the flow field of each discrete state's dynamics, projected into
    a 2D PCA space.

    model: fitted rSLDS/SLDS model (has model.dynamics.As, model.dynamics.bs,
           model.transitions.Rs, model.transitions.r)
    W:     (D, 2) PCA loading matrix (pca.components_.T)
    mu:    (D,) PCA mean (pca.mean_)
    xlim, ylim: bounds of the 2D PC grid to evaluate the flow field over
    """
    K = model.K
    D = model.D

    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    pc_grid = np.column_stack((X.ravel(), Y.ravel()))   # (G, 2) grid points in PC space

    # Map grid points back to full latent space to determine discrete state
    # at each location: x_full = W @ p + mu
    x_full = pc_grid.dot(W.T) + mu   # (G, D)

    # Discrete state assignment using the model's recurrent transition params
    # (same logic as plot_most_likely_dynamics, but evaluated at the
    # back-projected full-dimensional points)
    z = np.argmax(x_full.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = model.dynamics.As[k]   # (D, D)
        b = model.dynamics.bs[k]   # (D,)

        # Project affine dynamics into PC space:
        # p' = W^T A W p + W^T (A @ mu + b - mu)
        A_pc = W.T.dot(A).dot(W)                      # (2, 2)
        b_pc = W.T.dot(A.dot(mu) + b - mu)             # (2,)

        dpdt = pc_grid.dot(A_pc.T) + b_pc - pc_grid    # (G, 2)

        zk = z == k
        if zk.sum() > 0:
            ax.quiver(pc_grid[zk, 0], pc_grid[zk, 1],
                      dpdt[zk, 0], dpdt[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    return ax


if __name__ == "__main__":
    raw = "data/om/07538_M1_Day1_CCA_data.mat" # selected data

    # --- PCA ONLY---

    spikes = load_spikes(raw)
    full = full_session(spikes)
    num_neurons = np.shape(full)[0]
    num_obs = num_neurons
    data = full.T
    # y = bin_smooth(data).astype(int)

    # ------------
    
    disc_states = 5 # should depend on held-out cross validation
    latent_dims = 10
    
    run_rslds_pca_flowfield(data, disc_states, latent_dims, plot=True)