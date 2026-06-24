""" run RSLDS on dualshank spiking data"""

# ------------ IMPORTS ------------ #
import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import autograd.numpy as np
import autograd.numpy.random as npr
import seaborn as sns
from collections import namedtuple
import sklearn as skl
import math
import ssm
import copy
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, full_session
from rslds.rslds_util import plot_most_likely_dynamics, plot_observations, plot_trajectory, bin_only, bin_smooth, plot_dyn_down2D, plot_traj_down2D
from ssm.preprocessing import pca_with_imputation

color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")
npr.seed(42)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 14  
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 10  # For X and Y axis titles
plt.rcParams['xtick.labelsize'] = 9 # For X-axis tick numbers
plt.rcParams['ytick.labelsize'] = 9 # For Y-axis tick numbers

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


def run_rslds(data, disc_states, latent_dims, num_iters=50, plot: bool = False):
    # data = full.T

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
        """SELECT WHICH ONE BASED ON IF YOU HAVE >2 LATENT DIMENSIONS"""
        plot_traj_down2D(zhat_lem, xhat_lem, ax=ax3)
        # plot_trajectory(zhat_lem, xhat_lem, ax=ax3)
        plt.title("Inferred Trajectory, Sliced to 2 Dimensions, Laplace-EM")
        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(6,4))
        ax = plt.subplot(111)
        lim = abs(xhat_lem).max(axis=0) + 1
        """SELECT WHICH ONE BASED ON IF YOU HAVE >2 LATENT DIMENSIONS"""
        plot_dyn_down2D(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
        # plot_most_likely_dynamics(rslds_lem, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
        plt.title("Inferred Dynamics, Laplace-EM")
        plt.show()

    return rslds_lem, xhat_lem, zhat_lem, q_elbos_lem, q_lem

def run_rslds_pca_flowfield(data, disc_states, latent_dims, plot: bool = True,
                             nxpts=20, nypts=20, alpha=0.8, num_iters=50, margin=1.0):
    """
    Run rSLDS on a full session of data, then PCA-project the resulting latent trajectory
    to 2D and plot the flow field of each discrete state's dynamics in PC space.
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
    # print(f"eigs: {eigs}")
    # print(f"tc: {tc}")
    # print(f"vecs: {vecs}")

    if plot:
        # ELBO
        fig1, ax1 = plt.subplots(figsize=(7,5), layout='constrained')
        ax1.plot(q_elbos_lem[1:], label="Laplace-EM")
        ax1.legend()
        ax1.set_title("ELBO Plot")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("ELBO")

        # INFERRED TRAJECTORY
        fig2, ax2 = plt.subplots(figsize=(7,5), layout='constrained')
        plot_trajectory(zhat_lem, x_pc, ax=ax2)
        ax2.set_title("Inferred Trajectory in PC Space (Laplace-EM)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")

        # FLOW FIELD
        fig3, ax3 = plt.subplots(figsize=(6, 6), layout='constrained')
        lim = abs(x_pc).max(axis=0) + margin
        plot_pca_flowfield(rslds_lem, W, mu,
                            xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]),
                            nxpts=nxpts, nypts=nypts, alpha=alpha, ax=ax3)

        # EIGENSPECTRUM
        real = eigs.real
        imag = eigs.imag
        fig4, ax4 = plt.subplots(figsize=(6,6), layout='constrained')
        ax4.scatter(real, imag)
        ax4.grid()
        ax4.set_title("Eigenvalues from Dynamics Matrix of State")
        ax4.set_xlabel("Real Axis")
        ax4.set_ylabel("Imaginary Axis")
        

        # TIMECONSTANTS
        labels = eigs.astype(str)
        fig5, ax5 = plt.subplots(figsize=(6,6), layout='constrained')
        bars = ax5.bar(labels, tc)
        ax5.set_xticks([])
        ax5.bar_label(bars, padding=3)
        ax5.set_title('Bar Plot of Time Constants')
        ax5.set_ylabel('Milliseconds') # need to check this

        plt.show()
        # dynamic_velocity(rslds_lem)

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
    ax.set_title('Inferred Flow Field in PC Space (Laplace-EM)')

    return ax

def eigs_timeconstants(model, state_idx, dim_idx=None):
    """ Takes in dynamics matrix from model and returns eigenvalues in numpy array. Then calculates time constants per Nair
    equation and returns time constants as numpy array
    """
    A = model.dynamics.As # A shape is (num_states, 10, 10)
    print("A shape", np.shape(A))
    matrix = A[state_idx]
    tuple = np.linalg.eig(matrix)
    eigenvalues = tuple.eigenvalues
    eigenvectors = tuple.eigenvectors

    tc_list = []
    for i in range(np.shape(eigenvalues)[0]):
        val = eigenvalues[i]
        tc = abs(1 / math.log(abs(val)))
        tc_list.append(tc)
    
    if dim_idx is None:
        return eigenvalues, eigenvectors, tc_list
    
    if dim_idx == 'max':
        dim_idx = int(np.argmax(tc_list))

        eig_val = eigenvalues[dim_idx]
        eig_vec = eigenvectors[:, dim_idx]
        tc = tc_list[dim_idx]

        if not np.isclose(eig_val.imag, 0, atol=1e-6):
            print(f"Warning: dim {dim_idx} has nonzero imaginary part ({eig_val:.4f}) "
                f"— this is half of an oscillatory pair, not a pure integration mode.")

        return eig_val, eig_vec, tc, dim_idx


def dynamic_velocity(model):
    # Create 3D surface (dynamic velocity landscape) for a point attractor

    from matplotlib import cm

    xlim=(-20, 20)
    ylim=(-20, 20)
    nxpts=40
    nypts=40
    alpha=0.8
    ax=None
    figsize=(5, 5)

    # Create a grid of x, y values
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    dxydt_m = (xy).dot((model.dynamics.As).T)
    dxydt_m_norm = np.zeros((len(dxydt_m)))

    for xt in range(0,len(dxydt_m)): 
        dxydt_m_norm[xt] = np.linalg.norm(dxydt_m[xt,:])

    ax.plot_trisurf(xy[:, 0], xy[:, 1], dxydt_m_norm, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$ dynamic velocity$')

    fig.tight_layout()



if __name__ == "__main__":
    raw1 = "data/om/07538_M1_Day1_CCA_data.mat"
    day2 = "data/om/075356Mouse2_M1_Day2_CCA_data.mat"
    day5 = "data/om/075356_M1_Day5_CCA_data.mat"
    day6 = "data/om/075356_M1_Day6_CCA_data_74c141t.mat" # 74 neurons, 141 trials
    raw3 = "data/om/87564_M1R_Day12DualShank_41c324t.mat" # 41 cells, 324 trials
    # --- PCA ONLY---

    spikes = load_spikes(day6)
    full = full_session(spikes)
    num_neurons = np.shape(full)[0]
    num_obs = num_neurons
    data = full.T
    # y = bin_smooth(data).astype(int)

    # ------------
    
    disc_states = 4 # should depend on held-out cross validation
    latent_dims = 8
    
    run_rslds_pca_flowfield(data, disc_states, latent_dims, num_iters=50, plot=True)
    # run_rslds(data, disc_states, latent_dims, num_iters=50, plot=True)