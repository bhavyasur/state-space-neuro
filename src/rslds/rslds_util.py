# -------------- IMPORTS ---------------
import os
import pickle
import copy
from scipy.ndimage import gaussian_filter1d

import autograd.numpy as np
import autograd.numpy.random as npr
import math
import ssm

npr.seed(12345)

import matplotlib.pyplot as plt
from matplotlib import gridspec

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

plt.rcParams['axes.titlesize'] = 13

# --------------- HELPER FUNCTIONS ---------------

def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)
    return ax


"""This function was purely as a test, to run rSLDS with more than 2 latent dimensions and then only select 2 to plot
the trajectory. This is not very useful in practice."""
def plot_traj_down2D(z, x, ax=None, ls="-"):
    # Slice two dimensions explicitly
    x_dim1 = x[:, 2]
    x_dim2 = x[:, 3]
    
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
        
    for start, stop in zip(zcps[:-1], zcps[1:]):
        # Slice both the time segment [start:stop+1] AND the target dimensions
        ax.plot(x_dim1[start:stop + 1],
                x_dim2[start:stop + 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)
    return ax


def plot_observations(z, y, ax=None, ls="-", lw=1):

    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    T, N = y.shape
    t = np.arange(T)
    for n in range(N):
        for start, stop in zip(zcps[:-1], zcps[1:]):
            ax.plot(t[start:stop + 1], y[start:stop + 1, n],
                    lw=lw, ls=ls,
                    color=colors[z[start] % len(colors)],
                    alpha=1.0)
    return ax


def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    assert model.D >= 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As[:, :2, :2], model.dynamics.bs[:, :2])):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    return ax


"""This function was purely as a test, to run rSLDS with more than 2 latent dimensions and then only select 2 to plot
the flow field. This is not very useful in practice."""
def plot_dyn_down2D(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    D = model.D
    assert D >= 2
    
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Create a full D-dimensional coordinate array padded with zeros 
    # for dimensions 3 and beyond, so the transition dot product works.
    xy_full = np.zeros((xy.shape[0], D))
    xy_full[:, 2:4] = xy 

    # Get the probability of each state using the full D-dimensional coordinates
    z = np.argmax(xy_full.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # Slice the dynamics to use only the first two dimensions
    for k, (A, b) in enumerate(zip(model.dynamics.As[:, :2, :2], model.dynamics.bs[:, :2])):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax

"""This function was a test to compare binning vs binning and smoothing. Not really for practical use."""
def bin_only(spike_data, bin_size_ms = 10):
    """
    Compute continuous firing rates from binned spike data, no smoothing.
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

    return binned_spike_data


def bin_smooth(spike_data, sigma = 5, bin_size_ms = 10):
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


def plot_pca_flowfield(model, W, mu, plot_key,
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
    ax.set_title(f'Inferred Flow Field in PC Space (Laplace-EM): {plot_key}')

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


def softplus(x):
    return np.log1p(np.exp(x))


def plot_cv_heatmap(results, param_grid):
    """
    results: dict mapping (dim, state) -> mean_cv_score (fraction, not percent)
    param_grid: dict with 'dims' and 'states' lists
    """
    dims = param_grid['dims']
    states = param_grid['states']

    # Build matrix: rows = states, cols = dims
    grid = np.full((len(states), len(dims)), np.nan)
    for (dim, state), score in results.items():
        i = states.index(state)
        j = dims.index(dim)
        grid[i, j] = score * 100  # convert to percentage

    fig, ax = plt.subplots(figsize=(10, 6))
    finite_vals = grid[~np.isnan(grid)]
    vmin = finite_vals.min() if finite_vals.size > 0 else 0
    vmax = finite_vals.max() if finite_vals.size > 0 else 1
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels(states)
    ax.set_xlabel("Latent Dimensions")
    ax.set_ylabel("Discrete States")
    ax.set_title("Mean CV Variance Explained (%) by Hyperparameter Combination")

    # Annotate each cell with its value
    for i in range(len(states)):
        for j in range(len(dims)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color="black", fontsize=9)

    fig.colorbar(im, ax=ax, label="Variance Explained (%)")
    plt.tight_layout()
    plt.show()

def single_neuron_contribution(model, state_idx):
    """finds single neuron weights for the integration dimension (max eigenvalue) of a specific discrete state of your rSLDS model"""

    # determine which dimension you want to plot the single neuron contribution to. this function is set to plot single neuron contribution for largest eigenvalue dimension (integration dimension)
    eigs, vecs, tcs, max_idx = eigs_timeconstants(model, state_idx=state_idx, dim_idx='max') # returns max eigenvalue, corresponding eigenvector, time constant, index of max eigenvalue
    print("time constants", tcs)
    print("max_idx", max_idx)

    C = abs(np.squeeze(model.emissions.Cs))
    print("shape of C", np.shape(C))
    sorted_indices = np.argsort(-C[:, max_idx])  # Sort by contribution to the first dimension

    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 8], hspace=0.4)

    # Top: neuron weights for the first dimension
    ax0 = fig.add_subplot(gs[0, 0])
    abs_weights = np.abs(C[:, max_idx])
    markers, stemlines, baseline = ax0.stem(np.arange(len(abs_weights)), abs_weights, basefmt=" ")
    markers.set(markersize=5)
    stemlines.set(linewidth=0.75)

    ax0.set_ylabel("Weight (abs)", fontsize=11)
    ax0.set_xticks(np.arange(0, np.shape(C)[0], 15))
    
    ax0.set_title("Single-cell contribution\nof Integration Dimension", fontsize=12)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)

    # Bottom: heatmap of emission matrix, neurons sorted by first dimension
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(C[sorted_indices, :], aspect='auto', cmap='viridis', interpolation='none')
    ax1.set_ylabel("Neurons (sorted)", fontsize=11)
    ax1.set_xlabel("Dimension", fontsize=11)
    ax1.set_title('Emission matrix C', fontsize=13)
    ax1.grid(False)

    return fig