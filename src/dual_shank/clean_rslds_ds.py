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

    # ---- eigenvalues ----

def pca_for_rslds(D, datas, masks=None, num_iters=20, feature_names=None, group_labels=None, plot: bool = False):
    """
    Executes pca_with_imputation and generates a comprehensive 2D biplot.
    
    Parameters:
    - D: Number of PCA components to fit. Must be >= 2.
    - datas: Array or list/tuple of arrays to run PCA on.
    - masks: Optional missing data boolean mask(s).
    - num_iters: Number of EM-style imputation iterations.
    - feature_names: List of strings for feature names. (OPTIONAL)
    - group_labels: List of strings to label each distinct sub-dataset. (OPTIONAL)
    """
    if D < 2:
        raise ValueError("D (components) must be at least 2 to generate a 2D biplot.")

    pca, xs, ll = pca_with_imputation(D, datas, masks=masks, num_iters=num_iters)
    
    if plot == True:
        fig, (ax_var, ax_biplot) = plt.subplots(1, 2, figsize=(16, 7.5))

        # PANEL 1: VARIANCE EXPLAINED PLOT (SCREE PLOT)
        # ----------------------------------------------------
        var_exp = pca.explained_variance_ratio_ * 100
        cum_var_exp = np.cumsum(var_exp)
        pcs = np.arange(1, len(var_exp) + 1)
        
        # Bar plot for individual variance
        bars = ax_var.bar(pcs, var_exp, alpha=0.6, color='steelblue', label='Individual Variance')
        # Line plot for cumulative variance
        line = ax_var.plot(pcs, cum_var_exp, marker='o', color='darkorange', linewidth=2, label='Cumulative Variance')
        
        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax_var.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', 
                        ha='center', va='bottom', fontsize=9)
            
        ax_var.set_xlabel('Principal Components', fontsize=12)
        ax_var.set_ylabel('Variance Explained (%)', fontsize=12)
        ax_var.set_title('Variance Explained by Component', fontsize=13, weight='bold', pad=10)
        ax_var.set_xticks(pcs)
        ax_var.set_xticklabels([f'PC{i}' for i in pcs])
        ax_var.set_ylim(0, 105)
        ax_var.grid(True, linestyle='--', alpha=0.3)
        ax_var.legend(loc='upper left')

        # ----------------------------------------------------
        # PANEL 2: PCA BIPLOT
        # ----------------------------------------------------
        # Plot Data Points (Scores) group-by-group from xs
        colors = plt.cm.tab10(np.linspace(0, 1, len(xs)))
        for i, x_group in enumerate(xs):
            label = group_labels[i] if group_labels else f'Dataset {i+1}'
            ax_biplot.scatter(x_group[:, 0], x_group[:, 1], alpha=0.6, color=colors[i], label=label)
            
        # Compute and Plot Feature Loadings Vectors
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        num_features = loadings.shape[0]
        
        if feature_names is None:
            feature_names = [f'Var {j+1}' for j in range(num_features)]
            
        for j in range(num_features):
            # Draw vector arrows
            ax_biplot.arrow(0, 0, loadings[j, 0], loadings[j, 1], 
                            head_width=0.04, head_length=0.04, fc='darkred', ec='darkred', alpha=0.85, zorder=5)
            # Draw text labels
            ax_biplot.text(loadings[j, 0] * 1.15, loadings[j, 1] * 1.15, feature_names[j], 
                        color='darkred', fontsize=11, ha='center', va='center', weight='bold', zorder=6)

        # Biplot Formatting
        ax_biplot.set_xlabel(f'PC1 ({var_exp[0]:.1f}% Variance Explained)', fontsize=12)
        ax_biplot.set_ylabel(f'PC2 ({var_exp[1]:.1f}% Variance Explained)', fontsize=12)
        ax_biplot.set_title(f'PCA Biplot (Log-Likelihood: {ll:.2f})', fontsize=13, weight='bold', pad=10)
        ax_biplot.axhline(0, color='black', linestyle=':', alpha=0.4)
        ax_biplot.axvline(0, color='black', linestyle=':', alpha=0.4)
        ax_biplot.grid(True, linestyle='--', alpha=0.3)
        ax_biplot.legend(loc='best')
        ax_biplot.axis('equal')

        # Global adjustments
        plt.tight_layout()
        plt.show()
        
    return pca, xs, ll

if __name__ == "__main__":
    raw = "data/om/07538_M1_Day1_CCA_data.mat" # selected data

    # --- PCA ONLY---

    spikes = load_spikes(raw)
    full = full_session(spikes)
    num_neurons = np.shape(full)[0]
    num_obs = num_neurons
    data = full.T
    y = bin_smooth(data).astype(int)

    # ------------
    
    disc_states = 3 # should depend on held-out cross validation
    latent_dims = 8
    pca, x, y = pca_for_rslds(latent_dims, datas=y, masks=None, plot=True)
    print("pca:", np.shape(pca.components_))
    
    # run_rslds(data, disc_states, latent_dims, plot=True)