# ------------------------------ IMPORTS ------------------------------
# 1) path extension
import sys
from pathlib import Path
ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

# 2) loading functions + utilities for rSLDS. add to this if adding more data types
from src.rbp_cre.rbp_load_util import load_dfoverf, full_session_rbp, visualize_trace, trace_sanity_check
from src.dual_shank.ds_load_util import load_spikes, full_session, visualize_session
from src.l23.l23_load_util import load_dfoverf_l23, full_session_l23, trace_sanity_check_l23
from src.rslds.rslds_util import ( plot_trajectory, bin_smooth, plot_pca_flowfield, 
                                  eigs_timeconstants, plot_cv_heatmap, 
                                  softplus, single_neuron_contribution )

# 3) other necessary imports
import autograd.numpy as np
import autograd.numpy.random as npr
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import ssm
import copy
from enum import Enum

# 4) SEABORN
import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")
npr.seed(42)

# 5) MATPLOTLIB
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['font.size'] = 8
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 10  # For X and Y axis titles
plt.rcParams['xtick.labelsize'] = 9 # For X-axis tick numbers
plt.rcParams['ytick.labelsize'] = 9 # For Y-axis tick numbers
plt.rcParams['legend.fontsize'] = 11


# --------------- DATATYPE SETTER ---------------
class DataType(Enum):
    DualShank = "dualshank"
    RbpCre = "rbpcre"
    L23 = "l23"

# ------------------------------ MAIN FUNCTIONS ------------------------------

def run_rslds_pca_flowfield(raw_data, disc_states, latent_dims, plot_key, state_idx, type: DataType, plot: bool = True,
                             nxpts=20, nypts=20, alpha=0.8, num_iters=50, margin=1.0, roi=None):
    """
    Run rSLDS on a full session of data, then PCA-project the resulting latent trajectory
    to 2D and plot the flow field of each discrete state's dynamics in PC space.

    INPUT: raw_data is a .mat file. Structure depends on the type you've selected in the function call.

    OUTPUT: plots if set to True, plus rslds_lem, xhat_lem, x_pc, zhat_lem, q_elbos_lem, q_lem, pca. Most useful is rslds_lem,
    which returns the actual model item. You can extract the dynamics matrix with rslds_lem.dynamics.As. check ssm library on github
    for more assistance (specifically observations.py, the specifics depend on what type of dynamics you run the model with. 
    defaults to 'gaussian'.
    """

    if type is DataType.DualShank:
        spks = load_spikes(raw_data)
        full = full_session(spks)
        data = bin_smooth(full.T).astype(int)

    elif type is DataType.RbpCre:
        dfoverf = load_dfoverf(raw_data, roi)
        full = full_session_rbp(dfoverf)
        data = full.T.astype(int)

    elif type is DataType.L23:
        dfoverf = load_dfoverf_l23(raw_data)
        full = full_session_l23(dfoverf)
        data = full.T.astype(int)
    else:
        raise ValueError(f"Unsupported DataType: {type}")

    y = data
    print("y shape", np.shape(y))
    num_obs = np.shape(y)[1]

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
    if roi:
        output_folder = Path(f"output/{plot_key}/{disc_states}states_{latent_dims}dims_roi{roi}")
    else:
        output_folder = Path(f"output/{plot_key}/{disc_states}states_{latent_dims}dims")

    output_folder.mkdir(parents=True, exist_ok=True)

    # RAW DATA VISUALIZATION
    if type is DataType.DualShank:
        fig0, ax0 = plt.subplots(figsize=(7,5))
        visualize_session(full, ax=ax0)
        ax0.set_title(f"Spike Raster Plot: {plot_key}")
        fig0.tight_layout(pad=2)
        fig0.savefig(output_folder / "spikes.png")

    elif type is DataType.RbpCre:
        fig0, axes0 = trace_sanity_check(full, random_seed=42)
        fig0.suptitle(f"Calcium Trace of Neurons: {plot_key}")
        fig0.savefig(output_folder / "calcium_trace.png")

    elif type is DataType.L23:
        fig0, axes0 = trace_sanity_check_l23(full, random_seed=42)
        fig0.suptitle(f"Calcium Trace of Neurons: {plot_key}")
        fig0.savefig(output_folder / "calcium_trace.png")

    # ELBO
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.plot(q_elbos_lem[1:], label="Laplace-EM")
    ax1.legend()
    ax1.set_title(f"ELBO Plot: {plot_key}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ELBO")
    fig1.tight_layout(pad=2)

    fig1.savefig(output_folder / "elbo.png")

    # INFERRED TRAJECTORY
    fig2, ax2 = plt.subplots(figsize=(6,6))
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
    
    fig3.savefig(output_folder / "flowfield.png")

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

    # SUPERIMPOSE TRAJECTORY AND FLOW FIELDS
    fig6, ax6 = plt.subplots(figsize=(6,6))

    plot_trajectory(zhat_lem, x_pc, ax=ax6)

    lim = abs(x_pc).max(axis=0) + margin
    plot_pca_flowfield(rslds_lem, W, mu, plot_key,
                        xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]),
                        nxpts=nxpts, nypts=nypts, alpha=alpha, ax=ax6)
    
    ax6.set_title(f"Flow Field & Trajectory in PC Space: {plot_key}")
    ax6.set_xlabel("PC1")
    ax6.set_ylabel("PC2")
    fig6.tight_layout(pad=2)

    fig6.savefig(output_folder / "superimposedtraj.png")

    # SINGLE NEURON CONTRIBUTION
    fig7 = single_neuron_contribution(rslds_lem, state_idx=state_idx)
    # fig7.suptitle(f"Single Neuron Contributions of ROI {roi}: {plot_key}")
    fig7.savefig(output_folder / "single_neuron_contribution.png")

    # dynamic_velocity(rslds_lem)

    if plot:
        plt.show()

    return rslds_lem, xhat_lem, x_pc, zhat_lem, q_elbos_lem, q_lem, pca




def run_rslds_concat(data, disc_states, latent_dims):
    # data is one numpy array (num_timesteps, num_neurons)
    a = [data]
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

    return rslds_lem, xhat_lem, zhat_lem, q_elbos_lem, q_lem




def cross_val(raw_data, type: DataType, heldout_frac=0.1, n_repeats=3, var_threshold=0.8, **kwargs):
    """ datas is raw .mat file
    """
    if type is DataType.EPhys:
        spks = load_spikes(raw_data)
        full = full_session(spks)
        data = bin_smooth(full.T).astype(int)
        print("shape of data, ephys", np.shape(data))
    elif type is DataType.Calcium:
        dfoverf = load_dfoverf(raw_data, roi=1)
        full = full_session_rbp(dfoverf)
        data = full.T.astype(int)
        print("shape of data, calcium", np.shape(data))
    else:
        raise ValueError(f"Unsupported DataType: {type}")

    num_neurons = np.shape(data)[1]
    obs = num_neurons
    param_grid = {
        'dims': [6, 7, 8, 9, 10],
        'states': [2, 3, 4, 5, 6]
    }

    results = {}  # (dim, state) -> mean_cv_score (fraction)
    found_threshold = False

    # rslds on all hyperparameter combinations
    for dim in param_grid['dims']:
        if found_threshold:
            break
        for state in param_grid['states']:
            kfold = KFold()
            cv_scores = []
            for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
                train = list(data[train_idx])
                print("train shape", np.shape(train))
                test = [data[test_idx]]
                print("test shape", np.shape(train))
                model, b, c, d, e = run_rslds_concat(train, state, dim)
                predictions = []
                true_obs = []
                for trial in test:
                    elbo, q_x = model.approximate_posterior([trial], method="laplace_em")
                    x_mean = q_x.mean_continuous_states[0]
                    # project latents back to raw observation space
                    Cs = model.emissions.Cs
                    ds = model.emissions.ds
                    z_hat = model.most_likely_states(x_mean, trial)
                    num_states = Cs.shape[0]
                    z_hat = np.clip(z_hat, 0, num_states - 1)
                    y_pred = np.array([softplus(Cs[z] @ x_mean[t] + ds[z]) for t, z in enumerate(z_hat)])
                    print("Cs shape:", np.shape(Cs), "ds shape:", np.shape(ds))
                    print("z_hat unique:", np.unique(z_hat))
                    print("y_pred range:", y_pred.min(), y_pred.max())
                    print("y_true range:", trial.min(), trial.max())
                    predictions.append(y_pred)
                    true_obs.append(trial)

                #  R^2 for this fold
                y_true_fold = np.vstack(true_obs)
                y_pred_fold = np.vstack(predictions)
                rss = np.sum((y_true_fold - y_pred_fold) ** 2)
                tss = np.sum((y_true_fold - np.mean(y_true_fold, axis=0)) ** 2)
                fold_r2 = 1 - (rss / tss)
                cv_scores.append(fold_r2)
                print(f"Fold {fold+1} Variance Explained: {fold_r2 * 100:.2f}%")

            # Overall Performance
            print(f"\nParameters: dim={dim}, states={state}")
            mean_score = np.mean(cv_scores)
            results[(dim, state)] = mean_score
            if mean_score >= var_threshold:
                print(f"\nvariance threshold of {var_threshold} satisfied! mean CV variance explained: {mean_score * 100:.2f}% \u00b1 {np.std(cv_scores) * 100:.2f}%")
                found_threshold = True
                break
            else:
                print(f"\nmean CV variance explained: {mean_score * 100:.2f}% \u00b1 {np.std(cv_scores) * 100:.2f}%")

    plot_cv_heatmap(results, param_grid)
    return results
