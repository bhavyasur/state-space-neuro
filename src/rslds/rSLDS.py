# ----------------------------- IMPORTS -----------------------------

# 1) path extension
import sys
from pathlib import Path
ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

# 2) loading functions + utilities for rSLDS. add to this if adding more data types
from src.rbp_cre.rbp_load_util import ( load_dfoverf_rbp, full_session_rbp, 
                                       visualize_trace, trace_sanity_check, full_session_trialsliced_rbp, 
                                       load_trialtype_idx_rbp, gonogotrials_sliced_rbp, load_trialbreak_rbp )
from src.dual_shank.ds_load_util import load_spikes, full_session, visualize_session 
from src.l23.l23_load_util import ( load_dfoverf_l23, full_session_l23, 
                                   full_session_trialsliced_l23, load_trialtype_idx_l23,
                                   trace_sanity_check_l23, gonogotrials_sliced_l23, load_trialbreak_l23 )
from src.gcamp8.gcamp8_load_util import ( load_dfoverf_dendrite, full_session_dendrite, 
                                         full_session_trialsliced_dendrite, full_session_trialsliced_thresholded_dendrite, load_dfoverf_problemtest,
                                         trace_sanity_check_dendrite, session_concat_pipeline, spikes_smooth, load_trialbreak_dendrite )
from src.rslds.rslds_util import ( plot_trajectory, bin_smooth, plot_pca_flowfield, 
                                  eigs_timeconstants, plot_cv_heatmap, 
                                  softplus, single_neuron_contribution, run_rslds_concat, most_likely_state_plot )

# 3) other necessary imports
import autograd.numpy as np
import autograd.numpy.random as npr
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import ssm
import copy
from typing import Literal
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



# ----------------------------- DATATYPE SETTER -----------------------------
class DataType(Enum):
    DualShank = "dualshank"
    RbpCre = "rbpcre"
    L23 = "l23"
    GCaMP8 = "gcamp8"



# ----------------------------- rSLDS MAIN FUNCTION -----------------------------

# NOTE: if running on rbp-cre data, need to set an roi value.
# NOTE: if running on gcamp8 data, need to set a date of session.

def run_rslds_pca_flowfield(raw_data, disc_states, latent_dims, plot_key, state_idx, type: DataType, 
                            trial_selection: Literal["go", "nogo", "single_session", "single_session_thresholded", "session_concat", "spikes", None] = None,
                            path_type: Literal["manual", "suite2p", "binarized", "reconstructed", None] = None,
                            layer: Literal["L2", "L3", None] = None,
                            testing: bool = False,
                            trial_idx: int = None,
                            roi=None, date=None, plot: bool=True, nxpts=20, nypts=20, alpha=0.8, num_iters=50, margin=1.0):
    """
    Run rSLDS on a full session of data, then PCA-project the resulting latent trajectory
    to 2D and plot the flow field of each discrete state's dynamics in PC space.

    INPUT: raw_data is a .mat file. Structure depends on the type you've selected in the function call. most data types assume a certain folder structure
    and file structure, so check those utils and make sure your data aligns. if not, update function or write another loading function and add a new conditional
    or add a new datatype.

    OUTPUT: plots if set to True, plus rslds_lem, xhat_lem, x_pc, zhat_lem, q_elbos_lem, q_lem, pca. Most useful is rslds_lem,
    which returns the actual model item. You can extract the dynamics matrix with rslds_lem.dynamics.As. check ssm library on github
    for more help (specifically observations.py, the specifics depend on what type of dynamics you run the model with. 
    defaults to 'gaussian'.

    NOTE(s): 
        - trial_selection, roi, date, depend on the datatype. 
            - L23, RbpCre, GCaMP8 use trial_selection differently. 
            - roi is only for RbpCre (so far)
            - date is only for GCaMP8, and can either be a single date or a list of dates depending on your input to trial_selection. 
              will be used differently depending.
        - state_idx is for plotting. choose a single discrete state for eigendecomposition of A matrix. this will likely depend on your analysis / if 
          a certain state corresponds to a certain behavior.
    """

    print(f"\nInitializing rSLDS pipeline with PC flow field visualization, on dataset {plot_key}\n")
    
    if type is DataType.DualShank:
        spks = load_spikes(raw_data)
        full = full_session(spks)
        data = bin_smooth(full.T).astype(int)

    elif type is DataType.RbpCre:
        go, nogo = load_trialtype_idx_rbp(raw_data, path_type=path_type, roi=roi)
        trial_break = load_trialbreak_rbp(raw_data, path_type=path_type, roi=roi)

        dfoverf = load_dfoverf_rbp(raw_data, path_type=path_type, roi=roi)
        if path_type:
            print(f"Loaded RbpCre data with path_type={path_type}\n")
        if roi:
            print(f"Loaded RbpCre data with ROI: {roi}\n")

        full = full_session_trialsliced_rbp(dfoverf)
        go = gonogotrials_sliced_rbp(dfoverf, go)
        nogo = gonogotrials_sliced_rbp(dfoverf, nogo)

        if trial_selection == "go":
            data = go.T.astype(int)
            print("Running on go trials only.\n")
        elif trial_selection == "nogo":
            data = nogo.T.astype(int)
            print("Running on nogo trials only.\n")
        else:
            data = full.T.astype(int)
            print("Running on full session.\n")

    elif type is DataType.L23:
        trial_break = load_trialbreak_l23(raw_data)

        dfoverf = load_dfoverf_l23(raw_data, layer=layer)
        if layer:
            print(f"Loaded data for layer {layer}.\n")
        go, nogo = load_trialtype_idx_l23(raw_data)
        full = full_session_trialsliced_l23(dfoverf)
        go_trials = gonogotrials_sliced_l23(dfoverf, go)
        nogo_trials = gonogotrials_sliced_l23(dfoverf, nogo)

        if trial_selection == "go":
            data = go_trials.T.astype(int)
            print("Running on go trials only.\n")
        elif trial_selection == "nogo":
            data = nogo_trials.T.astype(int)
            print("Running on nogo trials only.\n")
        else:
            data = full.T.astype(int)
            print("Running on full session.\n")
        
    elif type is DataType.GCaMP8:
        trial_break = load_trialbreak_dendrite(raw_data, date=date)

        if testing:
            dfoverf = load_dfoverf_problemtest(raw_data, path_type=path_type)
            full = full_session_trialsliced_dendrite(dfoverf)
            data = full.T.astype(int)
        else:
            if trial_selection == "single_session":
                dfoverf = load_dfoverf_dendrite(raw_data, date)
                full = full_session_trialsliced_dendrite(dfoverf)
                data = full.T.astype(int)
            elif trial_selection == "single_session_thresholded":
                dfoverf = load_dfoverf_dendrite(raw_data, date)
                full = full_session_trialsliced_thresholded_dendrite(dfoverf)
                print(f"Thresholded full session shape: {np.shape(full)}\n")
                data = full.T.astype(int)
            elif trial_selection == "session_concat":
                full = session_concat_pipeline(raw_data, date) # FULL HERE = CONCATENATED SESSIONS
                data = full.T.astype(int)
            elif trial_selection == "spikes":
                data = spikes_smooth(raw_data, date).astype(int)
                full = data.T
                
            else:
                raise ValueError("trial_selection parameter must be set as 'session_concat' or 'single_session' for the GCamp8 datatype." \
                "'date' parameter must be set as a single date for single_session, list of dates for 'session_concat'")

    else:
        raise ValueError(f"Unsupported DataType: {type}")
    
    y = data
    print(f"Data was successfully found & cleaned/reshaped. Final shape before model training is: {np.shape(y)}\n")
    num_obs = np.shape(y)[1]

    if type is DataType.DualShank:
        rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                        transitions="recurrent_only",
                        emissions="poisson_orthog",
                        emission_kwargs=dict(link="softplus"))
        print(f"Instantiating model using Poisson Orthogonal emissions type.\n")
    elif type is DataType.RbpCre:
        rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                        transitions="recurrent_only",
                        emissions="gaussian_orthog")
        print(f"Instantiating model using Gaussian Orthogonal emissions type.\n")

    elif type is DataType.GCaMP8:
        if trial_selection == "spikes" or testing:
            rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                            transitions="recurrent_only",
                            emissions="poisson_orthog",
                            emission_kwargs=dict(link="softplus"))
            print(f"Instantiating model using Poisson Orthogonal emissions type.\n")

        else:
            rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                            transitions="recurrent_only",
                            emissions="gaussian_orthog")
            print(f"Instantiating model using Gaussian Orthogonal emissions type.\n")

    elif type is DataType.L23:
        rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                        transitions="recurrent_only",
                        emissions="gaussian_orthog")
        print(f"Instantiating model using Gaussian Orthogonal emissions type.\n")

    else:
        raise ValueError(f"Unsupported DataType: {type}")

    rslds.initialize(y, verbose=1)
    q_elbos_lem, q_lem = rslds.fit(y, method="laplace_em",
                                    variational_posterior="structured_meanfield",
                                    initialize=False, num_iters=num_iters)
    xhat_lem = q_lem.mean_continuous_states[0]          # (T, latent_dims) # POSTERIOR MEAN OF CONTINUOUS STATES (for our single trial)
    zhat_lem = rslds.most_likely_states(xhat_lem, y)    # (T,) # POSTERIOR OF DISCRETE STATES (made with viterbi algo for hmm)
    print("shape zhat", np.shape(zhat_lem), "shape xhat", np.shape(xhat_lem), "shape y", np.shape(y))
    print(zhat_lem[700:900])
    yhat_lem = rslds.smooth(xhat_lem, y)

    rslds_lem = copy.deepcopy(rslds)

    # ---- PCA on the latent trajectory ----
    pca = PCA(n_components=latent_dims)
    x_pc_numdims = pca.fit_transform(xhat_lem)   # (T, latent_dims)
    x_pc_2 = x_pc_numdims[:, :2]                     # (T, 2)
    W = pca.components_[:, :2]                # (latent_dims, 2) loading matrix
    mu = np.mean(xhat_lem, axis=0)                # (latent_dims,)
    print("shape x_pc_2", np.shape(x_pc_2), "shape W", np.shape(W), "shape mu", np.shape(mu))

    # ---- Eigenvalues Calculations -----
    eigs, vecs, tc = eigs_timeconstants(rslds_lem, 1) # select state you want to extract dynamics matrix from. each state has different dynamics matrix.

    # ---- PLOTS ----
    if date:
        if isinstance(date, str):
            key = f"{plot_key}/{date}"
        else:
            key = plot_key
    else:
        key = plot_key
    
    if path_type:
        key = f"{key}/{path_type}"

    if trial_selection:
        key = f"{key}/{trial_selection}"

    if layer:
        key = f"{key}/{layer}"

    if roi:
        output_folder = Path(f"output/{key}/{disc_states}states_{latent_dims}dims_roi{roi}")
    else:
        output_folder = Path(f"output/{key}/{disc_states}states_{latent_dims}dims")

    output_folder.mkdir(parents=True, exist_ok=True)

    # RAW DATA VISUALIZATION
    if type is DataType.DualShank:
        fig0, ax0 = plt.subplots(figsize=(7,5))
        visualize_session(full, ax=ax0)
        ax0.set_title(f"Spike Raster Plot: {key}")
        fig0.tight_layout(pad=2)
        fig0.savefig(output_folder / "spikes.png")

    elif type is DataType.RbpCre:
        fig0, axes0 = trace_sanity_check(full, random_seed=42)
        fig0.suptitle(f"Calcium Trace of Neurons: {key}")
        fig0.savefig(output_folder / "calcium_trace.png")

    elif type is DataType.L23:
        fig0, axes0 = trace_sanity_check_l23(full, random_seed=42)
        fig0.suptitle(f"Calcium Trace of Neurons: {key}")
        fig0.savefig(output_folder / "calcium_trace.png")
    
    elif type is DataType.GCaMP8:
        fig0, axes0 = trace_sanity_check_dendrite(full, random_seed=42)
        fig0.suptitle(f"Calcium Trace of Dendrites: {key}")
        fig0.savefig(output_folder / "calcium_trace.png")

    # ELBO
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.plot(q_elbos_lem[1:], label="Laplace-EM")
    ax1.legend()
    ax1.set_title(f"ELBO Plot: \n{key}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ELBO")
    fig1.tight_layout(pad=2)

    fig1.savefig(output_folder / "elbo.png")

    # PCA EXPLAINED VARIANCE FROM LATENT DIMS
    fig1b, ax1b = plt.subplots(figsize=(6,6))
    explained_variance_ratio = pca.explained_variance_ratio_
    bars = ax1b.bar(range(len(explained_variance_ratio)), explained_variance_ratio)
    ax1b.bar_label(bars, fmt='%.2f', padding=3)
    ax1b.set_title(f"Explained Variance Ratio of Latent Dims: \n{key}")
    ax1b.set_xlabel("Principal Component")
    ax1b.set_ylabel("Explained Variance Ratio")
    fig1b.tight_layout(pad=2)

    fig1b.savefig(output_folder / "expl_var_ratio.png")

    # PLOT OF PC1 AND PC2 OVER TIME
    fig1c, [ax1c_a, ax1c_b] = plt.subplots(figsize=(15,5), nrows=2, ncols=1)
    ax1c_a.plot(x_pc_2[:, 0], color='xkcd:windows blue', linewidth=0.5)
    ax1c_a.set_title(f"PC1 over Time: \n{key}")
    ax1c_a.set_xlabel("Time")
    ax1c_a.set_ylabel("PC1 Value")
    ax1c_b.plot(x_pc_2[:, 1], color='xkcd:red', linewidth=0.5)
    ax1c_b.set_title(f"PC2 over Time: \n{key}") 
    ax1c_b.set_xlabel("Time")
    ax1c_b.set_ylabel("PC2 Value")

    fig1c.tight_layout(pad=2)
    fig1c.savefig(output_folder / "pc_timeseries.png")

    # PLOT OF MOST LIKELY DISCRETE STATE OVER TIME
    if trial_idx:
        fig1d, ax1d = plt.subplots(figsize=(10, 4))
    else:
        fig1d, ax1d = plt.subplots(figsize=(20, 4))


    most_likely_state_plot(disc_states, zhat_lem, ax1d, trial_break=trial_break, trial_idx=trial_idx)
    if trial_idx:
        fig1d.suptitle(f"Most Likely Discrete State, Trial {trial_idx}: \n{key}")
    else:
        fig1d.suptitle(f"Most Likely Discrete State over Time: \n{key}")

    ax1d.set_xlabel("Time (frames)")
    ax1d.set_ylabel("Most Likely State")

    fig1d.tight_layout(pad=2)
    fig1d.savefig(output_folder / "most_likely_state.png")

    # INFERRED TRAJECTORY
    fig2, ax2 = plt.subplots(figsize=(6,6))
    plot_trajectory(zhat_lem, x_pc_2, ax=ax2)
    ax2.set_title(f"Inferred Trajectory in PC Space (Laplace-EM): \n {key}")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    fig2.tight_layout(pad=2)

    fig2.savefig(output_folder / "trajectory.png")

    # FLOW FIELD
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    lim = abs(x_pc_2).max(axis=0) + margin
    plot_pca_flowfield(rslds_lem, W, mu, key,
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
    ax4.set_title(f"Eigenvalues from Dynamics Matrix of State: \n{key}")
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
    ax5.set_title(f'Bar Plot of Time Constants: \n{key}')
    ax5.set_ylabel('Milliseconds') # need to check this
    fig5.tight_layout(pad=2)

    fig5.savefig(output_folder / "timeconstants.png")

    # SUPERIMPOSE TRAJECTORY AND FLOW FIELDS
    fig6, ax6 = plt.subplots(figsize=(6,6))

    plot_trajectory(zhat_lem, x_pc_2, ax=ax6)

    lim = abs(x_pc_2).max(axis=0) + margin
    plot_pca_flowfield(rslds_lem, W, mu, key,
                        xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]),
                        nxpts=nxpts, nypts=nypts, alpha=alpha, ax=ax6)
    
    ax6.set_title(f"Flow Field & Trajectory in PC Space: \n{key}")
    ax6.set_xlabel("PC1")
    ax6.set_ylabel("PC2")
    fig6.tight_layout(pad=2)

    fig6.savefig(output_folder / "superimposedtraj.png")

    # SINGLE NEURON CONTRIBUTION
    fig7 = single_neuron_contribution(rslds_lem, state_idx=state_idx)
    # fig7.suptitle(f"Single Neuron Contributions of ROI {roi}: {key}")
    fig7.savefig(output_folder / "single_neuron_contribution.png")

    # PLOT OF DISCRETE STATE CONTRIBUTIONS OVER TIME
    fig8, ax8 = plt.subplots(figsize=(6,6))


    # dynamic_velocity(rslds_lem)

    print(f"\nSAVED: plots at path {output_folder}\n")
    print("----------------------------------------------------------------------------------------------------\n")
    plt.close("all")

    if plot:
        plt.show()

    return rslds_lem, xhat_lem, x_pc_2, zhat_lem, q_elbos_lem, q_lem, pca



# ----------------------------- CROSS VALIDATION FUNCTION -----------------------------


def cross_val(raw_data, type: DataType, heldout_frac=0.1, n_repeats=3, var_threshold=0.8, **kwargs):
    """ datas is raw .mat file
    """
    
    # ---- REPLACE TYPE LOADING HERE WITH WHAT IS IN THE MAIN RSLDS FUNCTION. THIS IS PLACEHOLDER RN.
    data = None
    # ----

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

