# ----------------------------- IMPORTS -----------------------------

# 1) path extension
import sys
import warnings
import json
from pathlib import Path
ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))
import glob

# 2) loading functions + utilities for rSLDS. add to this if adding more data types
from src.rbp_cre.rbp_load_util import ( load_dfoverf_rbp, full_session_rbp, 
                                       visualize_trace, trace_sanity_check, full_session_trialsliced_rbp, 
                                       load_trialtype_idx_rbp, gonogotrials_sliced_rbp, load_trialbreak_rbp, behavioral_plot_rbp )
from src.dual_shank.ds_load_util import load_spikes, full_session, visualize_session 
from src.l23.l23_load_util import ( load_dfoverf_l23, full_session_l23, 
                                   full_session_trialsliced_l23, load_trialtype_idx_l23,
                                   trace_sanity_check_l23, gonogotrials_sliced_l23, behavioral_plot_l23, 
                                   load_trialbreak_l23, session_concat_pipeline_l23 )
from src.gcamp8.gcamp8_load_util import ( load_dfoverf_dendrite, full_session_dendrite, 
                                         full_session_trialsliced_dendrite, full_session_trialsliced_thresholded_dendrite, load_dfoverf_problemtest,
                                         trace_sanity_check_dendrite, session_concat_pipeline, spikes_smooth, load_trialbreak_dendrite, 
                                         gonogotrials_sliced_dendrite, load_trialtype_idx_dendrite, behavioral_plot_dendrite )
from src.rslds.rslds_util import ( plot_trajectory, bin_smooth, plot_pca_flowfield, 
                                  eigs_timeconstants, plot_cv_heatmap, select_trial_from_trial_break,
                                  softplus, single_neuron_contribution, most_likely_state_plot, trial_average_pc, trial_average_zhat, full_go_nogo )

# 3) other necessary imports
import autograd.numpy as np
import autograd.numpy.random as npr
from collections import namedtuple
from sklearn.decomposition import PCA
import ssm
from sklearn.model_selection import KFold
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

def run_rslds_pipeline(raw_data, disc_states, latent_dims, plot_key, type: DataType, 
                            trial_selection: Literal["go", "nogo", None] = None,
                            path_type: Literal["manual", "suite2p", "sliceTCA", "binarized", "reconstructed", None] = None,
                            layer: Literal["L2", "L3", None] = None,
                            testing: bool = False,
                            trial_idx: int = None,
                            trial_structure: Literal["single_trial", "full_sess", None] = None,
                            specific_loadtype: Literal["single_session_thresholded", "session_concat", "spikes", None] = None,
                            roi=None, date=None, l23_type: Literal["bessel", "etl", None] = None, plot: bool=False, save_output: bool=False, nxpts=20, nypts=20, alpha=0.8, num_iters=50, margin=1.0):
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
        go_idx, nogo_idx = load_trialtype_idx_rbp(raw_data, path_type=path_type, roi=roi)
        trial_break = load_trialbreak_rbp(raw_data, path_type=path_type, roi=roi)

        dfoverf = load_dfoverf_rbp(raw_data, path_type=path_type, roi=roi)
        if path_type:
            print(f"Loaded RbpCre data with path_type={path_type}\n")
        if roi:
            print(f"Loaded RbpCre data with ROI: {roi}\n")

        full, trial_break_sliced = full_session_trialsliced_rbp(dfoverf)
        go_trials = gonogotrials_sliced_rbp(dfoverf, go_idx)
        nogo_trials = gonogotrials_sliced_rbp(dfoverf, nogo_idx)

        if trial_selection == "go":
            data = go_trials.T.astype(int)
            print("Running on go trials only.\n")
            for_trace = go_trials
        elif trial_selection == "nogo":
            data = nogo_trials.T.astype(int)
            print("Running on nogo trials only.\n")
            for_trace = nogo_trials
        else:
            data = full.T.astype(int)
            print("Running on full session.\n")
            for_trace = full

    elif type is DataType.L23:

        if specific_loadtype == "session_concat":
            full, trial_break_sliced = session_concat_pipeline_l23(list_of_folder_paths=raw_data, trial_selection=trial_selection, layer=layer) # FULL HERE = CONCATENATED SESSIONS. raw_data is a list
            go_idx, nogo_idx = load_trialtype_idx_l23(raw_data, specific_loadtype="session_concat")
            data = full.T.astype(int)
            for_trace = full
        else:
            dfoverf = load_dfoverf_l23(raw_data, layer=layer)
            if layer:
                print(f"Loaded data for layer {layer}.\n")
            go_idx, nogo_idx = load_trialtype_idx_l23(raw_data)
            full, trial_break_sliced = full_session_trialsliced_l23(dfoverf)
            go_trials = gonogotrials_sliced_l23(dfoverf, go_idx)
            nogo_trials = gonogotrials_sliced_l23(dfoverf, nogo_idx)

            if trial_selection == "go":
                data = go_trials.T.astype(int)
                print("Running on go trials only.\n")
                for_trace = go_trials
            elif trial_selection == "nogo":
                data = nogo_trials.T.astype(int)
                print("Running on nogo trials only.\n")
                for_trace = nogo_trials
            else:
                data = full.T.astype(int)
                print("Running on full session.\n")
                for_trace = full
            
    elif type is DataType.GCaMP8:

        if testing: # if testing
            dfoverf = load_dfoverf_problemtest(raw_data, path_type=path_type)
            full, trial_break_sliced = full_session_trialsliced_dendrite(dfoverf)
            data = full.T.astype(int)
        else:
            if specific_loadtype == "single_session_thresholded":
                dfoverf = load_dfoverf_dendrite(raw_data, date)
                full = full_session_trialsliced_thresholded_dendrite(dfoverf)
                print(f"Thresholded full session shape: {np.shape(full)}\n")
                data = full.T.astype(int)
            elif specific_loadtype == "session_concat":
                full = session_concat_pipeline(raw_data, date, trial_selection=trial_selection, path_type=path_type) # FULL HERE = CONCATENATED SESSIONS
                data = full.T.astype(int)
            elif specific_loadtype == "spikes":
                data = spikes_smooth(raw_data, date).astype(int)
                full = data.T
            else: # NORMAL CASE (gcamp_specific_loadtype = None)
                if path_type == "sliceTCA":
                    dfoverf = load_dfoverf_dendrite(raw_data, date, path_type)
                    go_idx, nogo_idx = load_trialtype_idx_dendrite(raw_data, session_date=date, path_type=path_type)
                elif path_type is None:
                    dfoverf = load_dfoverf_dendrite(raw_data, date)
                    go_idx, nogo_idx = load_trialtype_idx_dendrite(raw_data, session_date=date)
                else:
                    raise ValueError("path type should be 'sliceTCA' or None for the normal, single-session case (specific_loadtype == None).")

                full, trial_break_sliced = full_session_trialsliced_dendrite(dfoverf)
                go_trials = gonogotrials_sliced_dendrite(dfoverf, go_idx)
                nogo_trials = gonogotrials_sliced_dendrite(dfoverf, nogo_idx)

                if trial_selection == "go":
                    data = go_trials.T.astype(int)
                    print("Running on go trials only.\n")
                    for_trace = go_trials
                elif trial_selection == "nogo":
                    data = nogo_trials.T.astype(int)
                    print("Running on nogo trials only.\n")
                    for_trace = nogo_trials
                else:
                    data = full.T.astype(int)
                    print("Running on full session.\n")
                    for_trace = full

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
    yhat_lem = rslds.smooth(xhat_lem, y)

    rslds_lem = copy.deepcopy(rslds)

    # ---- PCA on the latent trajectory ----
    pca = PCA(n_components=latent_dims)
    x_pc_numdims = pca.fit_transform(xhat_lem)   # (T, latent_dims)
    x_pc_2 = x_pc_numdims[:, :2]                     # (T, 2)
    W = pca.components_[:, :2]                # (latent_dims, 2) loading matrix
    mu = np.mean(xhat_lem, axis=0)                # (latent_dims,)

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
    else:
        key = f"{key}/full"
    
    if trial_idx and trial_structure == "single_trial":
        key = f"{key}/trial{trial_idx}"

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
        fig0, axes0 = trace_sanity_check(for_trace, random_seed=42)
        fig0.suptitle(f"Calcium Trace of Neurons: {key}")
        fig0.savefig(output_folder / "calcium_trace.png")

    elif type is DataType.L23:
        fig0, axes0 = trace_sanity_check_l23(for_trace, random_seed=42)
        fig0.suptitle(f"Calcium Trace of Neurons: {key}")
        fig0.savefig(output_folder / "calcium_trace.png")
    
    elif type is DataType.GCaMP8:
        fig0, axes0 = trace_sanity_check_dendrite(for_trace, random_seed=42)
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
    # RIGHT NOW THIS IS IRREGARDLESS OF GO, NOGO, OR FULL SET OF TRIALS. NEED TO FIX.
    if trial_structure == "single_trial":

        if trial_selection == "go" or trial_selection == "nogo":
            warnings.warn("Since you selected 'single_trial' as your trial_structure, this does not take into account whether you have selected go vs. " \
            "nogo trials, simply selects the trial index you set. If you want to take into account go vs nogo trials, us the trial_averaged choice for trial_structure.")

        if not trial_idx:
            raise ValueError("trial_idx must be set for trial_structure to be single trial.")
        fig1c, [ax1c_a, ax1c_b] = plt.subplots(figsize=(10,6), nrows=2, ncols=1)

        start_idx, end_idx = select_trial_from_trial_break(trial_break=trial_break_sliced, trial_idx=trial_idx)
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        ax1c_a.plot(x_pc_2[start_idx:end_idx, 0], color='xkcd:windows blue', linewidth=1.5)
        ax1c_a.set_title(f"PC1 over Time, Trial {trial_idx}: \n{key}")
        ax1c_a.set_xlabel("Time")
        ax1c_a.set_ylabel("PC1 Value")
        ax1c_a.set_xlim(0, end_idx-start_idx)

        ax1c_b.plot(x_pc_2[start_idx:end_idx, 1], color='xkcd:red', linewidth=1.5)
        ax1c_b.set_title(f"PC2 over Time, Trial {trial_idx}: \n{key}") 
        ax1c_b.set_xlabel("Time")
        ax1c_b.set_ylabel("PC2 Value")
        ax1c_b.set_xlim(0, end_idx-start_idx)

        fig1c.tight_layout(pad=2)
        fig1c.savefig(output_folder / "pc_timeseries.png")

    else:

        if trial_selection == "go":
            avg_trial = trial_average_pc(trial_break_sliced, x_pc_2, trial_selection=trial_selection, gonogo=go_idx)
        elif trial_selection == "nogo":
            avg_trial = trial_average_pc(trial_break_sliced, x_pc_2, trial_selection=trial_selection, gonogo=nogo_idx)
        else:
            avg_trial = trial_average_pc(trial_break_sliced, x_pc_2)
        
        length = np.shape(avg_trial)[0]

        fig1c, [ax1c_a, ax1c_b] = plt.subplots(figsize=(10,6), nrows=2, ncols=1)
        ax1c_a.plot(avg_trial[:, 0], color='xkcd:windows blue', linewidth=1.5)
        ax1c_a.set_title(f"PC1 over Time, Trial Averaged: \n{key}")
        ax1c_a.set_xlabel("Time")
        ax1c_a.set_ylabel("PC1 Value")
        ax1c_a.set_xlim(0, length)

        ax1c_b.plot(avg_trial[:, 1], color='xkcd:red', linewidth=1.5)
        ax1c_b.set_title(f"PC2 over Time, Trial Averaged: \n{key}") 
        ax1c_b.set_xlabel("Time")
        ax1c_b.set_ylabel("PC2 Value")
        ax1c_b.set_xlim(0, length)

        fig1c.tight_layout(pad=2)
        fig1c.savefig(output_folder / "pc_timeseries.png")
   
    # PLOT OF MOST LIKELY DISCRETE STATE OVER TIME
    
    if type is DataType.L23: # only l23 has the behavioral plot function set up, still need to do for the other datatypes
        fig1d, [ax1d_a, ax1d_b] = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

        if trial_structure == "single_trial":

            if trial_selection == "go" or trial_selection == "nogo":
                warnings.warn("Since you selected 'single_trial' as your trial_structure, this does not take into account whether you have selected go vs. " \
                "nogo trials, simply selects the trial index you set. If you want to take into account go vs nogo trials, us the trial_averaged choice for trial_structure.")

            if not trial_idx:
                raise ValueError("trial_idx must be set for trial_structure to be single trial.")
            most_likely_state_plot(disc_states, zhat_lem, ax1d_a, trial_break=trial_break_sliced, trial_structure=trial_structure, trial_idx=trial_idx)
            behavioral_plot_l23(trial_break_sliced, l23_type=l23_type, trial_structure=trial_structure, ax=ax1d_b, trial_idx=trial_idx)
            ax1d_a.set_title(f"Most Likely Discrete State, Trial {trial_idx}: \n{key}")
            ax1d_b.set_title(f"Online vs. Offline Phases of Trial, Trial {trial_idx}: \n{key}")

        elif trial_structure == "full_sess":
            if trial_selection == "go":
                full = full_go_nogo(trial_break_sliced, zhat_lem, trial_selection=trial_selection, gonogo=go_idx)
            elif trial_selection == "nogo":
                full = full_go_nogo(trial_break_sliced, zhat_lem, trial_selection=trial_selection, gonogo=nogo_idx)
            else:
                full = zhat_lem

            most_likely_state_plot(disc_states, full, ax1d_a, trial_break=trial_break_sliced, trial_structure=trial_structure)
            behavioral_plot_l23(trial_break_sliced, l23_type=l23_type, ax=ax1d_b, trial_structure=trial_structure)

            ax1d_a.set_title(f"Most Likely Discrete State, Full Session: \n{key}")
            ax1d_b.set_title(f"Online vs. Offline Phases of Trial, Full Session: \n{key}")

        else: # trial-averaged case
            
            if trial_selection == "go":
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem, trial_selection="go", gonogo=go_idx)
            elif trial_selection == "nogo":
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem, trial_selection="nogo", gonogo=nogo_idx)
            else:
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem) #zhat lem averaged - equal to length of one trial

            most_likely_state_plot(disc_states, avg_trial, ax1d_a, trial_break=trial_break_sliced)
            behavioral_plot_l23(trial_break_sliced, l23_type=l23_type, ax=ax1d_b)
            ax1d_a.set_title(f"Most Likely Discrete State, Trial Averaged: \n{key}")
            ax1d_b.set_title(f"Online vs. Offline Phases: \n{key}")


        ax1d_a.set_xlabel("Time (frames)")
        ax1d_b.set_xlabel("Time (frames)")
        
        fig1d.tight_layout(pad=2)
        fig1d.savefig(output_folder / "most_likely_state.png")

    elif type is DataType.GCaMP8:
        fig1d, [ax1d_a, ax1d_b] = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

        if trial_structure == "single_trial":
            if trial_selection == "go" or trial_selection == "nogo":
                warnings.warn("Since you selected 'single_trial' as your trial_structure, this does not take into account whether you have selected go vs. " \
                "nogo trials, simply selects the trial index you set. If you want to take into account go vs nogo trials, us the trial_averaged choice for trial_structure.")

            if not trial_idx:
                raise ValueError("trial_idx must be set for trial_structure to be single trial.")
            
            most_likely_state_plot(disc_states, zhat_lem, ax1d_a, trial_break=trial_break_sliced, trial_structure=trial_structure, trial_idx=trial_idx)
            if path_type == "sliceTCA":
                behavioral_plot_dendrite(raw_data, date, trial_break_sliced, path_type=path_type, trial_structure=trial_structure, ax=ax1d_b, trial_idx=trial_idx)
            else:
                behavioral_plot_dendrite(raw_data, date, trial_break_sliced, trial_structure=trial_structure, ax=ax1d_b, trial_idx=trial_idx)
            ax1d_a.set_title(f"Most Likely Discrete State, Trial {trial_idx}: \n{key}")
            ax1d_b.set_title(f"Online vs. Offline Phases of Trial, Trial {trial_idx}: \n{key}")

        else: # trial averaged
            
            if trial_selection == "go":
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem, trial_selection="go", gonogo=go_idx)
            elif trial_selection == "nogo":
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem, trial_selection="nogo", gonogo=nogo_idx)
            else:
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem) #zhat lem averaged - equal to length of one trial

            most_likely_state_plot(disc_states, avg_trial, ax1d_a, trial_break=trial_break_sliced)

            if path_type == "sliceTCA":
                behavioral_plot_dendrite(raw_data, date, trial_break_sliced, path_type=path_type, ax=ax1d_b)
            else:
                behavioral_plot_dendrite(raw_data, date, trial_break_sliced, ax=ax1d_b)

            ax1d_a.set_title(f"Most Likely Discrete State, Trial Averaged: \n{key}")
            ax1d_b.set_title(f"Online vs. Offline Phases: \n{key}")

        ax1d_a.set_xlabel("Time (frames)")
        ax1d_b.set_xlabel("Time (frames)")
        
        fig1d.tight_layout(pad=2)
        fig1d.savefig(output_folder / "most_likely_state.png")

    elif type is DataType.RbpCre:
        fig1d, [ax1d_a, ax1d_b] = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

        if trial_structure == "single_trial":
            if trial_selection == "go" or trial_selection == "nogo":
                warnings.warn("Since you selected 'single_trial' as your trial_structure, this does not take into account whether you have selected go vs. " \
                "nogo trials, simply selects the trial index you set. If you want to take into account go vs nogo trials, us the trial_averaged choice for trial_structure.")

            if not trial_idx:
                raise ValueError("trial_idx must be set for trial_structure to be single trial.")
            
            most_likely_state_plot(disc_states, zhat_lem, ax=ax1d_a, trial_break=trial_break_sliced, trial_structure=trial_structure, trial_idx=trial_idx)
            behavioral_plot_rbp(raw_data, trial_break_sliced, path_type='suite2p', ax=ax1d_b, trial_structure=trial_structure, trial_idx=trial_idx, roi=roi)
        
            ax1d_a.set_title(f"Most Likely Discrete State, Trial {trial_idx}: \n{key}")
            ax1d_b.set_title(f"Online vs. Offline Phases of Trial, Trial {trial_idx}: \n{key}")

        else: # trial averaged
            
            if trial_selection == "go":
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem, trial_selection="go", gonogo=go_idx)
            elif trial_selection == "nogo":
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem, trial_selection="nogo", gonogo=nogo_idx)
            else:
                avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem) #zhat lem averaged - equal to length of one trial

            most_likely_state_plot(disc_states, avg_trial, ax1d_a, trial_break=trial_break_sliced)
            behavioral_plot_rbp(raw_data, trial_break_sliced, path_type='suite2p', ax=ax1d_b, roi=roi)

            ax1d_a.set_title(f"Most Likely Discrete State, Trial Averaged: \n{key}")
            ax1d_b.set_title(f"Online vs. Offline Phases: \n{key}")

        ax1d_a.set_xlabel("Time (frames)")
        ax1d_b.set_xlabel("Time (frames)")
        
        fig1d.tight_layout(pad=2)
        fig1d.savefig(output_folder / "most_likely_state.png")

    else:
        fig1d, ax1d = plt.subplots(figsize=(10, 4))
        avg_trial = trial_average_zhat(trial_break_sliced, zhat_lem)

        if trial_structure == "single_trial":
            if not trial_idx:
                raise ValueError("trial_idx must be set for trial_structure to be single trial.")
            most_likely_state_plot(disc_states, zhat_lem, ax1d, trial_break=trial_break_sliced, trial_idx=trial_idx)
            ax1d.set_title(f"Most Likely Discrete State, Trial {trial_idx}: \n{key}")

        else:
            most_likely_state_plot(disc_states, avg_trial, ax1d, trial_break=trial_break_sliced)
            ax1d.set_title(f"Most Likely Discrete State\n")

        ax1d.set_xlabel("Time (frames)")
        
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

    for i in range(disc_states):
        state_key = f"{key}/state{i}"

        if roi:
            state_output_folder = Path(f"output/{state_key}/{disc_states}states_{latent_dims}dims_roi{roi}")
        else:
            state_output_folder = Path(f"output/{state_key}/{disc_states}states_{latent_dims}dims")

        state_output_folder.mkdir(parents=True, exist_ok=True)

        # ---- Eigenvalues Calculations -----
        eigs, vecs, tc = eigs_timeconstants(rslds_lem, i) # select state you want to extract dynamics matrix from. each state has different dynamics matrix.

        # EIGENSPECTRUM
        real = eigs.real
        imag = eigs.imag
        fig4, ax4 = plt.subplots(figsize=(6,6))
        ax4.scatter(real, imag)
        ax4.grid()
        ax4.set_title(f"Eigenvalues from Dynamics Matrix of State {i}: \n{state_key}")
        ax4.set_xlabel("Real Axis")
        ax4.set_ylabel("Imaginary Axis")
        fig4.tight_layout(pad=2)

        fig4.savefig(state_output_folder / "eigenspectrum.png")

        # TIMECONSTANTS
        labels = eigs.astype(str)
        fig5, ax5 = plt.subplots(figsize=(6,6))
        bars = ax5.bar(labels, tc)
        ax5.set_xticks([])
        ax5.bar_label(bars, padding=3)
        ax5.set_title(f'Bar Plot of Time Constants, State {i}: \n{state_key}')
        ax5.set_ylabel('Milliseconds') # need to check this
        fig5.tight_layout(pad=2)

        fig5.savefig(state_output_folder / "timeconstants.png")

        # SINGLE NEURON CONTRIBUTION
        fig7 = single_neuron_contribution(state_idx=i, model=rslds_lem)
        # fig7.suptitle(f"Single Neuron Contributions of ROI {roi}: {key}")
        fig7.savefig(state_output_folder / "single_neuron_contribution.png") 
     

    print(f"\nSAVED: plots at path {output_folder}\n")
    print("----------------------------------------------------------------------------------------------------\n")
    plt.close("all")

    if plot:
        plt.show()

    if save_output:
        save = Path(f"{output_folder}/rslds_lem")
        save.mkdir(parents=True, exist_ok=True)

        for i in range(disc_states):
            A = rslds_lem.dynamics.As # A shape is (num_states, 10, 10)
            matrix = A[i]
            np.savetxt(f"{save}/state{i}.csv", matrix, delimiter=",")

        C = abs(np.squeeze(rslds_lem.emissions.Cs))
        np.savetxt(f"{save}/emissions.csv", C, delimiter=",")

    return {
        "rslds_lem": rslds_lem, 
        "xhat_lem": xhat_lem, 
        "zhat_lem": zhat_lem, 
        "q_lem": q_lem}

"""this is if you want to run state-specific stuff after seeing the output of the general model plots, for example the most likely state plots. but you need to set
save_output=True on your pipeline call for this to work."""
def run_rslds_statespecificplots(state_idx: list[int], 
                                 disc_states: int = None,
                                 latent_dims: int = None,
                                 plot_key: str = None, 
                                 date: int = None, 
                                 path_type: Literal["manual", "suite2p", "sliceTCA", "binarized", "reconstructed", None] = None,
                                 trial_selection: Literal["go", "nogo", None] = None, trial_idx: int = None, 
                                 trial_structure: Literal["single_trial", None] = None, 
                                 layer: Literal["L2", "L3", None] = None, 
                                 roi: int = None,
                                 **kwargs):

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
    else:
        key = f"{key}/full"
    
    if trial_idx and trial_structure == "single_trial":
        key = f"{key}/trial{trial_idx}"

    if layer:
        key = f"{key}/{layer}"

    if roi:
        old_output_folder = Path(f"output/{key}/{disc_states}states_{latent_dims}dims_roi{roi}")
    else:
        old_output_folder = Path(f"output/{key}/{disc_states}states_{latent_dims}dims")

    outer = Path(old_output_folder)
    inner = Path(old_output_folder / "rslds_lem")
    A_files = glob.glob(f"{inner}/*state*.csv")
    C_files = glob.glob(f"{inner}/*emissions.csv")

    rslds_lem_A_dict = {
        Path(file).stem: np.loadtxt(file, delimiter=",")
        for file in A_files
    }

    rslds_lem_C_dict = {
        Path(file).stem: np.loadtxt(file, delimiter=",")
        for file in C_files
    }

    rslds_lem_C = rslds_lem_C_dict["emissions"]

    for i in state_idx:
        rslds_lem_A = rslds_lem_A_dict[f"state{i}"]

        state_key = f"{key}/state{i}"

        if roi:
            state_output_folder = Path(f"output/{state_key}/{disc_states}states_{latent_dims}dims_roi{roi}")
        else:
            state_output_folder = Path(f"output/{state_key}/{disc_states}states_{latent_dims}dims")

        state_output_folder.mkdir(parents=True, exist_ok=True)

        # ---- Eigenvalues Calculations -----
        eigs, vecs, tc = eigs_timeconstants(rslds_lem_A, i, model_type="csv") # select state you want to extract dynamics matrix from. each state has different dynamics matrix.

        # EIGENSPECTRUM
        real = eigs.real
        imag = eigs.imag
        fig4, ax4 = plt.subplots(figsize=(6,6))
        ax4.scatter(real, imag)
        ax4.grid()
        ax4.set_title(f"Eigenvalues from Dynamics Matrix of State {i}: \n{state_key}")
        ax4.set_xlabel("Real Axis")
        ax4.set_ylabel("Imaginary Axis")
        fig4.tight_layout(pad=2)

        fig4.savefig(state_output_folder / "eigenspectrum.png")

        # TIMECONSTANTS
        labels = eigs.astype(str)
        fig5, ax5 = plt.subplots(figsize=(6,6))
        bars = ax5.bar(labels, tc)
        ax5.set_xticks([])
        ax5.bar_label(bars, padding=3)
        ax5.set_title(f'Bar Plot of Time Constants, State {i}: \n{state_key}')
        ax5.set_ylabel('Milliseconds') # need to check this
        fig5.tight_layout(pad=2)

        fig5.savefig(state_output_folder / "timeconstants.png")

        # SINGLE NEURON CONTRIBUTION
        fig7 = single_neuron_contribution(state_idx=i, As=rslds_lem_A, Cs=rslds_lem_C, model_type='csv')
        # fig7.suptitle(f"Single Neuron Contributions of ROI {roi}: {key}")
        fig7.savefig(state_output_folder / "single_neuron_contribution.png") 

        print(f"\nSAVED: plots at path {state_output_folder}\n")
        print("----------------------------------------------------------------------------------------------------\n")




def run_rslds_concat(data, disc_states, latent_dims, type: DataType):
    # data is one numpy array (num_timesteps, num_neurons)
    a = [data]
    y = np.asarray(a)
    #y is (1, num_timesteps, num_neurons), dtype list of numpy arrays

    num_obs = np.shape(y)[2]
    num_neurons = num_obs

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


# ----------------------------- CROSS VALIDATION FUNCTION -----------------------------


def cross_val(full, plot_key, dims: list[int], states: list[int], type: DataType, heldout_frac=0.1, n_repeats=3, var_threshold=0.8, plot: bool = False, **kwargs):
    """ full is full! held consistaent across all datasets."""
    
    # ---- REPLACE TYPE LOADING HERE WITH WHAT IS IN THE MAIN RSLDS FUNCTION. THIS IS PLACEHOLDER RN.
    data = full.T.astype(int)
    # ----
    output_folder = Path(f"output/{plot_key}")
    output_folder.mkdir(parents=True, exist_ok=True)

    num_neurons = np.shape(data)[1]
    obs = num_neurons
    param_grid = {
        'dims': dims,
        'states': states
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
                model, b, c, d, e = run_rslds_concat(train, state, dim, type=type)
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

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_cv_heatmap(results, param_grid, ax=ax1)
    fig1.savefig(output_folder / "crossval.png")

    if plot:
        plt.tightlayout()
        plt.show()

    return results

