import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.rslds.rSLDS import run_rslds_pipeline, DataType
from src.gcamp8.gcamp8_load_util import load_dfoverf_dendrite, full_session_trialsliced_dendrite, concatenate_sessions

if __name__=="__main__":

    # run_rslds_pipeline(path, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA",  num_iters=50)
    # run_rslds_pipeline(path, disc_states, latent_dims, plot_key, testing=True, path_type="reconstructed", type=DataType.GCaMP8, num_iters=50)

    ## date sets per condition:
    # naive = ["050425_23819"]
    # naive_plot_key = '#23819M_GCaMP8m_rg_PoM-FOV2/naive_concat'

    # intermediate = ['050625_23819', '050825_23819']
    # intermediate_plot_key = '#23819M_GCaMP8m_rg_PoM-FOV2/intermediate_concat'

    # expert = ['051325_23819', '051925_23819']
    # expert_plot_key = '#23819M_GCaMP8m_rg_PoM-FOV2/expert_concat'

    # disc_states = 4 
    # latent_dims = 8

    # run_rslds_pipeline(folder_path, disc_states, latent_dims, intermediate_plot_key, date=intermediate, trial_selection="session_concat", type=DataType.GCaMP8, num_iters=50)


    folder = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2"
    plot_key = "#23819M_GCaMP8m_rg_PoM-FOV2"
    date1 = "050425" # NAIVE
    date2 = "050625" # INTERMEDIATE
    date3 = "050825" # INTERMEDIATE
    date4 = "051325" # EXPERT
    date5 = "051925" # EXPERT

    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4 
    latent_dims = 8

    # -------- RUN --------

    # full
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [0, 2], trial_structure="trial_averaged", date=date1, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [0,3], trial_structure="trial_averaged", date=date2, num_iters=50)
    run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = 0, trial_structure="trial_averaged", date=date3, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [1,3], trial_structure="trial_averaged", date=date4, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [2, 3], trial_structure="trial_averaged", date=date5, num_iters=50)

    # # go trials
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [0, 1], trial_selection="go", trial_structure="trial_averaged", date=date1, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [2, 3, 1], trial_selection="go", trial_structure="trial_averaged", date=date2, num_iters=50)
    run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [2, 1], trial_selection="go", trial_structure="trial_averaged", date=date3, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = 1, trial_selection="go", trial_structure="trial_averaged", date=date4, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [0, 3], trial_selection="go", trial_structure="trial_averaged", date=date5, num_iters=50)

    # # # nogo trials
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = 1, trial_selection="nogo", trial_structure="trial_averaged", date=date1, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = 3, trial_selection="nogo", trial_structure="trial_averaged", date=date2, num_iters=50)
    run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = 2, trial_selection="nogo", trial_structure="trial_averaged", date=date3, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = 2, trial_selection="nogo", trial_structure="trial_averaged", date=date4, num_iters=50)
    # run_rslds_pipeline(folder, disc_states, latent_dims, plot_key, type=DataType.GCaMP8, path_type="sliceTCA", state_idx = [2,3], trial_selection="nogo", trial_structure="trial_averaged", date=date5, num_iters=50)

    # -------------
