import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.rslds.rSLDS import run_rslds_pca_flowfield, DataType
from src.gcamp8.gcamp8_load_util import load_dfoverf_dendrite, full_session_trialsliced_dendrite, concatenate_sessions

if __name__=="__main__":

    path = "data/shulan/#23819M_GCaMP8m_rg_PoM-051325_23819-Ca_sliceTCA_reconstructed_data.mat"
    plot_key = "#23819-FOV2/051325_23819/sliceTCAreconstr"
    
    disc_states = 4 
    latent_dims = 8

    run_rslds_pca_flowfield(path, disc_states, latent_dims, plot_key, testing=True, path_type="binarized", type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)
    # run_rslds_pca_flowfield(path, disc_states, latent_dims, plot_key, testing=True, path_type="reconstructed", type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)


    ## date sets per condition:
    # naive = ["050425_23819"]
    # naive_plot_key = '#23819M_GCaMP8m_rg_PoM-FOV2/naive_concat'

    # intermediate = ['050625_23819', '050825_23819']
    # intermediate_plot_key = '#23819M_GCaMP8m_rg_PoM-FOV2/intermediate_concat'

    # expert = ['051325_23819', '051925_23819']
    # expert_plot_key = '#23819M_GCaMP8m_rg_PoM-FOV2/expert_concat'

    # disc_states = 4 
    # latent_dims = 8

    # run_rslds_pca_flowfield(folder_path, disc_states, latent_dims, intermediate_plot_key, date=intermediate, trial_selection="session_concat", type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)


    # folder = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2"
        # # plot_key = "#23819M_GCaMP8m_rg_PoM-FOV2"
        # # date1 = "050425"
        # # date2 = "050625"
        # # date3 = "050825"
        # # date4 = "051325"
        # # date5 = "051925"

        # folder = "data/shulan/51521F_GCaMP8m_rg_PoM-FOV2"
        # plot_key = "51521F_GCaMP8m_rg_PoM-FOV2"
        # date1 = "011426"
        # date2 = "012026"
        # date3 = "020626"


        # # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
        # disc_states = 4 
        # latent_dims = 8

        # # -- RUN --
        # run_rslds_pca_flowfield(folder, disc_states, latent_dims, plot_key, trial_selection="single_session", date=date1, type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)
        # run_rslds_pca_flowfield(folder, disc_states, latent_dims, plot_key, trial_selection="single_session", date=date2, type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)
        # run_rslds_pca_flowfield(folder, disc_states, latent_dims, plot_key, trial_selection="single_session", date=date3, type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)
        # # run_rslds_pca_flowfield(folder, disc_states, latent_dims, plot_key, date=date4, type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)
        # # run_rslds_pca_flowfield(folder, disc_states, latent_dims, plot_key, date=date5, type=DataType.GCaMP8, state_idx=1, num_iters=50, plot=False)

        # # -------------
