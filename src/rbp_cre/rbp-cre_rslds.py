import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.rslds.rSLDS import run_rslds_pca_flowfield, DataType

if __name__=="__main__":

    day1_5 = "data/shulan/#61635M_Rbp_GCaMP-mRuby/092324_61635" # SET MANUALLY
    plot_key1_5 = "#61635/092324_61635" # SET MANUALLY

    day2_5 = "data/shulan/#61635M_Rbp_GCaMP-mRuby/100124_61635" # SET MANUALLY
    plot_key2_5 = "#61635/100124_61635" # SET MANUALLY

    day3_5 = "data/shulan/#61635M_Rbp_GCaMP-mRuby/101624_61635" # SET MANUALLY
    plot_key3_5 = "#61635/101624_61635" # SET MANUALLY

    day4_5 = "data/shulan/#61635M_Rbp_GCaMP-mRuby/103124_61635" # SET MANUALLY
    plot_key4_5 = "#61635/103124_61635" # SET MANUALLY

    day5_5  = "data/shulan/#61635M_Rbp_GCaMP-mRuby/110324_61635" # SET MANUALLY
    plot_key5_5 = "#61635/110324_61635" # SET MANUALLY



    day1_7 = "data/shulan/#61637M_Rbp_GCaMP-mRuby/092324_61637" # SET MANUALLY
    plot_key1_7 = "#61637/092324_61637" # SET MANUALLY

    day2_7 = "data/shulan/#61637M_Rbp_GCaMP-mRuby/101624_61637" # SET MANUALLY
    plot_key2_7 = "#61637/101624_61637" # SET MANUALLY

    day3_7 = "data/shulan/#61637M_Rbp_GCaMP-mRuby/102124_61637" # SET MANUALLY
    plot_key3_7 = "#61637/102124_61637" # SET MANUALLY

    day4_7 = "data/shulan/#61637M_Rbp_GCaMP-mRuby/102524_61637" # SET MANUALLY
    plot_key4_7 = "#61637/102524_61637" # SET MANUALLY

    day5_7  = "data/shulan/#61637M_Rbp_GCaMP-mRuby/102824_61637" # SET MANUALLY
    plot_key5_7 = "#61637/102824_61637" # SET MANUALLY

    day6_7  = "data/shulan/#61637M_Rbp_GCaMP-mRuby/103024_61637" # SET MANUALLY
    plot_key6_7 = "#61637/103024_61637" # SET

    day7_7  = "data/shulan/#61637M_Rbp_GCaMP-mRuby/110724_61637" # SET MANUALLY
    plot_key7_7 = "#61637/110724_61637" # SET


    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4 
    latent_dims = 8

    # -- RUN --
    # if using rbp-cre data, include "roi". default of "roi" is None. unless it's suite2p data, then ignore roi.
    # set trial_selection to full, go, or nogo depending on what you want to run. if full, will run on all trials. if go or nogo, will run on only those trials.
    # can set path_type="manual" if working with a different data structure type than default, just want to load the .mat file directly rather than it searching the animal's folder.


    # 61635

    # day 1
    # run_rslds_pca_flowfield(day1_5, disc_states, latent_dims, plot_key1_5, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    # run_rslds_pca_flowfield(day1_5, disc_states, latent_dims, plot_key1_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    # run_rslds_pca_flowfield(day1_5, disc_states, latent_dims, plot_key1_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # # day 2
    # run_rslds_pca_flowfield(day2_5, disc_states, latent_dims, plot_key2_5, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    # run_rslds_pca_flowfield(day2_5, disc_states, latent_dims, plot_key2_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    # run_rslds_pca_flowfield(day2_5, disc_states, latent_dims, plot_key2_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 3
    run_rslds_pca_flowfield(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 4
    run_rslds_pca_flowfield(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 5
    run_rslds_pca_flowfield(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)



    # 61637

    # day 1
    run_rslds_pca_flowfield(day1_7, disc_states, latent_dims, plot_key1_7, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day1_7, disc_states, latent_dims, plot_key1_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day1_7, disc_states, latent_dims, plot_key1_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 2
    run_rslds_pca_flowfield(day2_7, disc_states, latent_dims, plot_key2_7, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day2_7, disc_states, latent_dims, plot_key2_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day2_7, disc_states, latent_dims, plot_key2_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 3
    run_rslds_pca_flowfield(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 4
    run_rslds_pca_flowfield(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 5
    run_rslds_pca_flowfield(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 6
    run_rslds_pca_flowfield(day6_7, disc_states, latent_dims, plot_key6_7, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day6_7, disc_states, latent_dims, plot_key6_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day6_7, disc_states, latent_dims, plot_key6_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)

    # day 7
    run_rslds_pca_flowfield(day7_7, disc_states, latent_dims, plot_key7_7, type=DataType.RbpCre, path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day7_7, disc_states, latent_dims, plot_key7_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(day7_7, disc_states, latent_dims, plot_key7_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", state_idx=1, num_iters=50, plot=False)
