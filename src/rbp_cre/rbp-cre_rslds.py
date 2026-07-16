import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.rslds.rSLDS import run_rslds_pipeline, DataType

if __name__=="__main__":

    day1_5 = "data/shulan/#61635_Rbp_GCaMP_mRuby/092324_61635" # SET MANUALLY
    plot_key1_5 = "#61635/092324_61635" # SET MANUALLY

    day2_5 = "data/shulan/#61635_Rbp_GCaMP_mRuby/100124_61635" # SET MANUALLY
    plot_key2_5 = "#61635/100124_61635" # SET MANUALLY

    day3_5 = "data/shulan/#61635_Rbp_GCaMP_mRuby/101624_61635" # SET MANUALLY
    plot_key3_5 = "#61635/101624_61635" # SET MANUALLY

    day4_5 = "data/shulan/#61635_Rbp_GCaMP_mRuby/103124_61635" # SET MANUALLY
    plot_key4_5 = "#61635/103124_61635" # SET MANUALLY

    day5_5  = "data/shulan/#61635_Rbp_GCaMP_mRuby/110324_61635" # SET MANUALLY
    plot_key5_5 = "#61635/110324_61635" # SET MANUALLY



    day1_7 = "data/shulan/#61637_Rbp_GCaMP_mRuby/092324_61637" # SET MANUALLY
    plot_key1_7 = "#61637/092324_61637" # SET MANUALLY

    day2_7 = "data/shulan/#61637_Rbp_GCaMP_mRuby/101624_61637" # SET MANUALLY
    plot_key2_7 = "#61637/101624_61637" # SET MANUALLY

    day3_7 = "data/shulan/#61637_Rbp_GCaMP_mRuby/102124_61637" # SET MANUALLY
    plot_key3_7 = "#61637/102124_61637" # SET MANUALLY

    day4_7 = "data/shulan/#61637_Rbp_GCaMP_mRuby/102524_61637" # SET MANUALLY
    plot_key4_7 = "#61637/102524_61637" # SET MANUALLY

    day5_7  = "data/shulan/#61637_Rbp_GCaMP_mRuby/102824_61637" # SET MANUALLY
    plot_key5_7 = "#61637/102824_61637" # SET MANUALLY

    day6_7  = "data/shulan/#61637_Rbp_GCaMP_mRuby/103024_61637" # SET MANUALLY
    plot_key6_7 = "#61637/103024_61637" # SET

    day7_7  = "data/shulan/#61637_Rbp_GCaMP_mRuby/110724_61637" # SET MANUALLY
    plot_key7_7 = "#61637/110724_61637" # SET


    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4 
    latent_dims = 8

    # -- RUN --
    # if using rbp-cre data, include "roi". default of "roi" is None. unless it's suite2p data and you want to do tuft vs. trunk
    # set trial_selection to full, go, or nogo depending on what you want to run. if full, will run on all trials. if go or nogo, will run on only those trials.
    # can set path_type="manual" if working with a different data structure type than default, just want to load the .mat file directly rather than it searching the animal's folder.


    # ---- ROI = 1 ----

    # 61635

    # naive
    run_rslds_pipeline(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=1, num_iters=50)

    # intermediate
    run_rslds_pipeline(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=1, num_iters=50)

     # expert
    run_rslds_pipeline(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=1, num_iters=50)



    # 61637


    # naive
    run_rslds_pipeline(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=1, num_iters=50)

    # intermediate
    run_rslds_pipeline(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=1, num_iters=50)

    # expert
    run_rslds_pipeline(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=1, num_iters=50)
    run_rslds_pipeline(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=1, num_iters=50)

    
    
    # ---- ROI=2 ------

    # 61635

    # naive
    run_rslds_pipeline(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day3_5, disc_states, latent_dims, plot_key3_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=2, num_iters=50)

    # intermediate
    run_rslds_pipeline(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day5_5, disc_states, latent_dims, plot_key5_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=2, num_iters=50)

     # expert
    run_rslds_pipeline(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day4_5, disc_states, latent_dims, plot_key4_5, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=2, num_iters=50)



    # 61637


    # naive
    run_rslds_pipeline(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day3_7, disc_states, latent_dims, plot_key3_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=2, num_iters=50)

    # intermediate
    run_rslds_pipeline(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day4_7, disc_states, latent_dims, plot_key4_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=2, num_iters=50)

    # expert
    run_rslds_pipeline(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, trial_selection="go", path_type="suite2p", roi=2, num_iters=50)
    run_rslds_pipeline(day5_7, disc_states, latent_dims, plot_key5_7, type=DataType.RbpCre, trial_selection="nogo", path_type="suite2p", roi=2, num_iters=50)

