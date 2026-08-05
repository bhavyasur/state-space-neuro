import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np
import matplotlib.pyplot as plt

from src.l23.l23_load_util import load_dfoverf_l23, full_session_trialsliced_l23
from src.rslds.rSLDS import run_rslds_pipeline, DataType, cross_val

if __name__=="__main__":
    
    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4
    latent_dims = 8

    # path_1357 = "data/dj/NeuronByDay_1357_qc_rescued.mat"
    # plot_1357 = "M2_1357"

    # naive_1357 = "0520"
    # interm_1357 = "0605"
    # expert_1357 = "0626"

    # run_rslds_pipeline(path_1357, disc_states, latent_dims, plot_1357, DataType.M2, date=naive_1357, num_iters=50)
    # run_rslds_pipeline(path_1357, disc_states, latent_dims, plot_1357, DataType.M2, date=interm_1357, num_iters=50)
    # run_rslds_pipeline(path_1357, disc_states, latent_dims, plot_1357, DataType.M2, date=expert_1357, num_iters=50)

    # plt.close('all')

    path_073723 = "data/dj/NeuronByDay_073723_qc_rescued.mat"
    plot_073723 = "M2_073723"

    naive_073723 = "0417" # 55%
    interm_073723 = "0424" 
    expert_073723 = "0501" # 83%

    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, m2_correct_only=True, date=naive_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, m2_correct_only=True, date=interm_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, m2_correct_only=True, date=expert_073723, num_iters=50, bin_size=3)

    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="right", date=naive_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="right", date=interm_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="right", date=expert_073723, num_iters=50, bin_size=3)

    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="right", m2_correct_only=True, date=naive_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="right", m2_correct_only=True, date=interm_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="right", m2_correct_only=True, date=expert_073723, num_iters=50, bin_size=3)

    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="left", date=naive_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="left", date=interm_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="left", date=expert_073723, num_iters=50, bin_size=3)

    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="left", m2_correct_only=True, date=naive_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="left", m2_correct_only=True, date=interm_073723, num_iters=50, bin_size=3)
    run_rslds_pipeline(path_073723, disc_states, latent_dims, plot_073723, DataType.M2, trial_selection="left", m2_correct_only=True, date=expert_073723, num_iters=50, bin_size=3)