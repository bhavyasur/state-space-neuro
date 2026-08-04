import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.l23.l23_load_util import load_dfoverf_l23, full_session_trialsliced_l23
from src.rslds.rSLDS import run_rslds_pipeline, DataType, cross_val

if __name__=="__main__":
    
    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4
    latent_dims = 8

    path_1357 = "data/dj/NeuronByDay_1357_qc_rescued.mat"
    plot_1357 = "M2_1357"

    naive_1357 = "0520"
    interm_1357 = "0605"
    expert_1357 = "0626"

    run_rslds_pipeline(path_1357, disc_states, latent_dims, plot_1357, DataType.M2, date=naive_1357, num_iters=50)
    run_rslds_pipeline(path_1357, disc_states, latent_dims, plot_1357, DataType.M2, date=interm_1357, num_iters=50)
    run_rslds_pipeline(path_1357, disc_states, latent_dims, plot_1357, DataType.M2, date=expert_1357, num_iters=50)