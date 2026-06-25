import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.rslds.rSLDS import run_rslds_pca_flowfield, DataType

if __name__=="__main__":
    
    naive_go = "data/shivam/Bessel_140_250/073593DR/Naive/GO"
    plot_naive_go = "Bessel_140_250_073593DR/Naive_GO"

    intermediate_go = "data/shivam/Bessel_140_250/073593DR/Intermediate/GO"
    plot_intermediate_go = "Bessel_140_250_073593DR/Intermediate_GO"

    expert_go = "data/shivam/Bessel_140_250/073593DR/Expert/GO"
    plot_expert_go = "Bessel_140_250_073593DR/Expert_GO"

    naive_nogo = "data/shivam/Bessel_140_250/073593DR/Naive/NOGO"
    plot_naive_nogo = "Bessel_140_250_073593DR/Naive_NOGO"

    intermediate_nogo = "data/shivam/Bessel_140_250/073593DR/Intermediate/NOGO"
    plot_intermediate_nogo = "Bessel_140_250_073593DR/Intermediate_NOGO"

    expert_nogo = "data/shivam/Bessel_140_250/073593DR/Expert/NOGO"
    plot_expert_nogo = "Bessel_140_250_073593DR/Expert_NOGO"

    # cross_val(raw, type=DataType.Calcium, roi=1)

    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4 
    latent_dims = 8

    # -- RUN --
    # if using Rbp-Cre data, include "roi". default of "roi" is None.  
    run_rslds_pca_flowfield(naive_go, disc_states, latent_dims, plot_naive_go, type=DataType.L23, state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(intermediate_go, disc_states, latent_dims, plot_intermediate_go, type=DataType.L23, state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(expert_go, disc_states, latent_dims, plot_expert_go, type=DataType.L23, state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(naive_nogo, disc_states, latent_dims, plot_naive_nogo, type=DataType.L23, state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(intermediate_nogo, disc_states, latent_dims, plot_intermediate_nogo, type=DataType.L23, state_idx=1, num_iters=50, plot=False)
    run_rslds_pca_flowfield(expert_nogo, disc_states, latent_dims, plot_expert_nogo, type=DataType.L23, state_idx=1, num_iters=50, plot=False)


    # -------------
