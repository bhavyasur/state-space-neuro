import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.rslds.rSLDS import run_rslds_pca_flowfield, DataType

if __name__=="__main__":
    raw = "data/shulan/091924_61601" # SET MANUALLY
    plot_key = "61601/091924_61601" # SET MANUALLY

    # cross_val(raw, type=DataType.Calcium, roi=1)

    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4 
    latent_dims = 8

    # -- RUN --
    # if using calcium data, include "roi". default of "roi" is None.  

    run_rslds_pca_flowfield(raw, disc_states, latent_dims, plot_key, type=DataType.RbpCre, roi=1, state_idx=1, num_iters=50, plot=False)