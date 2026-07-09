import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

import numpy as np

from src.rslds.rSLDS import run_rslds_pipeline, DataType, run_rslds_statespecificplots

if __name__=="__main__":
    
    # -- SET DESIRED HYPERPARAMETERS MANUALLY (determine with cross validation in rslds_crossval.py) --
    disc_states = 4 
    latent_dims = 8

    
    naive_go1 = "data/shivam/Bessel_140_250/1348DR/Naive/GO"
    plot_naive_go1 = "Bessel_140_250_1348DR/Naive_GO"

    intermediate_go1 = "data/shivam/Bessel_140_250/1348DR/Intermediate/GO"
    plot_intermediate_go1 = "Bessel_140_250_1348DR/Intermediate_GO"

    expert_go1 = "data/shivam/Bessel_140_250/1348DR/Expert/GO"
    plot_expert_go1 = "Bessel_140_250_1348DR/Expert_GO"

    naive_nogo1 = "data/shivam/Bessel_140_250/1348DR/Naive/NOGO"
    plot_naive_nogo1 = "Bessel_140_250_1348DR/Naive_NOGO"

    intermediate_nogo1 = "data/shivam/Bessel_140_250/1348DR/Intermediate/NOGO"
    plot_intermediate_nogo1 = "Bessel_140_250_1348DR/Intermediate_NOGO"

    expert_nogo1 = "data/shivam/Bessel_140_250/1348DR/Expert/NOGO"
    plot_expert_nogo1 = "Bessel_140_250_1348DR/Expert_NOGO"


    naive_go2 = "data/shivam/Bessel_140_250/073593DR/Naive/GO"
    plot_naive_go2 = "Bessel_140_250_073593DR/Naive_GO"

    intermediate_go2 = "data/shivam/Bessel_140_250/073593DR/Intermediate/GO"
    plot_intermediate_go2 = "Bessel_140_250_073593DR/Intermediate_GO"

    expert_go2 = "data/shivam/Bessel_140_250/073593DR/Expert/GO"
    plot_expert_go2 = "Bessel_140_250_073593DR/Expert_GO"

    naive_nogo2 = "data/shivam/Bessel_140_250/073593DR/Naive/NOGO"
    plot_naive_nogo2 = "Bessel_140_250_073593DR/Naive_NOGO"

    intermediate_nogo2 = "data/shivam/Bessel_140_250/073593DR/Intermediate/NOGO"
    plot_intermediate_nogo2 = "Bessel_140_250_073593DR/Intermediate_NOGO"

    expert_nogo2 = "data/shivam/Bessel_140_250/073593DR/Expert/NOGO"
    plot_expert_nogo2 = "Bessel_140_250_073593DR/Expert_NOGO"



    naive_go3 = "data/shivam/Bessel_250_340/44284DR/Naive/GO"
    plot_naive_go3 = "Bessel_250_340_44284DR/Naive_GO"

    intermediate_go3 = "data/shivam/Bessel_250_340/44284DR/Intermediate/GO"
    plot_intermediate_go3 = "Bessel_250_340_44284DR/Intermediate_GO"

    expert_go3 = "data/shivam/Bessel_250_340/44284DR/Expert/GO"
    plot_expert_go3 = "Bessel_250_340_44284DR/Expert_GO"

    naive_nogo3 = "data/shivam/Bessel_250_340/44284DR/Naive/NOGO"
    plot_naive_nogo3 = "Bessel_250_340_44284DR/Naive_NOGO"

    intermediate_nogo3 = "data/shivam/Bessel_250_340/44284DR/Intermediate/NOGO"
    plot_intermediate_nogo3 = "Bessel_250_340_44284DR/Intermediate_NOGO"

    expert_nogo3 = "data/shivam/Bessel_250_340/44284DR/Expert/NOGO"
    plot_expert_nogo3 = "Bessel_250_340_44284DR/Expert_NOGO"

    # run_rslds_pipeline(naive_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_naive_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_intermediate_go1,  type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_expert_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
   
    run_rslds_statespecificplots(state_idx=[0,1,2,3], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_naive_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    run_rslds_statespecificplots(state_idx=[0,1,2,3], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_intermediate_go1,  type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    run_rslds_statespecificplots(state_idx=[0,1,2,3], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_expert_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", num_iters=50)


    # run_rslds_pipeline(naive_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_naive_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_intermediate_go1,  type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_expert_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
   
    # run_rslds_pipeline(naive_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_naive_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_intermediate_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_expert_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    

    # run_rslds_statespecificplots(state_idx=[3,2], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_naive_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_statespecificplots(state_idx=[2,1,0], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_intermediate_go1,  type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_statespecificplots(state_idx=[1,0,2], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_expert_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)

    # run_rslds_statespecificplots(state_idx=[1,2], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_naive_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_statespecificplots(state_idx=[1,2], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_intermediate_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_statespecificplots(state_idx=[2,0], disc_states=disc_states, latent_dims=latent_dims, plot_key=plot_expert_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)


    # SET 1

    # # full trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=5)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=5)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)

    # go trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="go", state_idx=[3,2], l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, trial_selection="go", state_idx=[2,1,0], l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="go", state_idx=[1,0,2], l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)

    # nogo trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="nogo", state_idx=[1,2], l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1, type=DataType.L23, trial_selection="nogo", state_idx=[1,2], l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="nogo", state_idx=[2,0], l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", num_iters=50)


    # # SET 2

    # # full trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=5)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2,  type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=5)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2,  type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)

    # # SET 3

    # # full trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=5)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3,  type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=5)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3,  type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, trial_selection="go", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", trial_structure="trial_averaged", state_idx=1, num_iters=50)




    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    
    # naive_go4 = "data/shivam/ETL_140_250/075402/Naive/GO"
    # plot_naive_go4 = "ETL_140_250_075402/Naive_GO"

    # intermediate_go4 = "data/shivam/ETL_140_250/075402/Intermediate/GO"
    # plot_intermediate_go4 = "ETL_140_250_075402/Intermediate_GO"

    # expert_go4 = "data/shivam/ETL_140_250/075402/Expert/GO"
    # plot_expert_go4 = "ETL_140_250_075402/Expert_GO"

    # naive_nogo4 = "data/shivam/ETL_140_250/075402/Naive/NOGO"
    # plot_naive_nogo4 = "ETL_140_250_075402/Naive_NOGO"

    # intermediate_nogo4 = "data/shivam/ETL_140_250/075402/Intermediate/NOGO"
    # plot_intermediate_nogo4 = "ETL_140_250_075402/Intermediate_NOGO"

    # expert_nogo4 = "data/shivam/ETL_140_250/075402/Expert/NOGO"
    # plot_expert_nogo4 = "ETL_140_250_075402/Expert_NOGO"



    # # SET 1

    # full trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", trial_idx=20, state_idx=1, num_iters=5)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, l23_type="bessel", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", state_idx=1, trial_idx=20, num_iters=5)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, l23_type="bessel", state_idx=1, trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, l23_type="bessel", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, l23_type="bessel", trial_idx=20, state_idx=1, num_iters=50)


    # # go trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, l23_type="bessel", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, l23_type="bessel", trial_idx=20, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, l23_type="bessel", trial_structure="trial_averaged", trial_selection="go", trial_idx=20, state_idx=1, num_iters=5)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, l23_type="bessel", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, l23_type="bessel", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, l23_type="bessel", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, l23_type="bessel", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1, type=DataType.L23, l23_type="bessel", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, l23_type="bessel", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, l23_type="bessel", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, l23_type="bessel", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, l23_type="bessel", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)



    # # L2
    
    # # full set
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, state_idx=1, num_iters=5)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", trial_idx=20, layer="L2", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", trial_idx=20, layer="L2", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, state_idx=1, num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)


    # # L3

    # # full set
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, state_idx=1, num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, state_idx=1, num_iters=50)




    # ---------------------- OLD RUN -------------------------

    # naive_go1 = "data/shivam/Bessel_140_250/1348DR/Naive/GO"
    # plot_naive_go1 = "Bessel_140_250_1348DR/Naive_GO"

    # intermediate_go1 = "data/shivam/Bessel_140_250/1348DR/Intermediate/GO"
    # plot_intermediate_go1 = "Bessel_140_250_1348DR/Intermediate_GO"

    # expert_go1 = "data/shivam/Bessel_140_250/1348DR/Expert/GO"
    # plot_expert_go1 = "Bessel_140_250_1348DR/Expert_GO"

    # naive_nogo1 = "data/shivam/Bessel_140_250/1348DR/Naive/NOGO"
    # plot_naive_nogo1 = "Bessel_140_250_1348DR/Naive_NOGO"

    # intermediate_nogo1 = "data/shivam/Bessel_140_250/1348DR/Intermediate/NOGO"
    # plot_intermediate_nogo1 = "Bessel_140_250_1348DR/Intermediate_NOGO"

    # expert_nogo1 = "data/shivam/Bessel_140_250/1348DR/Expert/NOGO"
    # plot_expert_nogo1 = "Bessel_140_250_1348DR/Expert_NOGO"



    # naive_go2 = "data/shivam/Bessel_140_250/073593DR/Naive/GO"
    # plot_naive_go2 = "Bessel_140_250_073593DR/Naive_GO"

    # intermediate_go2 = "data/shivam/Bessel_140_250/073593DR/Intermediate/GO"
    # plot_intermediate_go2 = "Bessel_140_250_073593DR/Intermediate_GO"

    # expert_go2 = "data/shivam/Bessel_140_250/073593DR/Expert/GO"
    # plot_expert_go2 = "Bessel_140_250_073593DR/Expert_GO"

    # naive_nogo2 = "data/shivam/Bessel_140_250/073593DR/Naive/NOGO"
    # plot_naive_nogo2 = "Bessel_140_250_073593DR/Naive_NOGO"

    # intermediate_nogo2 = "data/shivam/Bessel_140_250/073593DR/Intermediate/NOGO"
    # plot_intermediate_nogo2 = "Bessel_140_250_073593DR/Intermediate_NOGO"

    # expert_nogo2 = "data/shivam/Bessel_140_250/073593DR/Expert/NOGO"
    # plot_expert_nogo2 = "Bessel_140_250_073593DR/Expert_NOGO"



    # naive_go3 = "data/shivam/Bessel_250_340/44284DR/Naive/GO"
    # plot_naive_go3 = "Bessel_250_340_44284DR/Naive_GO"

    # intermediate_go3 = "data/shivam/Bessel_250_340/44284DR/Intermediate/GO"
    # plot_intermediate_go3 = "Bessel_250_340_44284DR/Intermediate_GO"

    # expert_go3 = "data/shivam/Bessel_250_340/44284DR/Expert/GO"
    # plot_expert_go3 = "Bessel_250_340_44284DR/Expert_GO"

    # naive_nogo3 = "data/shivam/Bessel_250_340/44284DR/Naive/NOGO"
    # plot_naive_nogo3 = "Bessel_250_340_44284DR/Naive_NOGO"

    # intermediate_nogo3 = "data/shivam/Bessel_250_340/44284DR/Intermediate/NOGO"
    # plot_intermediate_nogo3 = "Bessel_250_340_44284DR/Intermediate_NOGO"

    # expert_nogo3 = "data/shivam/Bessel_250_340/44284DR/Expert/NOGO"
    # plot_expert_nogo3 = "Bessel_250_340_44284DR/Expert_NOGO"



    # naive_go4 = "data/shivam/ETL_140_250/075402/Naive/GO"
    # plot_naive_go4 = "ETL_140_250_075402/Naive_GO"

    # intermediate_go4 = "data/shivam/ETL_140_250/075402/Intermediate/GO"
    # plot_intermediate_go4 = "ETL_140_250_075402/Intermediate_GO"

    # expert_go4 = "data/shivam/ETL_140_250/075402/Expert/GO"
    # plot_expert_go4 = "ETL_140_250_075402/Expert_GO"

    # naive_nogo4 = "data/shivam/ETL_140_250/075402/Naive/NOGO"
    # plot_naive_nogo4 = "ETL_140_250_075402/Naive_NOGO"

    # intermediate_nogo4 = "data/shivam/ETL_140_250/075402/Intermediate/NOGO"
    # plot_intermediate_nogo4 = "ETL_140_250_075402/Intermediate_NOGO"

    # expert_nogo4 = "data/shivam/ETL_140_250/075402/Expert/NOGO"
    # plot_expert_nogo4 = "ETL_140_250_075402/Expert_NOGO"
     
    # # SET 1

    # # go trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)


    # # SET 2

    # # go trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, trial_selection="go", type=DataType.L23, state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)


    # # SET 3

    # # go trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)


    # # SET 4
    # # go trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, trial_selection="go", state_idx=1, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, trial_selection="nogo", state_idx=1, num_iters=50)

    # -------------
