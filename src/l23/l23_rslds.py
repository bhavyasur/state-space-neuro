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



    naive_go5 = "data/shivam/Bessel_140_250/092004DR/Naive/GO"
    plot_naive_go5 = "Bessel_140_250_092004DR/Naive_GO"

    intermediate_go5 = "data/shivam/Bessel_140_250/092004DR/Intermediate/GO"
    plot_intermediate_go5 = "Bessel_140_250_092004DR/Intermediate_GO"

    expert_go5 = "data/shivam/Bessel_140_250/092004DR/Expert/GO"
    plot_expert_go5 = "Bessel_140_250_092004DR/Expert_GO"

    naive_nogo5 = "data/shivam/Bessel_140_250/092004DR/Naive/NOGO"
    plot_naive_nogo5 = "Bessel_140_250_092004DR/Naive_NOGO"

    intermediate_nogo5 = "data/shivam/Bessel_140_250/092004DR/Intermediate/NOGO"
    plot_intermediate_nogo5 = "Bessel_140_250_092004DR/Intermediate_NOGO"

    expert_nogo5 = "data/shivam/Bessel_140_250/092004DR/Expert/NOGO"
    plot_expert_nogo5 = "Bessel_140_250_092004DR/Expert_NOGO"


    naive_go6 = "data/shivam/Bessel_250_340/0730DR/Naive/GO"
    plot_naive_go6 = "Bessel_250_340_0730DR/Naive_GO"

    intermediate_go6 = "data/shivam/Bessel_250_340/0730DR/Intermediate/GO"
    plot_intermediate_go6 = "Bessel_250_340_0730DR/Intermediate_GO"

    expert_go6 = "data/shivam/Bessel_250_340/0730DR/Expert/GO"
    plot_expert_go6 = "Bessel_250_340_0730DR/Expert_GO"

    naive_nogo6 = "data/shivam/Bessel_250_340/0730DR/Naive/NOGO"
    plot_naive_nogo6 = "Bessel_250_340_0730DR/Naive_NOGO"

    intermediate_nogo6 = "data/shivam/Bessel_250_340/0730DR/Intermediate/NOGO"
    plot_intermediate_nogo6 = "Bessel_250_340_0730DR/Intermediate_NOGO"

    expert_nogo6 = "data/shivam/Bessel_250_340/0730DR/Expert/NOGO"
    plot_expert_nogo6 = "Bessel_250_340_0730DR/Expert_NOGO"

    gauss_l2_in = "data/shivam/Gaussian_Control/L2/In"
    plot_gauss_l2_in = "Gaussian_Control/L2_In"

    gauss_l3_in = "data/shivam/Gaussian_Control/L3/In"
    plot_gauss_l3_in = "Gaussian_Control/L3_In"

    gauss_l2_out = "data/shivam/Gaussian_Control/L2/Out"
    plot_gauss_l2_out = "Gaussian_Control/L2_Out"

    gauss_l3_out = "data/shivam/Gaussian_Control/L3/Out"
    plot_gauss_l3_out = "Gaussian_Control/L3_Out"



    # # GAUSSIAN
    
    # # full trials

    # run_rslds_pipeline(gauss_l2_in, disc_states, latent_dims, plot_gauss_l2_in, type=DataType.L23, l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l3_in, disc_states, latent_dims, plot_gauss_l3_in,  type=DataType.L23, l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l2_out, disc_states, latent_dims, plot_gauss_l2_out, type=DataType.L23, l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l3_out, disc_states, latent_dims, plot_gauss_l3_out, type=DataType.L23, l23_type="bessel", num_iters=50)

    # run_rslds_pipeline(gauss_l2_in, disc_states, latent_dims, plot_gauss_l2_in, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l3_in, disc_states, latent_dims, plot_gauss_l3_in,  type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l2_out, disc_states, latent_dims, plot_gauss_l2_out, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l3_out, disc_states, latent_dims, plot_gauss_l3_out, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)

    # # go trials

    # run_rslds_pipeline(gauss_l2_in, disc_states, latent_dims, plot_gauss_l2_in, type=DataType.L23, trial_selection="go",  l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l3_in, disc_states, latent_dims, plot_gauss_l3_in,  type=DataType.L23, trial_selection="go",  l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l2_out, disc_states, latent_dims, plot_gauss_l2_out, type=DataType.L23, trial_selection="go",  l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l3_out, disc_states, latent_dims, plot_gauss_l3_out, type=DataType.L23, trial_selection="go",  l23_type="bessel", num_iters=50)

    # run_rslds_pipeline(gauss_l2_in, disc_states, latent_dims, plot_gauss_l2_in, type=DataType.L23, trial_selection="go",  l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l3_in, disc_states, latent_dims, plot_gauss_l3_in,  type=DataType.L23, trial_selection="go",  l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l2_out, disc_states, latent_dims, plot_gauss_l2_out, type=DataType.L23, trial_selection="go",  l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l3_out, disc_states, latent_dims, plot_gauss_l3_out, type=DataType.L23, trial_selection="go",  l23_type="bessel", plot_type='svg', num_iters=50)

    # # nogo trials

    # run_rslds_pipeline(gauss_l2_in, disc_states, latent_dims, plot_gauss_l2_in, type=DataType.L23, trial_selection="nogo",  l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l3_in, disc_states, latent_dims, plot_gauss_l3_in,  type=DataType.L23, trial_selection="nogo",  l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l2_out, disc_states, latent_dims, plot_gauss_l2_out, type=DataType.L23, trial_selection="nogo",  l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(gauss_l3_out, disc_states, latent_dims, plot_gauss_l3_out, type=DataType.L23, trial_selection="nogo",  l23_type="bessel", num_iters=50)

    # run_rslds_pipeline(gauss_l2_in, disc_states, latent_dims, plot_gauss_l2_in, type=DataType.L23, trial_selection="nogo",  l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l3_in, disc_states, latent_dims, plot_gauss_l3_in,  type=DataType.L23, trial_selection="nogo",  l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l2_out, disc_states, latent_dims, plot_gauss_l2_out, type=DataType.L23, trial_selection="nogo",  l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(gauss_l3_out, disc_states, latent_dims, plot_gauss_l3_out, type=DataType.L23, trial_selection="nogo",  l23_type="bessel", plot_type='svg', num_iters=50)


    

    # # SET 1 BINNED

    # full trials
    run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, l23_type="bessel", bin_size=2, num_iters=50)

    # go trials
    run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, trial_selection="go", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", bin_size=2, num_iters=50)

    # nogo trials
    run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", bin_size=2, num_iters=50)

    # full trials
    run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)

    # go trials
    run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1,  type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)

    # nogo trials
    run_rslds_pipeline(naive_go1, disc_states, latent_dims, plot_naive_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_go1, disc_states, latent_dims, plot_intermediate_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_go1, disc_states, latent_dims, plot_expert_go1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(naive_nogo1, disc_states, latent_dims, plot_naive_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(intermediate_nogo1, disc_states, latent_dims, plot_intermediate_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)
    run_rslds_pipeline(expert_nogo1, disc_states, latent_dims, plot_expert_nogo1, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', bin_size=2, num_iters=50)


    # # SET 2

    # # full trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2,  type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2,  type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go2, disc_states, latent_dims, plot_naive_go2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go2, disc_states, latent_dims, plot_intermediate_go2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go2, disc_states, latent_dims, plot_expert_go2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo2, disc_states, latent_dims, plot_naive_nogo2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo2, disc_states, latent_dims, plot_intermediate_nogo2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo2, disc_states, latent_dims, plot_expert_nogo2, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)

    # # SET 3

    # # full trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3,  type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3,  type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go3, disc_states, latent_dims, plot_naive_go3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go3, disc_states, latent_dims, plot_intermediate_go3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go3, disc_states, latent_dims, plot_expert_go3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo3, disc_states, latent_dims, plot_naive_nogo3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo3, disc_states, latent_dims, plot_intermediate_nogo3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo3, disc_states, latent_dims, plot_expert_nogo3, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)

    # # SET 5

    # # full trials
    # run_rslds_pipeline(naive_go5, disc_states, latent_dims, plot_naive_go5, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go5, disc_states, latent_dims, plot_intermediate_go5,  type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go5, disc_states, latent_dims, plot_expert_go5, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo5, disc_states, latent_dims, plot_naive_nogo5, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo5, disc_states, latent_dims, plot_intermediate_nogo5, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo5, disc_states, latent_dims, plot_expert_nogo5, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go5, disc_states, latent_dims, plot_naive_go5, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go5, disc_states, latent_dims, plot_intermediate_go5,  type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go5, disc_states, latent_dims, plot_expert_go5, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo5, disc_states, latent_dims, plot_naive_nogo5, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo5, disc_states, latent_dims, plot_intermediate_nogo5, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo5, disc_states, latent_dims, plot_expert_nogo5, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go5, disc_states, latent_dims, plot_naive_go5, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go5, disc_states, latent_dims, plot_intermediate_go5, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go5, disc_states, latent_dims, plot_expert_go5, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo5, disc_states, latent_dims, plot_naive_nogo5, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo5, disc_states, latent_dims, plot_intermediate_nogo5, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo5, disc_states, latent_dims, plot_expert_nogo5, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)

    # # SET 6

    # # full trials
    # run_rslds_pipeline(naive_go6, disc_states, latent_dims, plot_naive_go6, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go6, disc_states, latent_dims, plot_intermediate_go6,  type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go6, disc_states, latent_dims, plot_expert_go6, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo6, disc_states, latent_dims, plot_naive_nogo6, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo6, disc_states, latent_dims, plot_intermediate_nogo6, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo6, disc_states, latent_dims, plot_expert_nogo6, type=DataType.L23, l23_type="bessel", plot_type='svg', num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go6, disc_states, latent_dims, plot_naive_go6, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go6, disc_states, latent_dims, plot_intermediate_go6,  type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go6, disc_states, latent_dims, plot_expert_go6, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo6, disc_states, latent_dims, plot_naive_nogo6, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo6, disc_states, latent_dims, plot_intermediate_nogo6, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo6, disc_states, latent_dims, plot_expert_nogo6, type=DataType.L23, trial_selection="go", l23_type="bessel", plot_type='svg', num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go6, disc_states, latent_dims, plot_naive_go6, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_go6, disc_states, latent_dims, plot_intermediate_go6, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_go6, disc_states, latent_dims, plot_expert_go6, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(naive_nogo6, disc_states, latent_dims, plot_naive_nogo6, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(intermediate_nogo6, disc_states, latent_dims, plot_intermediate_nogo6, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)
    # run_rslds_pipeline(expert_nogo6, disc_states, latent_dims, plot_expert_nogo6, type=DataType.L23, trial_selection="nogo", l23_type="bessel", plot_type='svg', num_iters=50)


    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    
    """ ETL DATASET """

    # # L2
    
    # # full set
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", trial_idx=20, layer="L2", num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", trial_idx=20, layer="L2", num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_idx=20, trial_selection="go", num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="go", trial_idx=20, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L2", trial_selection="nogo", trial_idx=20, num_iters=50)


    # # L3

    # # full set
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_idx=20, num_iters=50)

    # # go trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="go", trial_idx=20, num_iters=50)

    # # nogo trials
    # run_rslds_pipeline(naive_go4, disc_states, latent_dims, plot_naive_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_go4, disc_states, latent_dims, plot_intermediate_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_go4, disc_states, latent_dims, plot_expert_go4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(naive_nogo4, disc_states, latent_dims, plot_naive_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(intermediate_nogo4, disc_states, latent_dims, plot_intermediate_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, num_iters=50)
    # run_rslds_pipeline(expert_nogo4, disc_states, latent_dims, plot_expert_nogo4, type=DataType.L23, l23_type="etl", layer="L3", trial_selection="nogo", trial_idx=20, num_iters=50)


    # -------------

    """ IN & OUT DATASETS """

    # naive_in = "data/shivam/Bessel_140_250/1348DR/Naive_to_expert/Naive/In"
    # plot_naive_in = "Bessel_140_250_1348DR/Naive_IN"
    # expert_in = "data/shivam/Bessel_140_250/1348DR/Naive_to_expert/Operant/In"
    # plot_expert_in = "Bessel_140_250_1348DR/Expert_IN"
    # naive_out = "data/shivam/Bessel_140_250/1348DR/Naive_to_expert/Naive/Out"
    # plot_naive_out = "Bessel_140_250_1348DR/Naive_OUT"
    # expert_out = "data/shivam/Bessel_140_250/1348DR/Naive_to_expert/Operant/Out"
    # plot_expert_out = "Bessel_140_250_1348DR/Expert_OUT"

    # plot_in = "Bessel_140_250_1348DR/naive_expert_concat_IN"
    # plot_out = "Bessel_140_250_1348DR/naive_expert_concat_OUT"

    # # CONCATENATED IN BARREL

    # run_rslds_pipeline([naive_in, expert_in], disc_states, latent_dims, plot_key=plot_in, type=DataType.L23, l23_type="bessel", specific_loadtype="session_concat", trial_idx=20, num_iters=50)
    # run_rslds_pipeline([naive_in, expert_in], disc_states, latent_dims, plot_key=plot_in, type=DataType.L23, trial_selection="go", l23_type="bessel", specific_loadtype="session_concat", num_iters=50)
    # run_rslds_pipeline([naive_in, expert_in], disc_states, latent_dims, plot_key=plot_in, type=DataType.L23, trial_selection="nogo", l23_type="bessel", specific_loadtype="session_concat", num_iters=50)


    # # CONCATENATED OUT BARREL
    # # run_rslds_pipeline([naive_out, expert_out], disc_states, latent_dims, plot_key=plot_out, type=DataType.L23, l23_type="bessel", specific_loadtype="session_concat", num_iters=50)
    # run_rslds_pipeline([naive_out, expert_out], disc_states, latent_dims, plot_key=plot_out, type=DataType.L23, trial_selection="go", l23_type="bessel", specific_loadtype="session_concat", num_iters=50)
    # run_rslds_pipeline([naive_out, expert_out], disc_states, latent_dims, plot_key=plot_out, type=DataType.L23, trial_selection="nogo", l23_type="bessel", specific_loadtype="session_concat", num_iters=50)


    # # IN BARREL, NAIVE ONLY
    # # run_rslds_pipeline(naive_in, disc_states, latent_dims, plot_key=plot_naive_in, type=DataType.L23, l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(naive_in, disc_states, latent_dims, plot_key=plot_naive_in, type=DataType.L23, trial_selection="go", l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(naive_in, disc_states, latent_dims, plot_key=plot_naive_in, type=DataType.L23, trial_selection="nogo", l23_type="bessel", num_iters=50)

    # # IN BARREL, EXPERT ONLY
    # # run_rslds_pipeline(expert_in, disc_states, latent_dims, plot_key=plot_expert_in, type=DataType.L23, l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(expert_in, disc_states, latent_dims, plot_key=plot_expert_in, type=DataType.L23, trial_selection="go", l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(expert_in, disc_states, latent_dims, plot_key=plot_expert_in, type=DataType.L23, trial_selection="nogo", l23_type="bessel", num_iters=50)

    # # OUT BARREL, NAIVE ONLY
    # # run_rslds_pipeline(naive_out, disc_states, latent_dims, plot_key=plot_naive_out, type=DataType.L23, l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(naive_out, disc_states, latent_dims, plot_key=plot_naive_out, type=DataType.L23, trial_selection="go", l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(naive_out, disc_states, latent_dims, plot_key=plot_naive_out, type=DataType.L23, trial_selection="nogo", l23_type="bessel", num_iters=50)

    # # OUT BARREL, EXPERT ONLY
    # # run_rslds_pipeline(expert_out, disc_states, latent_dims, plot_key=plot_expert_out, type=DataType.L23, l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(expert_out, disc_states, latent_dims, plot_key=plot_expert_out, type=DataType.L23, trial_selection="go", l23_type="bessel", num_iters=50)
    # run_rslds_pipeline(expert_out, disc_states, latent_dims, plot_key=plot_expert_out, type=DataType.L23, trial_selection="nogo", l23_type="bessel", num_iters=50)

