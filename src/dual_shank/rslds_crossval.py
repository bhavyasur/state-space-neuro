
import ssm
from ssm.util import ensure_args_are_lists
from ssm import SLDS
from ssm.util import random_rotation, find_permutation
from ssm.model_selection import cross_val_scores
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import copy

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, full_session
from utils.rslds_util import bin_smooth
    
@ensure_args_are_lists
def rslds_cross_val(
        model, datas, inputs=None, masks=None, tags=None,
        heldout_frac=0.1, n_repeats=3, **fit_kw):
    """
    Evaluate ELBO scores on heldout data using
    a speckled holdout pattern.

    Parameters
    ----------
    model : rSLDS
        Model instance.
    datas : ndarray or list of ndarray
        Data matrices with shape (n_obs, n_out).
    inputs : ndarray
        If applicable, matrix holding input variables with shape
        (n_obs, n_in).
    masks : ndarray
        If applicable, binary matrix specifying censored or
        unobserved entries in data. Entries of one correspond
        to observed data. Has shape (n_obs, n_out).
    tags : ???
        ???
    heldout_frac : float
        Number between zero and one specifying how much data
        to hold out on each cross-validation run.
    n_repeats : int
        Number of randomized cross-validation runs to perform.
    **fit_kw : dict
        Additional keyword arguments are passed to model.fit(...)

    Returns
    -------
    test_scores : ndarray
        Array holding normalized log-likelihood scores on test
        set. Has shape (n_repeats,).
    train_scores : ndarray
        Array holding log-likelihood scores on training set.
        Has shape (n_repeats,).
    """

    # Allocate space for train and test log-likelihoods.
    test_scores = np.empty(n_repeats)
    train_scores = np.empty(n_repeats)

    for r in range(n_repeats):

        # Create mask for training data.
        train_masks = []
        test_masks = []
        for m in masks:

            # Determine number of heldout points.
            total_obs = np.sum(m)
            obs_inds = np.argwhere(m)
            heldout_num = int(total_obs * heldout_frac)

            # Randomly hold out speckled data pattern.
            heldout_flat_inds = npr.choice(
                total_obs, heldout_num, replace=False)

            i, j = obs_inds[heldout_flat_inds].T

            train_m = m.copy()
            train_m[i,j] = False
            train_masks.append(train_m)

            test_m = np.zeros_like(m, dtype=bool)
            test_m[i,j] = True
            test_masks.append(test_m)
        
        model_r = copy.deepcopy(model)

        elbos, posterior = model_r.fit(
            datas, inputs=inputs, tags=tags, masks=train_masks, **fit_kw

        )
        # Fit model with training mask.
        train_elbos, train_posterior = model.fit(datas, inputs=inputs, tags=tags, masks=train_masks, **fit_kw)

        full_elbos, full_posterior = model.fit(datas, inputs=inputs, tags=tags, masks=masks, **fit_kw)

        print("train_elbos", np.shape(train_elbos))
        print("train_posterior", np.shape(train_posterior))
        print("full_elbos", np.shape(full_elbos))
        print("full_posterior", np.shape(full_posterior))

        # posterior is found with variational inference, maximizing ELBO means that we are approximating true posterior.

        # Save normalized log-likelihood scores.
        test_scores[r:,] = full_elbos - train_elbos
        train_scores[r:,] = train_elbos

        

        

    return test_scores, train_scores

def eigenvalues(array):
    eigs, eigvectors = np.linalg.eig(array)
    return eigs, eigvectors


def hyperparam_tuning():
    return


if __name__ == "__main__":

    # ---------- CROSS VALIDATION ------------- #

    data = "data/om/07538_M1_Day1_CCA_data.mat"

    spks = load_spikes(data)
    full = full_session(spks)
    data = bin_smooth(full.T).astype(int) # (num_neurons, num_trials * num_timesteps_per_trial)
    print("data shape", np.shape(data))

    """NOTE: cross_val_scores takes in datas as array or list of arrays 
    with (n_obs, n_out) = (n_neurons, n_timesteps) """

    num_neurons = np.shape(full)[0]
    num_obs = num_neurons
    disc_states = 4
    latent_dims = 2
    rslds = ssm.SLDS(num_obs, disc_states, latent_dims,
                    transitions="recurrent_only",
                    emissions="poisson_orthog",
                    emission_kwargs=dict(link="softplus"))

    test_scores, train_scores = rslds_cross_val(rslds, data, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False, num_iters=50)
    
    print("test scores of cross validation is:", str(test_scores))
    print("train scores of cross validation is:", str(train_scores))


    # --------------- HYPERPARAM TUNING ----------------
    
    disc_states_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # search space for # states hyperparameter
