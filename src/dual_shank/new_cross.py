import autograd.numpy as np
import pandas as pd
import ssm

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, full_session
from utils.rslds_util import bin_smooth
from src.dual_shank.clean_rslds_ds import run_rslds, run_rslds_binned

from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA

def cross_val(raw_data, heldout_frac=0.1, n_repeats=3, var_threshold = 0.9, **kwargs):
    """ datas is raw .mat file
    """
    spks = load_spikes(raw_data)
    full = full_session(spks)
    binned = bin_smooth(full.T).astype(int)
    print("NEW BINNED SHAPE", np.shape(binned))
    # BINNED is (num_timesteps, num_neurons)

    num_neurons = full.shape[0]
    obs = num_neurons

    param_grid = {
        'dims': [2,3,4,5,6,7,8,9,10],
        'states': [1,2,3,4,5]
    }

    # rslds on all hyperparameter combinations
    for dim in param_grid['dims']:
        for state in param_grid['states']:

            kfold = KFold()
            cv_scores = []

            for fold, (train_idx, test_idx) in enumerate(kfold.split(binned)):
                train = list(binned[train_idx])
                print("train shape", np.shape(train))
                test = [binned[test_idx]]
                print("test shape", np.shape(train))

                model, b, c, d, e = run_rslds_binned(train, state, dim)

                predictions = []
                true_obs = []

                for trial in test:
                    elbo, q_x = model.approximate_posterior([trial], method="laplace_em")
                    x_mean = q_x.mean[0]

                    z_hat = model.most_likely_states(x_mean, trial) 

                    # Project latents back to raw observation space
                    Cs = model.emissions.Cs
                    ds = model.emissions.ds
                    y_pred = np.array([Cs[z] @ x_mean[t] + ds[z] for t, z in enumerate(z_hat)])
    
                    predictions.append(y_pred)
                    true_obs.append(trial)
                
                # 5. Calculate R^2 for this specific fold
                y_true_fold = np.vstack(true_obs)
                y_pred_fold = np.vstack(predictions)
                
                rss = np.sum((y_true_fold - y_pred_fold) ** 2)
                tss = np.sum((y_true_fold - np.mean(y_true_fold, axis=0)) ** 2)
                fold_r2 = 1 - (rss / tss)
                
                cv_scores.append(fold_r2)
                print(f"Fold {fold+1} Variance Explained: {fold_r2 * 100:.2f}%")

            # Overall Performance
            print(f"\nParameters: dim={dim}, states={state}")
            if np.mean(cv_scores) >= var_threshold:
                print(f"\nvariance threshold of {var_threshold} satisfied! mean CV variance explained: {np.mean(cv_scores) * 100:.2f}% ± {np.std(cv_scores) * 100:.2f}%")
                break
            else:
                print(f"\nmean CV variance explained: {np.mean(cv_scores) * 100:.2f}% ± {np.std(cv_scores) * 100:.2f}%")
        

#     # find first combination that explains 90% of variance and return
# def plot_with_pca():
#     elbos, posterior = model.approximate_posterior(binned, method="laplace_em")
#     latent_states = posterior.mean
#     pca= PCA(n=2)

if __name__ == "__main__":
    raw_data = "data/om/07538_M1_Day1_CCA_data.mat"

    # spks = load_spikes(raw_data)
    # full = full_session(spks) #
    # binned = bin_smooth(full.T).astype(int)

    cross_val(raw_data)
