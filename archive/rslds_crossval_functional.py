import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssm
import sys
from pathlib import Path
ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))
from src.dual_shank.ds_load_util import load_spikes, full_session
from src.rslds.rslds_util import DataType, bin_smooth, run_rslds_concat, plot_cv_heatmap, softplus
from src.rbp_cre.rbp_load_util import load_dfoverf, full_session_rbp
from sklearn.model_selection import KFold


def cross_val(raw_data, type: DataType, heldout_frac=0.1, n_repeats=3, var_threshold=0.8, **kwargs):
    """ datas is raw .mat file
    """
    if type is DataType.EPhys:
        spks = load_spikes(raw_data)
        full = full_session(spks)
        data = bin_smooth(full.T).astype(int)
        print("shape of data, ephys", np.shape(data))
    elif type is DataType.Calcium:
        dfoverf = load_dfoverf(raw_data)
        full = full_session_rbp(dfoverf)
        data = full.T
        print("shape of data, calcium", np.shape(data))
    else:
        raise ValueError(f"Unsupported DataType: {type}")

    num_neurons = np.shape(data)[1]
    obs = num_neurons
    param_grid = {
        'dims': [6, 7, 8, 9, 10],
        'states': [2, 3, 4, 5, 6]
    }

    results = {}  # (dim, state) -> mean_cv_score (fraction)
    found_threshold = False

    # rslds on all hyperparameter combinations
    for dim in param_grid['dims']:
        if found_threshold:
            break
        for state in param_grid['states']:
            kfold = KFold()
            cv_scores = []
            for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
                train = list(data[train_idx])
                print("train shape", np.shape(train))
                test = [data[test_idx]]
                print("test shape", np.shape(train))
                model, b, c, d, e = run_rslds_concat(train, state, dim)
                predictions = []
                true_obs = []
                for trial in test:
                    elbo, q_x = model.approximate_posterior([trial], method="laplace_em")
                    x_mean = q_x.mean_continuous_states[0]
                    # project latents back to raw observation space
                    Cs = model.emissions.Cs
                    ds = model.emissions.ds
                    z_hat = model.most_likely_states(x_mean, trial)
                    num_states = Cs.shape[0]
                    z_hat = np.clip(z_hat, 0, num_states - 1)
                    y_pred = np.array([softplus(Cs[z] @ x_mean[t] + ds[z]) for t, z in enumerate(z_hat)])
                    print("Cs shape:", np.shape(Cs), "ds shape:", np.shape(ds))
                    print("z_hat unique:", np.unique(z_hat))
                    print("y_pred range:", y_pred.min(), y_pred.max())
                    print("y_true range:", trial.min(), trial.max())
                    predictions.append(y_pred)
                    true_obs.append(trial)

                #  R^2 for this fold
                y_true_fold = np.vstack(true_obs)
                y_pred_fold = np.vstack(predictions)
                rss = np.sum((y_true_fold - y_pred_fold) ** 2)
                tss = np.sum((y_true_fold - np.mean(y_true_fold, axis=0)) ** 2)
                fold_r2 = 1 - (rss / tss)
                cv_scores.append(fold_r2)
                print(f"Fold {fold+1} Variance Explained: {fold_r2 * 100:.2f}%")

            # Overall Performance
            print(f"\nParameters: dim={dim}, states={state}")
            mean_score = np.mean(cv_scores)
            results[(dim, state)] = mean_score
            if mean_score >= var_threshold:
                print(f"\nvariance threshold of {var_threshold} satisfied! mean CV variance explained: {mean_score * 100:.2f}% \u00b1 {np.std(cv_scores) * 100:.2f}%")
                found_threshold = True
                break
            else:
                print(f"\nmean CV variance explained: {mean_score * 100:.2f}% \u00b1 {np.std(cv_scores) * 100:.2f}%")

    plot_cv_heatmap(results, param_grid)
    return results


if __name__ == "__main__":
    raw_data = "data/om/87564_M1R_Day12DualShank_41c324t.mat"

    cross_val(raw_data, type=DataType.EPhys)