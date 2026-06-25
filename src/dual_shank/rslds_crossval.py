import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssm
import sys
from pathlib import Path
ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))
from src.dual_shank.ds_load_util import load_spikes, psth_firing, select_trial, visualize_trial, full_session
from rslds.rslds_util import bin_smooth
from src.dual_shank.clean_rslds_ds import run_rslds, run_rslds_binned
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA


def softplus(x):
    return np.log1p(np.exp(x))


def plot_cv_heatmap(results, param_grid):
    """
    results: dict mapping (dim, state) -> mean_cv_score (fraction, not percent)
    param_grid: dict with 'dims' and 'states' lists
    """
    dims = param_grid['dims']
    states = param_grid['states']

    # Build matrix: rows = states, cols = dims
    grid = np.full((len(states), len(dims)), np.nan)
    for (dim, state), score in results.items():
        i = states.index(state)
        j = dims.index(dim)
        grid[i, j] = score * 100  # convert to percentage

    fig, ax = plt.subplots(figsize=(10, 6))
    finite_vals = grid[~np.isnan(grid)]
    vmin = finite_vals.min() if finite_vals.size > 0 else 0
    vmax = finite_vals.max() if finite_vals.size > 0 else 1
    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels(states)
    ax.set_xlabel("Latent Dimensions")
    ax.set_ylabel("Discrete States")
    ax.set_title("Mean CV Variance Explained (%) by Hyperparameter Combination")

    # Annotate each cell with its value
    for i in range(len(states)):
        for j in range(len(dims)):
            val = grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color="black", fontsize=9)

    fig.colorbar(im, ax=ax, label="Variance Explained (%)")
    plt.tight_layout()
    plt.show()


def cross_val(full, heldout_frac=0.1, n_repeats=3, var_threshold=0.8, **kwargs):
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
        'dims': [6, 7, 8, 9, 10],
        'states': [2, 3, 4, 5]
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
                    x_mean = q_x.mean_continuous_states[0]
                    # Project latents back to raw observation space
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
            mean_score = np.mean(cv_scores)
            results[(dim, state)] = mean_score
            if mean_score >= var_threshold:
                print(f"\nvariance threshold of {var_threshold} satisfied! mean CV variance explained: {mean_score * 100:.2f}% \u00b1 {np.std(cv_scores) * 100:.2f}%")
                found_threshold = True
                break
            else:
                print(f"\nmean CV variance explained: {mean_score * 100:.2f}% \u00b1 {np.std(cv_scores) * 100:.2f}%")
    #     # find first combination that explains 90% of variance and return
    # def plot_with_pca():
    #     elbos, posterior = model.approximate_posterior(binned, method="laplace_em")
    #     latent_states = posterior.mean
    #     pca= PCA(n=2)

    plot_cv_heatmap(results, param_grid)
    return results


if __name__ == "__main__":
    raw_data = "data/om/87564_M1R_Day12DualShank_41c324t.mat"
    # spks = load_spikes(raw_data)
    # full = full_session(spks) #
    # binned = bin_smooth(full.T).astype(int)
    cross_val(raw_data)