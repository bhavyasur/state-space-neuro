import numpy as np
import matplotlib.pyplot as plt
import ssm

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from src.dual_shank.ds_load_util import load_spikes, full_session
from rslds.rslds_util import bin_smooth
from src.dual_shank.clean_rslds_ds import eigs_timeconstants

if __name__=='__main__':

    day6 = "data/om/075356_M1_Day6_CCA_data_74c141t.mat"
    fSave = "075356_M1_Day6_CCA_data"
    spikes = load_spikes(day6)
    full = full_session(spikes)
    num_neurons = np.shape(full)[0]
    num_obs = num_neurons
    
    # transposed_spikes = []
    # for i in range(len(spikes)):
    #     l = spikes[i]
    #     transposed_spikes.append(l.T)

    # print("transposed", np.shape(transposed_spikes))

    data = full.T
    disc_states = 4 # should depend on held-out cross validation
    latent_dims = 8
    
    y = bin_smooth(data).astype(int)

    slds = ssm.SLDS(num_obs, disc_states, latent_dims,
                      transitions="recurrent_only",
                      emissions="poisson_orthog",
                      emission_kwargs=dict(link="softplus"))
    
    slds.initialize(y, verbose=1)
    q_elbos_lem, q_lem = slds.fit(y, method="laplace_em",
                                    variational_posterior="structured_meanfield",
                                    initialize=False, num_iters=50)
    
    # %% Extract single neuron contributions from the emissions matrix

    eigs, vecs, tcs, max_idx = eigs_timeconstants(slds, 1, dim_idx='max')
    print("time constants", tcs)
    print("max_idx", max_idx)

    # C = abs(np.squeeze(slds.emissions.Cs))
    # sorted_indices_per_dim = [np.argsort(-C[:, d]) for d in range(C.shape[1])]
    # print("c shape", np.shape(C))

    # fig, axs = plt.subplots(1, 2, figsize=(8, 8), sharex=True, gridspec_kw={'wspace': 0.001})

    # # Plot binned spike counts
    # im1 = axs[0].imshow(C,
    #                 aspect='equal',
    #                 cmap='viridis',  # Scientific colormap that's colorblind-friendly
    #                 interpolation='none')  # Preserves exact values
    # axs[0].set_title("Emission matrix C", fontsize=14, fontweight='bold')
    # axs[0].set_ylabel("Neurons", fontsize=12)


    # # Add gridlines to help identify neuron positions
    # axs[0].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # im1 = axs[1].imshow(C[sorted_indices_per_dim[0],:],
    #                 aspect='equal',
    #                 cmap='viridis',  # Scientific colormap that's colorblind-friendly
    #                 interpolation='none')  # Preserves exact values
    # axs[1].set_title("Emission matrix C", fontsize=14, fontweight='bold')

    # # Add gridlines to help identify neuron positions
    # axs[0].grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

# Example emission matrix: C (replace with your actual data)
    C = abs(np.squeeze(slds.emissions.Cs))
    sorted_indices = np.argsort(-C[:, max_idx])  # Sort by contribution to the first dimension

    fig = plt.figure(figsize=(5, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 7], hspace=0.15)

    # Top left: neuron weights for the first dimension
    ax0 = fig.add_subplot(gs[0, 0])
    abs_weights = np.abs(C[:, max_idx])
    ax0.stem(np.arange(len(abs_weights)), abs_weights, basefmt=" ")
    ax0.set_ylabel("Weight (abs)", fontsize=11)
    ax0.set_xticks([])
    ax0.set_title("Single-cell contribution\nof Integration Dimension", fontsize=12)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)

    # Bottom left: heatmap of emission matrix, neurons sorted by first dimension
    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(C[sorted_indices, :], aspect='auto', cmap='viridis', interpolation='none')
    ax1.set_ylabel("Neurons (sorted)", fontsize=11)
    ax1.set_xlabel("Dimension", fontsize=11)
    ax1.set_title('Emission matrix C', fontsize=12)
    ax1.grid(False)

    # Add colorbar
    # # Bottom right: (Optional) Include unsorted version or another representation if desired
    # sorted_indices = np.argsort(-C[:, 1])  # Sort by contribution to the first dimension
    # ax2 = fig.add_subplot(gs[1, 1])
    # im2 = ax2.imshow(C[sorted_indices, :], aspect='auto', cmap='viridis', interpolation='none')
    # ax2.set_title('Emission matrix C', fontsize=12)

    # ax2.set_xlabel('Dimension', fontsize=11)
    # ax2.grid(False)
    # plt.colorbar(im2, ax=ax2, orientation='vertical', fraction=0.05, pad=0.04, label='Weight (abs)')

    # # Top right: hide axis if not needed
    # ax3 = fig.add_subplot(gs[0, 1])
    # abs_weights = np.abs(C[:, 1])
    # ax3.stem(np.arange(len(abs_weights)), abs_weights, basefmt=" ")

    # ax3.set_xticks([])
    # ax3.set_title("Single-cell contribution\nof dimension (L2)", fontsize=12)
    # ax3.spines['right'].set_visible(False)
    # ax3.spines['top'].set_visible(False)

    plt.tight_layout()
    filename = f"{fSave}_Single_neuron_contributions.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", transparent=True)
    plt.show()
    #%% Plot out top contributing neurons to each dimension
    # Get the weights for the first few dimensions (e.g., first 4)
    from matplotlib.gridspec import GridSpec
    C = abs(np.squeeze(slds.emissions.Cs))
    neuron_num = 16
    top_neurons_L1 = np.argsort(np.abs(C[:, 0]))[-neuron_num:]
    top_neurons_L2 = np.argsort(np.abs(C[:, 1]))[-neuron_num:]

    # ------------------------------------------------------------------
    # Set trials to analyze/plot here as a tuple: (start_trial, end_trial)
    # end_trial is excluded, just like standard Python slicing
    # Example: (0, 100) plots trials 0..99
    # ------------------------------------------------------------------
    trial_range = (0, 200)

    def smooth_rate(rate, sigma=15):
        radius = int(4 * sigma)
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return np.convolve(rate, kernel, mode='same')

    def plot_raster_and_rate(spike_data, top_neurons, region_name,
                            trial_range=(0, 100), sigma=15,
                            rate_style='filled', color='steelblue',
                            bin_size_ms=1):

        trial_start, trial_end = trial_range
        spike_data = spike_data[trial_start:trial_end, :, :]
        n_trials, n_time, _ = spike_data.shape
        time = np.arange(n_time)

        fig = plt.figure(figsize=(neuron_num * 4, 6))
        gs = GridSpec(2, neuron_num, height_ratios=[3, 1], hspace=0.12, wspace=0.3)

        for i, neuron_idx in enumerate(top_neurons):
            ax_raster = fig.add_subplot(gs[0, i])
            ax_rate = fig.add_subplot(gs[1, i], sharex=ax_raster)

            # Raster
            for trial_idx in range(n_trials):
                spike_times = np.where(spike_data[trial_idx, :, neuron_idx] > 0)[0]
                ax_raster.vlines(spike_times, trial_idx + 0.5, trial_idx + 1.5,
                                color='k', linewidth=0.7)

            ax_raster.set_ylim(0.5, n_trials + 0.5)
            ax_raster.set_title(f'Neuron {neuron_idx}')
            if i == 0:
                ax_raster.set_ylabel('Trial')
            else:
                ax_raster.set_yticklabels([])
            ax_raster.tick_params(axis='x', labelbottom=False)

            # Mean firing rate across selected trials
            mean_spikes = spike_data[:, :, neuron_idx].mean(axis=0)
            firing_rate = mean_spikes * (1000 / bin_size_ms)   # Hz
            firing_rate_smoothed = smooth_rate(firing_rate, sigma=sigma)

            # Filled/bar-like firing-rate panel
            if rate_style == 'filled':
                ax_rate.fill_between(time, firing_rate_smoothed, 0,
                                    color=color, alpha=0.95, linewidth=0)
                ax_rate.plot(time, firing_rate_smoothed, color=color, linewidth=1.0)

            elif rate_style == 'bar':
                ax_rate.bar(time, firing_rate_smoothed, width=1.0,
                            color=color, edgecolor=color, linewidth=0)

            else:  # fallback to line
                ax_rate.plot(time, firing_rate_smoothed, color=color, linewidth=1.5)

            if i == 0:
                ax_rate.set_ylabel('Hz')
            else:
                ax_rate.set_yticklabels([])

            ax_rate.set_xlabel('Time (ms)')
            ax_rate.spines['top'].set_visible(False)
            ax_rate.spines['right'].set_visible(False)
            ax_rate.set_xlim(time[0], time[-1])

        fig.suptitle(
            f'{region_name} Top Neurons in PC1: Raster + Firing Rate\nTrials {trial_start}:{trial_end}',
            y=1.02
        )
        plt.tight_layout()
        plt.savefig(f"{fSave}raster_rate_{region_name}_trials{trial_start}_{trial_end}.pdf")
        plt.show()

    # Example usage
    # plot_raster_and_rate(
    #     warp_spk[:,:,:,0], top_neurons_L1, 'L1',
    #     trial_range=trial_range,
    #     sigma=20,
    #     rate_style='filled',
    #     color='#1f77b4'
    # )

    # plot_raster_and_rate(
    #     warp_spk[:,:,:,0], top_neurons_L2, 'L2',
    #     trial_range=trial_range,
    #     sigma=20,
    #     rate_style='filled',
    #     color="#ca7d24"
    # )
    # #%%
    # def plot_raster_and_rate(spike_data, top_neurons, region_name,
    #                         trial_range=(0, 100), sigma=15,
    #                         rate_style='filled', color='steelblue',
    #                         bin_size_ms=1,
    #                         pull_times=None,
    #                         pull_color_first='tab:orange',
    #                         pull_color_last='tab:blue',
    #                         pull_marker_size=8):

    #     trial_start, trial_end = trial_range
    #     spike_data = spike_data[trial_start:trial_end, :, :]
    #     n_trials, n_time, _ = spike_data.shape

    #     # subset pull times if provided
    #     if pull_times is not None:
    #         pull_times_sub = pull_times[trial_start:trial_end, :]  # trials x 2
    #         first_pulls = pull_times_sub[:, 0]
    #         # mean first-pull time (in bins)
    #         t0 = np.nanmean(first_pulls)
    #     else:
    #         pull_times_sub = None
    #         t0 = 0.0

    #     # relative time axis (in ms)
    #     time = (np.arange(n_time) - t0) * bin_size_ms

    #     fig = plt.figure(figsize=(neuron_num * 4, 6))
    #     gs = GridSpec(2, neuron_num, height_ratios=[3, 1], hspace=0.12, wspace=0.3)

    #     for i, neuron_idx in enumerate(top_neurons):
    #         ax_raster = fig.add_subplot(gs[0, i])
    #         ax_rate = fig.add_subplot(gs[1, i], sharex=ax_raster)

    #         # Raster
    #         for trial_idx in range(n_trials):
    #             spike_bins = np.where(spike_data[trial_idx, :, neuron_idx] > 0)[0]
    #             spike_times_rel = (spike_bins - t0) * bin_size_ms
    #             ax_raster.vlines(spike_times_rel,
    #                             trial_idx + 0.5, trial_idx + 1.5,
    #                             color='k', linewidth=0.7)

    #             if pull_times_sub is not None:
    #                 t_first_bin, t_last_bin = pull_times_sub[trial_idx]

    #                 if not np.isnan(t_first_bin):
    #                     t_first_rel = (t_first_bin - t0) * bin_size_ms
    #                     ax_raster.scatter(t_first_rel, trial_idx + 1,
    #                                     s=pull_marker_size,
    #                                     color=pull_color_first,
    #                                     edgecolor='none',
    #                                     zorder=5)

    #                 if not np.isnan(t_last_bin):
    #                     t_last_rel = (t_last_bin - t0) * bin_size_ms
    #                     ax_raster.scatter(t_last_rel, trial_idx + 1,
    #                                     s=pull_marker_size,
    #                                     color=pull_color_last,
    #                                     edgecolor='none',
    #                                     zorder=5)

    #         ax_raster.set_ylim(0.5, n_trials + 0.5)
    #         ax_raster.set_title(f'Neuron {neuron_idx}')
    #         if i == 0:
    #             ax_raster.set_ylabel('Trial')
    #         else:
    #             ax_raster.set_yticklabels([])
    #         ax_raster.tick_params(axis='x', labelbottom=False)

    #         # Mean firing rate across selected trials
    #         mean_spikes = spike_data[:, :, neuron_idx].mean(axis=0)
    #         firing_rate = mean_spikes * (1000 / bin_size_ms)   # Hz
    #         firing_rate_smoothed = smooth_rate(firing_rate, sigma=sigma)

    #         if rate_style == 'filled':
    #             ax_rate.fill_between(time, firing_rate_smoothed, 0,
    #                                 color=color, alpha=0.95, linewidth=0)
    #             ax_rate.plot(time, firing_rate_smoothed, color=color, linewidth=1.0)
    #         elif rate_style == 'bar':
    #             ax_rate.bar(time, firing_rate_smoothed, width=bin_size_ms,
    #                         color=color, edgecolor=color, linewidth=0)
    #         else:
    #             ax_rate.plot(time, firing_rate_smoothed, color=color, linewidth=1.5)

    #         if i == 0:
    #             ax_rate.set_ylabel('Hz')
    #         else:
    #             ax_rate.set_yticklabels([])

    #         ax_rate.set_xlabel('Time (ms relative to mean first pull)')
    #         ax_rate.spines['top'].set_visible(False)
    #         ax_rate.spines['right'].set_visible(False)
    #         # set the xlim as the largest 3rd pull time
    #         ax_rate.set_xlim(time[500], time[-1])

    #     fig.suptitle(
    #         f'{region_name} Top Neurons in PC1: Raster + Firing Rate\n'
    #         f'Trials {trial_start}:{trial_end}, aligned to mean first pull',
    #         y=1.02
    #     )
    #     plt.tight_layout()
    #     plt.savefig(f"{fSave}raster_rate_{region_name}_aligned_trials{trial_start}_{trial_end}.pdf")
    #     plt.show()
    # # pull_times: full trials x 2 array for this warp_spk dataset

    # pull1aligned = np.squeeze(np.array([pull1_1, pull3_1])).T  # shape (n_trials, 2)

    # plot_raster_and_rate(
    #     warp_spk[:, :, :, 0],
    #     top_neurons_L1,
    #     'L1',
    #     trial_range=trial_range,
    #     sigma=20,
    #     rate_style='filled',
    #     color='#1f77b4',
    #     pull_times=pull1aligned  # pass here
    # )
    # plot_raster_and_rate(
    #     warp_spk[:, :, :, 0],
    #     top_neurons_L2,
    #     'L2',
    #     trial_range=trial_range,
    #     sigma=20,
    #     rate_style='filled',
    #     color="#c28120",
    #     pull_times=pull1aligned  # pass here
    # )
    