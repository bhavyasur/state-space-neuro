import multiprocessing
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from dysts.flows import Lorenz
import scipy.io

from pathlib import Path
import sys

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from external.LDNS.ldns.utils.utils import set_seed
from old.lorenz.latent_attractor import AttractorDataset, get_attractor_dataloaders
from utils.utils import spike_pca, plot_pca

# -----------------

def get_lorenz(n_neurons, sequence_length, n_ic, mean_spike_count):
    dataset = AttractorDataset(
        system_name="LorenzCoupled",
        n_neurons= n_neurons, #128
        sequence_length= sequence_length, #1024
        n_ic= n_ic, #100
        mean_spike_count= mean_spike_count, #200
        random_seed=42,
    )

    sample = dataset[1]
    print("np.shape(sample['signal']):", np.shape(sample["signal"]))
    print("np.shape(sample['rates'])", np.shape(sample["rates"]))
    return sample

def visualize_lorenz(sample):

    plt.figure(figsize=(6, 4))
    plt.imshow(sample["signal"], aspect="auto")
    plt.colorbar()
    plt.title("Spike counts: Lorenz, num_ic=500")

    plt.figure(figsize=(6, 4))
    plt.imshow(sample["rates"], aspect="auto")
    plt.colorbar()
    plt.title("Firing rates: Lorenz, num_ic=500")

    plt.show()
    return

if __name__ == "__main__":
    neurons = 128
    sequence = 1024
    nic = 100
    mean_spike = 200

    data = get_lorenz(neurons, sequence, nic, mean_spike)
    visualize_lorenz(data)
    scipy.io.savemat("output/lorenz/lorenz1.mat", {"sample": data})

    spikes = (data['signal']).T
    df, var = spike_pca(spikes, 10)

    print('explained variability per principal component: {}'.format(var))

    print(df.shape)

    plot_pca(spikes, df)

""" OLD CODE """

# --- DATATYPE DEBUGGING ---
# print(type(sample["signal"]))
# print(sample["signal"].shape)
# print(sample["signal"].dtype)
# print(torch.isnan(sample["signal"]).any())
# print(sample["signal"].min())
# print(sample["signal"].max())

# # test dataloader creation
# train_loader, valid_loader, test_loader = get_attractor_dataloaders(
#     "Lorenz", n_neurons=128, sequence_length=500, n_ic=100
# )

# # print shapes from each loader
# for loader, name in [(train_loader, "Train"), 
#                     (valid_loader, "Valid"), 
#                     (test_loader, "Test")]:
#     batch = next(iter(loader))
#     print(f"\n{name} loader shapes:")
#     print(f"Signal: {batch['signal'].shape}")
#     print(f"Latents: {batch['latents'].shape}")
#     print(f"Rates: {batch['rates'].shape}")