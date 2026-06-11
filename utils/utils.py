""" utility functions """
import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

def mat_to_dict(obj):
    """Recursively convert scipy mat_struct to nested dicts."""
    if isinstance(obj, scipy.io.matlab.mat_struct):
        return {field: mat_to_dict(getattr(obj, field))
                for field in obj._fieldnames}
    elif isinstance(obj, np.ndarray) and obj.dtype.names:
        # structured array
        return {name: mat_to_dict(obj[name]) for name in obj.dtype.names}
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        # array of structs
        return [mat_to_dict(item) for item in obj.flat]
    else:
        return obj  # base case: numbers, strings, plain np arrays
    
def spike_pca(data, n_components):
    """data parameter should be in form columns = each neuron, row = each time step"""
    print("if this doesn't match (num_timesteps x num_neurons), fix shape:", np.shape(data))

    norm_spike = StandardScaler().fit_transform(data)

    pca_spike = PCA(n_components = n_components)
    principalComponents_spike = pca_spike.fit_transform(norm_spike)


    pc_df = pd.DataFrame(data = principalComponents_spike,
                columns = [f'pc {i+1}'
                           for i in range(n_components)
                           ])

    var = pca_spike.explained_variance_ratio_

    return pc_df, var

def plot_pca(data, df):
    """plots first two principal components with full dataframe of PCA"""
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title("Principal Component Analysis of Spiking Dataset",fontsize=20)
    
    plt.scatter(df.loc['pc 1']
            , df.loc['pc 2'], s = 20)