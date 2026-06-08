"""
Description:  Functions to load multiple sessions of data for a given mouse into Jax 
            data structures for processing with Dynamax
Author:       Bhavya Surapaneni
Date :        2026-05-27
"""

# ------ IMPORTS ------ 
from autograd import scipy
import external.ssm
import external.ssm.ssm.util as util
import matplotlib.pyplot as plt
import os
import scipy.io
import numpy as np
import pandas as pd

# ------ LOAD DATA FILES ------

# change this based on the dataset we are working with
my_set = "#23819M_GCaMP8m_rg_PoM-FOV2"

# load sessions for naive, intermediate, or expert

# loads data straight from full mouse dataset (ex #23819M_GCaMP8m_rg_PoM-FOV2)
def load_dendrite_data(set):

    file_path = "data/" + my_set
    mat = scipy.io.loadmat(file_path)

    behavior = os.listdir((file_path + "/behavior_data", '**/behavior.mat')) 
    calcium = os.listdir((file_path + "/data", '**/data.mat'))
    manual = os.listdir((file_path + "/ManualResults", '**/manual.mat')) 

    return behavior, calcium, manual

# selects cleaned dataset depending on whether we are looking for naive, intermediate, or expert
# conditions to select from are [naive, intermediate, expert]
def select_condition_sessions(set, condition):
    if condition == "naive":
        mask1 = set[:, 0] == 1 # fix this indexing so it aligns with correct column
        return load_dendrite_data(mask1) 
    elif condition == "intermediate":
        mask2 = set[:, 0] == 2
        return load_dendrite_data(mask2)
    elif condition == "expert":
        mask3 = set[:, 0] == 3 
        return load_dendrite_data(mask3)
    else:
        raise ValueError("invalid condition, has to be [naive, intermediate, expert]")

# combines sessions from the condition you designated
def concatenate_sessions(cond_set):
    for i in set:
        # load the behavior, calcium, and manual data for each session
        behavior, calcium, manual = load_dendrite_data(i)
        # concatenate the data into a single array for each type of data
        behavior_array = np.concatenate(behavior)
        calcium_array = np.concatenate(calcium)
        manual_array = np.concatenate(manual)

        return behavior_array, calcium_array, manual_array

# run all behavior cleaning for the array with the pipeline (go, nogo, rejection, miss, hit rate, fa rate, dprime, 
# lick_time_concatenated, reward_time_concatenated, GO_time_concatenated, NOGO_time_concatenated, trial_break)

def behavior_data_pipeline(conc_set):
    return 0


def calcium_data_pipeline(conc_set):
    # note that this requires trial_break from the behavior set so will need some method of getting that from there 
    # (call the function internally perhaps, but that is inefficient?)

    return 0

mouse_23819 = load_dendrite_data(my_set)


# designate trial types

# GO_trial =
# NOGO_trial = 
# fa_trial = 
# hit_trial = 
# teach_trial = 
# missed_teach_trial = 

# rejection_trial = set.difference(NOGO_trial, fa_trial)
# miss_trial = set.difference(GO_trial, [hit_trial, teach_trial, missed_teach_trial]


# def main_load():
    
    
#     return final_array