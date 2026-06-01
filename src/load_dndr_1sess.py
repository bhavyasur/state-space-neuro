"""same as load_dendrite.py but for 1 session of data, preliminary testing"""

import sys
from pathlib import Path

from requests import session

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from autograd import scipy
import external.ssm
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import numpy as np

behavior_data = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/behavior_data/#23819M_GCaMP8m_rg_PoM-050425_23819-behavior.mat"
calcium_data = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/data/#23819M_GCaMP8m_rg_PoM-050425_23819-condition1-source1-FOV2-plane1-data.mat"
manual = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/ManualResults/cell_idx_map.mat" # does NOT change between sessions
range_full = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/range_full/range.mat" # does NOT change between sessions
pool_trunk = "data/shulan/pool_trunk_across_sessions.xlsx" # does NOT change between sessions


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

# BEHAVIOR
mat1 = scipy.io.loadmat(behavior_data)
print("behavior_data mat type: " + str(type(mat1))) # datatype debugging

arduino_data = mat_to_dict(mat1['arduino_data'])
behavior = mat_to_dict(mat1['behavior'])
trial_break = mat_to_dict(mat1['trial_break'])

print("nested behavior type: " + str(type(behavior)))

# from arduino (26 fields)
hit_rate = arduino_data["hit_rate"]
fa_rate = arduino_data["fa_rate"]
dprime = arduino_data["dprime"]

print("hit_rate type: " + str(type(hit_rate)))
print("fa_rate type: " + str(type(fa_rate)))
print("dprime type: " + str(type(dprime)))

# from behavior
lick = behavior["lick_time_concatenated"]
reward = behavior["reward_time_concatenated"]
go =  behavior["GO_time_concatenated"]
nogo = behavior["NOGO_time_concatenated"]

# trial break
print("trial_break type: " + str(type(trial_break))) # datatype debugging
print(np.sum(trial_break)) # check if sums to correct value

# CALCIUM
mat2 = scipy.io.loadmat(calcium_data)
print("calcium mat type" + str(type(mat2))) # datatype debugging
print("calcium keys: " + str(mat2.keys())) # check keys in calcium mat file

bSpikes = mat2['bSpikes']
deltaFoverF = mat2['deltaFoverF']

print("deltaFoverF type: " + str(type(deltaFoverF)))

# MANUAL
cell_idx_map = scipy.io.loadmat(manual)

print("manual mat type" + str(type(cell_idx_map))) # datatype debugging

allCentroid = cell_idx_map["allCentroid"]
session1_centroids = allCentroid[0, 0]
print("type(session1_centroids): " + str(type(session1_centroids)))
print(np.shape(session1_centroids))

allIdx = cell_idx_map["allIdx"]
session1_idx = allIdx[:, 0]
print("type(session1_idx): " + str(type(session1_idx)))
print("shape(session1_idx): " + str(np.shape(session1_idx)))

# RANGE_FULL
mat4 = scipy.io.loadmat(range_full)
print("range_full mat type" + str(type(mat4))) # datatype debugging

range = mat4["range"]
session1_range = range[0, 0]
print("shape(session1_range): " + str(np.shape(session1_range)))

range_sessions = mat4["range_sessions"]

# POOL TRUNK
pool = pd.read_excel(pool_trunk)
print("pool mat type" + str(type(pool))) # datatype debugging


# ============== now data cleaning

# hit_rate, fa_rate, dprime are per trial (1 x 254)
# lick, reward, go, nogo are per timepoint (1 x 22675) - these are parameter vectors
# bSpikes, deltaFoverF are per timepoint (51 x 22675)
# session1_centroids is 57 x 2, outlines xy coordinates for each dendrite
# session1_idx is the indices of each dendrite in session 1, (53,)
# session1_range is indices of dendrites that should be analyzed in session1
# range_sessions is the indices of sessions that should be analyzed from this animal - HAVE A CROSS CHECK WITH POOL TO MAKE SURE EACH SESSION SHOULD BE USED

# sample out to 22675



# i represents each trial
# for i in range(len(trial_break)):
#     len_trial = trial_break[i]
#     new = bSpikes[:, len_trial]

    
    # break each imaging dataset into trials based on trial_break
    # imaging_trial = x
    



