"""same as load_dendrite.py but for 1 session of data, preliminary testing"""

import sys
from pathlib import Path

ext = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ext))

from autograd import scipy
import ssm
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

behavior_data = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/behavior_data/#23819M_GCaMP8m_rg_PoM-050425_23819-behavior.mat"
calcium_data = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/data/#23819M_GCaMP8m_rg_PoM-050425_23819-condition1-source1-FOV2-plane1-data.mat"
manual = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/ManualResults/cell_idx_map.mat" # does NOT change between sessions
range_full = "data/shulan/#23819M_GCaMP8m_rg_PoM-FOV2/range_full/range.mat" # does NOT change between sessions
pool_trunk = "data/shulan/pool_trunk_across_sessions.xlsx" # does NOT change between sessions

# BEHAVIOR
mat1 = scipy.io.loadmat(behavior_data)
print("behavior_data mat type: " + str(type(mat1))) # datatype debugging

arduino_data = scipy.io.loadmat(mat1['arduino_data'])
behavior = scipy.io.loadmat(mat1['behavior'])
trial_break = scipy.io.loadmat(mat1['trial_break'])

print("nested behavior type: " + str(type(behavior)))
print(behavior.shape)

# from arduino (26 fields)
hit_rate = arduino_data[21]
fa_rate = arduino_data[22]
dprime = arduino_data[25]

# from behavior
lick = behavior[18]
reward = behavior[19]
go =  behavior[20]
nogo = behavior[21]

# CALCIUM
mat2 = scipy.io.loadmat(calcium_data)
print("calcium mat type" + str(type(mat2))) # datatype debugging

# MANUAL
mat3 = scipy.io.loadmat(manual)
print("manual mat type" + str(type(mat3))) # datatype debuggi1g

# RANGE_FULL
mat4 = scipy.io.loadmat(range_full)
print("range_full mat type" + str(type(mat4))) # datatype debugging``

# POOL TRUNK
pool = pd.read_excel(pool_trunk)
print("pool mat type" + str(type(pool))) # datatype debugging

print




