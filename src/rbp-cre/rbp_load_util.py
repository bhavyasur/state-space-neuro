"""
DATA DESCRIPTION:
Rbp-Cre data from Shulan. Once again working session-by-session. Each session contains data from Arduino, behavior data, and calcium imaging data.
I am ignoring the arduino data since there is a serial communication delay and working exclusively with behavior and Ca imaging data.

Rbp-Cre dataset is for L5 pyramidal cells, the retrograde virus injected ensures that it's only those neurons projecting to a specific region

Animal folder -> "behavior_output" -> "behavior_Intan" and "Ca_imaging_data".
behavior_Intan loads a mat struct called "behavior", Ca_imaging_data loads a mat struct called "Ca_data"

behavior:
- session_break_time is length of session in ms??
- Fs: sampling rate (usually 8192)
- GO_timeidx & GOend_timeidx: get trial times from this
- NOGO_timeidx & NOGOend_timeidx: get trial times
---> OR can use trial_timeidx and trialend_timeidx
- GO_lick: (num trials, x) based on downsampling -> HIT
---> need to convert this to binary & concatenate trials, then downsample?
- reward_time_idx: tells you when the mouse got water?
- hit_time: time indices of a hit
- fa_time: time indices of a false alarm
- hit_rate: rate of a hit as more trials occur
- fa_rate: rate of fa as more trials occur


Ca_data:
-  


EX DATASET: 090824_61601
- intan calcium Fs is 8192
- behavior Fs is 15


"""

import scipy.io
import os
import numpy as np
import sys
from pathlib import Path
import mat73
import matplotlib.pyplot as plt
import pandas as pd
import quantities as pq
import zipfile

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.utils import mat_to_dict


def load_behavior(folder_path):
    """
    INPUT: data is a folder for the session from a specific mouse. titled with session number and animal number, contains behavior_output folder which contains mat files
    OUTPUT: spikes is a list, each item represents a neuron and is a numpy array of (num trials x timesteps)
    """
    outer = Path(folder_path)
    inner = Path(outer / "behavior_output") 
    behavior = inner / "behavior_Intan"

    b = mat73.loadmat(behavior)

    print("mat type: " + str(type(b))) # datatype debugging
    print("mat keys: " + str(b.keys())) # check keys in mat file

    behavior_dict = {
        "session_break_time": b['behavior']['session_break_time'], # length of session in ms?
        "Fs": b['behavior']['Fs'], # sampling rate for intan
        "GO_timeidx": b['behavior']['GO_timeidx'], # use to get trial times
        "GOend_timeidx": b['behavior']['GOend_timeidx'], # use to get trial times
        "NOGO_timeidx": b['behavior']['NOGO_timeidx'], # use to get trial times
        "NOGOend_timeidx": b['behavior']['NOGOend_timeidx'], # use to get trial times
        "triallength": b['behavior']['triallength'], # ??
        "GOidx": b['behavior']['GOidx'], # ?? use to correspond trial data from time_idx to GO or NOGO
        "NOGOidx": b['behavior']['NOGOidx'], # ?? use to correspond trial data from time_idx to GO or NOGO
        "hit_idx": b['behavior']['hit_idx'], # identifies which trials were hits
        "fa_idx": b['behavior']['fa_idx'], # identifies which trials were false alarms
        "teach_idx": b['behavior']['teach_idx'], # identifies which trials were teach trials
        "missed_teach_idx": b['behavior']['missed_teach_idx'],
        "hit_rate": b['behavior']['hit_rate'],
        "fa_rate": b['behavior']['fa_rate'],
        "GO_lick": b['behavior']['GO_lick'],
        "NOGO_lick": b['behavior']['NOGO_lick'],
        "trial_map": b['behavior']['trial_map']
    }

    return behavior_dict

def load_calcium(folder_path):
    """
    INPUT: data is a folder for the session from a specific mouse. titled with session number and animal number, contains behavior_output folder which contains mat files
    OUTPUT: spikes is a list, each item represents a neuron and is a numpy array of (num trials x timesteps)
    """
    outer = Path(folder_path)
    inner = Path(outer / "behavior_output") 
    calcium = inner / "Ca_imaging_data"

    c = mat73.loadmat(calcium)

    roi1 = c['Ca_data']['ROI'][0]
    roi2 = c['Ca_data']['ROI'][1]
    roi3 = c['Ca_data']['ROI'][2]

    calcium_dict_roi1 = {
        "trial_break": roi1["trial_break"],
        "Fs": roi1["Fs"],
        "ROIcentroid": roi1["ROIcentroid"],
        "DeltaFoverF": roi1["DeltaFoverF"],
        "F": roi1["F"],
        "trial_start_time": roi1["trial_start_time"],
        "trial_end_time": roi1["trial_end_time"],
        
    }


# ---------- running things but ignore for now

if __name__ == "__main__":
    folder_path = "data/shulan/090824_61601"
    be = load_behavior(folder_path)
    ca = load_calcium(folder_path)
