# FIRST 23819 FOV_2: GO DATA
# SECOND 23819 FOV_2: NOGO DATA
# THIRD 51521 FOV_2: GO DATA
# FOURTH 51521 FOV_2: NOGO DATA

# LOADING SESSIONS FOR THE SAME MOUSE 

# IMPORTS
import os
import dynamax
import numpy as np

# change this based on the dataset we are working with
my_set = "#23819M_GCaMP8m_rg_PoM-FOV2"
file_path = "data/" + my_set

behavior = os.listdir((file_path + "/behavior_data", '**/behavior.mat')) 
calcium = os.listdir((file_path + "/data", '**/data.mat'))
manual = os.listdir((file_path + "/ManualResults", '**/manual.mat')) 

# debug

# move all the stuff for splitting into three sections
def load_dendrite_data(my_set):
    if idx


# load the encoding condition, go barrel vs nogo barrel, time index,
# licking behavior, reward dispensed, distance ran, running speed
# 
for idx in list(0, 254):
    if idx in go_idx:
    label trial as go
    else:
        % label trial as nogo


GO_trial =
NOGO_trial = 
fa_trial = 
hit_trial = 
teach_trial = 
missed_teach_trial = 

rejection_trial = set.difference(NOGO_trial, fa_trial)
miss_trial = set.difference(GO_trial, [hit_trial, teach_trial, missed_teach_trial]

label the trial based on the name of the files uploaded

create infrastructure for what we designate as inputs, outputs


% function to filter for naive data
function naive_data()
end

% function to filter for intermediate data
function intermediate_data()
end


% function to filter for expert data
function expert_data()
end
