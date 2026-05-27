# FIRST 23819 FOV_2: GO DATA
# SECOND 23819 FOV_2: NOGO DATA
# THIRD 51521 FOV_2: GO DATA
# FOURTH 51521 FOV_2: NOGO DATA


# LOADING SESSIONS FOR THE SAME MOUSE 

# change this based on the dataset we are working with
my_set = "#23819M_GCaMP8m_rg_PoM-FOV2";

# 1. Search for the folder (e.g., searching the C: drive for 'MyTargetFolder')
if ispc
    [status, result] = system('dir /s /b C:\' + my_set);
else
    [status, result] = system('find / -name "MyTargetFolder" -type d 2>/dev/null');
end

# 2. Extract the folder path from the result string
folderPath = strtrim(strtok(result, newline));

# 3. Check if found, navigate to it, and load files
if exist(folderPath, 'dir')
    cd(folderPath);
    % Load specific variable or workspace (e.g., load('data.mat'))
    load('data.mat'); 
else
    disp('Folder not found.');
end

behavior = dir(fullfile(my_set + "/behavior_data", '**/behavior.mat')); 
calcium = dir(fullfile(my_set + "/data", '**/data.mat'));
manual = dir(fullfile(my_set + "/ManualResults", '.mat')); 

for x = 1:length(behavior)
    filePath = fullfile(behavior(x).folder, behavior(x).name);
    [~, my_set] = fileparts(behavior(k).folder);
    behavior_set.(my_set) = load(filePath);
end

for k = 1:length(calcium)
    filePath = fullfile(calcium(x).folder, calcium(x).name);
    [~, my_set] = fileparts(calcium(k).folder);
    calcium_set.(my_set) = load(filePath);
end

for k = 1:length(manual)
    filePath = fullfile(manual(x).folder, manual(x).name);
    [~, my_set] = fileparts(manual(k).folder);
    manual_set.(my_set) = load(filePath);
end

# debug
disp(length(manual));


function load_dendrite_data(my_set)
% move all the stuff for splitting into three sections
end
% % load the encoding condition, go barrel vs nogo barrel, time index,
% % licking behavior, reward dispensed, distance ran, running speed
% 
% for idx in list(0, 254) 
%     if idx in go_idx:
%         % label trial as go
%     else:
%         % label trial as nogo
% 
% rejection_trial = setdiff(NOGO_trial, fa_trial);
% 
% miss_trial = setdiff(GO_trila, [hit_trial, teach_trial, missed_teach_trial]);cle
% 
% % label the trial based on the name of the files uploaded
% 
% % create infrastructure for what we designate as inputs, outputs


% function to filter for naive data
function naive_data()
end

% function to filter for intermediate data
function intermediate_data()
end


% function to filter for expert data
function expert_data()
end
