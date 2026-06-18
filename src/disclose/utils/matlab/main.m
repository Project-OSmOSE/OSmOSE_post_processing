clear;clc
% keep in mind that PAMGuard's MATLAB function must be located in the
% current working directory.
% It is available on https://github.com/PAMGuard/PAMGuardMatlab
addpath(genpath(fileparts(pwd)))

% user inputs
info_deploy.annotator = 'ANNOTATOR_NAME';
info_deploy.annotation = 'ANNOTATION_NAME';
info_deploy.dataset = 'DATASET_NAME';
info_deploy.timezone = '+00:00';
% info_deploy.dt_format = 'yyyyMMdd''T''HHmmss'; % MIRACETI filename format
% info_deploy.dt_format = 'yyMMddHHmmss'; % SoundTrap filename format
% info_deploy.dt_format = 'yyyy-MM-dd_HH-mm-ss'; % Sylence filename format
% info_deploy.dt_format = 'yyyy''y''MM''m''dd''d''_HH''h''mm''m''ss''s'''; % DORI filename format

% get wav files
%%% mode 'folder': the wav files are located on different folders
%%% mode 'file': the wav files are located on in the same folder
% mode = 'file';
mode = 'folder';

% path to start-location for the following pop-up menu
base_folder = 'L:\acoustock\Bioacoustique';

if isequal(mode, 'file')
    msg = sprintf('%s - select waves', info_deploy.dataset);
    [wav_file, folder_wav] = uigetfile('*.wav', msg, 'Multiselect', 'on', base_folder);
    wav_path = fullfile(folder_wav,  wav_file);
    wav_dir = cellfun(@dir, wav_path);
    folder_wav = {fileparts(folder_wav)};
elseif isequal(mode, 'folder')
    msg = sprintf('%s - select wav folders', info_deploy.dataset);
    folder_wav = uigetdir2(base_folder, msg);
    wav_dir = cellfun(@dir, fullfile(folder_wav, '**/*.wav'), 'UniformOutput', false);
    wav_dir = vertcat(wav_dir{:})';
end

binary_folder = uigetdir2(fileparts(folder_wav{1}), sprintf('%s - select binary folder', info_deploy.dataset));
binary_dir = cellfun(@dir, fullfile(binary_folder, '/**/*.pgdf'), 'UniformOutput', false);
binary_dir = vertcat(binary_dir{:})';

% read and export data from binary files to APLOSE csv / Raven txt files
if numel(binary_dir) == numel(wav_dir)
    for i=1:numel(binary_folder)
        PG2APLOSE(info_deploy, wav_dir, binary_dir, binary_folder{i});
    end
else
    error('Number of wav files (%.0f) is different than number of pgdf files (%.0f)', numel(wav_dir), numel(binary_dir))
end
