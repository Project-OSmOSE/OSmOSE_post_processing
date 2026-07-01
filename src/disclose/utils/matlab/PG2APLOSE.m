function [] = PG2APLOSE(info_deployment, wav_directory, binary_directory, binary_path)
% This function import .pgdf PAMGUARD binary files info and export it as an Aplose csv & Raven selection table

start = datetime("now");

% binary folder
dt_binary_file = char(extractfield(binary_directory, 'name')');
binary_struct.filename = string(extractfield(binary_directory, 'name')');
binary_struct.folder = string(extractfield(binary_directory, 'folder')');
binary_struct.datetime = datetime(dt_binary_file(:, end-19:end-5), 'InputFormat', 'yyyyMMdd_HHmmss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deployment.timezone);
binary_struct.path = char(unique(extractfield(binary_directory, 'folder')'));

% wav_folder
wav_struct.filename = string(extractfield(wav_directory, 'name')');
wav_struct.folder = string(extractfield(wav_directory, 'folder')');
wav_struct.datetime = convert_datetime(wav_struct.filename, info_deployment.dt_format, info_deployment.timezone);

% user input to select the datetime of deployment and recovery
[~, prompt_title, ~] = fileparts(binary_path);

% loop until valid inputs are provided or user cancels
while true
    % prompt user for deployment date and time
    deployment_start_str = inputdlg('Date & time of deployment (dd MM yyyy HH mm ss):', prompt_title);
    if isempty(deployment_start_str)  % check if user cancels
        return;
    end

    % prompt user for recovery date and time
    deployment_stop_str = inputdlg('Date & time of recovery (dd MM yyyy HH mm ss):', prompt_title);
    if isempty(deployment_stop_str)  % check if user cancels
        return;
    end

    % convert input strings to datetime
    deployment_start = datetime(string(deployment_start_str), 'InputFormat', 'dd MM yyyy HH mm ss', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deployment.timezone);
    deployment_stop = datetime(string(deployment_stop_str), 'InputFormat', 'dd MM yyyy HH mm ss', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deployment.timezone);

    % validate inputs
    if isempty(deployment_start) || isempty(deployment_stop)
        errordlg('Please enter valid date and time values', 'Invalid Input', 'modal');
    elseif deployment_stop <= deployment_start
        errordlg('Recovery date and time must be after deployment date and time', 'Invalid Input', 'modal');
    else
        break;
    end
end
dt_deploy = [deployment_start, deployment_stop];

% file selection based on previous user inputs
idx_file{1} = find(wav_struct.datetime < min(binary_struct.datetime), 1, 'last');
if isempty(idx_file{1})
    idx_file_beg = 1;
else
    [~, min_idx(1,1)]= min(abs(min(binary_struct.datetime) - wav_struct.datetime(idx_file{1}:idx_file{1}+1)));
    if min_idx(1,1) == 1
        idx_file_beg = idx_file{1};
    elseif min_idx(1,1) == 2
        idx_file_beg = idx_file{1} + 1;
    end
end

idx_file{2} = find(wav_struct.datetime > max(binary_struct.datetime), 1);
if isempty(idx_file{2})
    idx_file_end = length(wav_struct.datetime);
else
    [~, min_idx(1,2)]= min(abs(min(binary_struct.datetime) - wav_struct.datetime(idx_file{2}-1:idx_file{2})));
    if min_idx(1,2) == 1
        idx_file_end = idx_file{2}-1;
    elseif min_idx(1,2) == 2
        idx_file_end = idx_file{2};
    end
end
wav_struct.filename([1:idx_file_beg-1,idx_file_end+1:end]) = [];
wav_struct.folder([1:idx_file_beg-1,idx_file_end+1:end]) = [];
wav_struct.datetime([1:idx_file_beg-1,idx_file_end+1:end]) = [];

for i=1:numel(wav_struct.filename)
    wav_struct.wavinfo(i,1) = audioinfo(fullfile(wav_struct.folder(i,:), wav_struct.filename(i,:)));
    clc
    disp(['reading wav files...', num2str(i),'/', num2str(numel(wav_struct.filename))])
end

wav_struct.txt_filename = extractBefore(wav_struct.filename(1), '.wav');

% importation of data
detection_table_aplose = importBinary(info_deployment, wav_struct, binary_struct, binary_path, dt_deploy);

% name of file
if isequal([year(wav_struct.datetime(1)), month(wav_struct.datetime(1)), day(wav_struct.datetime(1))],...
        [year(wav_struct.datetime(end)), month(wav_struct.datetime(end)), day(wav_struct.datetime(end))])
    PG_title_datestr = string(datetime(wav_struct.datetime(1), 'Format','yyMMdd'));
else
    PG_title_datestr = strcat(string(datetime(wav_struct.datetime(1), 'Format','yyMMdd')), '_' , ...
        string(datetime(wav_struct.datetime(end), 'Format','yyMMdd')));
end

% export data as APLOSE csv file
writetable(detection_table_aplose, strcat(fileparts(binary_path), strcat('\PG_rawdata_', PG_title_datestr, '.csv')))

stop = datetime("now");
elapsed_time = seconds(stop - start);

clc
disp(['raw data file exported in: ', fileparts(binary_path)])
fprintf('\n')
disp(['elapsed time: ', num2str(uint64(elapsed_time)), ' s'])

end
