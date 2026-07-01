function data_table_Aplose = importBinary(info_deploy, wav_struct, binary_struct, binary_folder, dt_deploy)
% this function extracts and converts detections from PAMGuard binary files

% sampling frequency
sample_rate = wav_struct.wavinfo(1).SampleRate;

% load binary_data
binary_data = loadPamguardBinaryFolder(binary_folder, '*.pgdf', 1);
if isempty(binary_data)
    error('No detection found under binary folder %s', binary_struct.path)
end

detection_begin = datetime([binary_data.date]', 'ConvertFrom', 'datenum');
detection_begin.TimeZone = info_deploy.timezone;
detection_duration = cell2mat({binary_data(1:end).sampleDuration})' / sample_rate;
detection_end = detection_begin + seconds(detection_duration);
detection_begin_str = string(datetime(detection_begin, 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deploy.timezone));
detection_end_str = string(datetime(detection_end, 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', info_deploy.timezone));

% delete detections that are not within deployment and recovery datetimes
idx_to_delete = detection_begin < dt_deploy(1) | detection_begin > dt_deploy(2);
binary_data(idx_to_delete) = [];
detection_begin(idx_to_delete) = [];
detection_begin_str(idx_to_delete, :) = [];
detection_end_str(idx_to_delete, :) = [];
detection_duration(idx_to_delete) = [];

% min and max frequencies of each detection
freqs = cell2mat({binary_data(1:end).freqLimits}');
freq_low = freqs(:,1);
freq_high = freqs(:,2);

% index of the corresponding wav file for each detection
idx_wav = zeros(numel(detection_begin), 1);
for i = 1:numel(detection_begin)
    idx_wav(i) = find(detection_begin(i) >= wav_struct.datetime, 1, 'last');
end

% adjustment of the timestamps
wav_duration = extractfield(wav_struct.wavinfo, 'Duration')';

% time differences between consecutive datetimes and add wav_duration
filename_time_diff = diff(wav_struct.datetime);
adjust = [0; wav_duration(1:end-1) - seconds(filename_time_diff)];
cumsum_adjust = cumsum(adjust);

% adjusted datetime to match Raven functionning
detection_begin_adjusted = detection_begin + seconds(cumsum_adjust(idx_wav));

% number of seconds from start of files list to beginning of each detection
beg_sec_adjusted = seconds(detection_begin_adjusted - min(binary_struct.datetime));

% generate APLOSE table
L = length(binary_data);
annotator = repmat(info_deploy.annotator, [L,1]);
annotation = repmat(info_deploy.annotation, [L,1]);
dataset = repmat(info_deploy.dataset, [L,1]);
filename = wav_struct.filename(idx_wav);

start_time = zeros(numel(detection_begin), 1);
for i = 1:numel(detection_begin)
    if idx_wav(i) > 1
        start_time(i) = beg_sec_adjusted(i) - sum(wav_duration(1:idx_wav(i)-1));
    elseif idx_wav(i) == 1
        start_time(i) = beg_sec_adjusted(i);
    end
end
end_time = start_time + detection_duration;

data_table_Aplose = [ array2table([dataset, filename, start_time, end_time, freq_low, freq_high, annotation, annotator],...
    'VariableNames',{'dataset', 'filename', 'start_time','end_time','start_frequency','end_frequency', 'annotation', 'annotator'}),...
     table(detection_begin_str, detection_end_str, 'VariableNames',{'start_datetime','end_datetime'})];

data_table_Aplose = sortrows(data_table_Aplose, 'start_datetime');

clc
fprintf('Importation of %s detections done\n', num2str(length(binary_data)))
