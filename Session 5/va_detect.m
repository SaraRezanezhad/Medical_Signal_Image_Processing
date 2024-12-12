function [alarm, t] = va_detect(ecg_data, Fs)
% VA_DETECT Ventricular arrhythmia detection function
%  [ALARM,T] = VA_DETECT(ECG_DATA,FS) detects ventricular arrhythmia using
%  bandpower and median frequency thresholds.

    % Processing frames: adjust frame length & overlap here
    frame_sec = 10;  % Frame length in seconds
    overlap = 0.5;   % 50% overlap between consecutive frames

    % Set feature thresholds
    bandpower_threshold = 1000;  % Threshold for bandpower in 10-30 Hz
    median_freq_threshold = 3;  % Threshold for median frequency

    % Input argument checking
    if nargin < 2
        Fs = 250;  % Default sample rate
    end
    if nargin < 1
        error('You must enter an ECG data vector.');
    end
    ecg_data = ecg_data(:);  % Make sure that ecg_data is a column vector

    % Initialize Variables
    frame_length = round(frame_sec * Fs);  % Frame length in samples
    frame_step = round(frame_length * (1 - overlap));  % Step size between frames
    ecg_length = length(ecg_data);  % Length of input vector
    frame_N = floor((ecg_length - (frame_length - frame_step)) / frame_step); % Total number of frames
    alarm = zeros(frame_N, 1);  % Initialize alarm output
    t = ([0:frame_N - 1] * frame_step + frame_length) / Fs;  % Time vector

    % Analysis loop: each iteration processes one frame of data
    for i = 1:frame_N
        % Get the next data segment
        seg = ecg_data(((i - 1) * frame_step + 1):((i - 1) * frame_step + frame_length));
        
        % Calculate bandpower in the 10-30 Hz range
        bandpower_10_30Hz = bandpower(seg, Fs, [10 30]);
        
        % Calculate median frequency
        median_freq = medfreq(seg, Fs);
        
        % Set alarm if any of the thresholds are exceeded
        if bandpower_10_30Hz < bandpower_threshold || median_freq > median_freq_threshold
            alarm(i) = 1;
        end
    end
end
