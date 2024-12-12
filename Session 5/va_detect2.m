function [alarm, t] = va_detect2(ecg_data, Fs)
% VA_DETECT Ventricular arrhythmia detection function
%  [ALARM,T] = VA_DETECT(ECG_DATA,FS) detects ventricular arrhythmia using
%  morphological features: zero crossings and mean amplitude of R peaks.

    % Processing frames: adjust frame length & overlap here
    frame_sec = 10;  % Frame length in seconds
    overlap = 0.5;   % 50% overlap between consecutive frames

    % Set feature thresholds
    zero_crossing_threshold = 75;  % Threshold for zero crossings
    mean_R_amplitude_threshold = -30;  % Threshold for mean R amplitude

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
        
        % Calculate zero crossings
        zero_crossings = sum(abs(diff(seg > 0)));
        
        % Calculate mean amplitude of R peaks using findpeaks
        [pks, locs] = findpeaks(seg);        % Local maxima
        [valls, vall_locs] = findpeaks(seg); % Local minima
        valls = valls;                      % Convert minima back to positive values
        mean_R_amplitude = mean([pks; valls]);
        
        % Set alarm if any of the thresholds are exceeded ٪
        if  zero_crossings > zero_crossing_threshold
            alarm(i) = 1;  % VFIB
        end
    end
end
