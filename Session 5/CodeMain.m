%% Sara Rezanezhad - Ali Khosravipoor - Mohammadhossein Faramarzi
%% // Sara Rezanejad - 99101643 // MohamadHosein Faramarzi - 99104095 // Ali Khosravipour - 99101502 

%% Q1
time = normal(:, 1); 
ecg_signal = normal(:, 2); %

fs = 250;
segment_duration = 5; 
start_clean = 30; 
start_noisy = 240; 

% انتخاب بخش تمیز  
clean_segment = ecg_signal(time >= start_clean & time < start_clean + segment_duration);  
clean_time = time(time >= start_clean & time < start_clean + segment_duration);  

% انتخاب بخش نویزی  
noisy_segment = ecg_signal(time >= start_noisy & time < start_noisy + segment_duration);  
noisy_time = time(time >= start_noisy & time < start_noisy + segment_duration);  

% محاسبه طیف فرکانسی برای بخش تمیز  
[pxx_clean, f_clean] = pwelch(clean_segment, [], [], [], fs);  

% محاسبه طیف فرکانسی برای بخش نویزی  
[pxx_noisy, f_noisy] = pwelch(noisy_segment, [], [], [], fs);  

figure;  

subplot(2, 1, 1);  
plot(f_clean, 10*log10(pxx_clean));  
title('Power spectrum of clean ECG signal');  
xlabel('Frequency (Hz)');  
ylabel('Power(dB)');  
grid on;  

subplot(2, 1, 2);  
plot(f_noisy, 10*log10(pxx_noisy));  
title('Power spectrum of noisy ECG signal');  
xlabel('Frequency (Hz)');  
ylabel('Power(dB)');   
grid on;  

%% Q2

ecg_signal = double(ecg_signal);  
time = double(time);  

% محاسبه طیف توان برای تعیین فرکانسهای قطع  
[pxx, f] = pwelch(ecg_signal, [], [], [], fs);  

figure;  
plot(f, 10*log10(pxx));  
title('Power spectrum of ECG signal');  
xlabel('Frequency (Hz)');  
ylabel('Power(dB)');    
grid on;  

f_low = 0.5; 
f_high = 50; 

% طراحی فیلتر میان‌گذر  
order = 4; 
[b, a] = butter(order, [f_low, f_high] / (fs/2), 'bandpass'); 

filtered_signal = filtfilt(b, a, ecg_signal);  

figure;  
subplot(2, 1, 1);  
plot(time, ecg_signal);  
title('Original ECG signal');  
xlabel('Time(S)');  
ylabel('Voltage(V)');  
grid on;  

subplot(2, 1, 2);  
plot(time, filtered_signal);  
title('Filtered ECG signal');  
xlabel('Time(S)');  
ylabel('Voltage(V)');  
grid on;  


figure;  
[impulse_response, n] = impz(b, a); % 
stem(n, impulse_response, 'filled'); % رسم پاسخ ضربه با stem 
title('Pulse Response of Bandpass Filter');  
xlabel('Sample');  
ylabel('Transfer');  

figure;  
freqz(b, a, 1024, fs); % محاسبه پاسخ فرکانسی با نمونه 1024  
title('Frequency Response of Bandpass Filter');



%% Q3
% Select segments 
clean_segment = ecg_signal(1:2500); % First 10 seconds  
noisy_segment = ecg_signal(7500:10000); % Next 10 seconds  

% Calculate Power Spectral Density (PSD)  
[pxx_clean, f_clean] = pwelch(clean_segment, [], [], [], fs); % Clean segment  
[pxx_noisy, f_noisy] = pwelch(noisy_segment, [], [], [], fs); % Noisy segment  

% Normalize frequency for plotting  
normalized_frequency_clean = f_clean / (fs / 2);
normalized_frequency_noisy = f_noisy / (fs / 2); 

% Convert power to dB  
pxx_clean_dB = 10 * log10(pxx_clean);  
pxx_noisy_dB = 10 * log10(pxx_noisy);  

% Plotting  
figure;  

% Clean ECG Signal  
subplot(2, 1, 1);  
plot(normalized_frequency_clean, pxx_clean_dB, 'LineWidth', 1.5);  
title('Clean ECG Signal');  
xlabel('Normalized Frequency (× \pi rad/sample)');  
ylabel('Power/Frequency (dB/rad/sample)');  
grid on;  
axis([0 1 -60 0]); 
% Noisy ECG Signal  
subplot(2, 1, 2);  
plot(normalized_frequency_noisy, pxx_noisy_dB, 'LineWidth', 1.5);  
title('Noisy ECG Signal');  
xlabel('Normalized Frequency (× \pi rad/sample)');  
ylabel('Power/Frequency (dB/rad/sample)');  
grid on;  
axis([0 1 -60 0]); 

% Improve layout  
sgtitle('Power Spectral Density of ECG Signals'); % Main title

% Ensure both segments are of the same length for RMS calculation  
min_length = min(length(normalized_frequency_noisy), length(noisy_segment));  
normalized_frequency_noisy = normalized_frequency_noisy(1:min_length);  
noisy_segment = noisy_segment(1:min_length);  

% Calculate the root mean square (RMS) of the difference  
clean_noise_level = rms(noisy_segment - normalized_frequency_noisy);  
fprintf('Noise level (RMS) removed: %.4f\n', clean_noise_level);  
%% 
clean_segment = ecg_signal(time >= 10 & time <= 20);  
noisy_segment = ecg_signal(time >= 30 & time <= 40);  
time_clean = time(time >= 10 & time <= 20);  
time_noisy = time(time >= 30 & time <= 40);  

filtered_clean = filtfilt(b, a, clean_segment);  
filtered_noisy = filtfilt(b, a, noisy_segment);  

figure;  

subplot(2, 2, 1);  
plot(time_clean, clean_segment);  
title('ECG Clean Signal');  
xlabel('Time(S)');  
ylabel('Voltage(V)');
grid on;  

subplot(2, 2, 2);  
plot(time_clean, filtered_clean);  
title('ECG Clean Filtered Signal');  
xlabel('Time(S)');  
ylabel('Voltage(V)'); 
grid on;  

subplot(2, 2, 3);  
plot(time_noisy, noisy_segment);  
title('ECG Noisy Signal');  
xlabel('Time(S)');  
ylabel('Voltage(V)');
grid on;  

subplot(2, 2, 4);  
plot(time_noisy, filtered_noisy);  
title('ECG Noisy Filtered Signal');  
xlabel('Time(S)');  
ylabel('Voltage(V)');
grid on;  

% محاسبه و نمایش مقدار سیگنال و قدر مطلق اختلاف  
clean_noise_level = rms(noisy_segment - filtered_noisy); 
fprintf('Noise level (RMS) removed: %.4f\n', clean_noise_level);

% load the dataset
load("n_421.mat");
load("n_422.mat");
load("n_424.mat");
% plot them
ECG1 = n_422;  
ECG1 = n_424;  

fs = 250;  
t = (0:length(ECG1)-1) / fs;

figure;
subplot(2,1,1);  % First subplot for the first ECG signal
plot(t, ECG1, 'b');  
title('ECG Signal 1');
xlabel('Time (s)');
ylabel('Amplitude (mV)');
grid on;

subplot(2,1,2);  % Second subplot for the second ECG signal
plot(t, ECG2, 'r');  
title('ECG Signal 2');
xlabel('Time (s)');
ylabel('Amplitude (mV)');
grid on;

sgtitle('Comparison of Two ECG Signals');  
set(gcf, 'Color', 'w');  

%% Normal and abnormal
data=ECG1;

ECG1_Normal1=ECG1(1:2500,:);
ECG1_Abnormal1=ECG1(10711:13211,:);

ECG1_Normal2=ECG1(11211:11441,:);
ECG1_Abnormal2=ECG1(11442:13942,:);

% Create figure for plots
figure;

% Plot PSD for ECG1_Normal1
subplot(2, 2, 1);
pwelch(ECG1_Normal1, [], [], [], fs);
title('ECG1 Normal1 Segment');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Plot PSD for ECG1_Abnormal1
subplot(2, 2, 2);
pwelch(ECG1_Abnormal1, [], [], [], fs);
title('ECG1 Abnormal1 Segment');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Plot PSD for ECG1_Normal2
subplot(2, 2, 3);
pwelch(ECG1_Normal2, [], [], [], fs);
title('ECG1 Normal2 Segment');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Plot PSD for ECG1_Abnormal2
subplot(2, 2, 4);
pwelch(ECG1_Abnormal2, [], [], [], fs);
title('ECG1 Abnormal2 Segment');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Improve figure aesthetics
sgtitle('Frequency Domain Analysis of ECG1 Segments');
set(gcf, 'Color', 'w');  % Set background color to white

fs = 256;

normal_1_n422 = data(1:10710);
abnormal_1_n422 = data(10711:11210);
normal_2_n422 = data(11211:11442); 
abnormal_2_n422 = data(11442:59710); 
abnormal_3_n422 = data(61288:end);


[p_n_1, f_n_1] = pwelch(normal_1_n422, [], [], [], fs);
[p_n_2, f_n_2] = pwelch(normal_2_n422, [], [], [], fs);
[p_ab_2, f_ab_2] = pwelch(abnormal_2_n422, [], [], [], fs);
[p_ab_3, f_ab_3] = pwelch(abnormal_3_n422, [], [], [], fs);

% visualization

figure;
subplot(2,2,1)
plot(f_n_1, 10*log(p_n_1))
grid on;
title('Normal data 1')
xlabel('frequency (Hz)')
ylabel('power (dB)')
ylim([-60 100])

subplot(2,2,2)
plot(f_n_2, 10*log(p_n_2))
title('Normal data 2')
xlabel('frequency (Hz)')
ylabel('power (dB)')
grid on;

ylim([-60 100])

subplot(2,2,3)
plot(f_ab_2, 10*log(p_ab_2))
title('Abnormal data 2')
xlabel('frequency (Hz)')
ylabel('power (dB)')
grid on;

ylim([-60 100])

subplot(2,2,4)
plot(f_ab_3, 10*log(p_ab_3))
title('Abnormal data 3')
xlabel('frequency (Hz)')
ylabel('power (dB)')
grid on;

ylim([-60 100])





%% Q2
t1 = (0:length(ECG1)-1) / fs;  % Time vector

% Define new signal segments
ECG1_Normal1 = ECG1(1:2500);
t_Normal1 = t1(1:2500);  % Time vector for Normal1

ECG1_Abnormal1 = ECG1(10711:13211);
t_Abnormal1 = t1(10711:13211);  % Time vector for Abnormal1

ECG1_Normal2 = ECG1(11211:11441);
t_Normal2 = t1(11211:11441);  % Time vector for Normal2

ECG1_Abnormal2 = ECG1(11442:13942);
t_Abnormal2 = t1(11442:13942);  % Time vector for Abnormal2

% Create figure for plots
figure;

% Plot time-domain and frequency-domain for ECG1_Normal1
subplot(4, 2, 1);
plot(t_Normal1, ECG1_Normal1, 'b');
title('ECG1 Normal1 - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4, 2, 2);
pwelch(ECG1_Normal1, [], [], [], fs);
title('ECG1 Normal1 - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Plot time-domain and frequency-domain for ECG1_Abnormal1
subplot(4, 2, 3);
plot(t_Abnormal1, ECG1_Abnormal1, 'r');
title('ECG1 Abnormal1 - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4, 2, 4);
pwelch(ECG1_Abnormal1, [], [], [], fs);
title('ECG1 Abnormal1 - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Plot time-domain and frequency-domain for ECG1_Normal2
subplot(4, 2, 5);
plot(t_Normal2, ECG1_Normal2, 'b');
title('ECG1 Normal2 - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4, 2, 6);
pwelch(ECG1_Normal2, [], [], [], fs);
title('ECG1 Normal2 - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Plot time-domain and frequency-domain for ECG1_Abnormal2
subplot(4, 2, 7);
plot(t_Abnormal2, ECG1_Abnormal2, 'r');
title('ECG1 Abnormal2 - Time Domain');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(4, 2, 8);
pwelch(ECG1_Abnormal2, [], [], [], fs);
title('ECG1 Abnormal2 - Frequency Domain');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;

% Improve figure aesthetics
sgtitle('Time and Frequency Domain Analysis of Updated ECG1 Segments');
set(gcf, 'Color', 'w');  % Set background color to white


%% Q3
fs = 250;                   
segment_duration = 10 * fs; 
overlap = 5 * fs;           
signal_length = length(ECG1);

start_sample = 1;
endpoints = [];
segments = {};  

while start_sample + segment_duration - 1 <= signal_length
    end_sample = start_sample + segment_duration - 1;
    endpoints = [endpoints, end_sample];  % Store the end sample of each segment
    segments{end+1} = ECG1(start_sample:end_sample);  % Store the segment in the cell array
    start_sample = start_sample + (segment_duration - overlap);  % Shift by overlap
end

disp('Endpoints of each 10-second segment with 5-second overlap:');
disp(endpoints);

disp(['Total number of segments: ', num2str(length(segments))]);

%% Q4
Sample = [1, 10711, 11211, 11442, 59711, 61288];

fs = 250;                   
segment_duration = 10 * fs;  
overlap = 5 * fs;            
signal_length = length(ECG1);

% Define event periods
Norm1 = 1:10710;
VT1 = 10711:11210;
Norm2 = 11211:11441;
VT2 = 11442:59710;
Noise = 59711:61287;
VFIB = 61288:75000;

% Define labels for each event period
labels = {'Norm1', 'VT1', 'Norm2', 'VT2', 'Noise', 'VFIB'};
event_ranges = {Norm1, VT1, Norm2, VT2, Noise, VFIB};

start_sample = 1;
endpoints = [];
segments = {};  
segment_labels = {};  % Use a cell array for labels to allow mixed data types

while start_sample + segment_duration - 1 <= signal_length
    end_sample = start_sample + segment_duration - 1;
    endpoints = [endpoints, end_sample];  % Store the end sample of each segment
    segments{end+1} = ECG1(start_sample:end_sample);  % Store the segment in the cell array
    
    label = 0;  
    for i = 1:length(event_ranges)
        if all(start_sample >= min(event_ranges{i}) && end_sample <= max(event_ranges{i}))
            label = labels{i};  
            break;
        end
    end
    segment_labels{end+1} = label; 
    start_sample = start_sample + (segment_duration - overlap);  % Shift by overlap
end

% Display results
disp('Endpoints of each 10-second segment with 5-second overlap:');
disp(endpoints);

disp('Labels for each segment:');
disp(segment_labels);

disp(['Total number of segments: ', num2str(length(segments))]);


%% Q5 and Q6

bandpower_40_80Hz = [];
bandpower_10_30Hz = [];
median_freq = [];
mean_freq = [];
class_labels = [];

for i = 1:length(segments)
    segment = segments{i};
    label = segment_labels{i};
    
    % Calculate bandpower for 40-80 Hz and 10-30 Hz bands
    bandpower_40_80Hz(i) = bandpower(segment, fs, [40 80]);
    bandpower_10_30Hz(i) = bandpower(segment, fs, [10 30]);
    
    median_freq(i) = medfreq(segment, fs);
    mean_freq(i) = meanfreq(segment, fs);
    
    if strcmp(label, 'VFIB')
        class_labels(i) = 1;  % 1 for VFIB
    elseif strcmp(label, 'Norm1')
        class_labels(i) = 0;  % 0 for Normal
    elseif strcmp(label, 'Norm2')
        class_labels(i) = 0;  % 0 for Normal
    else
        class_labels(i)= 2;
    end
end

figure;

% Bandpower 40-80 Hz
subplot(2, 2, 1);
histogram(bandpower_40_80Hz(class_labels == 0),10, 'FaceColor', 'b'); hold on;
histogram(bandpower_40_80Hz(class_labels == 1),10, 'FaceColor', 'r');
title('Bandpower 40-80 Hz');
xlabel('Power'); ylabel('Count');
legend('Normal', 'VFIB');

% Bandpower 10-30 Hz
subplot(2, 2, 2);
histogram(bandpower_10_30Hz(class_labels == 0),10, 'FaceColor', 'b'); hold on;
histogram(bandpower_10_30Hz(class_labels == 1),10, 'FaceColor', 'r');
title('Bandpower 10-30 Hz');
xlabel('Power'); ylabel('Count');
legend('Normal', 'VFIB');

% Median Frequency
subplot(2, 2, 3);
histogram(median_freq(class_labels == 0),10, 'FaceColor', 'b'); hold on;
histogram(median_freq(class_labels == 1),10, 'FaceColor', 'r');
title('Median Frequency');
xlabel('Frequency (Hz)'); ylabel('Count');
legend('Normal', 'VFIB');

% Mean Frequency
subplot(2, 2, 4);
histogram(mean_freq(class_labels == 0),10, 'FaceColor', 'b'); hold on;
histogram(mean_freq(class_labels == 1),10, 'FaceColor', 'r');
title('Mean Frequency');
xlabel('Frequency (Hz)'); ylabel('Count');
legend('Normal', 'VFIB');



%% Q7
% Define sampling frequency
Fs = 250;

% Call the va_detect function to classify the segments
[alarm, t] = va_detect(ECG1, Fs);

% Define true labels (segment_labels) for VFIB and Normal segments
Norm1 = 1:10710;
Norm2 = 11211:11441;
VFIB = 61288:75000;

% Assign true labels for each segment
segment_labels = zeros(length(t), 1);  % Initialize with zeros (Normal = 0)
for i = 1:length(t)
    segment_end_sample = t(i) * Fs;  % Convert time to sample
    if ismember(segment_end_sample, VFIB)
        segment_labels(i) = 1;  % VFIB = 1
    elseif ismember(segment_end_sample, [Norm1, Norm2])
        segment_labels(i) = 0;  % Normal = 0
    else
        segment_labels(i) = 2;  % Other classes (excluded from metrics)
    end
end

% Filter out segments with label 2 (not classified as VFIB or Normal)
valid_indices = segment_labels ~= 2;
true_labels = segment_labels(valid_indices);
predicted_labels = alarm(valid_indices);

% Compute confusion matrix
confusion_mat = confusionmat(true_labels, predicted_labels);
disp('Confusion Matrix:');
disp(confusion_mat);

% Extract confusion matrix values
TP = confusion_mat(2, 2);  % True Positives (VFIB correctly classified)
TN = confusion_mat(1, 1);  % True Negatives (Normal correctly classified)
FP = confusion_mat(1, 2);  % False Positives (Normal misclassified as VFIB)
FN = confusion_mat(2, 1);  % False Negatives (VFIB misclassified as Normal)

% Calculate performance metrics
sensitivity = TP / (TP + FN);  % Sensitivity = Recall = TP / (TP + FN)
specificity = TN / (TN + FP);  % Specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN);  % Accuracy

% Display metrics
disp(['Sensitivity: ', num2str(sensitivity)]);
disp(['Specificity: ', num2str(specificity)]);
disp(['Accuracy: ', num2str(accuracy)]);






%%  New features

window_length = 10 * fs;
overlap = 5 * fs;

start_idx = 1:overlap:length(data)-window_length;
epoch_data = zeros(length(start_idx), 10*fs);
labels = zeros(length(start_idx), 1);
for i = 1:length(start_idx)
    end_idx = start_idx(i)+window_length-1;
    epoch_data(i, :) = data(start_idx(i):end_idx);
    
    if end_idx<=10711
        labels(i)=1 ;%normal
    elseif start_idx(i)>10711 &&  end_idx<=11211
        labels(i)=3 ;%VT
    elseif start_idx(i)>11211 &&  end_idx<=11442
        labels(i)=1 ;%normal
    elseif start_idx(i)>11442 &&  end_idx<=59711
        labels(i)=3 ;%VT
    elseif start_idx(i)>59711 &&  end_idx<=61288
        labels(i)=4 ;%noise
    elseif start_idx(i)>=61288
        labels(i)=2 ;%VFIB
    else
        labels(i)=0 ;%None
    end
end

%% Section d: Frequency Features

n_features = 5; % mean_freq, med_freq, bandpower(0, 20), bandpower(20, 40)
features = zeros(length(start_idx), 5);
band1 = [0 40];
band2 = [40 80];
band3 = [80 120];

for i = 1:length(start_idx)
    sig = epoch_data(i, :);
    features(i, 1) = meanfreq(sig, fs);
    features(i, 2) = medfreq(sig, fs);
    features(i, 3) = bandpower(sig, fs, band1);
    features(i, 4) = bandpower(sig, fs, band2);
    features(i, 5) = bandpower(sig, fs, band3);
end

%% Section e

feature_label = ["meanfreq", "medfreq", "band 0-40", "bandpower 40-80", "band 80-120"];
figure;
for i=1:n_features
    subplot(3,2,i)
    normal_data = features(labels == 1, i);
    histogram(normal_data, 10); hold on;
    VFIB_data = features(labels == 2, i);
    histogram(VFIB_data, 10); hold on;
    title(feature_label(i))
end
[alarm_medfreq ,t1] = va_detectSuper(data,fs, "medfreq");
[alarm_bandpower4080 ,t2] = va_detectSuper(data,fs, "bandpower40");

%% Confusion


idx = [find(labels == 1); find(labels == 2)];
true_labels = (labels(idx) - 1)';
pred_labels1 = alarm_medfreq(idx)';
[c1,cm1,ind1,per1] = confusion(true_labels, pred_labels1);

pred_labels2 = alarm_bandpower4080(idx)';
[c2,cm2,ind2,per2] = confusion(true_labels, pred_labels2);


disp('Confusion Matrix:');
disp(cm1);

disp('Feature = bandpower at 40-80 Hz')
acc_bandpower = sum(diag(cm2)) / sum(cm2, 'all');
sens_bandpower = cm2(1,1)/(cm2(1,1)+cm2(2,1));
spec_bandpower = cm2(2,2)/(cm2(2,2)+cm2(1,2));



% Display metrics
disp(['Sensitivity: ', num2str(sens_bandpower)]);
disp(['Specificity: ', num2str(spec_bandpower)]);
disp(['Accuracy: ', num2str(acc_bandpower)]);

%% 
% Initialize feature storage
max_amplitude = [];
min_amplitude = [];
peak_to_peak = [];
zero_crossings = [];
variance_amplitude = [];
mean_R_amplitude = [];

% Define Normal and VFIB labels for histogram plotting
norm_indices = find(segment_labels == 0);  % Indices of Normal segments
vfib_indices = find(segment_labels == 1);  % Indices of VFIB segments

% Process each segment
for i = 1:length(segments)
    segment = segments{i};
    
    % Calculate maximum and minimum amplitude
    max_amplitude(i) = max(segment);
    min_amplitude(i) = min(segment);
    
    % Calculate peak-to-peak value
    peak_to_peak(i) = max_amplitude(i) - min_amplitude(i);
    
    % Count zero crossings
    zero_crossings(i) = sum(abs(diff(segment > 0)));
    
    % Calculate variance of amplitudes
    variance_amplitude(i) = var(segment);
    
    % Calculate mean amplitude of R peaks using findpeaks
    [pks, locs] = findpeaks(segment);        % Find local maxima
    [valls, vall_locs] = findpeaks(segment); % Find local minima
    valls = valls;                          % Convert minima back to positive values
    
    % Mean amplitude of R peaks (average of local max and min amplitudes)
    mean_R_amplitude(i) = mean([pks; valls]);
end

% Plot histograms for each feature (compare Norm and VFIB)
figure;

% Maximum amplitude
subplot(3, 2, 1);
histogram(max_amplitude(norm_indices),10, 'FaceColor', 'b'); hold on;
histogram(max_amplitude(vfib_indices),10, 'FaceColor', 'r');
title('Maximum Amplitude');
xlabel('Amplitude'); ylabel('Count');
legend('Normal', 'VFIB');

% Minimum amplitude
subplot(3, 2, 2);
histogram(min_amplitude(norm_indices),10, 'FaceColor', 'b'); hold on;
histogram(min_amplitude(vfib_indices),10, 'FaceColor', 'r');
title('Minimum Amplitude');
xlabel('Amplitude'); ylabel('Count');
legend('Normal', 'VFIB');

% Peak-to-Peak
subplot(3, 2, 3);
histogram(peak_to_peak(norm_indices),10, 'FaceColor', 'b'); hold on;
histogram(peak_to_peak(vfib_indices),10, 'FaceColor', 'r');
title('Peak-to-Peak Amplitude');
xlabel('Amplitude'); ylabel('Count');
legend('Normal', 'VFIB');

% Zero Crossings
subplot(3, 2, 4);
histogram(zero_crossings(norm_indices),10, 'FaceColor', 'b'); hold on;
histogram(zero_crossings(vfib_indices),10, 'FaceColor', 'r');
title('Zero Crossings');
xlabel('Count'); ylabel('Count');
legend('Normal', 'VFIB');

% Variance of amplitudes
subplot(3, 2, 5);
histogram(variance_amplitude(norm_indices),10, 'FaceColor', 'b'); hold on;
histogram(variance_amplitude(vfib_indices),10, 'FaceColor', 'r');
title('Variance of Amplitudes');
xlabel('Variance'); ylabel('Count');
legend('Normal', 'VFIB');

% Mean amplitude of R peaks
subplot(3, 2, 6);
histogram(mean_R_amplitude(norm_indices),10, 'FaceColor', 'b'); hold on;
histogram(mean_R_amplitude(vfib_indices),10, 'FaceColor', 'r');
title('Mean Amplitude of R Peaks');
xlabel('Amplitude'); ylabel('Count');
legend('Normal', 'VFIB');



%% 
% Define sampling frequency
Fs = 250;

% Call the va_detect2 function to classify the segments
[alarm, t] = va_detect2(ECG1, Fs);

% Define true labels (segment_labels) for VFIB and Normal segments
Norm1 = 1:10710;
Norm2 = 11211:11441;
VFIB = 61288:75000;

% Assign true labels for each segment
segment_labels = zeros(length(t), 1);  % Initialize with zeros (Normal = 0)
for i = 1:length(t)
    segment_end_sample = t(i) * Fs;  % Convert time to sample
    if any(segment_end_sample >= VFIB(1) & segment_end_sample <= VFIB(end))
        segment_labels(i) = 1;  % VFIB = 1
    elseif any(segment_end_sample >= Norm1(1) & segment_end_sample <= Norm1(end)) || ...
           any(segment_end_sample >= Norm2(1) & segment_end_sample <= Norm2(end))
        segment_labels(i) = 0;  % Normal = 0
    else
        segment_labels(i) = 2;  % Other classes (excluded from metrics)
    end
end

% Filter out segments with label 2 (not classified as VFIB or Normal)
valid_indices = segment_labels ~= 2;
true_labels = segment_labels(valid_indices);
predicted_labels = alarm(valid_indices);

% Compute confusion matrix
confusion_mat = confusionmat(true_labels, predicted_labels);
disp('Confusion Matrix:');
disp(confusion_mat);

% Extract confusion matrix values
TP = confusion_mat(2, 2);  % True Positives (VFIB correctly classified)
TN = confusion_mat(1, 1);  % True Negatives (Normal correctly classified)
FP = confusion_mat(1, 2);  % False Positives (Normal misclassified as VFIB)
FN = confusion_mat(2, 1);  % False Negatives (VFIB misclassified as Normal)

% Calculate performance metrics
sensitivity = TP / (TP + FN);  % Sensitivity = Recall = TP / (TP + FN)
specificity = TN / (TN + FP);  % Specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN);  % Accuracy

% Display metrics
disp(['Sensitivity: ', num2str(sensitivity)]);
disp(['Specificity: ', num2str(specificity)]);
disp(['Accuracy: ', num2str(accuracy)]);

%% -------------------------------------------------

 clear; close all;

addpath('../Data/Lab 5_data/')

%% Load data

data = load('Lab 5_data\n_424.mat').n_424;
fs = 256;

normal_1_n424 = data(1:27248);
abnormal_1_n424 = data(27249:53673);
abnormal_2_n424 = data(53674:55134); 
abnormal_3_n424 = data(55135:end);

%% Pwelch

[p_n_1, f_n_1] = pwelch(normal_1_n424, [], [], [], fs);
[p_ab_1, f_ab_1] = pwelch(abnormal_1_n424, [], [], [], fs);
[p_ab_2, f_ab_2] = pwelch(abnormal_2_n424, [], [], [], fs);
[p_ab_3, f_ab_3] = pwelch(abnormal_3_n424, [], [], [], fs);

% visualization

figure;
subplot(2,2,1)
plot(f_n_1, 10*log(p_n_1))
title('Normal data 1')
xlabel('frequency (Hz)')
ylabel('power (dB)')
ylim([-60 100])

subplot(2,2,2)
plot(f_ab_1, 10*log(p_ab_1))
title('Abnormal data 2')
xlabel('frequency (Hz)')
ylabel('power (dB)')
ylim([-60 100])

subplot(2,2,3)
plot(f_ab_2, 10*log(p_ab_2))
title('Abnormal data 2')
xlabel('frequency (Hz)')
ylabel('power (dB)')
ylim([-60 100])

subplot(2,2,4)
plot(f_ab_3, 10*log(p_ab_3))
title('Abnormal data 3')
xlabel('frequency (Hz)')
ylabel('power (dB)')
ylim([-60 100])

%% Time plot (Section b)

figure;
subplot(2,2,1)
t = (1:size(normal_1_n424,1)) / fs - 1/fs;
plot(t, normal_1_n424)
title('Normal 1')
xlabel('time (s)')
ylabel('amplitude (v)')

subplot(2,2,2)
t = (1:size(abnormal_1_n424,1)) / fs - 1/fs;
plot(t, abnormal_1_n424)
title('Abnormal 1')
xlabel('time (s)')
ylabel('amplitude (v)')

subplot(2,2,3)
t = (1:size(abnormal_2_n424,1)) / fs - 1/fs;
plot(t, abnormal_2_n424)
title('Abnormal 2')
xlabel('time (s)')
ylabel('amplitude (v)')

subplot(2,2,4)
t = (1:size(abnormal_3_n424,1)) / fs - 1/fs;
plot(t, abnormal_3_n424)
title('Abnormal 3')
xlabel('time (s)')
ylabel('amplitude (v)')

%% Section (c) Labeling

data = load('n_424.mat').n_424;
window_length = 10 * fs;
overlap = 5 * fs;

start_idx = 1:overlap:length(data)-window_length;
epoch_data = zeros(length(start_idx), 10*fs);
labels = zeros(length(start_idx), 1);
for i = 1:length(start_idx)
    end_idx = start_idx(i)+window_length-1;
    epoch_data(i, :) = data(start_idx(i):end_idx);
    
    if end_idx<=27248
        labels(i)=1 ;%normal
    elseif start_idx(i)>27248 &&  end_idx<=53673
        labels(i)=2 ;%VFIB
    elseif start_idx(i)>53673 &&  end_idx<=55134
        labels(i)=4 ;%Noise
    elseif start_idx(i)>55135 &&  end_idx<=58288
        labels(i)=3 ;%NOD
    else
        labels(i)=0 ;%None
    end
end

%% Section d: Frequency Features

n_features = 4; % mean_freq, med_freq, bandpower(0, 20), bandpower(20, 40)
features = zeros(length(start_idx), 5);
band1 = [0 30];
band2 = [30 40];

for i = 1:length(start_idx)
    sig = epoch_data(i, :);
    features(i, 1) = meanfreq(sig, fs);
    features(i, 2) = medfreq(sig, fs);
    features(i, 3) = bandpower(sig, fs, band1);
    features(i, 4) = bandpower(sig, fs, band2);
end

%% Section e




feature_label = ["meanfreq Feature", "medfreq Feature", "bandPower 0-30 Feature", "bandpower 30-40 Feature", "band 100-120"];
figure;
for i=1:n_features
    subplot(2,2,i)
    normal_data = features(labels == 1, i);
    histogram(normal_data,10, 'FaceColor', 'b'); hold on;
    VFIB_data = features(labels == 2, i);
    histogram(VFIB_data,10, 'FaceColor', 'r'); hold on;
    title(feature_label(i))
end


%% section f



[alarm_meanfreq ,t1] = va_detectSuper(data,fs, "meanfreq");
[alarm_medfreq ,t2] = va_detectSuper(data,fs, "medfreq");

%% Confusion

idx = [find(labels == 1); find(labels == 2)];
true_labels = (labels(idx) - 1)';
pred_labels1 = alarm_meanfreq(idx)';
[c1,cm1,ind1,per1] = confusion(true_labels, pred_labels1);

pred_labels2 = alarm_medfreq(idx)';
[c2,cm2,ind2,per2] = confusion(true_labels, pred_labels2);

%% 
disp('Confusion Matrix:');
disp(cm1);

disp('Feature = Mean Frequency')
acc_bandpower = sum(diag(cm1)) / sum(cm1, 'all');
sens_bandpower = cm1(1,1)/(cm1(1,1)+cm1(2,1));
spec_bandpower = cm1(2,2)/(cm1(2,2)+cm1(1,2));



% Display metrics
disp(['Sensitivity: ', num2str(sens_bandpower)]);
disp(['Specificity: ', num2str(spec_bandpower)]);
disp(['Accuracy: ', num2str(acc_bandpower)]);


disp('Confusion Matrix:');
disp(cm2);

disp('Feature = Median Frequency')
acc_bandpower = sum(diag(cm2)) / sum(cm2, 'all');
sens_bandpower = cm2(1,1)/(cm2(1,1)+cm2(2,1));
spec_bandpower = cm2(2,2)/(cm2(2,2)+cm2(1,2));



% Display metrics
disp(['Sensitivity: ', num2str(sens_bandpower)]);
disp(['Specificity: ', num2str(spec_bandpower)]);
disp(['Accuracy: ', num2str(acc_bandpower)]);



%% Morphologic features



n_features_demog = 6; % max, min, peaktopeak, findpeaks, zeros, var
features_demog = zeros(length(start_idx), n_features_demog);

for i = 1:length(start_idx)
    sig = epoch_data(i, :);
    features_demog(i, 1) = max(sig);
    features_demog(i, 2) = min(sig);
    features_demog(i, 3) = max(sig) - min(sig);
    features_demog(i, 4) = mean(findpeaks(sig));
    features_demog(i, 5) = sum(sig == 0);
    features_demog(i, 6) = var(sig);
end

%%
feature_label = ["Maximum Amplitude", "Minimum Amplitude", "Peak-to-Peak Amplitude", "Mean Amplitude of R Peaks", "Zero Crossings", "Variance of Amplitudes"];
figure;
for i=1:n_features_demog
    subplot(3,2,i)

    normal_data = features_demog(labels == 1, i);
    histogram(normal_data,5, 'FaceColor', 'b'); hold on;
    VFIB_data = features_demog(labels == 2, i);
    histogram(VFIB_data,5, 'FaceColor', 'r'); hold on;
    title(feature_label(i))
end

% result: max and R peaks avg

%%

[alarm_max ,t1] = va_detectSuper(data,fs, "zeros");
[alarm_Rpeak ,t2] = va_detectSuper(data,fs, "peak-to-peak");

pred_labels1 = alarm_max(idx)';
[c1,cm1,ind1,per1] = confusion(true_labels, pred_labels1);

pred_labels2 = alarm_Rpeak(idx)';
[c2,cm2,ind2,per2] = confusion(true_labels, pred_labels2);

disp('Confusion Matrix:');
disp(cm1);

disp('Feature = Zero Crossings')
acc_bandpower = sum(diag(cm1)) / sum(cm1, 'all');
sens_bandpower = cm1(1,1)/(cm1(1,1)+cm1(2,1));
spec_bandpower = cm1(2,2)/(cm1(2,2)+cm1(1,2));



% Display metrics
disp(['Sensitivity: ', num2str(sens_bandpower)]);
disp(['Specificity: ', num2str(spec_bandpower)]);
disp(['Accuracy: ', num2str(acc_bandpower)]);


disp('Confusion Matrix:');
disp(cm2);

disp('Feature = Peak-Peak Amplitude')
acc_bandpower = sum(diag(cm2)) / sum(cm2, 'all');
sens_bandpower = cm2(1,1)/(cm2(1,1)+cm2(2,1));
spec_bandpower = cm2(2,2)/(cm2(2,2)+cm2(1,2));



% Display metrics
disp(['Sensitivity: ', num2str(sens_bandpower)]);
disp(['Specificity: ', num2str(spec_bandpower)]);
disp(['Accuracy: ', num2str(acc_bandpower)]);


%% 

[alarm_Rpeak ,t2] = va_detectSuper(data,fs, "R-peak-avg");

pred_labels2 = alarm_Rpeak(idx)';
[c2,cm2,ind2,per2] = confusion(true_labels, pred_labels);

disp('Confusion Matrix:');
disp(cm2);

disp('Feature = Peak-Peak Amplitude')
acc_bandpower = sum(diag(cm2)) / sum(cm2, 'all');
sens_bandpower = cm2(1,1)/(cm2(1,1)+cm2(2,1));
spec_bandpower = cm2(2,2)/(cm2(2,2)+cm2(1,2));



% Display metrics
disp(['Sensitivity: ', num2str(sens_bandpower)]);
disp(['Specificity: ', num2str(spec_bandpower)]);
disp(['Accuracy: ', num2str(acc_bandpower)]);





%% part Final

data = load('n_426.mat').n_426;
fs = 256;
window_length = 10 * fs;
overlap = 5 * fs;

start_idx = 1:overlap:length(data)-window_length;
epoch_data = zeros(length(start_idx), 10*fs);
labels = zeros(length(start_idx), 1);
for i = 1:length(start_idx)
    end_idx = start_idx(i)+window_length-1;
    epoch_data(i, :) = data(start_idx(i):end_idx);
    
    if end_idx<= 26432
        labels(i)=1 ;%normal
    elseif start_idx(i)> 26432
        labels(i)=2 ;%VF
    else
        labels(i)=0 ;%None
    end
end


clc;

[alarm_Rpeak ,t2] = va_detectSuper(data,fs, "R-peak-avg");

idx = [find(labels == 1); find(labels == 2)];
true_labels = (labels(idx) - 1)';
pred_labels2 = alarm_Rpeak(idx)';
[c2,cm2,ind2,per2] = confusion(true_labels, pred_labels2);

disp('Confusion Matrix:');
disp(cm2);

disp('Feature = Peak-Peak Amplitude')
acc_bandpower = sum(diag(cm2)) / sum(cm2, 'all');
sens_bandpower = cm2(1,1)/(cm2(1,1)+cm2(2,1));
spec_bandpower = cm2(2,2)/(cm2(2,2)+cm2(1,2));



% Display metrics
disp(['Sensitivity: ', num2str(sens_bandpower)]);
disp(['Specificity: ', num2str(spec_bandpower)]);
disp(['Accuracy: ', num2str(acc_bandpower)]);