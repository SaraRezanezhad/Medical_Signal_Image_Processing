%  Sara Rezanezhad - Mohamad Hosein Faramarzi - Ali Khosravipour 
% Load the EMG signal data  
load("EMG_sig.mat");  

%% Part 1: Time Domain Plots  
% Healthy Signal  
numSamplesHealthy = length(emg_healthym);  
timeVectorHealthy = (0:1/fs:numSamplesHealthy/fs-1/fs);  
subplot(3, 1, 1);  
plot(timeVectorHealthy, emg_healthym);  
xlim([0, numSamplesHealthy/fs-1/fs]);  
ylabel('Healthy Signal');  
grid minor;  
title('Healthy EMG Signal');  

% Myopathy Signal  
numSamplesMyopathy = length(emg_myopathym);  
timeVectorMyopathy = (0:1/fs:numSamplesMyopathy/fs-1/fs);  
subplot(3, 1, 2);  
plot(timeVectorMyopathy, emg_myopathym);  
xlim([0, numSamplesMyopathy/fs-1/fs]);  
ylabel('Myopathy Signal');  
grid minor;  
title('Myopathy EMG Signal');  

% Neuropathy Signal  
numSamplesNeuropathy = length(emg_neuropathym);  
timeVectorNeuropathy = (0:1/fs:numSamplesNeuropathy/fs-1/fs);  
subplot(3, 1, 3);  
plot(timeVectorNeuropathy, emg_neuropathym);  
xlim([0, numSamplesNeuropathy/fs-1/fs]);  
ylabel('Neuropathy Signal');  
grid minor;  
title('Neuropathy EMG Signal');  

%% Part 2: Frequency Domain Plots  
windowLength = 128;  % Length of the Hamming window  
hammingWindow = hamming(windowLength);  
overlapSamples = 64;  % Number of overlapping samples  

figure;  

% Frequency Domain for Healthy Signal  
frequencyVectorHealthy = (-numSamplesHealthy/2:numSamplesHealthy/2-1) .* fs / numSamplesHealthy;  
subplot(3, 1, 1);  
plot(frequencyVectorHealthy, fftshift(abs(fft(emg_healthym))));  
xlabel('Frequency (Hz)');  
ylabel('Magnitude');  
grid minor;  
title('Healthy Signal DFT');  

% Frequency Domain for Myopathy Signal  
frequencyVectorMyopathy = (-numSamplesMyopathy/2:numSamplesMyopathy/2-1) .* fs / numSamplesMyopathy;  
subplot(3, 1, 2);  
plot(frequencyVectorMyopathy, fftshift(abs(fft(emg_myopathym))));  
xlabel('Frequency (Hz)');  
ylabel('Magnitude');  
grid minor;  
title('Myopathy Signal DFT');  

% Frequency Domain for Neuropathy Signal  
frequencyVectorNeuropathy = (-numSamplesNeuropathy/2:numSamplesNeuropathy/2-1) .* fs / numSamplesNeuropathy;  
subplot(3, 1, 3);  
plot(frequencyVectorNeuropathy, fftshift(abs(fft(emg_neuropathym))));  
xlabel('Frequency (Hz)');  
ylabel('Magnitude');  
grid minor;  
title('Neuropathy Signal DFT');  

%% Part 3: Spectrogram Plots  
figure;  

% Spectrogram for Healthy Signal  
subplot(3, 1, 1);  
spectrogram(emg_healthym, hammingWindow, overlapSamples, numSamplesHealthy, fs, 'yaxis');  
title('Healthy Signal Spectrogram');  
grid minor;  

% Spectrogram for Myopathy Signal  
subplot(3, 1, 2);  
spectrogram(emg_myopathym, hammingWindow, overlapSamples, numSamplesMyopathy, fs, 'yaxis');  
title('Myopathy Signal Spectrogram');  
grid minor;  

% Spectrogram for Neuropathy Signal  
subplot(3, 1, 3);  
spectrogram(emg_neuropathym, hammingWindow, overlapSamples, numSamplesNeuropathy, fs, 'yaxis');  
title('Neuropathy Signal Spectrogram');  
grid minor;