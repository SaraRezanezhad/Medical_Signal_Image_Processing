%  Sara Rezanezhad - Mohamad Hosein Faramarzi - Ali Khosravipour 
% Load the EOG signal data  
load("EOG_sig.mat");  

%% Part 1: Time Domain Plots  
% Define parameters  
numSamples = length(Sig);  
timeVector = (0:1/fs:numSamples/fs-1/fs);  
frequencyVector = (-numSamples/2:numSamples/2-1) .* fs / numSamples;  
windowLength = 128;  
hammingWindow = hamming(windowLength);  
overlapSamples = 64;  

figure;  
subplot(2, 1, 1);  
plot(timeVector, Sig(1, :));  
xlim([0, 60]);  
xlabel('Time (seconds)');  
ylabel('Right Eye Signal');  
grid minor;  
title('Right Eye EOG Signal');  

subplot(2, 1, 2);  
plot(timeVector, Sig(2, :));  
xlim([0, 60]);  
xlabel('Time (seconds)');  
ylabel('Left Eye Signal');  
grid minor;  
title('Left Eye EOG Signal');  

%% Part 2.A: Frequency Domain Plots  
figure;  
subplot(2, 1, 1);  
plot(frequencyVector, fftshift(abs(fft(Sig(1, :)))));  
xlabel('Frequency (Hz)');  
ylabel('Magnitude');  
grid minor;  
title('Right Eye Signal DFT');  

subplot(2, 1, 2);  
plot(frequencyVector, fftshift(abs(fft(Sig(2, :)))));  
xlabel('Frequency (Hz)');  
ylabel('Magnitude');  
grid minor;  
title('Left Eye Signal DFT');  

%% Part 2.B: Spectrogram Plots  
figure;  
subplot(2, 1, 1);  
spectrogram(Sig(1, :), hammingWindow, overlapSamples, numSamples, fs, 'yaxis');  
title('Right Eye Signal Spectrogram');  
caxis([-50,50]);

grid minor;  

subplot(2, 1, 2);  
spectrogram(Sig(2, :), hammingWindow, overlapSamples, numSamples, fs, 'yaxis');  
title('Left Eye Signal Spectrogram');  
caxis([-50,50]);

grid minor;