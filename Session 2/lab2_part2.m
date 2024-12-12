%% Ali Khosravipour - 99101502 // Sara Rezanejad - 99101643 // MohamadHosein Faramarzi - 99104095
clc;
clear;
load("Electrodes.mat")
sig1 = load("NewData1.mat").EEG_Sig;
sig2 = load("NewData2.mat").EEG_Sig;
fs = 250;
xCoords = Electrodes.X;
yCoords = Electrodes.Y;
chLabels = Electrodes.labels;
%% 2.1
disp_eeg(sig1,[], 250, Electrodes.labels, 'Signal 1');
disp_eeg(sig2,[], 250, Electrodes.labels, 'Signal 2');
%% 2.2
figure;
for ch = 1:size(sig1, 1)
    subplot(3,7,ch);
    [pxx, f] = pwelch(sig1(ch, :), [], [], [], fs);
    plot(f, 10*log10(pxx), 'r');
    title(['Channel ', num2str(ch)]);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0 fs/2]);  
    grid on;
end
figure;
for ch = 1:size(sig2, 1)
    subplot(3,7,ch);
    [pxx, f] = pwelch(sig2(ch, :), [], [], [], fs);
    plot(f, 10*log10(pxx), 'b');
    title(['Channel ', num2str(ch)]);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0 fs/2]);  
    grid on;
end
%% 2.3
% W = pinv(F)
[F1,W1,K1] = COM2R(sig1, 21);
sig1_component = W1*sig1;
[F2,W2,K2] = COM2R(sig2, 21);
sig2_component = W2*sig2;
%% 2.4
disp_eeg(sig1_component,[], 250, Electrodes.labels, 'Components of Signal 1');
disp_eeg(sig2_component,[], 250, Electrodes.labels, 'Components of Signal 2');
%%
figure;
for ch = 1:size(sig1_component, 1)
    subplot(3,7,ch);
    [pxx, f] = pwelch(sig1_component(ch, :), [], [], [], fs);
    plot(f, 10*log10(pxx), 'r');
    title(['Component ', num2str(ch)]);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0 fs/2]);  
    xticks(0:20:fs/2); 
    grid on;
end
figure;
for ch = 1:size(sig2_component, 1)
    subplot(3,7,ch);
    [pxx, f] = pwelch(sig2_component(ch, :), [], [], [], fs);
    plot(f, 10*log10(pxx), 'b');
    title(['Component ', num2str(ch)]);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0 fs/2]);  
    xticks(0:20:fs/2);
    grid on;
end
%%
figure;
for ch = 1:size(F1, 1)
    subplot(3,7,ch);
    plottopomap(xCoords,yCoords,chLabels,F1(:, ch), ch) 
end
figure;
for ch = 1:size(F2, 1)
    subplot(3,7,ch);
    plottopomap(xCoords,yCoords,chLabels,F2(:, ch), ch) 
end
%% 2.5 Removing Unwanted Components
selectedSources1 = sig1_component;
selectedSources1([1, 4, 7, 11, 18, 20, 9, 10, 12, 6], :) = [];
W1(: , [1, 4, 7, 11, 18, 20, 9, 10, 12, 6]) = [];
sig1_denoised = W1 * selectedSources1;
disp_eeg(sig1,[], 250, Electrodes.labels, 'Signal 1');
disp_eeg(sig1_denoised,[], 250, Electrodes.labels, 'Signal 1 Denoised');

%%
selectedSources2 = sig2_component;
% selectedSources2([1:8 , 11, 12, 15:17], :) = [];
% W2(: , [1:8 , 11, 12, 15:17]) = [];
selectedSources2([1:13 , 15:17], :) = [];
W2(: , [1:13 , 15:17]) = [];
sig2_denoised = W2 * selectedSources2;
disp_eeg(sig2,[], 250, Electrodes.labels, 'Signal 2');
disp_eeg(sig2_denoised,[], 250, Electrodes.labels, 'Signal 2 Denoised');



