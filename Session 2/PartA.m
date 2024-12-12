% PART A - LAB2
% // Sara Rezanejad - 99101643 // MohamadHosein Faramarzi - 99104095// Ali Khosravipour - 99101502 
%%
clear all; close all;
%%
load("X_org.mat");
load("X_noise.mat");
load("Electrodes.mat");
%% Q1 Plot X_orig
offset = max(max(abs(X_org)))/3 ;
feq = 250 ;
ElecName = Electrodes.labels;
disp_eeg(X_org,offset,feq,ElecName,'All Channels Time Series') ;

%% Q2 Plot X_noise
offset = max(max(abs(X_org)))/3 ;
feq = 250 ;
ElecName = Electrodes.labels;
disp_eeg(X_noise,offset,feq,ElecName,'All Channels Time Series Noise') ;
%% Q3 Add noise to X_orig with -15 and -5 dB values of SNR
% Compute the power of the original signal
P_signal = sum(sum(X_org.^2)); 
P_noise= sum(sum(X_noise.^2));

% SNR values (in dB)
SNR_values = [-15,-5];

for i = 1:length(SNR_values)
    SNR_dB = SNR_values(i);
    
    sigma = sqrt((P_signal/P_noise)*(10^(-SNR_dB/10)));
    
    % Add the noise to the original signal
    X_noisy(i,:,:) = X_org + sigma.*X_noise;
    
    fprintf('Added noise with SNR = %d dB\n', SNR_dB);
    
end

feq = 250 ;
ElecName = Electrodes.labels;
X_NoisyDisp=squeeze(X_noisy(1,:,:));
offset = max(max(abs(X_NoisyDisp)))/3 ;

disp_eeg(X_NoisyDisp,offset,feq,ElecName,'All Channels Time Series Noise') ;


%% Q4 COM2R ICA 
Pest = 32;  
% Apply COM2 ICA to extract the sources
[F, W, K] = COM2R(X_NoisyDisp, Pest);

% Extract the independent components (sources)
Z_ICA = W * X_NoisyDisp;  % Reconstructed sources

offset = max(max(abs(Z_ICA)))/2 ;

disp_eeg(Z_ICA,offset,feq,[],'ICA All Components Time Series Noise') ;


%% Q5 and Q6
% Spiky_Indices=[2,5,12];
Spiky_Indices=[15,18,19,20];

Z_ICA_Spiky=Z_ICA(Spiky_Indices,:);

F_Spiky=F(:,Spiky_Indices);


% Back to sensor 
X_den=F_Spiky*Z_ICA_Spiky;
offset = max(max(abs(X_den)))/2 ;

disp_eeg(X_den,offset,feq,ElecName,"Recunstructed Signal after ICA -15dB")
% disp_eeg(X_den,offset,feq,ElecName,"Recunstructed Signal after ICA -5dB")

%% Q7

N = numel(X_org); % Total number of elements
mse = sum((X_org(:) - X_den(:)).^2) / N;
rmse = sqrt(mse); 
rms_org = sqrt(sum(X_org(:).^2) / N);

% Compute RRMSE
RRMSE = rmse / rms_org;

% Display the result
fprintf('The RRMSE between the original and reconstructed signals is: %.6f\n', RRMSE);

%% 

% Assuming X_den, X_org, and X_NoisyDisp are your 32x10240 matrices

channels = [13, 24]; % Channels to be plotted
t = 1:10240; % Time or sample points (assuming 10240 samples)

%% Plot for Channel 13
figure; % Open a new figure for Channel 13
subplot(3, 1, 1); % Create a subplot for the Original signal
plot(t, X_org(13, :), 'Color', [0.1, 0.6, 0.8], 'LineWidth', 2); % Original signal (cyan)
xlabel('Time or Sample Points'); 
ylabel('Amplitude');
title('Channel 13 - Original Signal');
grid on;

subplot(3, 1, 2); % Create a subplot for the Denoised signal
plot(t, X_den(13, :), 'Color', [0.85, 0.33, 0.1], 'LineWidth', 2); % Denoised signal (orange)
xlabel('Time or Sample Points'); 
ylabel('Amplitude');
title('Channel 13 - Denoised Signal');
grid on;

subplot(3, 1, 3); % Create a subplot for the Noisy signal
plot(t, X_NoisyDisp(13, :), 'Color', [0.47, 0.67, 0.19], 'LineWidth', 2); % Noisy signal (olive green)
xlabel('Time or Sample Points'); 
ylabel('Amplitude');
title('Channel 13 - Noisy Signal');
grid on;

%% Plot for Channel 24
figure; % Open a new figure for Channel 24
subplot(3, 1, 1); % Create a subplot for the Original signal
plot(t, X_org(24, :), 'Color', [0.1, 0.6, 0.8], 'LineWidth', 2); % Original signal (cyan)
xlabel('Time or Sample Points'); 
ylabel('Amplitude');
title('Channel 24 - Original Signal');
grid on;

subplot(3, 1, 2); % Create a subplot for the Denoised signal
plot(t, X_den(24, :), 'Color', [0.85, 0.33, 0.1], 'LineWidth', 2); % Denoised signal (orange)
xlabel('Time or Sample Points'); 
ylabel('Amplitude');
title('Channel 24 - Denoised Signal');
grid on;

subplot(3, 1, 3); % Create a subplot for the Noisy signal
plot(t, X_NoisyDisp(24, :), 'Color', [0.47, 0.67, 0.19], 'LineWidth', 2); % Noisy signal (olive green)
xlabel('Time or Sample Points'); 
ylabel('Amplitude');
title('Channel 24 - Noisy Signal');
grid on;



