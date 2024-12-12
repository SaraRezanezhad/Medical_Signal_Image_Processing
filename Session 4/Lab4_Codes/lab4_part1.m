%% Ali Khosravipour - 99101502 // Sara Rezanejad - 99101643 // MohamadHosein Faramarzi - 99104095
% Lab4 - Part A EEG_ERP
clear all; close all; clc;
load("ERP_EEG.mat");
trial_numbers=2550;
sampling_freq=240;
%% Q1
trial_numbers=2550;
sampling_freq=240;
% Trial counts we want to test
N_values = 100:100:2500; 
num_N = length(N_values); 

% Create a figure for plotting
figure;
hold on;

for i = 1:num_N
    N = N_values(i); 
    
    selected_trials = ERP_EEG(:, 1:N);
    
    mean_trials = mean(selected_trials, 2);
    
    time = (0:sampling_freq-1) / sampling_freq;
    
    plot(time, mean_trials, 'DisplayName', ['N = ' num2str(N)]);
end

% Labeling the plot
xlabel('Time (seconds)');
ylabel('Mean Amplitude');
title('Mean ERP for Different N Values');
legend show;
grid on;
hold off;

%% Q2
max_abs_values = zeros(1, trial_numbers);

for N = 1:trial_numbers
    % Select the first N trials from the EEG data
    selected_trials = ERP_EEG(:, 1:N);
    
    % Compute the mean across the selected trials
    mean_trials = mean(selected_trials, 2);
    
    % Calculate the maximum of the absolute signal values
    max_abs_values(N) = max(abs(mean_trials));
end

% Plotting
figure;
plot(1:trial_numbers, max_abs_values, 'LineWidth', 2);
xlabel('Number of Trials (N)');
ylabel('Max of Absolute Signal Values');
title('Max of Absolute Signal Values for Different N');
grid on;

%% Q3
% Array to store RMSE values for each N
rmse_values = zeros(1, trial_numbers-1);

for N = 2:trial_numbers
    current_trials = ERP_EEG(:, 1:N);
    prev_trials = ERP_EEG(:, 1:(N-1));
    
    mean_current = mean(current_trials, 2);
    mean_prev = mean(prev_trials, 2);
    
    rmse_values(N-1) = sqrt(mean((mean_current - mean_prev).^2));
end

% Plotting the RMSE values
figure;
plot(2:trial_numbers, rmse_values, 'LineWidth', 2);
xlabel('Number of Trials (N)');
ylabel('RMSE between N and N-1');
title('RMSE between i-th and (i-1)-th Mean Trials for Different N');
grid on;

%% Q5 

% Mean based on 2550 trials
mean_2550_trials = mean(ERP_EEG, 2);
N0=1500;
% Mean based on N0/3 trials (first N0/3 trials)
mean_N0_3_fixed = mean(ERP_EEG(:, 1:round(N0/3)), 2);

% Mean based on N0 trials selected randomly
random_N0_trials = ERP_EEG(:, randperm(trial_numbers, N0));
mean_N0_random = mean(random_N0_trials, 2);

% Mean based on N0/3 trials selected randomly
random_N0_3_trials = ERP_EEG(:, randperm(trial_numbers, round(N0/3)));
mean_N0_3_random = mean(random_N0_3_trials, 2);

time = (0:sampling_freq-1) / sampling_freq;

% Plot all means
figure;
hold on;
plot(time, mean_2550_trials, 'LineWidth', 2, 'DisplayName', 'Mean (2550 Trials)');
plot(time, mean_N0_3_fixed, 'LineWidth', 2, 'DisplayName', ['Mean (N_0/3 = ' num2str(round(N0/3)) ' Fixed Trials)']);
plot(time, mean_N0_random, 'LineWidth', 2, 'DisplayName', ['Mean (N_0 = ' num2str(N0) ' Random Trials)']);
plot(time, mean_N0_3_random, 'LineWidth', 2, 'DisplayName', ['Mean (N_0/3 = ' num2str(round(N0/3)) ' Random Trials)']);

xlabel('Time (seconds)');
ylabel('Mean Amplitude');
title('Mean ERP for Different Trial Counts and Selections');
legend show;
grid on;
hold off;



