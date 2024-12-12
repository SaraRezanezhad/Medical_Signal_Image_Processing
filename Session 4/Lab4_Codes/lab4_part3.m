%% Ali Khosravipour 99101502 - MohamadHosein Faramarzi 99104095 - Sara Rezanejad 99101643
clc; clear;
load("FiveClass_EEG.mat");
fs = 256;
X = X';
% elec_labels = {'Pz', 'Oz', 'P7', 'P8', 'O2', 'O1'}; %% Complete it later with
% actal labels
%% Part 3.1
fs = 256;
X = X';
filtered_delta = myBandpass(X, 1, 4, fs); % delta
filtered_theta = myBandpass(X, 4, 8, fs); % theta
filtered_alpha = myBandpass(X, 8, 13, fs); % alpha
filtered_beta = myBandpass(X, 13, 30, fs); % beta
%% checking the filter (plotting the first five secs)
bands = {
    'Raw Signal', [1, 4];
    'Delta', [1, 4];
    'Theta', [4, 8];
    'Alpha', [8, 13];
    'Beta', [13, 30];
};
all_data1 = {X, filtered_delta, filtered_theta, filtered_alpha, filtered_beta};
time_window = 5; 
num_samples = time_window * fs;
%disp(num_samples);
figure;
for i = 1:length(all_data1)
    my_data = all_data1{i};
    signal_segment = my_data(:, 1:num_samples);
    band_name = bands{i, 1};
    subplot(5, 1, i);
    plot((0:num_samples-1) / fs, signal_segment(1, :));
    title([band_name ' Band (1st Channel)']);
    xlabel('Time (s)');
    ylabel('Amplitude');
end
%% Part 3.2
seg_delta = zeros(200,30,2560);
seg_theta = zeros(200,30,2560);
seg_alpha = zeros(200,30,2560);
seg_beta = zeros(200,30,2560);
for i=1:200
    seg_delta(i,:,:) = filtered_delta(:,trial(i): (trial(i) + 2560) - 1);
    seg_theta(i,:,:) = filtered_theta(:,trial(i): (trial(i) + 2560) - 1);
    seg_alpha(i,:,:) = filtered_alpha(:,trial(i): (trial(i) + 2560) - 1);
    seg_beta(i,:,:) = filtered_beta(:,trial(i): (trial(i) + 2560) - 1);
end
%% Part 3.3
squared_data_delta = seg_delta.^2;
squared_data_theta = seg_theta.^2;
squared_data_alpha = seg_alpha.^2;
squared_data_beta = seg_beta.^2;
%% Part 3.4
class1 = []; 
class2 = [];  
class3 = [];  
class4 = [];  
class5 = [];  
all_data = {squared_data_delta, squared_data_theta, squared_data_alpha, squared_data_beta};
for j=1:length(all_data)
    squared_data = all_data{j};
    for i = 1:200
        if y(i) == 1
            class1 = cat(1, class1, squared_data(i, :, :));
        elseif y(i) == 2
            class2 = cat(1, class2, squared_data(i, :, :));
        elseif y(i) == 3
            class3 = cat(1, class3, squared_data(i, :, :));
        elseif y(i) == 4
            class4 = cat(1, class4, squared_data(i, :, :));
        else
            class5 = cat(1, class5, squared_data(i, :, :));
        end
    end
    mean1 = mean(class1(:,:,:));
    mean2 = mean(class2(:,:,:));
    mean3 = mean(class3(:,:,:));
    mean4 = mean(class4(:,:,:));
    mean5 = mean(class5(:,:,:));
    if j==1
        Delta_X_avg = cat(1, mean1, mean2, mean3, mean4, mean5);    
    elseif j==2
        Theta_X_avg = cat(1, mean1, mean2, mean3, mean4, mean5);
    elseif j==3
        Alpha_X_avg = cat(1, mean1, mean2, mean3, mean4, mean5);
    else
        Beta_X_avg = cat(1, mean1, mean2, mean3, mean4, mean5);    
    end
end
%% Part 3.5
newWin = ones(1,200)/sqrt(200);
Delta_X_avg_conv = zeros(size(Delta_X_avg));
Theta_X_avg_conv = zeros(size(Theta_X_avg));
Alpha_X_avg_conv = zeros(size(Alpha_X_avg));
Beta_X_avg_conv = zeros(size(Beta_X_avg));
for class = 1:size(Delta_X_avg, 1)
    for channel = 1:size(Delta_X_avg, 2)
        Delta_X_avg_conv(class, channel, :) = conv(squeeze(Delta_X_avg(class, channel, :)), newWin, 'same');
        Theta_X_avg_conv(class, channel, :) = conv(squeeze(Theta_X_avg(class, channel, :)), newWin, 'same');
        Alpha_X_avg_conv(class, channel, :) = conv(squeeze(Alpha_X_avg(class, channel, :)), newWin, 'same');
        Beta_X_avg_conv(class, channel, :) = conv(squeeze(Beta_X_avg(class, channel, :)), newWin, 'same');
    end
end
%% Part 3.6
% CPz --> 16th channel
all_data2 = {Delta_X_avg_conv, Theta_X_avg_conv, Alpha_X_avg_conv, Beta_X_avg_conv};
for j = 1:4
    my_data_orig = all_data2{j};
    figure;
    hold on;
    for class = 1:5
        sig = squeeze(my_data_orig(class, 16, :));
        plot(sig, 'DisplayName', ['Class ' num2str(class)]);
    end
    xlabel('Time (samples)');
    ylabel('Amplitude');
    if j==1
        title('Five Classes in Delta Band (CPz Channel)');
    elseif j==2
        title('Five Classes in Theta Band (CPz Channel)');
    elseif j==3
        title('Five Classes in Alpha Band (CPz Channel)');
    else
        title('Five Classes in Beta Band (CPz Channel)');
    end
    grid on;
    xlim([1 2560]); 
    legend('show');
end



%%
function X = myBandpass(sig, lower_edge, higher_edge, fs)
N = 4;
Apass = 1;
h = fdesign.bandpass('N,Fp1,Fp2,Ap', N, lower_edge, higher_edge, Apass, fs);
Hd = design(h, 'cheby1');
for c = 1 : 30
    X(c,:) = filter(Hd, sig(c,:));
end
end