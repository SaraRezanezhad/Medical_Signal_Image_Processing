%% Ali Khosravipour - 99101502 // MohamadHosein Faramarzi - 99104095 // Sara Rezanejad - 99101643
clc; clear;
mecg1 = load("D:\term9\SignalLab\New folder\Lab3Codes_99101502_99104095_99101643\mecg1.dat");
fecg1=load("D:\term9\SignalLab\New folder\Lab3Codes_99101502_99104095_99101643\fecg1.dat");
noise1=load("D:\term9\SignalLab\New folder\Lab3Codes_99101502_99104095_99101643\noise1.dat");
total_sig1 = fecg1' + mecg1' + noise1';
fs = 256;
%% 1.1
figure;
subplot(4,1,1);
plot(mecg1);
title('Mother');
ylabel('mV');
xlim([0 2560]);
subplot(4,1,2);
plot(fecg1);
title('Child');
ylabel('mV');
xlim([0 2560]);
subplot(4,1,3);
plot(noise1);
title('Noise');
ylabel('mV');
xlim([0 2560]);
subplot(4,1,4);
plot(total_sig1);
title('Total (Mother + Child + Noise)');
ylabel('mV');
xlim([0 2560]);
%% 1.2
figure;
subplot(4,1,1);
pwelch(mecg1, [], [], [], fs)
title('Mother');
subplot(4,1,2);
pwelch(fecg1, [], [], [], fs)
title('Child');
subplot(4,1,3);
pwelch(noise1, [], [], [], fs)
title('Noise');
subplot(4,1,4);
pwelch(total_sig1, [], [], [], fs)
title('Total (Mother + Child + Noise)');
%% 1.3
mean_mother = mean(mecg1);
var_mother = var(mecg1);
mean_child = mean(fecg1);
var_child = var(fecg1);
mean_noise = mean(noise1);
var_noise = var(noise1);
%% 1.4
subplot(2,2,1);
histogram(mecg1, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'NumBins', 100, 'EdgeAlpha', 0.5);
title("Mother Hist");
grid on;
subplot(2,2,2);
histogram(fecg1, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'NumBins', 100, 'EdgeAlpha', 0.5);
title("Child Hist");
grid on;
subplot(2,2,3);
histogram(noise1, 'FaceColor', 'green', 'FaceAlpha', 0.5, 'NumBins', 100, 'EdgeAlpha', 0.5);
title("Noise Hist");
grid on;
subplot(2,2,4);
histogram(total_sig1, 'FaceColor', 'yellow', 'FaceAlpha', 0.5, 'NumBins', 100, 'EdgeAlpha', 0.5);
title("Total Hist");
grid on;
% 4th moment
fourthMoment_m1 = kurtosis(mecg1);
fourthMoment_f1 = kurtosis(fecg1);
fourthMoment_n1 = kurtosis(noise1);
fourthMoment_t1 = kurtosis(total_sig1);








