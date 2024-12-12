%% Ali Khosravipour 99101502 - MohamadHosein Faramarzi 99104095 - Sara Rezanejad 99101643
clc; clear;
load("SSVEP_EEG.mat");
fs = 250;
elec_labels = {'Pz', 'Oz', 'P7', 'P8', 'O2', 'O1'};
%% 2.1 
fs = 250;
elec_labels = {'Pz', 'Oz', 'P7', 'P8', 'O2', 'O1'};
offset = max(abs(SSVEP_Signal(:)));
disp_eeg(SSVEP_Signal,offset,fs,elec_labels,'Original Signals');
ssvep_filtered = zeros(size(SSVEP_Signal));
for i = 1: size(SSVEP_Signal,1)
    ssvep_filtered(i, :) = bandpass(SSVEP_Signal(i, :), [1 40], fs);
end
offset1 = max(abs(ssvep_filtered(:)));
disp_eeg(ssvep_filtered,offset,fs,elec_labels,'Filtered Signals');
%% 2.2 Window Extraction
winLength = 5 * fs; 
eventWindows = [];
for i = 1 : size(Event_samples,2)
    start_idx = Event_samples(i);
    event_win = ssvep_filtered(:, start_idx:start_idx + winLength - 1);
    eventWindows = cat(3, eventWindows, event_win);
end
%% 2.3 Each Channel in Frequency Domain
figure;
for i = 1 : 15
    subplot(3,5,i)
    my_SIG = squeeze(eventWindows(:,:,i));
    hold on;
    for ch = 1:size(my_SIG, 1)
        [pxx, f] = pwelch(my_SIG(ch, :), [], [], [], fs);
        % plot(f, 10*log10(pxx), 'DisplayName', elec_labels{ch});
        plot(f, pxx, 'DisplayName', elec_labels{ch});
    end
    hold off;
    if(i < 4)
        title(sprintf('Experiment %d - Freq %.2f', i, Events(i)));
    elseif(i < 7)
        title(sprintf('Experiment %d - Freq %.2f', i, Events(i)));
    elseif(i < 10)
        title(sprintf('Experiment %d - Freq %.2f', i, Events(i)));
    elseif(i < 13)
        title(sprintf('Experiment %d - Freq %.2f', i, Events(i)));
    else
        title(sprintf('Experiment %d - Freq %.2f', i, Events(i)));
    end
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    xlim([0 30]);
    xticks(0:5:30);
    grid on;
    legend show;
end
%% 
function t = disp_eeg(X,offset,feq,ElecName,titre)
% function t = disp_eeg(X,offset,feq,ElecName,titre)
%
% inputs
%     X: dynamics to display. (nbchannels x nbsamples) matrix
%     offset: offset between channels (default max(abs(X)))
%     feq: sapling frequency (default 1)
%     ElecName: cell array of electrode labels (default {S1,S2,...})
%     titre: title of the figure
%
% output
%     t: time vector
%
% G. Birot 2010-02


%% Check arguments
[N K] = size(X);

if nargin < 4
    for n = 1:N
        ElecName{n}  = ['S',num2str(n)];
    end
    titre = [];
end

if nargin < 5
    titre = [];
end

if isempty(feq)
    feq = 1;
end

if isempty(ElecName)
    for n = 1:N
        ElecName{n}  = ['S',num2str(n)];
    end
end

if isempty(offset)
    offset = max(abs(X(:)));
end


%% Build dynamic matrix with offset and time vector
X = X + repmat(offset*(0:-1:-(N-1))',1,K);
t = (1:K)/feq;
graduations = offset*(0:-1:-(N-1))';
shiftvec = N:-1:1;
Ysup = max(X(1,:)) + offset;
Yinf = min(X(end,:)) - offset;
% YLabels = cell(N+2) ElecName(shiftvec)

%% Display
figure1 = figure;
% a1 = axes('YAxisLocation','right');
a2 = axes('YTickLabel',ElecName(shiftvec),'YTick',graduations(shiftvec),'FontSize',7);
ylim([Yinf Ysup]);
box('on');
grid('on')
hold('all');
plot(t,X');
xlabel('Time (seconds)','FontSize',10);
ylabel('Channels','FontSize',10);
title(titre);
hold off


end

