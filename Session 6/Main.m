%% Sara Rezanejad - 99101643/Ali Khosravipour - 99101502/MohamadHosein Faramarzi - 99104095 
clear; clc ;

load('D:\term9\SignalLab\Session 6\Codes and Data\ElecPosXYZ.mat') ;
load('D:\term9\SignalLab\Session 6\Codes and Data\Interictal.mat') ;

%% Q1
%Forward Matrix
ModelParams.R = [8 8.5 9.2] ; % Radius of diffetent layers
ModelParams.Sigma = [3.3e-3 8.25e-5 3.3e-3]; 
ModelParams.Lambda = [.5979 .2037 .0237];
ModelParams.Mu = [.6342 .9364 1.0362];

Resolution = 1 ;
[LocMat,GainMat] = ForwardModel_3shell(Resolution, ModelParams) ;
scatter3(LocMat(1,:), LocMat(2,:), LocMat(3,:)); 
hold on;

%% Q2

numElectrodes = numel(ElecPos);  
ElectrodePosNumeric = zeros(numElectrodes, 3); % Preallocate for positions  
ElectrodeLabelS = cell(1,21);

for i = 1:numElectrodes  
    for j=1:3
    ElectrodePosNumeric(i, j) = ElecPos{1,i}.XYZ(j).* ModelParams.R(3); % Extract XYZ coordinates  
   
    end
        ElectrodeLabelS{i} = ElecPos{1,i}.Name;

end  

% plot using ElectrodePosNumeric  
scatter3(ElectrodePosNumeric(:,1), ElectrodePosNumeric(:,2), ElectrodePosNumeric(:,3), 'r', 'filled'); % Plot electrodes  

% Add labels to each electrode  
for i = 1:numElectrodes  
    text(ElectrodePosNumeric(i, 1), ElectrodePosNumeric(i, 2), ElectrodePosNumeric(i, 3), ...  
        ['Electrode ' ElectrodeLabelS(i)], 'FontSize', 8, 'Color', 'k');  
    
end

%% Q3
% Create LocMat from the electrode position coordinates  
%LocMat = [ElectrodePosNumeric(:,1); ElectrodePosNumeric(:,2); ElectrodePosNumeric(:,3)];  

% Randomly select a dipole index  
rand_index = int32(21 * rand(1, 1) + 1); % Adjust your max index here if needed  

% Plot the selected random dipole location  
scatter3(LocMat(1, rand_index), LocMat(2, rand_index), LocMat(3, rand_index), 'g', 'filled');  

% Create a radial line extending from the dipole position  
startPoint = LocMat(:, rand_index); % Get the random dipole location  
lineLength = 1; % Length of the line  

% Define the end point of the line in radial direction  
endPoint = startPoint + lineLength * normalize(startPoint);   

% Plot the line  
plot3([startPoint(1), endPoint(1)], [startPoint(2), endPoint(2)], [startPoint(3), endPoint(3)], 'g', 'LineWidth', 2);  

% Label the random dipole  
text(LocMat(1, rand_index), LocMat(2, rand_index), LocMat(3, rand_index), ' Random Dipole', 'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold');  

hold off;  
xlabel('X-axis');  
ylabel('Y-axis');  
zlabel('Z-axis');  
title('Dipole Locations and Electrode Positions');  
grid on;  
axis equal; % Maintain the aspect ratio  



%% Q4
signal = Interictal(6,:);
% Randomly select a dipole index from electrode locations  
dipoleIndex = randi(21); % Assuming you have 21 electrodes  

% Calculate the dipole direction  
dipole_loc = [LocMat(1,dipoleIndex), LocMat(2,dipoleIndex), LocMat(3,dipoleIndex)];  
signal_dir = dipole_loc ./ norm(dipole_loc); % Normalize the dipole location  
%%
% Calculate the potential at each electrode  
M = GainMat(:, (3*dipoleIndex-2:3*dipoleIndex)) * signal_dir' * signal;  

% Visualize the potentials  
disp_eeg(M, max(abs(M(:))), [], ElectrodeLabelS, 'Potential of Electrodes');  

hold off;  
xlabel('X-axis');  
ylabel('Y-axis');  
zlabel('Z-axis');  
title('Dipole Locations and Electrode Positions');  
grid on;  
axis equal; % Maintain the aspect ratio  

%% Q5 
figure;
mean_Pot = zeros(21, 1);
for i=1:21
    [pks, locs] = findpeaks(M(i,:), 'MinPeakProminence', 0.9*max(M(i,:)));
    epochs = zeros(length(locs), 7);
    for j=1:length(locs)
       epochs(j, :) = M(i, locs(j)-3:locs(j)+3);
    end
    mean_Pot(i) = mean(epochs, 'all');
end

Display_Potential_3D(ModelParams.R(3), mean_Pot)
colorbar;
caxis([-20, 20])


%% Q6
alpha = 0.5;
Q = GainMat' * (inv(GainMat * GainMat' + alpha * eye(21))) * M;

%% Q7
% Calculate average Q values  
avgQ = max(Q, [], 2);  

% Initializing norms array  
norms = zeros(1, 1317);  

% Calculate norms for each dipole vector  
for idx = 1:1317  
    vec = avgQ(3*idx-2:3*idx); % Get dipole components  
    norms(idx) = norm(vec);     % Compute norm  
end  

% maximum norm and its index  
[maxNorm, maxIdx] = max(norms);  

% Get the direction corresponding to the maximum norm  
normDir = avgQ(3*maxIdx-2:3*maxIdx) / maxNorm; % Normalize direction  
disp("Normalized Direction : "+normDir);

%% Q8
dirError = norm(dipole_loc - normDir);
posError = norm(LocMat(:, rand_index) - LocMat(:, maxIdx));  
disp("Location estimation error: "+posError);
disp("Bipolar direction error: "+dirError);

%% Q9
%% Part 9: Analyzing Dipole Locations and Potentials  
% First index (cortex)  
[max_cortex, idx_cortex] = max(sqrt(sum(LocMat.^2, 1)), [], 2);  
dir_cortex = LocMat(:, idx_cortex) / norm(LocMat(:, idx_cortex));  

% Ensure dimensions match: GainMat should have appropriate size  
M_cortex=GainMat(:, (3*idx_cortex-2:3*idx_cortex)) * dir_cortex * signal;  
% Transpose sig_interictal  
disp_eeg(M_cortex, max(abs(M_cortex(:))), [], ElectrodeLabelS, 'Electrode Potential (Cortex)');  

% Average potential across electrodes  
avg_cortex = zeros(1, 21);  
for i = 1:21  
    [~, peaks_t_cortex] = findpeaks(M_cortex(i,:), 'MinPeakHeight', 0.3 * max(M_cortex(i,:)));  
    win_cortex = repmat(peaks_t_cortex, [1 size(peaks_t_cortex, 1)]) + repmat((-3:3)', size(peaks_t_cortex, 1));  
    avg_cortex(i) = mean(M_cortex(i, win_cortex), 'all');  
end  
figure;  
Display_Potential_3D(ModelParams.R(3), avg_cortex);  

% Compute Q values  
Q_cortex = GainMat' * (inv(GainMat * GainMat' + 0.6 * eye(size(21)))) * M_cortex;  

% Average Q values and direction  
Q_avg_cortex = max(Q_cortex, [], 2);  
Q_norm_cortex = zeros(1, 1317);  
for i = 1:1317  
    Q_norm_cortex(i) = norm(Q_avg_cortex(3*i-2:3*i));  
end  
[Q_max_cortex, index_Q_cortex] = max(Q_norm_cortex);  
Q_dir_cortex = Q_avg_cortex(3*index_Q_cortex-2:3*index_Q_cortex) / Q_max_cortex;  

% Position and direction error  
pos_error_cortex = norm(LocMat(:, idx_cortex) - LocMat(:, index_Q_cortex));  
dir_error_cortex = norm(dir_cortex - Q_dir_cortex);

%% Second index (temporal)  
idx_temp = 104;  % Fixed index for temporal dipole  
dir_temp = LocMat(:, idx_temp) / norm(LocMat(:, idx_temp));  
M_temp = GainMat(:, (3*idx_temp-2:3*idx_temp)) * dir_temp * signal;  
disp_eeg(M_temp, max(abs(M_temp(:))), [], ElectrodeLabelS, 'Electrode Potential (Temporal)');  

% Average potential calculation  
avg_temp = zeros(1, 21);  
for i = 1:21  
    [~, peaks_t_temp] = findpeaks(M_temp(i,:), 'MinPeakHeight', 0.3 * max(M_temp(i,:)));  
    win_temp = repmat(peaks_t_temp, [1 size(peaks_t_temp, 1)]) + repmat((-3:1:3)', size(peaks_t_temp, 1));  
    avg_temp(i) = mean(M_temp(i, win_temp), 'all');  
end  
figure;  
Display_Potential_3D(ModelParams.R(3), avg_temp);  

% Compute Q values  
Q_temp = GainMat' * (inv(GainMat * GainMat' + 0.6 * eye(21))) * M_temp;  

% Average Q values and direction  
Q_avg_temp = max(Q_temp, [], 2);  
Q_norm_temp = zeros(1, 1317);  
for i = 1:1317  
    Q_norm_temp(i) = norm(Q_avg_temp(3*i-2:3*i));  
end  
[Q_max_temp, index_Q_temp] = max(Q_norm_temp);  
Q_dir_temp = Q_avg_temp(3*index_Q_temp-2:3*index_Q_temp) / Q_max_temp;  

% Position and direction error  
pos_error_temp = norm(LocMat(:, idx_temp) - LocMat(:, index_Q_temp));  
dir_error_temp = norm(dir_temp - Q_dir_temp);  

%% Third index (depth)  
[min_depth, idx_depth] = min(sqrt(sum(LocMat.^2, 1)), [], 2);  
dir_depth = LocMat(:, idx_depth) / norm(LocMat(:, idx_depth));  
M_depth = GainMat(:, (3*idx_depth-2:3*idx_depth)) * dir_depth * signal;  
disp_eeg(M_depth, max(abs(M_depth(:))), [], ElectrodeLabelS, 'Electrode Potential (Depth)');  

% Average potential calculation  
avg_depth = zeros(1, 21);  
for i = 1:21  
    [~, peaks_t_depth] = findpeaks(M_depth(i,:), 'MinPeakHeight', 0.3 * max(M_depth(i,:)));  
    win_depth = repmat(peaks_t_depth, [1 size(peaks_t_depth, 1)]) + repmat((-3:1:3)', size(peaks_t_depth, 1));  
    avg_depth(i) = mean(M_depth(i, win_depth), 'all');  
end  
figure;  
Display_Potential_3D(ModelParams.R(3), avg_depth);  

% Compute Q values  
Q_depth = GainMat' * (inv(GainMat * GainMat' + 0.6 * eye(21))) * M_depth;  

% Average Q values and direction  
Q_avg_depth = max(Q_depth, [], 2);  
Q_norm_depth = zeros(1, 1317);  
for i = 1:1317  
    Q_norm_depth(i) = norm(Q_avg_depth(3*i-2:3*i));  
end  
[Q_max_depth, index_Q_depth] = max(Q_norm_depth);  
Q_dir_depth = Q_avg_depth(3*index_Q_depth-2:3*index_Q_depth) / Q_max_depth;  

% Position and direction error  
pos_error_depth = norm(LocMat(:, idx_depth) - LocMat(:, index_Q_depth));  
dir_error_depth = norm(dir_depth - Q_dir_depth);  

%% Output Errors  
disp(['Cortex Position Error: ', num2str(pos_error_cortex)]);  
disp(['Cortex Direction Error: ', num2str(dir_error_cortex)]);  
disp(['Temporal Position Error: ', num2str(pos_error_temp)]);  
disp(['Temporal Direction Error: ', num2str(dir_error_temp)]);  
disp(['Depth Position Error: ', num2str(pos_error_depth)]);  
disp(['Depth Direction Error: ', num2str(dir_error_depth)]);
%%
% Normalize function to get direction vector  
function normVec = normalize(vec)  
    normVec = vec / norm(vec);  
end  