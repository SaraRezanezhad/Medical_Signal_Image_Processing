%% Ali Khosravipour - 99101502 // MohamadHosein Faramarzi - 99104095 // Sara Rezanejad - 99101643

%% Load data
fecg2 = load('D:\term9\SignalLab\New folder\Lab3Codes_99101502_99104095_99101643\fecg2.dat');

%% 4.1
Fs = 256;  
% original data  
plot3ch(X, Fs, 'Scatter plot of data');   

%denoised data using ICA  
plot3ch(Xdenoised.', Fs, 'Scatter plot of ICA');  

%reduced data using SVD  
plot3ch(newY, Fs, 'Scatter plot of SVD');

%3D vectors from matrix A  
figure;  
plot3dv(A_apprx(:,1), [], 'r'); hold on;  
plot3dv(A_apprx(:,2), [], 'g');  
plot3dv(A_apprx(:,3), [], 'b');  
title('3D Vectors from Matrix A');  
xlabel('X-axis');  
ylabel('Y-axis');  
zlabel('Z-axis');  

%3D vectors from matrix V  
figure;  
plot3dv(V(:,1), [], 'r'); hold on;  
plot3dv(V(:,2), [], 'g');  
plot3dv(V(:,3), [], 'b');  
title('3D Vectors from Matrix V');  
xlabel('X-axis');  
ylabel('Y-axis');  
zlabel('Z-axis');  

% Calculating angles and norms for matrix V  
angle_V12 = V(:,1).' * V(:,2);  
angle_V13 = V(:,1).' * V(:,3);  
angle_V32 = V(:,3).' * V(:,2);  
norm_V1 = norm(V(:,1));  
norm_V2 = norm(V(:,2));  
norm_V3 = norm(V(:,3));  

% Calculating angles and norms for matrix A  
angle_A12 = A_apprx(:,1).' * A_apprx(:,2);  
angle_A13 = A_apprx(:,1).' * A_apprx(:,3);  
angle_A32 = A_apprx(:,3).' * A_apprx(:,2);  
norm_A1 = norm(A_apprx(:,1));  
norm_A2 = norm(A_apprx(:,2));  
norm_A3 = norm(A_apprx(:,3));

%% 4.2
% Define time vector  
t = 0 : 1/fs : 10 - (1/fs);  

figure;  
% Plot Idle Signal  
subplot(3,1,1);  
plot(t, fecg2);  
title('Idle Signal');  
xlabel('Time (s)');
ylabel('Amplitude');  
grid minor;  

% Plot Denoised Signal using SVD  
subplot(3,1,2);  
plot(t, newY(:,1));  
title('Denoised by SVD');  
xlabel('Time (s)');
ylabel('Amplitude');  
grid minor;  

% Plot Denoised Signal using ICA  
subplot(3,1,3);  
plot(t, Xdenoised(2,:));  
title('Denoised by ICA');  
xlabel('Time (s)');
ylabel('Amplitude');  
grid minor;


%% 4.3
% Calculate correlation coefficients for ICA denoised signal  
icaCorrelation = corrcoef(Xdenoised(2, :), fecg2);  

% Calculate correlation coefficients for SVD denoised signal  
svdCorrelation = corrcoef(newY(:, 1), fecg2);
display( icaCorrelation);

display( svdCorrelation);

%%
function plot3ch(X,Fs,plot_title)
%PLOT3CH  Plot 3 channel data in the time-domain and on a 3D scatter plot
%  PLOT3CH(X,FS,'TITLE') plots the three columns of data matrix X on a
%  time-domain plot with sample rate FS on and plots each column against the
%  other on a 3D scatter plot. The default value for FS is 256 Hz. The optional
%  'TITLE' input allows the user to specify the plot title string.
% Created by: G.D. Clifford 2004 gari AT mit DOT edu
% Modified 5/6/05, Eric Weiss. Documentation updated. Plot title input added.
% Input argument checking
%------------------------
if nargin < 2
    Fs = 256;
end;
if nargin < 3
    plot_title = '3 Channel Data';
end;
[M,N] = size(X);
if N ~= 3;
    error('Input matrix must have 3 columns!');
end;
% Generate time-domain plot
%--------------------------
t = [1/Fs:1/Fs:M/Fs];
figure;
for i = 1:N
    subplot(N,1,i)
    plot(t,X(:,i)); ylabel(['Ch',int2str(i)]);
    axis([0 max(t) min(X(:,i))-abs(0.1*max(X(:,i))) max(X(:,i))+abs(0.1*max(X(:,i)))]);
    %axis([0 max(t) min(X(:,i)) max(X(:,i))])
end;
xlabel('Time (sec)');
subplot(N,1,1); title(plot_title);
figure;
plot3(X(:,1), X(:,2), X(:,3),'.m');
xlabel('Ch1'); ylabel('Ch2'); zlabel('Ch3');
title(plot_title);
grid on;
end 

function plot3dv(v, s, col)
%PLOT3DV  Plots the specified vector onto a 3D scatter plot
%  PLOT3DV(V, S, 'COL') plots the eigenvector +/-V with singular value S and
%  color 'COL' onto a 3D plot of the currently displayed figure. The length of
%  the plotted eigenvector is equal to the square root of the singular value. If
%  the singular value S is not specified, the default scaling length is 10. If
%  the color 'COL' is not specified, the default color is 'k' (black).

% Created by: GD Clifford 2004 gari AT alum DOT mit DOT edu
% Last modified 5/7/06, Eric Weiss. Documentation updated.

% Input argument checking
%------------------------
if nargin < 2 | isempty(s)
    s = 100;
end;
if nargin < 3
    col = 'k';
end;
v = v(:); % ensure that eigenvector is in column format
[m, n] = size(v);
if (n ~= 1 | m ~= 3)
    error('vector must be 3x1')
end;
if s == 1  % legacy code: does not affect function
    ln = 1/sqrt((v(1)*v(1))+(v(2)*v(2))+(v(3)*v(3)));
end;

% Plot eigenvector on 3D plot
%----------------------------
sn = sqrt(s);
hold on;
plot3(sn*[-1*v(1) v(1)],sn*[-1*v(2) v(2)],sn*[-1*v(3) v(3)],col);
grid on;
view([1,1,1])
end
