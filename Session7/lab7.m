% Lab 7 
% Ali Khosravipour 99101502 - Mohamad Hosein Faramarzi 99104095 - Sara
% Rezanejad 99101643
%% Q1
clc;
t1_image = imread("t1.jpg");
first_slice = t1_image(:,:,1);
figure;
imshow(first_slice);
%%
row_128 = first_slice(128,:);
dft_128 = fft(row_128);
mag_128 = abs(dft_128); 
phase_128 = angle(dft_128);   
disp('128th Row DFT Magnitude:'), disp(mag_128);
disp('128th Row DFT Phase:'), disp(phase_128);
figure;
subplot(2, 1, 1); 
plot(mag_128, 'LineWidth', 1.5);
title('Magnitude');
xlabel('Frequency Index');
ylabel('Magnitude');
grid on;
subplot(2, 1, 2);          
plot(phase_128, 'LineWidth', 1.5);
title('Phase');
xlabel('Frequency Index');
ylabel('Phase');
grid on;
%%
first_slice_double = double(first_slice);
dft_img = fft2(first_slice_double);
magnitude = abs(dft_img);
figure;
subplot(1,2,1);
imshow(log(1 + magnitude), []);
title('Magnitude');
subplot(1,2,2);
imshow(first_slice);
title('Original Image');
%% fftshift
dft_shifted = fftshift(dft_img);
magnitude2 = abs(dft_shifted);
figure;
subplot(1,2,1);
imshow(log(1 + magnitude2), []);
title('Shifted Magnitude');
subplot(1,2,2);
imshow(first_slice);
title('Original Image');
%% Q2
rows = 256;
cols = 256;
centerX = rows / 2;
centerY = cols / 2;
radius = 15;
[x, y] = ndgrid(1:rows, 1:cols);
G = ((x - centerX).^2 + (y - centerY).^2) <= radius^2;
G = double(G);
F = zeros(256,256);
F(100,50)=1;
F(120,48)=2;
G_fft = fft2(G);
F_fft = fft2(F);
P = G_fft .* F_fft;
P_inverse = ifft2(P);
P_inverse = P_inverse/max(max(P_inverse));
P_inverse = ifftshift(P_inverse);
figure;
subplot(1,3,1);
imshow(P_inverse);
title("Convolved Image");
subplot(1,3,2);
imshow(G);
title("First Image");
subplot(1,3,3);
imshow(F);
title("Second Image");
%% pd.jpg
my_pd = imread("pd.jpg");
first_slice_pd = my_pd(:,:,1);
first_slice_pd = double(first_slice_pd);
first_slice_pd_fft = fft2(first_slice_pd);
pd_conv = first_slice_pd_fft .* G_fft;
pd_conv_inverse = ifft2(pd_conv);
pd_conv_inverse = pd_conv_inverse/max(max(pd_conv_inverse));
pd_conv_inverse = ifftshift(pd_conv_inverse);
figure;
subplot(1,2,1);
imshow(my_pd);
title("Original Image");
subplot(1,2,2);
imshow(pd_conv_inverse);
title("Convolved Image");
%% Q3
ct = imread("ct.jpg");
figure;
imshow(ct);
ct_first_slice = ct(:,:,1);
ct_fft = fft2(ct_first_slice);
ct_fft = fftshift(ct_fft);
zoomed_ct = zeros(512,512);
start_row = rows / 2 + 1;
end_row = 3 * rows / 2;
start_col = cols / 2 + 1;
end_col = 3 * cols / 2;
zoomed_ct(start_row:end_row, start_col:end_col) = ct_fft;
zoomed_ct = ifftshift(zoomed_ct);
ifft_zoomed_ct = ifft2(zoomed_ct);
ifft_zoomed_ct = ifft_zoomed_ct / max(max(ifft_zoomed_ct)); 
figure;
imshow(ifft_zoomed_ct(start_row:end_row, start_col:end_col));
title('Zoomed-in Image (IFFT of Zero-Padded FFT)');
%% Q4.1
ct = imread("ct.jpg");
ct_first_slice = ct(:,:,1);
[rows, cols] = size(ct_first_slice);
ct_fft = fft2(ct_first_slice);
shift_x = 20; % Shift to the right
shift_y = 40; % Shift down
[x, y] = meshgrid(1:cols, 1:rows);
shift_kernel = exp(-2*pi*1i * ((x - 1) * shift_x / cols + (y - 1) * shift_y / rows));
shifted_fft = ct_fft .* shift_kernel;
shifted_image = ifft2(shifted_fft);
shifted_image = real(shifted_image);
shifted_image = mat2gray(shifted_image);
figure;
subplot(1,3,1);
imshow(ct_first_slice);
title('Original Image');
subplot(1,3,2);
imshow(shifted_image);
title('Shifted Image (20 Right, 40 Down)');
subplot(1,3,3);
shift_kernel_fft = fft2(shift_kernel);
subplot(1,3,3);
imshow(log(1 + abs(shift_kernel_fft)), []); 
title('Shifted Image Magnitude');
%% Q4.2
ct = imread("ct.jpg");
ct_first_slice = ct(:,:,1);
ct_rotate = imrotate(ct_first_slice,30);
figure;
subplot(1,2,1);
imshow(ct);
title('Original Image');
subplot(1,2,2);
imshow(ct_rotate);
title('Rotated Image');
figure;
subplot(1,2,1);
ct_fft = fft2(ct_first_slice);
ct_fft = fftshift(ct_fft);
magnitude = abs(ct_fft);
magnitude_log = log(1 + magnitude);  
magnitude_log = magnitude_log / max(magnitude_log(:));
imshow(magnitude_log, []);
title('Mag FFT Original Image');
subplot(1,2,2);
ct_rotate_fft = fft2(ct_rotate);
ct_rotate_fft = fftshift(ct_rotate_fft);
magnitude = abs(ct_rotate_fft);
magnitude_log = log(1 + magnitude);  
magnitude_log = magnitude_log / max(magnitude_log(:));
imshow(magnitude_log, []);
title('Mag FFT Rotated Image');
%%
ct_fft = fftshift(fft2(ifftshift(ct_first_slice)));
ct_fft_rotate = imrotate(ct_fft,30);
ct_ifft_rotate = ifftshift(ifft2(fftshift(ct_fft_rotate)));
ct_ifft_rotate = ct_ifft_rotate / max(max(ct_ifft_rotate)); 
figure;
imshow(real(ct_ifft_rotate));
%% Q5
t1_img = imread("t1.jpg");
t1_img_first_slice = double(t1_img(:,:,1)); 
y = circshift(t1_img_first_slice, 1, 2);
y1 = circshift(t1_img_first_slice, -1, 2);
horizontal_derivative = abs(y - y1);
z = circshift(t1_img_first_slice, 1, 1);
z1 = circshift(t1_img_first_slice, -1, 1);
vertical_derivative = abs(z - z1);
gradient_magnitude = sqrt(horizontal_derivative.^2 + vertical_derivative.^2);
figure;
subplot(2,2,1);
imshow(t1_img_first_slice, []);
title('Original Image');
subplot(2,2,2);
imshow(horizontal_derivative, []);
title('Horizontal Derivative');
subplot(2,2,3);
imshow(vertical_derivative, []);
title('Vertical Derivative');
subplot(2,2,4);
imshow(gradient_magnitude, []);
title('Gradient Magnitude');
%% Q6: Sobel & Canny
t1_img = imread("t1.jpg");
t1_img_first_slice = double(t1_img(:,:,1)); 
sobel_edges = edge(t1_img_first_slice, 'Sobel');
canny_edges = edge(t1_img_first_slice, 'Canny');
figure;
subplot(1,3,1);
imshow(t1_img_first_slice, []);
title('Original Image');
subplot(1,3,2);
imshow(sobel_edges, []);
title('Sobel Edge Detection');
subplot(1,3,3);
imshow(canny_edges, []);
title('Canny Edge Detection');








