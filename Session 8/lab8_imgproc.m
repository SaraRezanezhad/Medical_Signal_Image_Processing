%% Ali Khosravipour 99101502 // MohamadHosein Faramarzi 99104095 // Sara Rezanejad 99101643
%% Q1
t2_img = imread('D:\term9\SignalLab\Session 8\Lab 8_data\S2_Q1_utils\t2.jpg');
first_slice = t2_img(:,:,1);
var = 15;
noisy_first_slice = imnoise(first_slice, 'gaussian', 0, var / 255^2);
rows = 256; 
cols = 256; 
binary_img = zeros(rows, cols);
start_row = floor((rows - 4) / 2) + 1;
start_col = floor((cols - 4) / 2) + 1;
binary_img(start_row:start_row+3, start_col:start_col+3) = 1;
F_first_slice = fft2(double(first_slice));
F_binary_img = fft2(double(binary_img));
F_result = F_first_slice .* F_binary_img;
result_img = real(ifftshift(ifft2(F_result)));
figure;
subplot(1, 3, 1);
imshow(first_slice, []);
title('Orig Img');
subplot(1, 3, 2);
imshow(noisy_first_slice, []);
title('Noisy Img');
subplot(1, 3, 3);
imshow(result_img, []);
title('Kernel 1');
%%
start_row = floor((rows - 4) / 2) + 1;
start_col = floor((cols - 4) / 2) + 1;
binary_img2 = zeros(rows, cols);
binary_img2(start_row:start_row+3, start_col:start_col+3) = 1 / (4 * 4);
F_binary_img2 = fft2(double(binary_img2));
F_result2 = F_first_slice .* F_binary_img2;
result_img2 = real(ifftshift(ifft2(F_result2)));
figure;
subplot(1, 3, 1);
imshow(first_slice, []);
title('Orig Img');
subplot(1, 3, 2);
imshow(noisy_first_slice, []);
title('Noisy Img');
subplot(1, 3, 3);
imshow(result_img2, []);
title('Kernel 2');
%%
gaussfilt_img = imgaussfilt(double(first_slice), sqrt(1)); 
figure;
subplot(1,2,1);
imshow(first_slice, []);
subplot(1,2,2);
imshow(gaussfilt_img, []);
%% Q2
t2_img = imread('D:\term9\SignalLab\Session 8\Lab 8_data\S2_Q2_utils\t2.jpg');
f = t2_img(:,:,1);
h = Gaussian(0.5,[256,256]);
g = conv2(double(f),h,'same');
G = fft2(g);
H = fft2(h);
F = G ./ H;
recon_f = abs(fftshift(ifft2(F)));
%
g_noised = imnoise(g,'gaussian',0,0.001);
G_noised = fft2(g_noised);
F2 = G_noised ./ H;
recon_f2 = abs(fftshift(ifft2(F)));
%
figure;
subplot(1,3,1);
imshow(f,[]);
subplot(1,3,2);
imshow(recon_f,[]);
subplot(1,3,3);
imshow(recon_f2,[]);



%% Q3
image = imread('D:\term9\SignalLab\Session 8\Lab 8_data\S2_Q2_utils\t2.jpg');  
image_resized = imresize(image, [64, 64]); 
imshow(image_resized);
title('Changed Picture');

% Convert to grayscale if the image has 3 channels (RGB)  
if size(image_resized, 3) == 3  
    image_resized = rgb2gray(image_resized);  
end  

h = [0 1 0; 1 2 1; 0 1 0;];  
K = zeros(64, 64); 
K(1:3, 1:3) = h; 
imshow(K);
title('Picture with Gaussian filtered');

N = 64;  % ابعاد تصویر بعد از کاهش
D = zeros(N^2, N^2);  % ماتریس D به ابعاد 4096x4096
for c = 1:64
    for r = 1:64
        new_k = circshift(K, [r-1,c-1]);
        D(64*(c-1) + r, :) = reshape(new_k, 1, 64*64);
    end
end 


image_1 = double(reshape(image_resized, N^2, 1));  % تبدیل تصویر به بردار ستونی
g = D * image_1;
disp(size(g));   
disp(size(D));  
NoisyImage = g + random('Normal', 0, 0.05, N^2, 1);
pinvD = pinv(D);
denoised = pinvD * NoisyImage;
DenoisedImage = (reshape(denoised, N, N));
subplot(1,2,1)
imshow(image)
title('original')
subplot(1,2,2)
imshow(DenoisedImage / max(max(DenoisedImage)))
title('denoised image')
disp(size(image));  % Should be [height, width, channels]  
disp(size(image_resized)); 


%% Q4

image = imread('D:\term9\SignalLab\Session 8\Lab 8_data\S2_Q2_utils\t2.jpg');  
image_resized = imresize(image, [64, 64]); 
if size(image_resized, 3) == 3  
    image_resized = rgb2gray(image_resized);  
end  

h = [0 1 0; 1 2 1; 0 1 0];  
K = zeros(64, 64); 
K(1:3, 1:3) = h; 


N = 64;  
D = zeros(N^2, N^2);  
for c = 1:64
    for r = 1:64
        new_k = circshift(K, [r-1,c-1]);
        D(64*(c-1) + r, :) = reshape(new_k, 1, 64*64);
    end
end 

image_1 = double(reshape(image_resized, N^2, 1));  
g = D * image_1;

NoisyImage = g + random('Normal', 0, 0.05, N^2, 1);

% پیاده‌سازی الگوریتم Descent Gradient
alpha = 0.01;  
iterations = 100;  
f = zeros(N^2, 1); 

errors = zeros(iterations, 1);

for k = 1:iterations
    gradient = D' * (D * f - NoisyImage);  % محاسبه گرادیان
    f = f - alpha * gradient;  % به‌روزرسانی تصویر با گرادیان نزولی
    
    errors(k) = 0.5 * norm(D * f - NoisyImage)^2;
end

% بازسازی تصویر دنوایشن‌شده
DenoisedImage = reshape(f, N, N);

subplot(1,2,1)
imshow(image_resized)
title('original Picture');

subplot(1,2,2)
imshow(DenoisedImage / max(max(DenoisedImage)))
title('Denoised Picture with Descent Gradient')

% نمایش همگرایی تابع هزینه
figure;
plot(1:iterations, errors);
xlabel('Repetition number');
ylabel('MSE');
title('Convergence of Gradient Descent Algorithm');











