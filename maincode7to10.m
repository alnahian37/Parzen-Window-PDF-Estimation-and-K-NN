clc;
clear all;
close all;


% Read image files
image_folder = 'C:\Users\nahia\Google Drive (nahian.buet11@gmail.com)\Fall-21 Semester Drive Folder\ECSE 6610 Pattern\HW\HW 7 2021\hw7data\train_data\'; % Change
%this to your directory location
file_pattern = fullfile(image_folder, '*.jpg');
image_files = dir(file_pattern);
nfiles = length(image_files);
data = zeros(nfiles, 28 * 28); % Matrix with vectorized images along rows
filename_num = zeros(nfiles, 1); % Vector of filenames
for i = 1:nfiles
filename = image_files(i).name;
filename_num(i) = str2double(filename(1:(end - 4)));
im = imread([image_folder, filename]); % Read i th image
data(i, :) = reshape(im, [1, 28 * 28]); % Save image along i th row
end
[filename_num, order] = sort(filename_num); % Sort filenames
data = data(order, :); % Rearrange data matrix to correct order
data = data / 255; % Divide each pixel by 255


%Read labels from the train_labels.txt file into a vector.
% Read labels files
file_id = fopen('train_label.txt', 'r');
format_specification = '%d';
labels = fscanf(file_id, format_specification); % Read labels into a vector
fclose(file_id);

% Arrange the images and their corresponding labels according to class
index_1 = find(labels == 1); % Find indices for label 1
index_5 = find(labels == 5); % Find indices for label 5
labels = [labels(index_1) ; labels(index_5)]; % Modify label vector
data = [data(index_1, :) ; data(index_5, :)]; % Modify data matrix



data=data';
labels=labels';
digit1=data(:,1:100);
digit5=data(:,101:end);

n1 = size(digit1,2); %Total number of digit 1
n5 = size(digit5,2); %Total number of digit 5
mu_1 = mean(digit1,2);
mu_5 = mean(digit5,2);
d = size(data,1);
[W,Z,lamda] = pca(data,length(data(:,1))); % Priniciple Basis Vectors from both digit
%disp(size(W))

%%%%%%%%% part 7 and 8%%%%%%%%

for i = 1 : n1
x1 = digit1(:,i);
z1(i) = W(:,1)'*(x1 - mu_1); % First Priniciple Component
end

for i=1:n5
x5 = digit5(:,i);
z5(i) = W(:,1)'*(x5 - mu_5); % First Priniciple Component
end

figure()
scatter(z1,zeros(size(z1)),'r');
hold on

scatter(z5,zeros(size(z5)),'b');
title('First principle component for Digit 1 and 5')
legend('Digit 1 component', 'Digit 5 component');




%%%%%%%%%%%% Part 9 and 10 %%%%%%%%%%%


dim1 = size(digit1,1);
dim5 = size(digit5,1);
Sigma1 = zeros(dim1,dim1);
Sigma5 = zeros(dim5,dim5);

%sigma of 1
for i = 1 : n1
x1 = digit1(:,i);
Sigma1 = Sigma1 + (x1 - mu_1)*(x1 - mu_1)';
end

%sigma of 5
for i=1:n5
x5 = digit5(:,i);
Sigma5 = Sigma5 + (x5 - mu_5)*(x5 - mu_5)';
end

Sigma_Within = Sigma1 + Sigma5; % within Class

Sigma_Between = (mu_1 - mu_5)*(mu_1 - mu_5)';% Between Class


w = pinv(Sigma_Within)*(mu_1 - mu_5);  %Fisher discriminant vector


for i = 1 : n1
x1 = digit1(:,i);
yk1(i) = w'*x1; % Digit 1 FDA Component
end

for i=1:n5
x5 = digit5(:,i);
yk5(i) = w'*x5; % Digit 5 FDA Component

end
figure()
scatter(yk1,zeros(size(yk1)),'r');
hold on
scatter(yk5,zeros(size(yk5)),'b');
title('Fisher Discriminant Components of Digit 1 and Digit 5')
legend('Digit 1 component', 'Digit 5 component');






