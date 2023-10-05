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

%d = size(digit1,1); % initial feature length
[W_1,Z_1,D] = pca(digit1, length(digit1(1,:))); % Computing Prinicipal Vectors
%will all the eigenvectors
figure
%stem(D);
scatter([1:length(D)],D)
title('Eigenvalues from largest to lowest')
xlabel('Index')
ylabel('Eigenvalues')

%[W,Z,D] = pca(data,10);

%Question 5
[W_1,Z_1,D] = pca(digit1,10);
figure
for k = 1:10
basis_vector = W_1(:,k);
basis_image = reshape(basis_vector,28,28);
subplot(4,3,k)
imshow(double(basis_image));
title(strcat('Principle Basis Image ',' ', num2str(k)));
end


%Part 6

basis=[1 3 10 50 100];

for i=1:length(basis)
%L=basis(i);
[W,Z,D]= pca(digit1,basis(i)); % taking the first L principle vectors
disp(strcat('for L= ',num2str(basis(i))))
disp(size(W))
mean_n = mean(digit1,2);
n = size(digit1,2);
for k = 1:n
xk = data(:,k);
zk = Z(:,k);
xk_hat = W*zk + mean_n;
X_hat(:,k) = xk_hat;
end

figure()

for j = 1 : 100
subplot(10,10,j)
imshow(double(reshape(X_hat(:,j),28,28)));
end

title(strcat('Reconstructed using ','  ', num2str(basis(i)),' vectors'));
end


