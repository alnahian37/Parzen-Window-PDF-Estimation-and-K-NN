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


%%%%%%%%%%%% part 4 %%%%%%%%%%%

data=data';
labels=labels';

digit1=data(:,1:100);
digit5=data(:,101:end);


[W,Z,lambda] = pca(digit1, length(digit1(:,1))); % Computing Prinicipal Vectors 

figure()

plot([1:length(lambda)],lambda,marker='o')
title('Eigenvalue Plot from largest to lowest')
xlabel('index of lambda')
ylabel('Value of Lambda')



%%%%%%%%% Part 5 %%%%%%%


figure()
for i = 1:10
basis_vector = reshape(W(:,i),28,28);
subplot(4,3,i)
imshow(basis_vector);
title(strcat('Basis Image#  ', num2str(i)));
end


%%%%%%%%% Part 6 %%%%%%%%



mu_1 = mean(digit1,2);
n = length(digit1(1,:));

basis=[1 3 10 50 100];

for i=1:length(basis)

%[W1,Z1,lambda]= pca(digit1,basis(i)); % taking the first L principle vectors

W1=W(:,1:basis(i));
Z1=Z(1:basis(i),:);
%continue;

for k = 1:n
z = Z1(:,k);
x_hat = W1*z + mu_1;
X_recon(:,k) = x_hat;
end

figure()

for j = 1 : length(digit1(1,:))
subplot(10,10,j)
imshow(reshape(X_recon(:,j),28,28));
end

title(strcat('Reconstructed using ','  ', num2str(basis(i)),' vectors'));
end


