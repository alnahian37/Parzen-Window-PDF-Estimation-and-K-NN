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

digit1 = data(:,1:100);
digit5 = data(:,101:200);
n1 = size(digit1,2);
n2 = size(digit5,2);
mu_1 = mean(digit1,2);
mu_5 = mean(digit5,2);
d = size(data,1);
[W,Z,lambda] = pca(data,d); % Priniciple Basis Vectors from Digit 1 and Digit 5


%%%% 2d main code %%%%
for k = 1 : n1
xk1 = digit1(:,k);
zk1(:,k) = W(:,1:3)'*(xk1 - mu_1); % First 3 Priniciple Components of Digit 1
end
for k=1:n2
xk5 = digit5(:,k);
zk5(:,k) = W(:,1:3)'*(xk5 - mu_5); % First 3 Principle Components of Digit 5
end
data_train = [zk1(:,1:25) zk5(:,1:25)];
datatest_1 = zk1(:,26:100);
datatest_5 = zk5(:,26:100);


delT = delaunayTriangulation(data_train');
[V,R] = voronoiDiagram(delT);



for i = 1:25
if all(R{i} ~= 1)
XR = V(R{i},:);
trisurf(convhull(XR), XR(:,1) ,XR(:,2) ,XR(:,3) ,'FaceColor', [0.5 0.3 0.3], 'FaceAlpha',0.8)
hold on
end
end

for i = 26:50
if all(R{i} ~= 1)
XR = V(R{i},:);
trisurf(convhull(XR), XR(:,1) ,XR(:,2) ,XR(:,3) , 'FaceColor', [0.3 0.3 0.5], 'FaceAlpha',0.8)
hold on
end
end

h_1 =scatter3(data_train(1,1:25),data_train(2,1:25),data_train(3,1:25),'r','filled');
hold on
h_5 =scatter3(data_train(1,26:50),data_train(2,26:50),data_train(3,1:25),'b','filled');
axis([-5 5 -5 5 -8 8])
legend([h_1,h_5],{'Digit 1','Digit 5'});
xlabel('1st component');
ylabel('2nd component');
zlabel('3rd component');
title('3D Voronoi Diagram for Digit 1 and Digit5');

%KNN classification

K = 1;

labels_test_1 = knn_classifier(data_train,datatest_1,K);

labels_test_5 = knn_classifier(data_train,datatest_5,K);

PF1 = sum(labels_test_1 == 5)/length(labels_test_1);
PD1 = sum(labels_test_1 == 1)/length(labels_test_1);
PF5 = sum(labels_test_5 == 1)/length(labels_test_5);
PD5 = sum(labels_test_5 == 5)/length(labels_test_5);

fprintf('for K=1\n')
fprintf(strcat("Detecetion for Digit 1 = ",num2str(PD1*100),"%%","\n"));
fprintf(strcat("Misclassification for Digit 1 = ",num2str(PF1*100),"%%","\n"));
fprintf(strcat("Detecetion for Digit 5 = ",num2str(PD5*100),"%%","\n"));
fprintf(strcat("Misclassification for Digit 5 = ",num2str(PF5*100),"%%","\n\n"));



%%%%% 2C %%%%%
K = 5;
% Digit 1 Classification
labels_test_1 = knn_classifier(data_train,datatest_1,K);
%Digit 5 Classification
labels_test_5 = knn_classifier(data_train,datatest_5,K);
PF1 = sum(labels_test_1 == 5)/length(labels_test_1);
PD1 = sum(labels_test_1 == 1)/length(labels_test_1);
PF5 = sum(labels_test_5 == 1)/length(labels_test_5);
PD5 = sum(labels_test_5 == 5)/length(labels_test_5);
fprintf('for K=5\n')
fprintf(strcat("Detecetion for Digit 1 = ",num2str(PD1*100),"%%","\n"));
fprintf(strcat("Misclassification for Digit 1 = ",num2str(PF1*100),"%%","\n"));
fprintf(strcat("Detecetion for Digit 5 = ",num2str(PD5*100),"%%","\n"));
fprintf(strcat("Misclassification for Digit 5 = ",num2str(PF5*100),"%%","\n\n"));






function testLabel = knn_classifier(train_data,test_data,K)
for i = 1:size(test_data,2)
test_sample = test_data(:,i);
for j = 1:size(train_data,2)
train_sample = train_data(:,j);
distance(j) = sqrt(sum((test_sample - train_sample).^2));
end

[sort_distance, index] = sort(distance);
index_label(index <= 25) = 1;
index_label(index > 25) = 5;
K_labels = index_label(1:K);
label_1_count = sum(K_labels == 1);
label_5_count = sum(K_labels == 5);
if label_1_count > label_5_count
testLabel(i) = 1;
else
testLabel(i) = 5;
end
end
end
