function [W,Z,lambda] = pca(image,num_of_vector)

n = length(image(1,:));

dim = length(image(:,1)); %number of features
mu_1 = mean(image,2);
Sigma = zeros(dim,dim);
for k = 1 : n
x = image(:,k);
Sigma = Sigma + (x - mu_1)*(x - mu_1)';
end

Sigma = Sigma./n;
[phi,lambda] = eig(Sigma,'vector');
[lambda, ind_sorted] = sort(lambda,'descend');
phi = phi(:,ind_sorted);
W = phi(:,1:num_of_vector);

for k = 1:n
    x = image(:,k);
    z = W'*(x - mu_1);
    Z(:,k) = z;
end