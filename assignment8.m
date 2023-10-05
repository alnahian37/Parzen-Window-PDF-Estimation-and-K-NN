clc;
clear all;
close all;


x1 = load('X1.mat').X1;
x2 = load('X2.mat').X2;

h1 = 3; 

samples = [1,16,36,64];
x = -20:0.01:20;
p = 0;
for i=1:length(samples)
n=samples(i);

p=p+1;
parzen_1 = parzen_calc(h1,n,x,x1);
subplot(4,2,p)
plot(x,parzen_1,'b');
title(strcat('PDF of X1, number of samples = ',num2str(n)));

p=p+1;
parzen_2 = parzen_calc(h1,n,x,x2);
subplot(4,2,p)
plot(x,parzen_2,'r');
title(strcat('PDF of X2, number of samples = ',num2str(n)));

end



%%%%% Part 1C %%%%%%

n = 64;
figure();

testdata = x2(65:100);
p_1 = parzen_calc(h1,n,testdata,x1);
p_2 = parzen_calc(h1,n,testdata,x2);

label_predicted = p_2./p_1> 1;
x = -15:0.01:15;
y=parzen_calc(h1,n,x,x1);
plot(x,y,'b');
hold on

y=parzen_calc(h1,n,x,x2);
plot(x,y,color='r');
hold on

x_false=testdata(label_predicted==0);
y_false=zeros(length(x_false),1);

x_detect=testdata(label_predicted ==1);
y_detect=zeros(length(x_detect),1);

plot(x_detect,y_detect,'o', color='g');
hold on
plot(x_false,y_false,'x',color='magenta');

xlabel('x values')
ylabel('PDF Estimate');
legend('PDF estimate of X1', 'PDF estimate of X2','True Detection','False Alarm');

PF = length(x_false)./length(testdata);
PD = length(x_detect)./length(testdata);

disp(strcat("PD for h1 = ", num2str(h1)," is ",num2str(PD),"\n"));
disp(strcat("PF for h1 = ", num2str(h1)," is ",num2str(PF),"\n\n"));






%%%%%% 1d %%%%%%


for h1=[1 0.5 10]
figure()
samples = [1,16,36,64];
x = -20:0.01:20;

p = 0;
for i=1:length(samples)
n=samples(i);

p=p+1;
parzen_1 = parzen_calc(h1,n,x,x1);
subplot(4,2,p)
plot(x,parzen_1,'b');
title(strcat('PDF of X1  for sample = ',num2str(n),' and h1=',num2str(h1)));

p=p+1;
parzen_2 = parzen_calc(h1,n,x,x2);
subplot(4,2,p)
plot(x,parzen_2,'r');
title(strcat('PDF of X2 for samples = ',num2str(n),' and h1= ',num2str(h1)));
end

end

for h1=[1 0.5 10]
PD_all=[];
PF_all=[];
n = 64;
figure();

testdata = x2(65:100);
p_1 = parzen_calc(h1,n,testdata,x1);
p_2 = parzen_calc(h1,n,testdata,x2);

label_predicted = p_2./p_1> 1;
x = -15:0.01:15;
y=parzen_calc(h1,n,x,x1);
plot(x,y,'b');
hold on

y=parzen_calc(h1,n,x,x2);
plot(x,y,'r');
hold on
x_false=testdata(label_predicted == 0);
y_false=zeros(length(x_false),1);

x_detect=testdata(label_predicted == 1);
y_detect=zeros(length(x_detect),1);

plot(x_detect,y_detect,'o', color='g');
hold on
plot(x_false,y_false,'x',color='magenta');

xlabel('x values')
ylabel('PDF Estimate');

title(strcat('for h1=',num2str(h1)))
legend('PDF estimate of X1', 'PDF estimate of X2','True Detection','False Alarm');

PF = length(x_false)/length(testdata);
PD = length(x_detect)/length(testdata);

PD_all=[PD_all, PD];
PF_all=[PF_all, PF];

fprintf(strcat("PD for h1 = ", num2str(h1)," is ",num2str(PD),"\n"));
fprintf(strcat("PF for h1 = ", num2str(h1)," is ",num2str(PF),"\n\n"));

end
%%%end of 1%%%



%Parzen Estimate function
function parz = parzen_calc(h1,n,x,xi)
hn = h1/sqrt(n);
parz = zeros(size(x));

for i = 1:n
z = (x - xi(i))./hn; %Gaussian
parz = parz + (1/(sqrt(2*pi)*hn)).*exp(-z.^2/2);
end
parz = parz/n;
end

