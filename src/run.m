close all; clc; home;

load('test.mat');
load('train.mat');
[number, numVisible, numbatches] = size(testbatchdata);

%--------------------------------------------------------------------------------------------------------------------------
% Calculate log Z  h10.mat
%--------------------------------------------------------------------------------------------------------------------------
% using h10.m to generate log_Z
fprintf('Start h10.mat \n');
load('h10.mat');
% using AIS
logZ_10 = AIS(parameter_a, parameter_b, parameter_W, MNIST_for_log);
% log-likehood of P(v)
p_10 = 0;
for i = 1 : number
    for j = 1 : numbatches
        p_10 = p_10 + exp(testbatchdata(i, :, j) * parameter_b' + sum(log(1 + exp(parameter_a + testbatchdata(i, :, j) * parameter_W)))-logZ_10);
    end
end
fprintf('Estimated  log-partition function : %f \n', logZ_10);
fprintf('Average estimated log_prob on the test testdata_a : %d \n', p_10);

%--------------------------------------------------------------------------------------------------------------------------
% Calculate log Z  h20.mat
%--------------------------------------------------------------------------------------------------------------------------
% using h10.m to generate log_Z
fprintf('Start h20.mat \n');
load('h20.mat');

% using AIS
logZ_20 = AIS(parameter_a, parameter_b, parameter_W, MNIST_for_log);
% log-likehood of P(v)
p_20 = 0;
for i = 1 : number
    for j = 1 : numbatches
        p_20 = p_20 + exp(testbatchdata(i, :, j) * parameter_b' + sum(log(1 + exp(parameter_a + testbatchdata(i, :, j) * parameter_W)))-logZ_20);
    end
end
fprintf('Estimated  log-partition function : %f \n', logZ_20);
fprintf('Average estimated log_prob on the test testdata_a : %d \n', p_20);

%--------------------------------------------------------------------------------------------------------------------------
% Calculate log Z  h100.mat
%--------------------------------------------------------------------------------------------------------------------------
% using h10.m to generate log_Z
fprintf('Start h100.mat \n');
load('h100.mat');

% using AIS
logZ_100 = AIS(parameter_a, parameter_b, parameter_W, MNIST_for_log);
% log-likehood of P(v)
p_100 = 0;
for i = 1 : number
    for j = 1 : numbatches
        p_100 = p_100 + exp(testbatchdata(i, :, j) * parameter_b' + sum(log(1 + exp(parameter_a + testbatchdata(i, :, j) * parameter_W)))-logZ_100);
    end
end
fprintf('Estimated  log-partition function : %f \n', logZ_100);
fprintf('Average estimated log_prob on the test testdata_a : %d \n', p_100);

%--------------------------------------------------------------------------------------------------------------------------
% Calculate log Z  h500.mat
%--------------------------------------------------------------------------------------------------------------------------
% using h10.m to generate log_Z
fprintf('Start h500.mat \n');
load('h500.mat');

% using AIS
logZ_500 = AIS(parameter_a, parameter_b, parameter_W, MNIST_for_log);
% log-likehood of P(v)
p_500 = 0;
for i = 1 : number
    for j = 1 : numbatches
        p_500 = p_500 + exp(testbatchdata(i, :, j) * parameter_b' + sum(log(1 + exp(parameter_a + testbatchdata(i, :, j) * parameter_W)))-logZ_500);
    end
end
fprintf('Estimated  log-partition function : %f \n', logZ_500);
fprintf('Average estimated log_prob on the test testdata_a : %d \n', p_500);

z = [ logZ_10, logZ_20, logZ_100, logZ_500, p_10, p_20, p_100, p_500];
save('z.mat', 'z');
