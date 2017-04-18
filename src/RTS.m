% We sample beta from (0, 1), and the range is divided into K. When ck is
% closed to rk, means that we have already sampled enough points. So we
% should iterate several times.
%  -----------------------------------------------------------------------------
%  step 1 : sample x
%  step 2 : sample beta using x
%  step 3 : update ck
%  step 4 : update Zk
%  ----------------------------------------------------------------------------
%  for h10.m  --- iter = 100
%  for h20.m  --- iter = 40
%  for h100.m --- iter = 200
%  for h500.m --- iter = 10
%  ----------------------------------------------------------------------------

close all; clc; clear all; home;
load('h500.mat');
load('test.mat');
load('train.mat');

[train_number, train_size] = size(MNIST_for_log);
[size_v, size_h] = size(parameter_W);
zero_count = zeros(train_size, 1);
for n = 1 : train_size
    zero_count(n) = length(find(MNIST_for_log(:, n) == 0)) - 1;
end
parameter_b_A = (log(train_number - zero_count) - log(zero_count))';
parameter_a_A = zeros(1, length(parameter_a));
fprintf('the process of train data completed \n');

% parameter initialization
iter = 100;
N = 30;
K = 100;
log_Z = zeros(1, K);  % Z(k) is 1 at the beggining, so log Z is 0
v = rand(1, size_v);
h = rand(size_h, 1);
drawPlot = zeros(1, iter);
ck = zeros(1, K);  % every c(k)
betaPosition = randi([1, K], 1, 1);
% generate a uniform random value using K
range_K = linspace(0, 1, K);
tic
for n = 1 : iter
    for i = 1 : N
        % gibbs sample vector v
        [v, h] = gibbsSample_v(v, range_K(betaPosition), parameter_a, parameter_b_A, parameter_b, parameter_W);
        % sample nest beta given v   beta | x  ~  (beta | x)
        % a new beta should have a smaller logZ, this part is similar to
        % AIS, whose beta is from 0 to 1
        betaPosition = gibbsSample_beta(log_Z, v', size_h, range_K, parameter_b_A, parameter_a_A, parameter_b, parameter_a, parameter_W, betaPosition);
        % ck is being close to rk which is 1/K;
        ck = ck + 1/N *q_beta_x(range_K, parameter_a, parameter_b_A, parameter_a_A, parameter_W, v, size_h, log_Z);
    end
    % update log_Z
    log_Z = log_Z + log(ck/ck(1));
    drawPlot(n) = log_Z(K);
    ck = ck / sum(ck);
    fprintf('This is %d step. The log_Z(K) is %f  \n', n, log_Z(K));
end
toc

% get the final log_Z, using the last Z(k) and Z_A
logZ_A = 0; 
for n = 1 : size_v
    logZ_A = logZ_A + log(1+exp(parameter_b_A(n)));
end
for n = 1 : size_h
    logZ_A = logZ_A + log(1+exp(parameter_a_A(n)));
end
log_Z_final = logZ_A + log_Z(K);
fprintf('the final log_Z is %f  \n', log_Z_final);

% draw stars and line at the same plot
figure(1);
plot(drawPlot, 'g');
hold on;
plot(drawPlot, 'b*');
xlabel('iteration number');
ylabel('value of log Zk');

