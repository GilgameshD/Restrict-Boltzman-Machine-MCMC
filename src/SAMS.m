close all; clc; clear all; home;
load('h20.mat');
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

% parameter initialization
iter = 4000;
K = 100;
log_Z = zeros(1, K);  % Z(k) is 1 at the beggining, so log Z is 0
v = rand(1, size_v);
h = rand(size_h, 1);
drawPlot = zeros(1, iter);
betaPosition = randi([1, K], 1, 1);
% generate a uniform random value using K
range_K = linspace(0, 1, K);
% burn-in step
t_0 = 0.8 * iter;
tic
for n = 1 : iter
    % sample nest beta given v   beta | x  ~  (beta | x)
    betaPosition = gibbsSample_beta(log_Z, v', size_h, range_K, parameter_b_A, parameter_a_A, parameter_b, parameter_a, parameter_W, betaPosition);
    % gibbs sample vector v
    [v, h] = gibbsSample_v(v, range_K(betaPosition), parameter_a, parameter_b_A, parameter_b, parameter_W);
    
    % -----------------------------------------------   theory 1    -------------------------------------------------
    if n < t_0  % burn-in
        log_Z(betaPosition) = log_Z(betaPosition) +  K .* min(1/K, n^(-0.6));
    else
        log_Z(betaPosition) = log_Z(betaPosition) +  K .* min(1/K, 1/(n - t_0 + t_0^0.6));
    end
    log_Z = log_Z - log_Z(1);
    %---------------------------------------------------------------------------------------------------------------
    
    % -----------------------------------------------   theory 2    --------------------------------------------------
    % update log_Z
    % ¦Â is set to 0.6 or 0.8 and t0 is set such that the proportions of Lt = j are within 50% ~ 20%
%     if n < t_0  % burn-in 
%         log_Z = log_Z +  K .* q_beta_x(range_K, parameter_a, parameter_b, parameter_W, v, size_h, log_Z) .* min(1/K, n^(-0.6));
%     else
%         log_Z = log_Z +  K .* q_beta_x(range_K, parameter_a, parameter_b, parameter_W, v, size_h, log_Z) .* min(1/K, 1/(n - t_0 + t_0^0.6));
%     end
%     log_Z = log_Z - log_Z(1);
    %---------------------------------------------------------------------------------------------------------------
    
    drawPlot(n) = log_Z(K);
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
plot(drawPlot + logZ_A, 'b.');
xlabel('iteration number');
ylabel('value of log Zk');

