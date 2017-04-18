% 一般来说，可视节点的数目要多于隐藏节点.
% 这个算法区别于一般的MC方法的原因是使用了退火的思想，如果直接估计PA和PB两个可能
% 相差很大，但是如果采用一个平稳的序列就可以使得PA和PB十分接近。
% 整个过程是从一个基本的状态开始不断的退火收敛到需要的结果,而Gibbs抽样是MCMC的一个特例，
% 它交替的固定某一维度xi，然后通过其他维度x‘i的值来抽样该维度的值
% 注意，gibbs采样只对z是高维（2维以上）情况有效。
%-----------------------------------------------------------------------------------------------------------
% logZZ_est                  -- estimate of Z
% parameter_W            -- a matrix of RBM weights [numvis, numhid]
% parameter_a              -- a row vector of hidden  biases [1 numhid]
% parameter_b              -- a row vector of visible biases [1 numvis]
% numruns                   -- number of AIS runs
% beta                          -- a row vector containing beta's, (the inverse temperature）
% testbatchdata            -- the data that is divided into batches (numcases numdims numbatches)  
%------------------------------------------------------------------------------------------------------------

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
visibleBiases_A = (log(train_number - zero_count) - log(zero_count))';
%visibleBiases_A = 0*parameter_b;
fprintf('the process of train data completed \n');

numruns = 10;

% in different stage, we can have different step
beta = [0 : 0.001 : 0.5   0.5 : 0.0001 : 0.95   0.95 : 0.000001 : 1];

% Base model
[numVisible, numHidden] = size(parameter_W); 

% copy the three paramters to run repeatly
visibleBias_A = repmat(visibleBiases_A, numruns, 1); % biases of base model.  
hiddenBias_B = repmat(parameter_a, numruns, 1); 
visibleBias_B = repmat(parameter_b, numruns, 1);  

% Sample from the base model
logw_AIS = zeros(numruns, 1);
visible_B = repmat(1./(1+exp(-visibleBiases_A)), numruns, 1);  
visible_B = visible_B > rand(numruns, numVisible);
logw_AIS  =  logw_AIS - (visible_B*visibleBiases_A' + numHidden*log(2));

% update parameters
WightAndHidden = visible_B*parameter_W + hiddenBias_B; 
Bv_base = visible_B*visibleBiases_A';
Bv = visible_B*parameter_b';   
logZ_A = sum(log(1 + exp(visibleBiases_A))) + (numHidden)*log(2);  % the log-likehood of Z of PA

% the core process of  AIS, using a random sequence to transfer from PA to PB
temp = 1;
drawPlot_mean = zeros(size(beta)-1);  % for drawing
drawPlot_var = zeros(size(beta)-1);  % for drawing
tic
for eachBeta = beta(2 : end-1)
    % iteration 1
    expWh = exp(eachBeta*WightAndHidden);
    logw_AIS  =  logw_AIS + (1 - eachBeta)*Bv_base + eachBeta*Bv + sum(log(1+expWh), 2);
    
    % gibbs sample the new v', using losgistic function
    hidden_B = expWh ./ (1 + expWh) > rand(numruns, numHidden); 
    visible_B = 1 ./ (1 + exp(-(1-eachBeta)*visibleBias_A - eachBeta*(hidden_B*parameter_W' + visibleBias_B))); 
    visible_B = visible_B > rand(numruns, numVisible); % random number between 0 and 1
    
    % update parameters
    WightAndHidden = visible_B*parameter_W + hiddenBias_B;
    Bv_base = visible_B*visibleBiases_A';
    Bv = visible_B*parameter_b';
    
    % draw current value
    if mod(temp, 500) == 0
        fprintf('In step %d. The variance of weight is : %f . The mean of weight is %f \n', temp/500, var(logw_AIS( : )), mean(logw_AIS( : )));
    end
    drawPlot_var(temp) = var(logw_AIS( : ));
    drawPlot_mean(temp) = mean(logw_AIS( : ));
    temp = temp + 1;
        
    % iteration 2
    expWh = exp(eachBeta*WightAndHidden);
    logw_AIS  =  logw_AIS - ((1-eachBeta)*Bv_base + eachBeta*Bv + sum(log(1+expWh), 2));
end 
toc
% draw the variance of logw_AIS
figure(1)
plot(drawPlot_mean, 'b');
hold on
plot(drawPlot_var, 'r');
grid on
legend('mean value of log_w', 'variance value of low_w');
xlabel('the step of beta');
ylabel('log value');

logw_AIS  = logw_AIS + visible_B*parameter_b' + sum(log(1+expWh), 2);
% logAndSum returns the log of sum of logs, use w_AIS to get r_AIS (the ratio of two state)
% the function is log(sum(exp(x))) = alpha + log(sum(exp(x-alpha)));
alpha = max(logw_AIS, [ ], 1) - log(realmax) / 2;
logsum = alpha + log(sum(exp(logw_AIS - repmat(alpha, 1, 1)), 1));
% get r_AIS
r_AIS = logsum - log(numruns);   
% exp(r_AIS) = PB/PA, so logZZ = log(exp(r_AIS)) + log_base 
logZ_B = r_AIS + logZ_A;  
fprintf('final log_Z is %f \n', logZ_B);


