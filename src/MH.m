% 当Markov链经过burn-in阶段，消除初始参数的影响，到达平稳状态后，每一次状态转移都可以生成待模拟分布的一个样本。

clear all; clc; close all; home;

% parameters
N = 40000;                      % sample points
acceptPercent = 0;           % record accetped rate
stateStore = zeros(2, N);   % store the markov chain
x_t = [1 1];                        % random start point

% target normal function pdf
muX = 5;   varX = 1;  
muY = 10;   varY = 4;  covXY = 1;
muMatrix = [muX muY]; 
covMatrix = [varX covXY; covXY varY]; 
p = @(x) mvnpdf(x, muMatrix, covMatrix); 

% matrix Q -- a random tansfer matrix
varX_transfer  = 1;    
varY_transfer = 1;  
corrXY_transfer = 0.5;   
covXY_transfer = sqrt(varX_transfer)*sqrt(varY_transfer)*corrXY_transfer;   
cov_transfer = [varX_transfer      covXY_transfer; 
                          covXY_transfer   varY_transfer];         
% use mvnpdf to generate a pdf function of normal, (two parameters) 
% mu is the state we have known, and x is a point we want to know, so this
% is a condition propobility
q = @(x, mu) mvnpdf(x, mu, cov_transfer);          
% mvnpdf, (one parameter) 
% generate a 2-D normal function whose mean is mu and cov is cov_transfer
q_s = @(mu) mvnrnd(mu, cov_transfer);  

% Metropolis-Hastings process 
for i = 1 : N   
    if mod(i, 1000) == 0
        fprintf('current percent is: %f \n', i/N);
    end
    % sampling
    y = q_s(x_t);   
    % calculate iterator result
    alpha = (p(y) * q(x_t, y)) / (p(x_t) * q(y, x_t));    
    
    %---------------------------------------------
    %alpha =  (p(y)) / (p(x_t));    
    %---------------------------------------------
    
    % uniform pdf, judging the accepted result
    if rand <= min(alpha, 1)
        x_t_next = y;       % Accept the candidate
        accepted = 1;                      
    else
        x_t_next = x_t;     % Reject the candidate and use the same state
        accepted = 0;                    
    end
    % for next iteration
    x_t = x_t_next;
    stateStore(:, i) = x_t;    % store all accepted states
    acceptPercent = acceptPercent + accepted;   % sum of the points accepted
end

% 'average accepted point
accrate = acceptPercent/N;
fprintf('average accepted point is %f \n', accrate);
% two kinds of correlation
correlationCalculate = covXY/(sqrt(varX)*sqrt(varY));
fprintf('correlation which is calculated is  %f \n', correlationCalculate);
startPoint =N - N/2;
endPoint = N;
correlationSimulate = corrcoef(stateStore(1, startPoint : endPoint), stateStore(2, startPoint : endPoint));
fprintf('correlation which is simulated is  %f \n', correlationSimulate(1, 2));

% 3-D plot
X = (2 : 0.1 : 16)';
Y = (2 : 0.1 : 16)';
[X_axis, Y_axis] = meshgrid(X, Y);
% get a rectangle part of function p
Z = p([X_axis(:) Y_axis(:)]);  
Z = reshape(Z, length(Y_axis), length(X_axis));
figure(1);
surf(X, Y, Z);  grid on; shading interp;
xlabel('X', 'FontSize', 12);  
ylabel('Y', 'FontSize', 12);  
title('f_{XY}(x,y)', 'FontSize', 15);

% draw the step-points
figure(2);
step1 = 9*N/10;
step2 = 99*N/100;
plot(stateStore(1, 1 : step1), stateStore(2, 1 : step1), 'y.'); hold on; 
plot(stateStore(1, step1+1 : step2), stateStore(2, step1+1 : step2), 'g.'); hold on; 
plot(stateStore(1, step2+1 : N-100), stateStore(2, step2+1 : N-100), 'r.'); hold on; 
plot(stateStore(1, N-100 : N), stateStore(2, N-100 : N), 'b.'); hold on; 
xlabel('X', 'FontSize', 12);   
ylabel('Y', 'FontSize', 12);
title('sample points', 'FontSize', 15, 'LineWidth', 2);


