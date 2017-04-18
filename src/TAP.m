%         Compute the pseudo-likelihood of v using second order TAP
%         Parameters
%         ----------
%         a random vector v
%         parameter_a : the biaes of hidden
%         parameter_b : the biaes of visible
%         parameter_W : weight matrix
%
%         Notes
%         -----
%         according to the size of the parameters, the result can be
%         periodic or convergent

close all; clc; clear all; home;
load('h20.mat');
load('test.mat');

logistic = @(x) 1./(1+exp(-x));
[size_v, size_h] = size(parameter_W);

mh = rand(size_h, 1); 
mv = rand(1, size_v);
iters = 100;

states = zeros(1, iters);
tic
% use cross iteration to get stable state
for n = 1 : iters
    v_fluc = diag(mv - mv.^2);
    mh = logistic(parameter_a' + ...
                        (mv*parameter_W)' - sum(((parameter_W .^ 2) * diag(mh - 1/2))' * v_fluc, 2) + ...
                        2*sum((parameter_W.^3 * diag(mh.^2 - mh +1/6))' * diag((mv-mv.^2) .* (1/2 - mv)), 2));       % d(F_three)/dm
    h_fluc = diag(mh - mh.^2);
    mv = logistic(parameter_b + ...
                        (parameter_W*mh)' - (sum(diag(mv - 1/2) * (parameter_W .^ 2) * h_fluc, 2))' + ...
                        (2*sum(((parameter_W.^3)' * diag(mv.^2 - mv +1/6))' * diag((mh-mh.^2) .* (1/2 - mh)), 2))');  % d(F_three)/dm

    % TAP £¨beta = 1£©
    F_MF = mv*parameter_b' + parameter_a*mh + mv*parameter_W*mh;  
    
    % avoid the case of log(0)                       
    check_mh = 1 - mh;
    check_mh((check_mh == 0)) = 1;
    check_mv = 1 - mv;
    check_mv((check_mv == 0)) = 1;
    
    % calculate the entropy
    Entropy = sum(mv .* log(mv) + check_mv .* log(check_mv)) + sum(log(mh) .* mh + log(check_mh) .* check_mh);
    % calculate the F_onsager
    Onsager = (mh - mh.^2)' * (parameter_W.^2)' * (mv - mv.^2)' *0.5;
    % avoid  shave
    F_three = 2/3*sum(sum((parameter_W .^ 3 * diag((mh - mh.^2) .* (1/2 - mh)))' * diag((mv - mv.^2) .* (1/2 - mv)))); 

    freeEnergy_TAP = F_MF + Onsager - Entropy + F_three;
    fprintf('In the step %d. The free energy is : %f \n', n, freeEnergy_TAP);
    states(n) = freeEnergy_TAP;
end
toc
% if the iterstion result is a periodic value, we calculate a mean value
% else the result is convergent, so we get a true value
% if abs(states(n) - states(n-1)) > 1
%     finalFreeEnergy = mean(states(n/2 : n));
% else
%     finalFreeEnergy = states(n);
% end
finalFreeEnergy = states(n);
 
figure;
plot(states, 'g');
hold on
plot(states, 'b*');
xlabel('step count');
ylabel('log of every free energy point');
fprintf('the final free energy is : %f \n', finalFreeEnergy);


