function [new_v, new_h] = gibbsSample_v(v, beta_k, parameter_a, parameter_b_A, parameter_b, parameter_W)

    % 1 / (1 + exp(-x)) = (1 + tanh(x / 2)) / 2
    % This way of computing the logistic is both fast and stable.
    % sample h from v 
    p = beta_k*(v * parameter_W + parameter_a);
    p = 1./(1 + exp(-p));
    new_h = rand(size(p)) < p;
    
    % sample new v from h
    p = beta_k * (new_h * parameter_W' + parameter_b) + (1 - beta_k)*parameter_b_A;
    p = 1./(1 + exp(-p));
    new_v = rand(size(p)) < p;
end
    