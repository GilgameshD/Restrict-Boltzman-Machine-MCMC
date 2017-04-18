function p_beta = q_beta_x(range_K, parameter_a, parameter_b, parameter_W, v, size_h, logZ)
    Z = exp(logZ);
    [~, K] = size(logZ);
    p_beta = zeros(1, K);
    normalizationNumber = 0;
    % some p is too big, we should divide a number
    if size_h == 500
        scale_number = 10^140;
    else
        scale_number = 1;
    end
    
    for n = 1 : K
        p = parameter_b*v' / 2;
        for m = 1 : size_h
            p = p + ...
                   log(1 + exp((1-range_K(n)) * parameter_a(m))) / 2 + ...
                   log(1 + exp(range_K(n) * (parameter_a(m) + v * parameter_W( : , m)))) / 2;
        end
        % (exp(p/2)/scale)^2 can get a linear scale for normalization
        p_beta(n) = (exp(p)/scale_number)^2 / Z(n);
        normalizationNumber = normalizationNumber + p_beta(n);
    end
    p_beta = p_beta / normalizationNumber;
end