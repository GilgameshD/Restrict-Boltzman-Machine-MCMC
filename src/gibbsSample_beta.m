function nextBeta = gibbsSample_beta(Zk, v, size_h, range_K, parameter_b_A, parameter_a_A, parameter_b, parameter_a, perameter_W, nowBeta)
    accepted = 1;
    K = length(range_K);
    % p*(k+1) / p*(k) = Z(k)  that is  log_Z = P(k+1) - P(k)
    % P(k) = P*(k) - log_Z 
    % this is the sequence in AIS, now we should find a new Zk in the whole Zk      
    % pk_now
    p_now = range_K(nowBeta)*parameter_b*v + (1-range_K(nowBeta))*parameter_b_A*v - Zk(nowBeta);
    for n = 1 : size_h
        p_now = p_now + ...
                       log(1 + exp((1-range_K(nowBeta)) * parameter_a_A(n))) + ...
                       log(1 + exp(range_K(nowBeta) * (parameter_a(n)+perameter_W( : , n)' * v)));
    end
    while(accepted ~= 0)       
        % pk_next. get a random new beta
        nextBeta = randi([1, K]);
        p_new = range_K(nextBeta)*parameter_b*v + (1-range_K(nextBeta))*parameter_b_A*v - Zk(nextBeta);
        for n = 1 : size_h
            p_new = p_new + ...
                          log(1 + exp((1 - range_K(nextBeta)) * parameter_a_A(n))) + ...
                          log(1 + exp(range_K(nextBeta) * (parameter_a(n) + perameter_W( : , n)' * v)));
        end
        % sample
       accepted = exp(p_new - p_now) < rand;
    end
end

