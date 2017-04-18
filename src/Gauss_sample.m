clear all; clc; close all;

n = 25000;
x = zeros(n, 1);
x(1) = 0.5;

for i = 1 : n - 1
    % rand returns a number between (0, 1)
    % normpdf is normal school propbility dense function
    x_c = normrnd(x(1), 0.05);   
    if rand < min(1, normpdf(x_c)/normpdf(x(i)))
        x(i+1) = x_c;    
    else
        x(i+1) = x(i);    
    end
end
plot(x);