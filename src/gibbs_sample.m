nSamples = 10000;  
   
mu = [0 0]; % TARGET MEAN目标均值  
rho(1) = 0.8; % rho_21目标方差  
rho(2) = 0.8; % rho_12目标方差  
   
% INITIALIZE THE GIBBS SAMPLER    
minn = [-3 -3];  
maxx = [3 3];  
   
% INITIALIZE SAMPLES  
x = zeros(nSamples,2);  
x(1,1) = unifrnd(minn(1), maxx(1));%unifrnd生成连续均匀分布的随机数  
x(1,2) = unifrnd(minn(2), maxx(2));  
   
dims = 1:2; % INDEX INTO EACH DIMENSION  
   
% RUN GIBBS SAMPLER  
t = 1;  
while t < nSamples
    t = t + 1;  
    T = [t-1, t];  
    for iD = 1 : 2 % 2-dimension
        % 更新采样点  
        nIx = dims ~= iD; % *NOT* THE CURRENT DIMENSION找到另外一维nIx=[0 1]logical类型  
        % CONDITIONAL MEAN  
        muCond = mu(iD) + rho(iD)*(x(T(iD),nIx)-mu(nIx));%计算均值 = 表达式μ(1)+ρ(1)*(x(n,2)-μ(2)) 其中x(n,2)代表样本第n个数据的第二维  
        % CONDITIONAL VARIANCE  
        varCond = sqrt(1 - rho(iD)^2);  %计算方差  
        % DRAW FROM CONDITIONAL  
        x(t, iD) = normrnd(muCond, varCond);%正态分布随机函数，计算得到当前第t个数据的第1维数据value  
    end  
end  
   
% 画图
figure;  
h1 = scatter(x(:,1),x(:,2),'r.');%scatter描绘散点图，x为横坐标，y为纵坐标  
   
% CONDITIONAL STEPS/SAMPLES  
hold on;   %画出前五十个采样点  
for t = 1:50  
    plot([x(t,1),x(t+1,1)],[x(t,2),x(t,2)],'k-');  
    plot([x(t+1,1),x(t+1,1)],[x(t,2),x(t+1,2)],'k-');  
    h2 = plot(x(t+1,1),x(t+1,2),'ko');  
end  
   
h3 = scatter(x(1,1),x(1,2),'go','Linewidth',3);  
legend([h1,h2,h3],{'Samples','1st 50 Samples','x(t=0)'},'Location','Northwest')  
hold off;  
xlabel('x_1');  
ylabel('x_2');  
axis square  