clear all;
clc;
%% 读取数据
% 取机组 启停时间 启停成本 燃料成本 爬坡速率
NGdata = xlsread("118bus.xlsx","机组参数");
NG = size(NGdata,1);
Ton = NGdata(:,10);
Toff = NGdata(:,9);
cost = NGdata(:,2:4);
cost(:,[1,3]) = cost(:,[3,1]);
STcost = NGdata(:,14);
Grate = NGdata(:,13);
Pmin = NGdata(:,6);
Pmax = NGdata(:,5);

% 取多场景下风力发电数据
Swinddata = xlsread("2022.xlsx","多场景下的风电出力");
Swinddata = Swinddata(2:11,:);
Snum = size(Swinddata,1);
NT = size(Swinddata,2);

% 取线路电感 传输容量 起始节点 末端节点
busdata = xlsread("118bus.xlsx", "线路参数");
X = busdata(:,5);
Plimit = busdata(:,7);
NodeNum = 118;
I = busdata(:,2);
J = busdata(:,3);
G = digraph(I,J);
IN = -incidence(G);
busNum = size(busdata,1);

% 取线路有功和无功负荷
loaddata = xlsread("2022.xlsx","预测负荷和风电出力");
loadP = loaddata(2,:);
% loadQ = loaddata(3,:);

% 系统备用容量
RST = 200;
ROT = 200;

%% 模型搭建
% 求解变量
I = binvar(NG,NT,Snum);
P = sdpvar(NG,NT,Snum);
Start = binvar(NG,NT,Snum);
Stop = binvar(NG,NT,Snum);
RS = sdpvar(NG,NT,Snum);
RO = sdpvar(NG,NT,Snum);
Xon = binvar(NG,NT,Snum);
Xoff = binvar(NG,NT,Snum);
theta = sdpvar(NodeNum,NT,Snum);
% g = sdpvar(NodeNum,NT,Snum);
% d = sdpvar(NodeNum,NT,Snum);
p = sdpvar(busNum,NT,Snum);

% % 发点节点
% g(1,:,:) = P(1,:,:);
% g(2,:,:) = P(2,:,:); 
% g(3,:,:) =0;
% g(4,:,:) = Swinddata';
% g(5,:,:) =0;
% g(6,:,:) = P(3,:,:);     

% % 负载节点
% d(1,:,:) = 0;
% d(2,:,:) = 0; 
% d(4,:,:) = 0;
% d(6,:,:) = 0;
% for s = 1:Snum
%     k = 0.9;
%     d(3,:,s) = k * loadP;
%     d(5,:,s) = (1-k) * loadP;
% end



% 目标函数
z = 0;
for s = 1:Snum
    for i = 1:NG
        for t = 1:NT
            FP = cost(i,1) * P(i,t,s)^2 + cost(i,2) * P(i,t,s) + cost(i,3);
            SU = Start(i,t,s) * STcost(i,1);
            SD = Stop(i,t,s) * STcost(i,1);
            z = z+(FP + SU + SD)/Snum;
        end
    end
end

% % 约束条件
Cons = [];                                         
for s = 1:Snum
    for t = 1:NT
        SumRS = 0;
        SumRO = 0;
        PI = 0;
        for i = 1:NG
            PI = PI + I(i,t,s) * P(i,t,s);
            Pf = Swinddata(s,t);
            SumRS = SumRS + RS(i,t,s) * I(i,t,s);
            SumRO = SumRO + RO(i,t,s) * I(i,t,s);
%             Cons = Cons + [IN * p(:,t,s) == g(:,t,s) - d(:,t,s)];   % 网络约束-功率约束
            Cons = Cons + [IN'*theta(:,t,s) == p(:,t,s).*X];        % 网络约束-功角约束
            Cons = Cons + [theta(1,t,s) == 0];                      % 网络约束-参考节点
            Cons = Cons + [p(:,t,s) <= Plimit];                     % 网络约束-传输容量
            if(t >= 2)
                Cons = Cons + [P(i,t,s) - P(i,t-1,s) <= Grate(i)];                                                                   % 爬坡约束
                Cons = Cons + [P(i,t-1,s) - P(i,t,s) <= Grate(i)];                                                                   % 爬坡约束   
                Cons = Cons + [(Xon(i,t-1,s) - Ton(i)) * (I(i,t-1,s) - I(i,t,s)) >= 0];                                              % 公式(7)
                Cons = Cons + [(Xoff(i,t-1,s) - Toff(i)) * (I(i,t,s) - I(i,t-1,s)) >= 0];                                            % 公式(8)
                Cons = Cons + [Pmin(i) * I(i,t,s) <= P(i,t,s)];                                                                      % 公式(9)           
                Cons = Cons + [Pmax(i) * I(i,t,s) >= P(i,t,s)];                                                                      % 公式(9) 
                Cons = Cons + [I(i,t,s) - I(i,t-1,s) == Start(i,t,s) - Stop(i,t,s)];
            end
        end
        Cons = Cons + [PI + Pf == loadP(t)];   % 公式(2)
        Cons = Cons + [SumRS >= RST];       % 公式(3)
        Cons = Cons + [SumRO >= ROT];       % 公式(4)
    end
end
Cons = Cons + [P >= 0];


% 求解
options=sdpsettings('solver', 'gurobi', 'verbose', 1);
tic
result=optimize(Cons,z,options);
toc

obj=double(z)
P_t=double(P);

% I = value(I);
% P = value(P);
% save("P.mat","P")