clear all
clc
% 118节点系统标准场景（可用作基准）

%%  参数读取与设置
netpara=xlsread('118bus','线路参数');
loadpoint=xlsread('118bus','负荷数据');  
unitpara=xlsread('118bus','机组参数');
p_wind=xlsread('2022','多场景下的风电出力');
fprintf("data loading completed\n")
p_wind=p_wind(2:11,1:24);

gennum_num=size(unitpara);
N_tp=gennum_num(1,1);   % 火电机组数
cost=unitpara(:,2:4);  %% 机组燃料成本 cost(:,1) cost(:,2) cost(:,3) 常数 一次 二次 每次都写反！！！！
stardown =unitpara(:,14);   %机组启停成本
tlimit=unitpara(:,9:10);    %最小开机时间

Ramp=unitpara(:,12);    %机组爬坡功率
T=24;
r=netpara(:,4);   %%电阻
x=netpara(:,5);    %电抗
Plimit=netpara(:,7);  %% 线路传输容量限制
limit=unitpara(:,5:8);  %机组出力上下限
load=loadpoint(:,2);  %有功负荷预测
w_p=[44	70.2	76	82	84	84	100	100	78	64	100	92	84	80	78	32	4	8	10	5	6	56	82	52];%风电出力预测
branch_num=size(netpara);
branch_num=branch_num(1,1);
O=netpara(:,2);
E=netpara(:,3);
G=digraph(O,E);
NI=incidence(G);
NI=-NI;
ops = sdpsettings('solver', 'gurobi', 'verbose', 1);
Cons=[];%%约束条件
fprintf("paremeter setting completed\n")
%% 决策变量 
p_t=sdpvar(N_tp,T,'full');%火电机组功率
P=sdpvar(branch_num,T,'full');%支路功率
u=binvar(N_tp,T,'full');%状态变量 01
v=binvar(N_tp,T,'full');%启动动作 开启为1 其他为0
w=binvar(N_tp,T,'full');%关停动作 关停为1 其他为0
theta=sdpvar(118,T);%% 节点电压相角
fprintf("variables setting completed\n")

%% 目标函数与约束
% obj=0;
% for i=1:N_tp
%     for t=1:T
%         obj=obj+cost(i,3)*p_t(i,t)^2+cost(i,2)*p_t(i,t)+cost(i,1)+v(i,t)*stardown(i)+w(i,t)*stardown(i); 
%     end
% end
% 
% for t=1:T 
%     Cons=Cons+[u(:,t).*limit(:,2)<=p_t(:,t),p_t(:,t)<=u(:,t).*limit(:,1)];%机组机组出力上下限约束 
%     Cons =Cons+[sum(p_t(:,t))+w_p(t) == load(t)];
%     Cons=Cons+[NI'*theta(:,t)==P(:,t).*x];%直流潮流
%     Cons=Cons+[theta(1,:)==0]; %PV节点
% end
% for t=2:T
%     Cons=Cons+[u(:,t)-u(:,t-1)==v(:,t)-w(:,t)];%机组启停状态逻辑约束
% end
% % %起停机约束 
% for i=1:N_tp
% %for t=2:T
%     for t=tlimit(:,1):T
%         Cons=[Cons,sum(v(:,(t-tlimit(:,1)+1):t))<=u(i,t)];
%     end
%     for t=tlimit(:,2):T
%         Cons=[Cons,sum(w(:,(t-tlimit(:,2)+1):t))<=1-u(i,t)];
%     end
% end
% for j=1:branch_num
%     Cons=Cons+[abs(P(j,:))<=Plimit(j)]; %支路容量
% end
% for t=2:T
%         for i=1:N_tp
% %             Cons=Cons+[p(i,t)-p(i,t-1)<=Ramp*(1-u(i,t)*(1-u(i,t-1)))+u(i,t)*(1-u(i,t-1))*limit(i,1)]; %%上坡约束
% %             Cons=Cons+[p(i,t-1)-p(i,t)<=Ramp*(1-u(i,t-1)*(1-u(i,t)))+u(i,t-1)*(1-u(i,t))*limit(i,2)]; %%下坡约束
%             Cons=Cons+[p_t(i,t)-p_t(i,t-1)<=Ramp(i)];
%             Cons=Cons+[p_t(i,t-1)-p_t(i,t)<=Ramp(i)];
%         end
% end
% %改写的爬坡约束
% % for i=1:N_tp
% %     Cons=Cons+[p_t(i,t)-p(i,t-1)<=ramp(i)*u(i,t-1)];
% %     Cons=Cons+[p_t(i,t-1)-p(i,t)<=ramp(i)*u(i,t)];
% %     for t=
% % end
% 
% fprintf("Constraints setting completed\n")
% tic
% result=optimize(Cons,obj,ops);
% toc
% fprintf("Solve completed \n")
% 
% Power_t=double(p_t);
% goal=double(obj);
% fprintf('min total cost is %4.3f',goal);

%% 多场景
goal=zeros(10,1);
Power_t=zeros(10,N_tp,T);
for s=1:10
    w_p=p_wind(s,:);
    obj=0;
    for i=1:N_tp
        for t=1:T
            obj=obj+cost(i,3)*p_t(i,t)^2+cost(i,2)*p_t(i,t)+cost(i,1)+v(i,t)*stardown(i)+w(i,t)*stardown(i); 
        end
    end
    
    for t=1:T 
        Cons=Cons+[u(:,t).*limit(:,2)<=p_t(:,t),p_t(:,t)<=u(:,t).*limit(:,1)];%机组机组出力上下限约束 
        Cons =Cons+[sum(p_t(:,t))+w_p(t) == load(t)];
        Cons=Cons+[NI'*theta(:,t)==P(:,t).*x];%直流潮流
        Cons=Cons+[theta(1,:)==0]; %PV节点
    end
    for t=2:T
        Cons=Cons+[u(:,t)-u(:,t-1)==v(:,t)-w(:,t)];%机组启停状态逻辑约束
    end
    % %起停机约束 
    for i=1:N_tp
    %for t=2:T
        for t=tlimit(:,1):T
            Cons=[Cons,sum(v(:,(t-tlimit(:,1)+1):t))<=u(i,t)];
        end
        for t=tlimit(:,2):T
            Cons=[Cons,sum(w(:,(t-tlimit(:,2)+1):t))<=1-u(i,t)];
        end
    end
    for j=1:branch_num
        Cons=Cons+[abs(P(j,:))<=Plimit(j)]; %支路容量
    end
    for t=2:T
            for i=1:N_tp
    %             Cons=Cons+[p(i,t)-p(i,t-1)<=Ramp*(1-u(i,t)*(1-u(i,t-1)))+u(i,t)*(1-u(i,t-1))*limit(i,1)]; %%上坡约束
    %             Cons=Cons+[p(i,t-1)-p(i,t)<=Ramp*(1-u(i,t-1)*(1-u(i,t)))+u(i,t-1)*(1-u(i,t))*limit(i,2)]; %%下坡约束
                Cons=Cons+[p_t(i,t)-p_t(i,t-1)<=Ramp(i)];
                Cons=Cons+[p_t(i,t-1)-p_t(i,t)<=Ramp(i)];
            end
    end
    %改写的爬坡约束
    % for i=1:N_tp
    %     Cons=Cons+[p_t(i,t)-p(i,t-1)<=ramp(i)*u(i,t-1)];
    %     Cons=Cons+[p_t(i,t-1)-p(i,t)<=ramp(i)*u(i,t)];
    %     for t=
    % end
    
    fprintf("Constraints setting completed\n")
    tic
    result=optimize(Cons,obj,ops);
    toc
    fprintf("Solve completed \n")
    
    Power_t(s,:,:)=double(p_t);
    goal(s)=double(obj);
    fprintf("Objective function of Scenario %d is %4.3f",s,goal(s))
end
