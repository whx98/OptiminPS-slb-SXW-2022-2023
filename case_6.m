clear all
clc
%% data load and parameter setting %%
netpara=xlsread('6bus','线路参数');
loadpoint=xlsread('6bus','预测负荷和风电出力');  
gunmumcos=xlsread('6bus','机组参数');
s_pw=xlsread('6bus','多场景下的风电出力');
fprintf("data loading completed\n")
s_pw=s_pw(2:11,1:24);%%行表示场景数，列表示小时
gennum_num=size(gunmumcos);
NG=gennum_num(1,1);   % 火电机组数
cost=gunmumcos(:,11:13);  %% 机组燃料成本 cost(:,1) cost(:,2) cost(:,3) 分别为常数项 一次项和二次项
startup=gunmumcos(:,14);  %机组启成本
shutdown = gunmumcos(:,15);   %机组停成本
t_on=gunmumcos(:,7);    %  min on time
t_off=gunmumcos(:,8);      % min off time
ramp=gunmumcos(:,6);    %机组爬坡功率  13是上坡和14是下坡 up and down is the same so no single deal 
T=24;
power=zeros(3,24,10);
r=netpara(:,4);%%电阻
x=netpara(:,5);%电抗   
P_limit=netpara(:,6);%% 线路传输容量限制  
limit=gunmumcos(:,2:5);%机组有功无功出力上下限 有上有下
load=loadpoint(2,:);%有功负荷预测
%w_p=loadpoint(4,:);%% 风电出力
branch_num=size(netpara);
branch_num=branch_num(1,1);
sys_R1= 200 ;  %系统旋转备用初始化
sys_R2= 200 ; %系统运行备用初始化
fprintf("paremeter setting completed\n")
%% set variables %%
S=10;
I=binvar(NG,T); 
P=sdpvar(NG,T,S);%% 机组出力调度 P p
p=sdpvar(branch_num,T,S);%支路功率
x1=sdpvar(NG,T);%机组关机时间
x2=sdpvar(NG,T);%机组开机时间
R1=sdpvar(NG,T,S);%旋转备用
R2=sdpvar(NG,T,S);%运行备用
v=binvar(NG,T);%启动动作 关停为1 其他为0
w=binvar(NG,T);%关停动作 关停为1 其他为0
theta=sdpvar(6,T,S);%% 节点电压相角
% O=[1 1 2 5]; % 这个结构是没有考虑变压器和分接开关
% E=[2 4 4 6];
O=netpara(:,2);
E=netpara(:,3);
G=digraph(O,E);
NI=incidence(G);
NI=-NI;
d=sdpvar(2,T,S);%%表示3、5节点的负载
ops = sdpsettings('solver', 'gurobi', 'verbose', 1);
% obj=0;
fprintf("variables setting completed\n")
for s=1:S
    obj=0;
    for t=1:T
        Cons=[sum(P(:,t,s))+s_pw(s,t)==load(t)]; %功率平衡
        Cons=[Cons,NI'*theta(:,t,s)==p(:,t,s).*x];%直流潮流
        Cons=[Cons,theta(1,:,s)==0]; %PV节点
        Cons=[Cons,sum(d(:,t,s))==load(t)];
        Cons=[Cons,sum(R1(:,t,s))>=sys_R1,sum(R2(:,t,s))>=sys_R2]; %备用约束
        Cons=[Cons,limit(:,2).*I(:,t)<=P(:,t,s),P(:,t,s)<=limit(:,1).*I(:,t)]; %出力约束
        for i=1:NG
%             obj=obj+I(i,t)*(cost(i,1)*P(i,t)^2+cost(i,2)*P(i,t)+cost(i,3))+v(i,t)*stardown(i)+w(i,t)*stardown(i);
            obj=obj+(cost(i,3)*P(i,t)^2+cost(i,2)*P(i,t)+cost(i,1))+v(i,t)*startup(i)+w(i,t)*shutdown(i);
        end
    end
    for j=1:branch_num
        Cons=[Cons,abs(p(j,:,s))<=P_limit(j)]; %支路容量
    end
    for t=2:T
        for i=1:NG
            Cons=[Cons,P(i,t,s)-P(i,t-1,s)<=ramp*(1-I(i,t)*(1-I(i,t-1)))+I(i,t)*(1-I(i,t-1))*limit(:,1)]; %ramp up
            Cons=[Cons,P(i,t-1,s)-P(i,t,s)<=ramp*(1-I(i,t-1)*(1-I(i,t)))+I(i,t-1)*(1-I(i,t))*limit(:,2)]; %ramp down
            Cons=[Cons,I(i,t)-I(i,t-1)==v(i,t)-w(i,t)]; %启停约束
            Cons=[Cons,(x1(i,t-1)-t_on(i))*(I(i,t-1)-I(i,t))>=0]; %开机时间约束
            Cons=[Cons,(x2(i,t-1)-t_off(i))*(I(i,t)-I(i,t-1))>=0];%关机时间约束
        end
    end
    fprintf("Constraints setting completed\n")
    tic
    result=optimize(Cons,obj,ops);
    toc
    fprintf("Scenario %d Solution completed \n",s)
    res(s)=double(obj);
    power(:,:,s)=double(P(:,:,s));
    fprintf("Objective function of Scenario %d is %4.3f",s,res(s))
end