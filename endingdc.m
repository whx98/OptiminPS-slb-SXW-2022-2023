clc%%% 小算例 加DC
netpara=xlsread('2022','线路参数');
loadpoint=xlsread('2022','预测负荷和风电出力');  %%
gunmumcos=xlsread('2022','机组参数');

gennum_num=size(gunmumcos);%
Gnumber=gennum_num(1,1);   % 机组数
cost=gunmumcos(:,11:13);  %% 机组燃料成本 cost(:,1) cost(:,2) cost(:,3)
stardown =gunmumcos(:,6);   %机组启停成本
tlimit=gunmumcos(:,7:8);    % zuixiao kaiji shijian

Ramp=gunmumcos(:,6);    %机组爬坡功率
T=24;
r=netpara(:,4);   %%电阻
x=netpara(:,5);    %电抗
Plimit=netpara(:,6);  %% 线路传输容量限制
limit=gunmumcos(:,2:5);  %机组出力上下限//limit(:,1)表示有功上限，limit(:,2)表示有功下限  limit(:,3)和limit(:,4)分别表示无功上下限
load=loadpoint(2,:);  %有功负荷预测
w_p=loadpoint(4,:);  %% 风电出力
branch_num=size(netpara);
branch_num=branch_num(1,1);


%limit=G(:,3:4);%机组出力上下限//limit(:,1)表示上限，limit(:,2)表示下限
%para=G(:,5:8);%成本系数//para(:,1)表示线形成本,para(:,2)表示空载成本,para(:,3)表示关停成本，,para(:,4)表示启动成本。
%tlimit=G(:,9:10);%%tlimit(:,1)表示最小开启 tlimit(:,2)关停表示最小


D=[];c=[];d=[];
st=[];%%约束条件
%% 设定的成本系数
% Q = diag([.04 .01 .02]);
% C = [10 20 20];
%% 决策变量 不考虑随机性
P=sdpvar(branch_num,T);%支路功率
p=sdpvar(Gnumber,T,'full');
u=binvar(Gnumber,T,'full');%状态变量 01
v=binvar(Gnumber,T,'full');%启动动作 关停为1 其他为0
w=binvar(Gnumber,T,'full');%关停动作 关停为1 其他为0
% x=sdpvar(Gnumber*T,1);
% y=binvar(3*Gnumber*T,1);
% zlower=sdpvar(1,1);
theta=sdpvar(6,T);%% 节点电压相角
% O=[1 1 2 5]; % 这个结构是没有考虑变压器和分接开关
% E=[2 4 4 6];
O=netpara(:,2);
E=netpara(:,3);
G=digraph(O,E);
NI=incidence(G);
NI=-NI;

%% 目标函数
totalcost=0;
for i=1:Gnumber
for t=1:T
totalcost=totalcost+cost(i,3)*p(i,t)^2+cost(i,2)*p(i,t)+cost(i,1)+v(i,t)*stardown(i)+w(i,t)*stardown(i); 
% totalcost=totalcost+para(i,1)*p(i,t)+u(i,t)*para(i,2)+v(i,t)*para(i,3)+w(i,t)*para(i,4);
% % totalcost=totalcost+u(i,t)*(para(i,2)*limit(i,2)+para(i,1)*limit(i,2)^2+para(i,3))+para(i,2)*x(t)+para(i,1)*x(t)^2+para(i,3);%加上表示机组开机并以最小出力 运行产生的煤耗
% % totalcost=totalcost+costH(i,t)+costJ(i,t);%加上机组启停产生的开停机成本 totalcost=punishcost+totalcost+costH(i,t)+costJ(i,t)
end
end
%for k = 1:T
% totalcost = totalcost + p(:,k)'*Q*p(:,k) + C*p(:,k);
%end
%% 各种约束
for i=1:branch_num
        st=st+[abs(P(i,:))<=Plimit(i)];
    end
for t=1:T 
    st=st+[NI'*theta(:,t)==P(:,t).*x,theta(1,:)==0];%,p<=20];%% 一号节点是平衡节点
       
      %  st=st+[sum(d(:,t))==load(t)];%value(0)
st=st+[u(:,t).*limit(:,2)<=p(:,t)<=u(:,t).*limit(:,1)];%机组机组出力上下限约束 
end
for t=2:T
st=st+[u(:,t)-u(:,t-1)==v(:,t)-w(:,t)];%机组启停状态逻辑约束
end
% for i=1:Gnumber
% st=st+[u(i,1)-u(i,T)==v(i,1)-w(i,1)];%机组启停状态逻辑约束
% end
for t = 1:T
st = st+[ sum(p(:,t))+w_p(t) == load(t)];%% 系统负荷约束
end
% for t = 1:T
% st = st+[ sum(u(:,t).*limit(:,1)) >= load(t)];%% 系统负荷约束
% end
%% 启停时间约束
% for i=1:Gnumber
% for t=2:T
% %%% st=st+[u(i,range)>=v(i,t)];
% st=st+[v(i,)]
% end
% end
% %起停机约束 换种写法
for i=1:Gnumber
%for t=2:T
    for t=tlimit(:,1):T
   st=[st,sum(v(:,(t-tlimit(:,1)+1):t))<=u(i,t)];
    end
    for t=tlimit(:,2):T
    st=[st,sum(w(:,(t-tlimit(:,2)+1):t))<=1-u(i,t)];
    end
end
for t=2:T
        for i=1:Gnumber
%             
            st=st+[p(i,t)-p(i,t-1)<=Ramp*u(i,t-1)];%+v(i,t)*%(1-I(i,t)*(1-I(i,t-1)))+I(i,t)*(1-I(i,t-1))*limit(:,1)]; %%上坡约束
            st=st+[p(i,t-1)-p(i,t)<=Ramp*u(i,t)];%(1-I(i,t-1)*(1-I(i,t)))+I(i,t-1)*(1-I(i,t))*limit(:,2)]; %% 下坡约束
             
        end
end
ops=sdpsettings('solver', 'cplex');
result=solvesdp(st,totalcost,ops);
if   result.problem == 0
    value(u)
    value(totalcost)
s_p=value(P);
G=digraph(O,E,s_p(:,1));
plot(G, 'EdgeLabel', G.Edges.Weight, 'linewidth', 2);

else
    display('错了亲！');
    result.info
    yalmiperror(result.problem)
end