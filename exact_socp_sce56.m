clear all
close all
clc
%% With/Without S_volt
WithSvolt=1; % 1表示有Svolt 
%% %% 系统参数设置
WithSvolt=1;
Ubase=12e3; % unit:V
Sbase=1e6;   % unit:VA
Ibase=Sbase/sqrt(3)/Ubase; % unit:A
Zbase=Ubase/Ibase/sqrt(3);  % unit: 
Netpara=xlsread('SCE56','网络参数');
Loadpoint=xlsread('SCE56','节点负荷');
L=size(Netpara,1);%支路数量
R=Netpara(:,4)/Zbase;%支路电阻,unit: p.u.
X=Netpara(:,5)/Zbase;%支路电抗,unit: p.u.
So=Loadpoint(:,3)*1e6/Sbase;%节点负荷
Judge=Loadpoint(:,4);%节点负荷类型
I_max=12* 1e6 / Ubase / 1.732/ Ibase ; 
v_max=1.1^2;%电压平方上限
v_min=0.9^2;%电压平方下限
w=sqrt(-1);
z=R+X*w; %支路电抗的标幺值   
z_c=conj(z); % 共轭阻抗

%% 网络图生成
I=Netpara(:,2);
J=Netpara(:,3);
GN=graph(I,J);
h1=figure;
p=plot(GN);
highlight(p,find(Judge==1),'Marker','s','NodeColor','c','Markersize',10);  % Capacitator
highlight(p,find(Judge==0),'Marker','v','NodeColor','y','Markersize',10);  % PV panels
In=myincidence(I,J); % 关联矩阵生成
Inn=In;
Inn(Inn>0)=0;    %Inn是In的非正部分
%% 节点数
N=56;
%% 变量设置
P_ij=sdpvar(L,1); 
Q_ij=sdpvar(L,1); 
U=sdpvar(N,1); 
V=sdpvar(N,1);
L_ij=sdpvar(L,1) ;
Pi=sdpvar(N,1);
Qi=sdpvar(N,1);
%% 约束生成
Cons=[];
Cons_Load=[];
Eta=2.8;  % PV渗透率
Pi_max=zeros(N,1);
Qi_max=zeros(N,1);
for i=1:N
    if Judge(i)==1    %1表示电容负荷
        Cons_Load=[Cons_Load,Pi(i)==0];
        Qi_max(i)=So(i)*Eta;
        Cons_Load=[Cons_Load,0<=Qi(i)<=Qi_max(i)]; 
    elseif Judge(i)==2 %2表示普通节点负荷
        Pi_max(i)=-So(i)*0.9;
        Qi_max(i)=-So(i)*sqrt(1-0.9^2);
        Cons_Load=[Cons_Load,Pi(i)==Pi_max(i)];  
        Cons_Load=[Cons_Load,Qi(i)==Qi_max(i)];              
    elseif Judge(i)==0  %0表示PV负荷
        Pi_max(i)=So(i)*Eta;
        Cons_Load=[Cons_Load,0<=Pi(i)<=Pi_max(i)];
        Cons_Load=[Cons_Load,0==Qi(i)];
    elseif Judge(i)==3  %%3表示端节点
        Cons_Load=[Cons_Load,-So(i)<=Pi(i)<=So(i)];
        Cons_Load=[Cons_Load,-So(i)*0.5<=Qi(i)<=So(i)*0.5];
    end
end
Cons=[Cons,Cons_Load];
display('已生成识别节点负荷类型的约束')
% Cons_Load

%判断是否满足C1条件
Deg = degree(GN); % degree of each node
Deg
LeafNodes=find(Deg==1);
LeafNodes=LeafNodes(2:end); % 去除节点1
Pij_hat=abs(inv(In(2:end,:))*Pi_max(2:end));
Qij_hat=abs(inv(In(2:end,:))*Qi_max(2:end));
IJ=[I J];
IJ
Al=cell(length(LeafNodes),1);
Path_L=cell(length(LeafNodes),1);
Al_uij=zeros(length(LeafNodes),2);
for i=1:length(LeafNodes)
    lt=LeafNodes(i);
    Path = shortestpath(GN,lt,1);
    Al{i}=zeros(2,2,length(Path)-1);
    length(Path)-1
    for j=1:length(Path)-1 % for all node in Path from lt锟斤拷1, calculate A_l
%         nl=[]; % the set of 1,2,...,nl in C1
        for k=1:L
            if (IJ(k,1)==Path(j) && IJ(k,2)==Path(j+1))||(IJ(k,2)==Path(j) && IJ(k,1)==Path(j+1))
                Path_L{i}=[Path_L{i},Path(j+1)];
                Al{i}(:,:,j)=diag([1 1])-2/v_min*mtimes([R(k);X(k)],[Pij_hat(k),Qij_hat(k)]);
            end
        end
    end
    Al_uij_temp=diag([1 1]);
    for m=2:length(Path)-1
        Al_uij_temp=Al_uij_temp*Al{i}(:,:,m);
    end
    Al_uij(i,:)=(Al_uij_temp*[R(find(J==lt));X(find(J==lt))])';
end
if find(Al_uij<0)
    disp('不满足C1条件')
else
    disp('满足C1条件!')
end

%% 电压与电流约束
Cons_LimitI=[I_max^2>=L_ij>=0];
Cons=[Cons,Cons_LimitI];
Cons_LimitU=[V(2:end)>=v_min];
Cons=[Cons,Cons_LimitU]

% Cons_Limits=[I_max^2>=L_ij>=0, V(2:end)>=v_min];
% Cons=[Cons,Cons_Limits];
disp('已生成电压与电流约束')
% Cons_Limits
Cons_V=[];
Cons_V=[Cons_V, V(1)==v_max];
Cons_V=[Cons_V, In'*V == 2*R.*P_ij+2*X.*Q_ij-(R.^2+X.^2).*L_ij];
Cons=[Cons,Cons_V];
disp('已生成电压约束！')
% Cons_V

%% SOC约束
Cons_SOC=[];
for k=1:L
    i=I(k);%锟阶节碉拷
    Cons_SOC=[Cons_SOC,(L_ij(k)+V(i)).^2 >= 4*P_ij(k).^2 + 4*Q_ij(k).^2 + (L_ij(k)-V(i)).^2];
end
Cons=[Cons,Cons_SOC];
disp('已生成SOCP约束！')
% Cons_SOC

%% 潮流约束
Cons_PF=[In*P_ij - Inn*(R.*L_ij) == Pi, In*Q_ij - Inn*(X.*L_ij) == Qi];
Cons=[Cons,Cons_PF];
disp('已生成潮流平衡约束!')
% Cons_PF

%% Svolt约束
if WithSvolt==1
    P_ij_s=sdpvar(L,1); %the solution of the Linear DistFlow model
    Q_ij_s=sdpvar(L,1); 
    v_s=sdpvar(N,1); 
    Cons_Svolt=[];
    Cons_Svolt=[Cons_Svolt, v_max>=v_s(2:end)];
    Cons_Svolt=[Cons_Svolt, v_s(1)==v_max];
    Cons_Svolt=[Cons_Svolt, In'*v_s == 2*R.*P_ij_s+2*X.*Q_ij_s];
    Cons_Svolt=[Cons_Svolt, In(2:end,:)*P_ij_s == Pi(2:end), In(2:end,:)*Q_ij_s == Qi(2:end)];
    Cons=[Cons, Cons_Svolt];
    display('已生成Svolt的约束!')
end

%% 目标函数设置
Pr_sub=0.9;
Pr_pv=[0.8];
Pr_pv*Pi([find(Judge==0)]);
Pr_p=1;
disp(find(Judge==0));
Obj=Pr_pv*Pr_p*Pi([find(Judge==0)])+Pr_sub*Pi(1);
% C=sum(Pi); 
%% 模型求解
ops=sdpsettings('solver', 'gurobi');
result=optimize(Cons,Obj,ops);
result.info

if result.problem==0
 %% 求解结果
    s_P_ij=value(P_ij)*Sbase/1e6;
    s_Q_ij=value(Q_ij)*Sbase/1e6;
    s_v=value(V);
    s_l_ij=value(L_ij);
    s_Pi=value(Pi)*Sbase/1e6;
    s_Qi=value(Qi)*Sbase/1e6;
    display(['分布式光伏电源',num2str(sum(s_Pi([find(Judge==0)])),3),' MW']);
    display(['与主网相连的根节点',num2str(sum(s_Pi(1)),3),' MW']);
    if WithSvolt==1
        s_P_ij_s=value(P_ij_s)*Sbase/1e6;
        s_Q_ij_s=value(Q_ij_s)*Sbase/1e6;
        s_v_s=value(v_s);
    end
    
  %%  作图
    for k=1:L
        if norm([s_P_ij(k) s_Q_ij(k)])~=0
            highlight(p,I(k),J(k),'LineWidth',norm([s_P_ij(k) s_Q_ij(k)]));  % 锟斤拷锟斤拷路锟斤拷锟斤拷锟斤拷锟斤拷为锟斤拷锟斤拷锟斤拷小
        end
    end
    os=0.1; % offset value to print text in the figure
    text(p.XData-os, p.YData-os, num2str(sqrt(s_v)),'HorizontalAlignment', 'center','FontSize', 10, 'Color','g'); % printf the nodes' voltage (sqrt(v_j)).
    N_cap=find(Judge==1);
    N_pv=find(Judge==0);
    text(p.XData(N_cap)-os, p.YData(N_cap)+os, num2str(s_Qi(N_cap)*1e6/Sbase,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','r'); % printf the nodes' Qi if it's a node with Capacitator.
    text(p.XData(N_pv)+os, p.YData(N_pv)-3*os, num2str(s_Pi(N_pv)*1e6/Sbase,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','k'); % printf the nodes' Pi if it's a node with PVs.
    for k=1:L
        Coor1_PQ=(p.XData(I(k))+p.XData(J(k)))/2;
        Coor2_PQ=(p.YData(I(k))+p.YData(J(k)))/2;
        text(Coor1_PQ, Coor2_PQ, num2str(s_P_ij(k)+s_Q_ij(k)*w,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','b'); % printf the complex power in distribution lines(if any).
    end
 %% SOCP松弛的Exact判断
    for k=1:L
        i=I(k);
        SOC_gap(k)=(s_l_ij(k)+s_v(i)).^2-(4*(s_P_ij(k)*1e6/Sbase).^2+4*(s_Q_ij(k)*1e6/Sbase).^2+(s_l_ij(k)-s_v(i)).^2);
        if round(SOC_gap(k),3)==0 
          
        else
            disp(['Line ',num2str(I(k)),' ',num2str(J(k)),' SOCP松弛不Exact,' 'Line impedance z = ',num2str(z(k)*Zbase,3), 'Ohm'])
        end
    end
    SOC_gap=SOC_gap';
    %%  Svolt判断
   if WithSvolt==1
        title("对于sce56节点算例的精确SOCP松弛， With Svolt")
    else
        title("对于sce56节点算例的SOCP松弛， WITHOUT Svolt")
    end
end


