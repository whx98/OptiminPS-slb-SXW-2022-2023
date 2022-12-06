%% SOCP relaxation for distribution system optimal power flow
% [1] Gan L.,Li N.,Topcu U.and H. Low S. 2015. Exact Convex Relaxation of Optimal Power Flow in Radial Networks.
% SCOP是否exact的充分条件参考了沈老师在github上的代码（https://github.com/a280558071/SOCPR-and-Linear-Disrflow-based-DNP/blob/main/ExactSOCPRforDOPF_sxw.m）
clc;
clear;

%% 首先确定是要求SOCP问题还是SOCP-m问题
judge=1; % 0为OPF；1为OPF-m

%% 参数设定
Ubase=12.35e3; % 单位: 伏
Sbase=1e7;   % 单位: 伏安
Ibase=Sbase/sqrt(3)/Ubase; % 单位: 安
Zbase=Ubase^2/Sbase;  % 单位: 欧
I_max=560.98/Ibase; % 取P_max为1.2MW,I_max=12e5/Ubase/sqrt(3)= 560.98 A,
v_max=1.1^2;
v_min=0.9^2;
parameter_line=xlsread('SCE56bus','Line Data');
parameter_bus=xlsread('SCE56bus','Load Data');
r=parameter_line(:,4)/Zbase;% p.u.
x=parameter_line(:,5)/Zbase;% p.u.
w=sqrt(-1); %定义j
z=r+x*w;
z_conj=conj(z); %z的共轭
So=parameter_bus(:,3)*1e6/Sbase;% p.u.
Nodetype=parameter_bus(:,4); % 0表示接入PV的节点，1表示接入并联电容器组的节点，2表示普通节点负荷，3表示变电站节点
B=56; % Bus的数量
L=B-1; %Line数量(对于配电网，没有环网，支路数=节点数-1)

%% 初始化网络图
I=parameter_line(:,2);
J=parameter_line(:,3);
G=graph(I,J);
p=plot(G);
highlight(p,find(Nodetype==1),'Marker','s','NodeColor','r','Markersize',10);  % Shunt Capacitator
highlight(p,find(Nodetype==3),'Marker','s','NodeColor','r','Markersize',10);  % 节点1既是变电站节点，也是有Capacitator的节点
highlight(p,find(Nodetype==0),'Marker','v','NodeColor','y','Markersize',10);  % PV panels
In=myincidence(I,J); 
Innegative=In;
Innegative(Innegative>0)=0;    % the lines end nodes

%% 决策变量设定
P_ij=sdpvar(L,1); % 节点i到节点j的支路有功功率
Q_ij=sdpvar(L,1); % 节点i到节点j的支路无功功率
u=sdpvar(B,1); % 电压幅值
v=sdpvar(B,1); % 电压幅值的平方
l_ij=sdpvar(L,1) ;% 电流幅值的平方
Pi=sdpvar(B,1); % 节点的注入有功功率
Qi=sdpvar(B,1); % 节点的注入无功功率

%% 开始添加约束
Cons=[];

%% 节点功率平衡约束
Cons_PF=[In*P_ij - Innegative*(r.*l_ij) == Pi, In*Q_ij - Innegative*(x.*l_ij) == Qi]; %Inn代表支路流入节点
Cons=[Cons,Cons_PF];
display('Constraints on power flow caculation completed!')

%% 欧姆定律约束
Cons_V=[];
Cons_V=[Cons_V, v(1)==v_max];
Cons_V=[Cons_V, In'*v == 2*r.*P_ij+2*x.*Q_ij-(r.^2+x.^2).*l_ij];
Cons=[Cons,Cons_V];
display('Constraints on voltage calculation completed!')

%% 支路首端功率约束（SOCP松弛即添加到此约束中）
Cons_SOC=[];
for k=1:L %对每一条支路
    i=I(k);% Starting node of line k
    Cons_SOC=[Cons_SOC,(l_ij(k)+v(i)).^2  >= 4*P_ij(k).^2 + 4*Q_ij(k).^2 + (l_ij(k)-v(i)).^2];
end
Cons=[Cons,Cons_SOC];
display('Constraints on SOCP relaxation completed!')

%% 节点注入功率约束（根据不同的节点类型）
Cons_Load=[];
Eta=0.15;  % Scale up the PV/Capacitator rate
Pi_max=zeros(B,1);
Qi_max=zeros(B,1);
for i=1:B %Pi_max(i)与Qi_max(i)
    if Nodetype(i)==1     % Capacitator
        Cons_Load=[Cons_Load,Pi(i)==0]; %并联电容器组补偿无功
        Qi_max(i)=So(i)*Eta; 
        Cons_Load=[Cons_Load,0<=Qi(i)<=Qi_max(i)];  
    elseif Nodetype(i)==2  % Load node（功率因数为0.9）
        Pi_max(i)=-So(i)*0.9; 
        Qi_max(i)=-So(i)*sqrt(1-0.9^2);
        Cons_Load=[Cons_Load,Pi(i)==Pi_max(i)];  
        Cons_Load=[Cons_Load,Qi(i)==Qi_max(i)];  
    elseif Nodetype(i)==0  %PV panel
        Pi_max(i)=So(i)*Eta;
        Cons_Load=[Cons_Load,Qi(i)==0];
        Cons_Load=[Cons_Load,0<=Pi(i)<=Pi_max(i)];
    elseif Nodetype(i)==3  %substation node
        Cons_Load=[Cons_Load,-So(i)<=Pi(i)<=So(i)];
        Cons_Load=[Cons_Load,-So(i)*0.5<=Qi(i)<=So(i)*0.5];
    end
end
Cons=[Cons,Cons_Load];
display('Constraints on load type of nodes completed!')

%% 电压电流约束
Cons_Limits=[0<=l_ij<=I_max^2, v(2:end)>=v_min]; %满足电压无上限约束，故一定是满足C2条件的
Cons=[Cons,Cons_Limits];
display('Constraints on voltage and current limits completed!')

%% OPF-m比OPF多的约束（OPF无此约束）
if judge==1
    P_ij_s=sdpvar(L,1); % the solution of the Linear DistFlow model, to constitue Svolt
    Q_ij_s=sdpvar(L,1); % the solution of the Linear DistFlow model
    v_s=sdpvar(B,1); % the solution of the Linear DistFlow model(无损)
    Cons_Svolt=[];
    Cons_Svolt=[Cons_Svolt, v_s(2:end)<=v_max];
    Cons_Svolt=[Cons_Svolt, v_s(1)==v_max];
    Cons_Svolt=[Cons_Svolt, In'*v_s == 2*r.*P_ij_s+2*x.*Q_ij_s]; %欧姆定律忽略损耗
    Cons_Svolt=[Cons_Svolt, In(2:end,:)*P_ij_s == Pi(2:end), In(2:end,:)*Q_ij_s == Qi(2:end)]; %功率平衡方程忽略损耗
    Cons=[Cons, Cons_Svolt];
    display('Constraints on s \in S_{volt} completed!')
end

%% 目标函数
% Pr_sub=0.9;
% Pr_pv=[0.8 0.7 0.6 0.5 0.4];
% C=Pr_pv*Pi([find(Nodetype==0)])+Pr_sub*Pi(1);
C=sum(Pi([find(Nodetype==0);1]));

%% 决策变量、约束、目标函数均已有，现对其求解：
ops=sdpsettings('solver', 'gurobi');
result=optimize(Cons,C,ops);
result.info
if result.problem==0
 s_P_ij=value(P_ij)*Sbase/1e6; %单位MW
 s_Q_ij=value(Q_ij)*Sbase/1e6; %单位MVar
 s_v=value(v);
 s_l_ij=value(l_ij);
 s_Pi=value(Pi)*Sbase/1e6;
 s_Qi=value(Qi)*Sbase/1e6;
 display(['分布式光伏电源：',num2str(sum(s_Pi([find(Nodetype==0)])),3),' MW']);
 display(['与主网相连变电站节点：',num2str(sum(s_Pi(1)),3),' MW']);
 if judge==1
    s_P_ij_s=value(P_ij_s)*Sbase/1e6;
    s_Q_ij_s=value(Q_ij_s)*Sbase/1e6;
    s_v_s=value(v_s);
 end
    
%% 绘制完整的潮流图
for k=1:L
  if norm([s_P_ij(k) s_Q_ij(k)])~=0 % 不等于0
            highlight(p,I(k),J(k),'LineWidth',norm([s_P_ij(k) s_Q_ij(k)]));  % 把线路宽度设置为潮流大小
  end
  end
    os=0.1; % offset value to print text in the figure
    text(p.XData-os, p.YData-os, num2str(sqrt(s_v)),'HorizontalAlignment', 'center','FontSize', 10, 'Color','m'); % printf 节点电压幅值
    N_cap=find(Nodetype==1);
    N_pv=find(Nodetype==0);
    text(p.XData(N_cap)-os, p.YData(N_cap)+os, num2str(s_Qi(N_cap)*1e6/Sbase,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','c'); % printf 接入并联电容器的节点的无功功率
    text(p.XData(N_pv)+os, p.YData(N_pv)-3*os, num2str(s_Pi(N_pv)*1e6/Sbase,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','g'); % printf 接入PV的节点的有功功率
  for k=1:L
        Coor1_PQ=(p.XData(I(k))+p.XData(J(k)))/2;
        Coor2_PQ=(p.YData(I(k))+p.YData(J(k)))/2;
        text(Coor1_PQ, Coor2_PQ, num2str(s_P_ij(k)+s_Q_ij(k)*w,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','b'); % printf 支路的复功率
  end
    
%% Check whether C1 holds for the network
De = degree(G); % degree of each node（连接到每个节点的支路的数量）
LeafNodes=find(De==1); %末尾节点（该节点只连接一个支路）
LeafNodes=LeafNodes(2:end); % delete node 1（不能算与主网相连的节点）
Pij_hat=abs(inv(In(2:end,:))*Pi_max(2:end));
Qij_hat=abs(inv(In(2:end,:))*Qi_max(2:end));
IJ=[I J];
A_l=cell(length(LeafNodes),1);
Path_L=cell(length(LeafNodes),1);
Al_uij=zeros(length(LeafNodes),2);
for i=1:length(LeafNodes)
    lt=LeafNodes(i);
    Path = shortestpath(G,lt,1); %到主网节点的最短路径
    A_l{i}=zeros(2,2,length(Path)-1);
    for j=1:length(Path)-1 
        for k=1:L
            if (IJ(k,1)==Path(j) && IJ(k,2)==Path(j+1))||(IJ(k,2)==Path(j) && IJ(k,1)==Path(j+1))
                Path_L{i}=[Path_L{i},Path(j+1)];
                A_l{i}(:,:,j)=diag([1 1])-2/v_min*mtimes([r(k);x(k)],[Pij_hat(k),Qij_hat(k)]);
            end
        end
    end
    Al_uij_temp=diag([1 1]);
    for m=2:length(Path)-1
        Al_uij_temp=Al_uij_temp*A_l{i}(:,:,m);
    end
    Al_uij(i,:)=(Al_uij_temp*[r(find(J==lt));x(find(J==lt))])';
end
if find(Al_uij<0)
    display('Important message: C1 doesn''t hold !')
else
    display('Important message: C1 holds !')
end

%% Check whether or not the SOCP relaxation is exact
for k=1:L
    i=I(k);% Starting node of line k
    SOC_gap(k)=(s_l_ij(k)+s_v(i)).^2-(4*(s_P_ij(k)*1e6/Sbase).^2+4*(s_Q_ij(k)*1e6/Sbase).^2+(s_l_ij(k)-s_v(i)).^2);
     if round(SOC_gap(k),3)==0 % always some errors in duality gap
            %display(['Line ',num2str(I(k)),' ',num2str(J(k)),' SOCP relaxation is exact']) 为了方便看结果，注释掉了这一行
     else
            display(['Line ',num2str(I(k)),' ',num2str(J(k)),' SOCP relaxation is not exact'])
     end
     end
 
%% Figure title according to WithSvolt
if judge==1
   title("SOCP relaxation for OPF-m: SCE 56 bus")
else
   title("SOCP relaxation for OPF: SCE 56 bus")
end

end