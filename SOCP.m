clear all;close all;
%% 可调参数
NetType = 1;  % 1：SCE47； 2：SCE56
[Sb,Ub,Ib,Zb,filename,Imax,vmax,vmin] = BaseValue(NetType);
FlagSvolt = 1;% 1：考虑Svolt约束； 0：不考虑Svolt约束

%% 网络参数
Data = xlsread(filename,'网络参数');
Load = xlsread(filename,'节点负荷');
LineNum = size(Data,1);% 线路数
NodeNum = LineNum + 1; % 节点数
LoadType = Load(1:NodeNum,4); % 节点类型 0：太阳能板 1：电容负荷 2：可控负载 3：松弛节点
r = Data(:,4)/Zb;% 线路电阻：标幺值
x = Data(:,5)/Zb;% 线路电抗：标幺值
z = r + 1i * x;
s = Load(:,3)*1e6/Sb;% 节点注入潮流：MVA
%% 网络拓扑
headNode = Data(:,2);
endNode = Data(:,3);
tree = graph(headNode,endNode);
p = plot(tree);
NodeBranchMat = myincidence(headNode,endNode);

%% 决策变量
Pij = sdpvar(LineNum,1); % Pij=sdpvar(numnodes,numnodes,'full');% Active power from node i to node j
Qij = sdpvar(LineNum,1); % Qij=sdpvar(numnodes,numnodes,'full');% Reactive power from node i to node j
u = sdpvar(NodeNum,1); % Voltage
v = sdpvar(NodeNum,1); % u^2
lij = sdpvar(LineNum,1) ;% Currents' magnitudes' square, I^2
Pi = sdpvar(NodeNum,1);% Active power injection of nodes
Qi = sdpvar(NodeNum,1);% Reactive power injection of nodes

%% 约束条件
PenetrateRate = 0.566;  % DG渗透率
Eta = 0.9; % 可控负载的效率
[Cons,Pimax,Qimax] = ConsOfSOCP(Pij,Qij,v,lij,Pi,Qi,s,Imax,vmax,vmin,LoadType,headNode,endNode,NodeBranchMat,LineNum,r,x,PenetrateRate,Eta,NodeNum);
if(FlagSvolt)
    Pij_s=sdpvar(LineNum,1); 
    Qij_s=sdpvar(LineNum,1);
    v_s=sdpvar(NodeNum,1); 
    
    ConsSvolt=[];
    ConsSvolt=[ConsSvolt, vmax>=v_s(2:end)];
    ConsSvolt=[ConsSvolt, v_s(1)==vmax];
    ConsSvolt=[ConsSvolt, NodeBranchMat'*v_s == 2*r.*Pij_s+2*x.*Qij_s];
    ConsSvolt=[ConsSvolt, NodeBranchMat(2:end,:)*Pij_s == Pi(2:end), NodeBranchMat(2:end,:)*Qij_s == Qi(2:end)];
    Cons=[Cons, ConsSvolt]; 
end

%% 目标函数
Pr_sub=0.9;
Pr_pv=[0.8 0.7 0.6 0.7 0.8];
% C=Pr_pv*Pi([find(judge==0)])+Pr_sub*Pi(1);
C=sum(Pi);
% % C=sum(Pi([find(judge==0);1]));
%% C1先验计算
flagC1 = isC1Hold(tree,NodeBranchMat,LineNum,Pimax,Qimax,vmin,headNode,endNode,r,x);

%% 求解
ops=sdpsettings('solver', 'gurobi');
result=optimize(Cons,C,ops);
result.info

%% 结果
if result.problem==0
    %% save the results
    s_Pij=value(Pij)*Sb/1e6;
    s_Qij=value(Qij)*Sb/1e6;
    s_v=value(v);
    s_lij=value(lij);
    s_Pi=value(Pi)*Sb/1e6;
    s_Qi=value(Qi)*Sb/1e6;
    display(['PV panel generation：',num2str(sum(s_Pi([find(LoadType==0)])),3),' MW']);
    display(['Substation power：',num2str(sum(s_Pi(1)),3),' MW']);
    if flag==1
        s_Pij_s=value(Pij_s)*Sb/1e6;
        s_Qij_s=value(Qij_s)*Sb/1e6;
        s_v_s=value(v_s);
    end
    
    %% Put some data in the graph
    for k=1:LineNum
        if norm([s_Pij(k) s_Qij(k)])~=0
            highlight(p,headNode(k),endNode(k),'LineWidth',norm([s_Pij(k) s_Qij(k)]));  % 把线路宽度设置为潮流大小
        end
    end
    os=0.1; % offset value to print text in the figure
    text(p.XData-os, p.YData-os, num2str(sqrt(s_v)),'HorizontalAlignment', 'center','FontSize', 10, 'Color','g'); % printf the nodes' voltage (sqrt(v_j)).
    N_cap=find(LoadType==1);
    N_pv=find(LoadType==0);
    text(p.XData(N_cap)-os, p.YData(N_cap)+os, num2str(s_Qi(N_cap)*1e6/Sb,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','r'); % printf the nodes' Qi if it's a node with Capacitator.
    text(p.XData(N_pv)+os, p.YData(N_pv)-3*os, num2str(s_Pi(N_pv)*1e6/Sb,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','k'); % printf the nodes' Pi if it's a node with PVs.
    for k=1:LineNum
        Coor1_PQ=(p.XData(headNode(k))+p.XData(endNode(k)))/2;
        Coor2_PQ=(p.YData(headNode(k))+p.YData(endNode(k)))/2;
        text(Coor1_PQ, Coor2_PQ, num2str(s_Pij(k)+s_Qij(k)*1i,2),'HorizontalAlignment', 'center','FontSize', 10, 'Color','b'); % printf the complex power in distribution lines(if any).
    end
    
    %% Wether or not the SOCP relaxation is exact
    for k=1:LineNum
        i=headNode(k);% Starting node of line k
        SOC_gap(k)=s_lij(k)*s_v(i)-((s_Pij(k)*1e6/Sb).^2+(s_Qij(k)*1e6/Sb).^2);
        if round(SOC_gap(k),3)==0 % always some errors in duality gap
            %display(['Line  ',num2str(I(k)),' ',num2str(J(k)),' SOCP relaxation is exact'])
        else
            display(['Line ',num2str(headNode(k)),' ',num2str(endNode(k)),' SOCP relaxation is not exact'])
        end
    end
    SOC_gap=SOC_gap';
end