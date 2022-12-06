%% Distribution Network Planning (DNP) based on Second-Order Cone Programming Relaxation (SOCPR) Optimal Power Flow (OPF) Model in 
% Exact convex relaxation of optimal power flow in radial networks L Gan, N Li, U Topcu, SH LowIEEE Transactions on Automatic Control 60 (1), 72-87, 2014

clear all
close all
clc
%% ***********Parameters **********
mpc=case33bw;
N=33; % number of load nodes
L=37; % number of dis. lines
Sbase=1e6;  % base value of complex power, unit:VA
Ubase=10e3;  % base value of voltage, unit:V, i.e. this is a 10 kV distribution network
Ibase=Sbase/Ubase/1.732;  % unit: A
Zbase=Ubase/Ibase/1.732;  % unit: Ω
LineInf=xlsread('16bus33lines.xlsx','F3:L39');
netpara=xlsread('IEEE33bw','网络dnp');
% NodeInf=xlsread('16bus33lines.xlsx','A3:D18');
% s=mpc.branch(:,1);
% t=mpc.branch(:,2);
s=netpara(:,2);
t=netpara(:,3);
N_gen=1;  % Subs nodes
N_loads=2:33; % Load nodes
v_min=0.95^2; % unit: p.u.
v_max=1.05^2; % unit: p.u.
I_max=560.98/Ibase; % max current in distribution line, unit: p.u. I_max≈S_max in p.u. because U≈1 (p.u.)
S_max=20 ; % max power in any distribution line
M=1e8;
% line investment cost denoted by 
Cost_invest=LineInf(:,7).*LineInf(:,4);
n=0.8;
P_load=mpc.bus(N_loads,3);
Q_load=mpc.bus(N_loads,4);
SS=sqrt(P_load.^2+Q_load.^2);
% make impedance z become r and x assuming a rx rate n_rx
% line impedance unit:p.u. 
n_rx=1;
% r=mpc.branch(:,3)/Zbase*16.0276;
% x=mpc.branch(:,4)/Zbase*16.0276;
r=mpc.branch(:,3)/Zbase;
x=mpc.branch(:,4)/Zbase;
z=r.^2+x.^2;
% power loss cost denoted by 
Cost_loss=98.0375;%tariff doller /Mwh
% OLTC parameters
Vbulk=1;vbulk=1;% the voltage of the bulk network
r_oltcmax=1.05;r_oltcmin=0.85;
SR_oltc=20;rjs_oltc=0.01;%20 steps,ranging from 0.85 to 1.05,0.01 per step.
Nmax_oltc=6; %maximum number of regulations per T
% CB parameters
q_cbmax=0.375;q_cbmin=0;%375 kvar to 0.375Mvar.
SR_cb=15;rjs_cb=0.025;
Nmax_cb=5;
Cost_cb=22500;%3000 cost per 50kw
%% ***********Graph of the Distribution System***********
G=graph(s,t);  % G=(V,E), G:graph V:vertex E:edge
In=myincidence(s,t);  % 节支关联矩阵，node-branch incidence matrix
% I=I(:,idxOut);   % make line index follow the order of s and t
Inn=In;
Inn(Inn>0)=0;    % Inn is the negative part of I

% In=incidence(G);  % 节支关联矩阵，node-branch incidence matrix
% In=-In;
% Inn=In;
% Inn(Inn>0)=0;    % Inn is the negative part of I
% figure;
% p=plot(G,'Layout','force');
% p.XData=[1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;2;3;4;5;3;4;5;6;7;8;9;10;11;12;13];
% p.YData=[5;5;5;5;5;5;5;5;5;6;6;6;6;6;5;5;5;5;7;7;7;7;2;2;2;3;3;3;3;3;3;3;3];
% hold on;
% labelnode(p,N_gen,{'1'});
% highlight(p,N_loads,'NodeColor','y','Markersize',20,'NodeFontSize',20);
% highlight(p,N_gen,'Marker','s','NodeColor','c','Markersize',30,'NodeFontSize',40);
% highlight(p,s,t,'EdgeColor','k','LineStyle','-.','LineWidth',2,'EdgeFontSize',8);
% text(p.XData, p.YData, p.NodeLabel,'HorizontalAlignment', 'center','FontSize', 15); % put nodes' label in right position.
% p.NodeLabel={};
% hold off;
%% ***********Variable statement**********
v_i=sdpvar(N,1, 'full'); % v=|U|^2=U·U*, "*" means conjugate
l_ij=sdpvar(L,1, 'full'); % l=|I|^2=I·I*
P_ij=sdpvar(L,1, 'full'); 
Q_ij=sdpvar(L,1, 'full');
P_shed=sdpvar(length(N_loads),1, 'full'); % shedded load
Q_shed=sdpvar(length(N_loads),1, 'full'); % shedded load
g_gen_P=sdpvar(length(N_gen),1,'full'); % generation from substations
g_gen_Q=sdpvar(length(N_gen),1,'full'); % generation from substations
y_ij=binvar(L,1,'full'); % decision variable denoting whether this line should be invested
% OLTC
r_i_oltc=sdpvar(1,1,'full');%node 1
step_i_oltc=binvar(1,SR_oltc,'full');
% stepin_i_oltc=binvar(1,1,'full');
% stepde_i_oltc=binvar(1,1,'full');
% CB
Q_i_cb=sdpvar(N,1,'full');% nodes 29 and 33
step_i_cb=binvar(N,SR_cb,'full');
% stepin_i_cb=binvar(N,1,'full');
% stepde_i_cb=binvar(N,1,'full');
y_ij_cb=binvar(N,1,'full');% which nodes build CB

%% ***********Constraints*************
Cons=[];
Cons=[Cons,sum(y_ij)==32];
Cons=[Cons,y_ij(1,1)==1];
s1=3;s2=2;s3=1;s4=3;s5=3;s6=8;s7=4;s9=2;s10=4;s11=4;            
Cons=[Cons,s1-sum(y_ij(3:5,1))<=1,s2-sum(y_ij(6:7,1))<=1];
Cons=[Cons,s4-sum(y_ij(9:11,1))<=1,s5-sum(y_ij(12:14,1))<=1];
Cons=[Cons,s6-sum(y_ij(15:17,1))-sum(y_ij(29:32,1))-sum(y_ij(36,1))<=1];
Cons=[Cons,s7-sum(y_ij(18:20,1))-sum(y_ij(2,1))<=1,s9-sum(y_ij(21,1))-sum(y_ij(35,1))<=1];
Cons=[Cons,s10-sum(y_ij(22:24,1))-sum(y_ij(37,1))<=1,s11-sum(y_ij(25:28,1))<=1];
Cons=[Cons,sum(y_ij(3:5,1))+sum(y_ij(22:24,1))+sum(y_ij(25:28,1))+sum(y_ij(37,1))<=10];
Cons=[Cons,sum(y_ij(2,1))+sum(y_ij(12:24,1))+sum(y_ij(29:32,1))+sum(y_ij(35:37,1))<=20];
Cons=[Cons,y_ij(8,1)+y_ij(30,1)+y_ij(35,1)>=1];
Cons=[Cons,sum(y_ij(9:14,1))+y_ij(34,1)<=6];
% Cons=[Cons,sum(y_ij(2:7,1))+sum(y_ij(18:20,1))+y_ij(33,1)<=9,sum(y_ij(18:20,1))++y_ij(33,1)>=3];
%% 1. Power balance
% S_ij=s_i+\sum_(h:h→i)(S_hi-z_hi·l_hi) for any (i,j)∈E, 
% denoted by node-branch incidence matrix I
Cons_S=[];
Cons_S=[In*P_ij-Inn*(r.*l_ij)==[g_gen_P;-(P_load-P_shed)],In*Q_ij-Inn*(x.*l_ij)==[g_gen_Q;-(Q_load-Q_shed)]+Q_i_cb, P_shed>=0,Q_shed==P_shed*sqrt(1-n^2)/n];
Cons=[Cons,Cons_S,P_ij>=0,Q_ij>=0];
% Cons=[Cons,Cons_S];
%% 2. Voltage Calculation
% v_i-v_j=2Re(z_ij·S_ij*)+(r.^2+x.^2).*l_ij=2(r·P_ij,i+x·Q_ij,i)+(r.^2+x.^2).*l_ij
% → |v_i-v_j-2(r·P_ij+x·Q_ij)-(r.^2+x.^2).*l_ij|≤(1-y_ij)*M
Cons_V=[v_i(N_gen,:)==vbulk*r_i_oltc];
Cons_V=[Cons_V,abs(In'*v_i-2*r.*P_ij-2*x.*Q_ij+z.^2.*l_ij)<=(1-y_ij)*M];
Cons=[Cons,Cons_V];
%% 3. Voltage limits
% v_min<=v<=v_max
Cons=[Cons,v_min<=v_i<=v_max];
%% 4. l_ij<=l_ij_max (I_max.^2)
Cons_l=[0<=l_ij<=y_ij.*(I_max^2)];
Cons=[Cons, Cons_l];
%% 5. l_ij<=S_ij^2/v_i, for any (i,j)
Cons_SOC=[];
for l=1:L
    sl=s(l); % sl is the node index of starting node of line l
    Cons_SOC=[Cons_SOC,(l_ij(l)+v_i(sl)).^2 >= 4*(P_ij(l).^2+Q_ij(l).^2)+(l_ij(l)-v_i(sl)).^2];
end
Cons=[Cons, Cons_SOC];
% 4. P_ij^2+Q_ij^2<=S_ij_max
Cons_PQ=[abs(P_ij)<=y_ij*S_max,abs(Q_ij)<=y_ij*S_max];
Cons_PQ=[P_ij.^2+Q_ij.^2<=y_ij*S_max.^2];
Cons=[Cons, Cons_PQ];
%% ***********ADG Constraints*************
%% 6. OLTC limits
stepsum=0;
for k=1:SR_oltc
    stepsum=stepsum+rjs_oltc*step_i_oltc(:,k);
end
r_i_oltc=r_oltcmin+stepsum;
for k=1:(SR_oltc-1)
    Cons_oltc=[step_i_oltc(:,k)>=step_i_oltc(:,k+1)];
end
Cons_oltc=[Cons_oltc,r_oltcmin<=r_i_oltc<=r_oltcmax];
Cons_oltc=[Cons_oltc,v_min<=vbulk*r_i_oltc<=v_max];
Cons=[Cons,Cons_oltc];
%% 7. CB limits
stepsum_cb=0;
for k=1:SR_cb
    stepsum_cb=stepsum_cb+rjs_cb*step_i_cb(:,k);
end
Q_i_cb=q_cbmin+stepsum_cb;
for k=1:(SR_cb-1)
    Cons_cb=[step_i_cb(:,k)>=step_i_cb(:,k+1)];
end
Cons_cb=[Cons_cb,y_ij_cb.*q_cbmin<=Q_i_cb<=y_ij_cb.*q_cbmax,1<=sum(y_ij_cb(:,1))<=4,y_ij_cb(1,1)==0];
Cons=[Cons,Cons_cb];
%% **********Objectives*************
Obj_inv=sum(Cost_invest.*y_ij); %investment cost, related to line investment
Obj_ope=M*sum(P_shed);
P_loss=sum(P_ij.*r)*24*365;
Obj_loss=sum(P_loss*Cost_loss);% power loss
Obj_cb=sum(y_ij_cb)*Cost_cb;%cost of building cb
Obj=Obj_inv+Obj_ope+Obj_loss+Obj_cb;
%% ********* Solve the probelm
ops=sdpsettings('solver','gurobi', 'gurobi.Heuristics',0,'gurobi.Cuts',0,'gurobi.TimeLimit',7000); %, 'gurobi.Heuristics',0,'gurobi.Cuts',0,'usex0',1,'gurobi.MIPGap',5e-2
sol=optimize(Cons,Obj,ops);
%% Save the solution with "s_" as start
s_y_ij=value(y_ij);
s_v_i=value(v_i);
s_P_ij=value(P_ij);
s_Q_ij=value(Q_ij);
s_P_shed=value(P_shed);
s_Q_shed=value(Q_shed);
s_g_subs_P=value(g_gen_P);
s_g_subs_Q=value(g_gen_Q);
s_l_ij=value(l_ij);
s_r=value(r_i_oltc);
s_loss=value(P_loss);
s_Q_cb=value(Q_i_cb);
s_y_ij_cb=value(y_ij_cb);
%OBJ
s_Obj=value(Obj);
s_Obj_inv=value(Obj_inv);
s_Obj_ope=value(Obj_ope);
s_obj_loss=value(Obj_loss);
s_obj_cb=value(Obj_cb);
%% Print the results in the command line
disp('————————规划方案如下——————————————');
disp(['   建设线路方案： ',num2str(round(s_y_ij',2))]);
disp([' (1/0 表示 建设/不建设 该线路, 线路编号参见输入文件)']);
disp('————————规划建设成本如下——————————————');
disp(['   建设配电线路成本： ',num2str(value(Obj_inv)),' 美元']);
disp(['   失负荷成本:  ',num2str(round(value(Obj_ope),2)),'  美元']);
disp(['   网络损耗成本:  ',num2str(round(value(Obj_loss),2)),'  美元']);
disp(['   CB建设成本:  ',num2str(round(value(Obj_cb),2)),'  美元']);
%disp(['   ESS建设成本:  ',num2str(round(value(Obj_ess),2)),'  美元']);
% disp(['   EV Station建设成本:  ',num2str(round(value(Obj_station),2)),'  美元']);
disp(['>> 规划建设总成本:  ',num2str(value(Obj(1))),'  美元']);
%% Plot the results
Gi=digraph(s,t);
os=0.2; % offset value to print text in the figure
figure;
pi=plot(Gi,'Layout','force');
%pi.LineWidth(idxOut) = 10*abs(s_P_ij)/max(abs(s_P_ij))+0.01;
pi.XData=[1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;2;3;4;5;3;4;5;6;7;8;9;10;11;12;13];
pi.YData=[5;5;5;5;5;5;5;5;5;6;6;6;6;6;5;5;5;5;7;7;7;7;2;2;2;3;3;3;3;3;3;3;3];
labelnode(pi,N_gen,{'1'});
highlight(pi,N_loads,'NodeColor','y','Markersize',20);
highlight(pi,N_gen,'Marker','s','NodeColor','c','Markersize',30);
text(pi.XData, pi.YData, pi.NodeLabel,'HorizontalAlignment', 'center','FontSize', 15); % put nodes' label in right position.
text(pi.XData+os, pi.YData+os, num2str([zeros(length(N_gen),1);round(mean(s_P_shed,2),4)]),'HorizontalAlignment', 'center','FontSize', 12, 'Color','r'); % printf the shedded load (if any).
text(pi.XData-os, pi.YData-os, num2str(mean(sqrt(s_v_i),2)),'HorizontalAlignment', 'center','FontSize', 12, 'Color','g'); % printf the nodes' voltage (sqrt(v_j)).
for l=1:L
    Coor1_PQ=(pi.XData(s(l))+pi.XData(t(l)))/2;
    Coor2_PQ=(pi.YData(s(l))+pi.YData(t(l)))/2;
    text(Coor1_PQ, Coor2_PQ, num2str(round(mean(s_P_ij(l,:)+s_Q_ij(l,:)*sqrt(-1),2),4)),'HorizontalAlignment', 'center','FontSize', 12, 'Color','b'); % printf the complex power in distribution lines(if any).
end 
pi.NodeLabel={};
%% Save the results
warning off
save('result_socp');