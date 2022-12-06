%% Distribution Network Planning (DNP) based on Linear Distflow Optimal Power Flow (OPF) Model in 
% M. E. Baran and F. F. Wu, “Optimal capacitor placement on radial distribution systems,” IEEE Trans. Power Delivery, vol. 4, no. 1, pp. 725–734,1989.
% M. E. Baran and F. F. Wu, “Optimal sizing of capacitors placed on a radial distribution system,” IEEE Trans. Power Delivery, vol. 4, no. 1, pp. 735–743, 1989
% A compact form is described in Section III. of 
% Exact convex relaxation of optimal power flow in radial networksL Gan, N Li, U Topcu, SH LowIEEE Transactions on Automatic Control 60 (1), 72-87, 2014

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
% % ESS parameters
% Pcharge_max=0.2;Pcharge_min=0;
% Pdischarge_max=0.3;Pdischarge_min=0;
% E_max=1.5;E_min=0;
% a_charge=0.9; a_discharge=1.1;
% Cost_ess=20000;%cost per ess 18000+2000
%% ***********Graph of the Distribution System***********
G=digraph(s,t);  % G=(V,E), G:graph V:vertex E:edge
idxOut = findedge(G,s,t); % !!!!!!*************** index of edge is not the same with that of in mpc.branch !!!!!!*******
In=incidence(G);  % 节支关联矩阵，node-branch incidence matrix
In=-In;
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
% % ESS
% u_charge=binvar(N,T,'full');% nodes 7 and 24
% u_discharge=binvar(N,T,'full');
% Ess_i=sdpvar(N,T,'full');
% Pcharge_i=sdpvar(N,T,'full');
% Pdischarge_i=sdpvar(N,T,'full');
% y_ij_ess=binvar(N,1,'full')*ones(1,T);% which nodes build ESS
%% ***********Topology Constraints*************
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
%% ***********BFM Constraints*************
%% 1. Power balance
% S_ij=s_i+\sum_(h:h→i) S_hi for any (i,j)∈E, 
% denoted by node-branch incidence matrix I
Cons_S=[In*P_ij==([g_gen_P;-(P_load-P_shed)]),In*Q_ij==([g_gen_Q;-(Q_load-Q_shed)]+Q_i_cb),P_shed>=0,Q_shed==P_shed*sqrt(1-n^2)/n];
Cons=[Cons,Cons_S];
Cons=[Cons,Cons_S,P_ij>=0,Q_ij>=0];
%% 2. Voltage Calculation
% v_i-v_j=2Re(z_ij·S_ij*)=2(r·P_ij+x·Q_ij) 
% → |v_i-v_j-2(r·P_ij+x·Q_ij)|≤(1-y_ij)*M
Cons_V=[v_i(N_gen,:)==vbulk*r_i_oltc,v_i>=0];%OLTC at node 1
Cons_V=[Cons_V,abs(In'*v_i-2*r.*P_ij-2*x.*Q_ij)<=(1-y_ij)*M];
Cons=[Cons,Cons_V];
%% 3. Voltage limits
% v_min<=v<=v_max
Cons=[Cons,v_min<=v_i<=v_max];
%% 4. P_ij^2+Q_ij^2<=S_ij_max
% Cons_PQ=[abs(P_ij)<=y_ij*S_max,abs(Q_ij)<=y_ij*S_max];
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
% %% 7. ESS limits
% Cons_ess=[u_charge+u_discharge <= 1];% charge limit
% Cons_ess=[Cons_ess,u_discharge.*Pdischarge_min <= Pdischarge_i <= u_discharge.*Pdischarge_max,
%     u_charge.*Pcharge_min <= Pcharge_i <= u_charge.*Pcharge_max];% power limits
% for k=1:T-1
%     Cons_ess=[Cons_ess,Ess_i(:,k+1)==Ess_i(:,k)+a_charge.*Pcharge_i(:,k)-a_discharge.*Pdischarge_i(:,k)];
% end
% Cons_ess=[Cons_ess,y_ij_ess.*E_min<=Ess_i<=y_ij_ess.*E_max,1<=sum(y_ij_ess(:,1))<=3,y_ij_ess(1,1)==0];
% Cons_ess=[Cons_ess,sum(Pcharge_i,2)==sum(Pdischarge_i,2)];
% Cons=[Cons,Cons_ess];
%% **********Objectives*************
Obj_inv=sum(Cost_invest.*y_ij); %investment cost, related to line investment
Obj_ope=M*sum(P_shed);
P_loss=sum(P_ij.*r)*24*365;
Obj_loss=sum(P_loss*Cost_loss);% power loss
Obj_cb=sum(y_ij_cb)*Cost_cb;%cost of building cb
Obj=Obj_inv+Obj_ope+Obj_loss+Obj_cb;
%% ********* Solve the probelm
ops=sdpsettings('solver','gurobi', 'gurobi.Heuristics',0,'gurobi.Cuts',0); %,'usex0',1,'gurobi.MIPGap',5e-2,
sol=optimize(Cons,Obj,ops)
%% Save the solution with "s_" as start
s_y_ij=value(y_ij);
s_v_i=value(v_i);
s_P_ij=value(P_ij);
s_Q_ij=value(Q_ij);
s_P_shed=value(P_shed);
s_Q_shed=value(Q_shed);
s_g_subs_P=value(g_gen_P);
s_g_subs_Q=value(g_gen_Q);
s_r=value(r_i_oltc);
s_loss=value(P_loss);
s_Q_cb=value(Q_i_cb);
s_y_ij_cb=value(y_ij_cb);
% s_Pcharge=value(Pcharge_i);
% s_Pdischarge=value(Pdischarge_i);
% s_y_ij_ess=value(y_ij_ess);

%OBJ
s_Obj=value(Obj);
s_Obj_inv=value(Obj_inv);
s_Obj_ope=value(Obj_ope);
s_obj_loss=value(Obj_loss);
s_obj_cb=value(Obj_cb);
% s_obj_ess=value(Obj_ess);
%% Print the results in the command line
disp('————————规划方案如下——————————————');
disp(['   建设线路方案： ',num2str(round(s_y_ij(:,1)',2))]);
disp([' (1/0 表示 建设/不建设 该线路, 线路编号参见输入文件)']);
disp('————————规划建设成本如下——————————————');
disp(['   建设配电线路成本： ',num2str(value(Obj_inv(1))),' 美元']);
disp(['   失负荷成本:  ',num2str(round(value(Obj_ope),2)),'  美元']);
disp(['   网络损耗成本:  ',num2str(round(value(Obj_loss),2)),'  美元']);
disp(['   CB建设成本:  ',num2str(round(value(Obj_cb),2)),'  美元']);
% disp(['   ESS建设成本:  ',num2str(round(value(Obj_ess),2)),'  美元']);
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
save('result_linear');