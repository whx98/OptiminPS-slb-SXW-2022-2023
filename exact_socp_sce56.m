clear all
close all
clc
%% With/Without S_volt
WithSvolt=1; % 1��ʾ��Svolt 
%% %% ϵͳ��������
WithSvolt=1;
Ubase=12e3; % unit:V
Sbase=1e6;   % unit:VA
Ibase=Sbase/sqrt(3)/Ubase; % unit:A
Zbase=Ubase/Ibase/sqrt(3);  % unit: 
Netpara=xlsread('SCE56','�������');
Loadpoint=xlsread('SCE56','�ڵ㸺��');
L=size(Netpara,1);%֧·����
R=Netpara(:,4)/Zbase;%֧·����,unit: p.u.
X=Netpara(:,5)/Zbase;%֧·�翹,unit: p.u.
So=Loadpoint(:,3)*1e6/Sbase;%�ڵ㸺��
Judge=Loadpoint(:,4);%�ڵ㸺������
I_max=12* 1e6 / Ubase / 1.732/ Ibase ; 
v_max=1.1^2;%��ѹƽ������
v_min=0.9^2;%��ѹƽ������
w=sqrt(-1);
z=R+X*w; %֧·�翹�ı���ֵ   
z_c=conj(z); % �����迹

%% ����ͼ����
I=Netpara(:,2);
J=Netpara(:,3);
GN=graph(I,J);
h1=figure;
p=plot(GN);
highlight(p,find(Judge==1),'Marker','s','NodeColor','c','Markersize',10);  % Capacitator
highlight(p,find(Judge==0),'Marker','v','NodeColor','y','Markersize',10);  % PV panels
In=myincidence(I,J); % ������������
Inn=In;
Inn(Inn>0)=0;    %Inn��In�ķ�������
%% �ڵ���
N=56;
%% ��������
P_ij=sdpvar(L,1); 
Q_ij=sdpvar(L,1); 
U=sdpvar(N,1); 
V=sdpvar(N,1);
L_ij=sdpvar(L,1) ;
Pi=sdpvar(N,1);
Qi=sdpvar(N,1);
%% Լ������
Cons=[];
Cons_Load=[];
Eta=2.8;  % PV��͸��
Pi_max=zeros(N,1);
Qi_max=zeros(N,1);
for i=1:N
    if Judge(i)==1    %1��ʾ���ݸ���
        Cons_Load=[Cons_Load,Pi(i)==0];
        Qi_max(i)=So(i)*Eta;
        Cons_Load=[Cons_Load,0<=Qi(i)<=Qi_max(i)]; 
    elseif Judge(i)==2 %2��ʾ��ͨ�ڵ㸺��
        Pi_max(i)=-So(i)*0.9;
        Qi_max(i)=-So(i)*sqrt(1-0.9^2);
        Cons_Load=[Cons_Load,Pi(i)==Pi_max(i)];  
        Cons_Load=[Cons_Load,Qi(i)==Qi_max(i)];              
    elseif Judge(i)==0  %0��ʾPV����
        Pi_max(i)=So(i)*Eta;
        Cons_Load=[Cons_Load,0<=Pi(i)<=Pi_max(i)];
        Cons_Load=[Cons_Load,0==Qi(i)];
    elseif Judge(i)==3  %%3��ʾ�˽ڵ�
        Cons_Load=[Cons_Load,-So(i)<=Pi(i)<=So(i)];
        Cons_Load=[Cons_Load,-So(i)*0.5<=Qi(i)<=So(i)*0.5];
    end
end
Cons=[Cons,Cons_Load];
display('������ʶ��ڵ㸺�����͵�Լ��')
% Cons_Load

%�ж��Ƿ�����C1����
Deg = degree(GN); % degree of each node
Deg
LeafNodes=find(Deg==1);
LeafNodes=LeafNodes(2:end); % ȥ���ڵ�1
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
    for j=1:length(Path)-1 % for all node in Path from lt��1, calculate A_l
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
    disp('������C1����')
else
    disp('����C1����!')
end

%% ��ѹ�����Լ��
Cons_LimitI=[I_max^2>=L_ij>=0];
Cons=[Cons,Cons_LimitI];
Cons_LimitU=[V(2:end)>=v_min];
Cons=[Cons,Cons_LimitU]

% Cons_Limits=[I_max^2>=L_ij>=0, V(2:end)>=v_min];
% Cons=[Cons,Cons_Limits];
disp('�����ɵ�ѹ�����Լ��')
% Cons_Limits
Cons_V=[];
Cons_V=[Cons_V, V(1)==v_max];
Cons_V=[Cons_V, In'*V == 2*R.*P_ij+2*X.*Q_ij-(R.^2+X.^2).*L_ij];
Cons=[Cons,Cons_V];
disp('�����ɵ�ѹԼ����')
% Cons_V

%% SOCԼ��
Cons_SOC=[];
for k=1:L
    i=I(k);%�׽ڵ�
    Cons_SOC=[Cons_SOC,(L_ij(k)+V(i)).^2 >= 4*P_ij(k).^2 + 4*Q_ij(k).^2 + (L_ij(k)-V(i)).^2];
end
Cons=[Cons,Cons_SOC];
disp('������SOCPԼ����')
% Cons_SOC

%% ����Լ��
Cons_PF=[In*P_ij - Inn*(R.*L_ij) == Pi, In*Q_ij - Inn*(X.*L_ij) == Qi];
Cons=[Cons,Cons_PF];
disp('�����ɳ���ƽ��Լ��!')
% Cons_PF

%% SvoltԼ��
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
    display('������Svolt��Լ��!')
end

%% Ŀ�꺯������
Pr_sub=0.9;
Pr_pv=[0.8];
Pr_pv*Pi([find(Judge==0)]);
Pr_p=1;
disp(find(Judge==0));
Obj=Pr_pv*Pr_p*Pi([find(Judge==0)])+Pr_sub*Pi(1);
% C=sum(Pi); 
%% ģ�����
ops=sdpsettings('solver', 'gurobi');
result=optimize(Cons,Obj,ops);
result.info

if result.problem==0
 %% �����
    s_P_ij=value(P_ij)*Sbase/1e6;
    s_Q_ij=value(Q_ij)*Sbase/1e6;
    s_v=value(V);
    s_l_ij=value(L_ij);
    s_Pi=value(Pi)*Sbase/1e6;
    s_Qi=value(Qi)*Sbase/1e6;
    display(['�ֲ�ʽ�����Դ',num2str(sum(s_Pi([find(Judge==0)])),3),' MW']);
    display(['�����������ĸ��ڵ�',num2str(sum(s_Pi(1)),3),' MW']);
    if WithSvolt==1
        s_P_ij_s=value(P_ij_s)*Sbase/1e6;
        s_Q_ij_s=value(Q_ij_s)*Sbase/1e6;
        s_v_s=value(v_s);
    end
    
  %%  ��ͼ
    for k=1:L
        if norm([s_P_ij(k) s_Q_ij(k)])~=0
            highlight(p,I(k),J(k),'LineWidth',norm([s_P_ij(k) s_Q_ij(k)]));  % ����·��������Ϊ������С
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
 %% SOCP�ɳڵ�Exact�ж�
    for k=1:L
        i=I(k);
        SOC_gap(k)=(s_l_ij(k)+s_v(i)).^2-(4*(s_P_ij(k)*1e6/Sbase).^2+4*(s_Q_ij(k)*1e6/Sbase).^2+(s_l_ij(k)-s_v(i)).^2);
        if round(SOC_gap(k),3)==0 
          
        else
            disp(['Line ',num2str(I(k)),' ',num2str(J(k)),' SOCP�ɳڲ�Exact,' 'Line impedance z = ',num2str(z(k)*Zbase,3), 'Ohm'])
        end
    end
    SOC_gap=SOC_gap';
    %%  Svolt�ж�
   if WithSvolt==1
        title("����sce56�ڵ������ľ�ȷSOCP�ɳڣ� With Svolt")
    else
        title("����sce56�ڵ�������SOCP�ɳڣ� WITHOUT Svolt")
    end
end


