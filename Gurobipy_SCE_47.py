# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 01:03:35 2022

@author: 29951
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 19:28:05 2022

@author: 29951
"""

"""
Created on Mon Nov 14 18:22:31 2022

@author: 29951
"""

# -*- coding: utf-8 -*-
from gurobipy import *
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

"""
Created on Sun Nov 13 22:21:16 2022

@author: 29951
"""


def myincidence(s, t):
    MaxNode = max(max(s), max(t))
    I = np.zeros((MaxNode, len(s)))
    for j in range(len(s)):
        I[s[j] - 1, j] = 1
        I[t[j] - 1, j] = -1
    return I


##para
WithSvolt = 1
N = 47  # number of nodes
Sbase = 1e6  # base value of complex power, unit:VA
Ubase = 12.35e3  # base value of voltage, unit:V, i.e. this is a 10 kV distribution network
Ibase = Sbase / Ubase / 1.732  # unit: A
Zbase = Ubase / Ibase / 1.732  # unit: Ω
path = r'C:\Users\29951\PycharmProjects\pythonProject4\SCE47.xls'
netpara = pd.read_excel(path, sheet_name='网络参数')
loadpoint = pd.read_excel(path, sheet_name='节点负荷')
L = len(netpara)  # number of dis. lines
r = netpara.iloc[:, 3] / Zbase  # branch resistance
r = np.array(r)
# print(r)
# print(r)
x = netpara.iloc[:, 4] / Zbase  # branch reactance
x = np.array(x)
# print(x)
So = loadpoint.iloc[:, 2] * 1e6 / Sbase  # unit: MVA
So = np.array(So)
# print(So)
judge = loadpoint.iloc[:, 3]  # types of nodes
judge = np.array(judge)
# print(judge)
I_max = 560.98/Ibase  # 12MW?
v_min = float(0.9 ** 2)  # unit: p.u.
v_max = float(1.1 ** 2)  # unit: p.u.
##Graph
N_loads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
LineInf = netpara.iloc[:48, 3:5]
NodeInf = netpara.iloc[:48, 1:3]
LineInf = np.array(LineInf)
NodeInf = np.array(NodeInf)
print(NodeInf)
Pr_pv = [0.8, 0.7, 0.6, 0.7, 0.8]
pr_judge=0
Pr_p=1.935
Pr_sub = 0.9
index=0
Obj=0
Cost = netpara.iloc[:48, 6]
LM = netpara.iloc[:48, 7]
Cost = np.array(Cost)
LM = np.array(LM)
Cost = Cost * LM
print(Cost)
I = NodeInf[:, 0]
# print(len(I))
J = NodeInf[:, 1]
# IJ=np.zeros((2*N,1))
# for i in range(len(I)):
# IJ[i,1]=I[i]
# IJ[len(I)+i,1]=J[i]
# print(IJ)
'''
N_loads=[]
N_subs=[]
b1=1
b2=1
for i=1:N
    if So(i)==0
        N_subs(b1)=i;
        b1=b1+1;
    else 
        N_loads(b2)=i;
        b2=b2+1;
    end
end
S=loadpoint(N_loads,3);
S;
length(N_loads);
n=0.8;  % power factor
P_load=S*n;
Q_load=S*sqrt(1-n^2);
% calculate the maximum complex power in each distribution line
S_max=sqrt(v_max)*I_max;
'''
# b1 = 0
# b2 = 0
# for i in range(N):
#     if So[i] == 0:
#         b1 = b1 + 1
#     else:
#         b2 = b2 + 1
#
# N_loads = np.zeros((b2, 1))
# N_subs = np.zeros((b1, 1))
# S = np.zeros((b2, 1))
# # print(b1)
# # print(b2)
# c1 = 0
# c2 = 0
# for i in range(N):
#     if So[i] == 0:
#         N_subs[c1] = i + 1
#         c1 = c1 + 1
#     else:
#         N_loads[c2] = i + 1
#         S[c2] = So[i]
#         c2 = c2 + 1
# # print(N_subs)
# # print(N_loads)
# # S=So[N_loads]
# # print(S)
# n = 0.8
# P_load = S * n
# Q_load = S * math.sqrt(1 - n ** 2)
# S_max = math.sqrt(v_max) * I_max
# M = 1e4

edges = pd.DataFrame()
# print(s)
# print(t)
edges['sources'] = I
edges['targets'] = J
G = nx.from_pandas_edgelist(edges, source='sources', target='targets')
nx.draw(G, with_labels=True, pos=None, arrows=True)
# plt.savefig("undirected_graph.png")
plt.show()
In = myincidence(I, J)
print(In)
Inn = In
# Inn[Inn>0]=0    # Inn is the negative part of I
for i in range(len(Inn)):
    for j in range(len(Inn[0])):
        if Inn[i][j] > 0:
            Inn[i][j] = 0
        else:
            Inn[i][j] = Inn[i][j]

print(Inn)
De = G.degree()
print(De)
##
In = myincidence(I, J)
print(In)
print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
# 变量声明
'''
P_ij=sdpvar(L,1); % Pij=sdpvar(numnodes,numnodes,'full');% �ڵ�1��ڵ�j���͵��й�����
Q_ij=sdpvar(L,1); % Qij=sdpvar(numnodes,numnodes,'full');%�ڵ�i��ڵ�j���͵��޹�����
u=sdpvar(N,1); % ���ڵ��ѹ
v=sdpvar(N,1); % u^2
l_ij=sdpvar(L,1) ;% ������ֵ��ƽ��
Pi=sdpvar(N,1);% �ڵ�ע���й����� ppoint(i)
Qi=sdpvar(N,1);%�ڵ�ע��wu������
w=sqrt(-1);
z=r+x*w;
z_c=conj(z); % ��zȡ����

Cons=[];
'''

model = Model('mip')
u = model.addMVar((N, 1), vtype=GRB.CONTINUOUS, name='u')
v = model.addMVar((N, 1), vtype=GRB.CONTINUOUS, name='v')
P_ij = model.addMVar((L, 1), vtype=GRB.CONTINUOUS, name='P_ij')
Q_ij = model.addMVar((L, 1), vtype=GRB.CONTINUOUS, name='Q_ij')
l_ij = model.addMVar((L, 1), vtype=GRB.CONTINUOUS, name='l_ij')
Pi = model.addMVar((N, 1), vtype=GRB.CONTINUOUS, name='Pi')
Qi = model.addMVar((N, 1), vtype=GRB.CONTINUOUS, name='Qi')
# P_shed = model.addMVar((len(N_loads), 1), vtype=GRB.CONTINUOUS, name='P_shed')
# Q_shed = model.addMVar((len(N_loads), 1), vtype=GRB.CONTINUOUS, name='Q_shed')
# g_subs_P = model.addMVar((len(N_subs), 1), vtype=GRB.CONTINUOUS, name='g_subs_P')
# g_subs_Q = model.addMVar((len(N_subs), 1), vtype=GRB.CONTINUOUS, name='g_subs_Q')
# y_ij = model.addMVar((L, 1), vtype=GRB.BINARY, name='y_ij')
model.update()

w = 1j
z = r + x * w
z_c = r - x * w

# 约束条件
'''
Cons=[];
Cons_Load=[];
Eta=2.8;  % Scale up the PV penetration rate, 5 means 5 times than Table I in [1]
Pi_max=zeros(N,1);
Qi_max=zeros(N,1);
for i=1:N
    if judge(i)==1     %���ݲ���
        Cons_Load=[Cons_Load,Pi(i)==0];
        Qi_max(i)=So(i)*Eta;
        Cons_Load=[Cons_Load,0<=Qi(i)<=Qi_max(i)];   %st=[pl(i)==0,0<=ql(i)<=So(i)];
    elseif judge(i)==2 %��ͨ�ڵ㸺��
        Pi_max(i)=-So(i)*0.9;
        Qi_max(i)=-So(i)*sqrt(1-0.9^2);
        Cons_Load=[Cons_Load,Pi(i)==Pi_max(i)];  % cos(0.9)???
        Cons_Load=[Cons_Load,Qi(i)==Qi_max(i)];               %st=[pl(i)==-So(i)*cos(0.9);ql(i)==-So(i)*sin(0.9)];
    elseif judge(i)==0  %PV panel
        Pi_max(i)=So(i)*Eta;
        Cons_Load=[Cons_Load,0<=Pi(i)<=Pi_max(i)];
        Cons_Load=[Cons_Load,0==Qi(i)];
    elseif judge(i)==3  %slack node (�׶˽ڵ㣬�����վ)
        Cons_Load=[Cons_Load,-So(i)<=Pi(i)<=So(i)];
        Cons_Load=[Cons_Load,-So(i)*0.5<=Qi(i)<=So(i)*0.5];
    end
end
Cons=[Cons,Cons_Load];
'''
Eta = 2.8
Pi_max = np.zeros((N, 1))
Qi_max = np.zeros((N, 1))
# if judge[i] == 1:
# model.addConstrs(Pi[i] == 0 for i in N)
# model.addConstrs(Qi[i] >= 0 for i in N)
# model.addConstrs(Qi[i] <= So[i] * Eta for i in N)

for i in range(N):
    if judge[i] == 1:
        model.addConstr(Pi[i] == 0)
        Qi_max[i] = So[i] * Eta
        model.addConstr(0 <= Qi[i], name='约束1.1' + str(i))
        model.addConstr(Qi[i] <= Qi_max[i], name='约束1.2' + str(i))
    elif judge[i] == 2:
        Pi_max[i] = -So[i] * 0.9
        Qi_max[i] = -So[i] * math.sqrt(1 - 0.9 ** 2)
        model.addConstr(Pi[i] == Pi_max[i], name='约束1.3' + str(i))
        model.addConstr(Qi[i] == Qi_max[i], name='约束1.4' + str(i))
    elif judge[i] == 0:
        Pi_max[i] = So[i] * Eta
        model.addConstr(0 <= Pi[i], name='约束1.5' + str(i))
        model.addConstr(Pi[i] <= Pi_max[i], name='约束1.6' + str(i))
        model.addConstr(0 == Qi[i], name='约束1.7' + str(i))
    elif judge[i] == 3:
        model.addConstr(-So[i] <= Pi[i], name='约束1.8' + str(i))
        model.addConstr(Pi[i] <= So[i], name='约束1.9' + str(i))
        model.addConstr(-So[i] * 0.5 <= Qi[i], name='约束1.10 ' + str(i))
        model.addConstr(Qi[i] <= So[i] * 0.5, name='约束1.11 ' + str(i))
'''
%Check whether C1 holds for the network
De = degree(G); % degreee of each node
LeafNodes=find(De==1);
LeafNodes=LeafNodes(2:end); % delete node 1

Pij_hat=abs(inv(In(2:end,:))*Pi_max(2:end));
Qij_hat=abs(inv(In(2:end,:))*Qi_max(2:end));
IJ=[I J];

A_l=cell(length(LeafNodes),1);
Path_L=cell(length(LeafNodes),1);
Al_uij=zeros(length(LeafNodes),2);
for i=1:length(LeafNodes)
    lt=LeafNodes(i);
    Path = shortestpath(G,lt,1);
    A_l{i}=zeros(2,2,length(Path)-1);
    for j=1:length(Path)-1 % for all node in Path from lt��1, calculate A_l
%         nl=[]; % the set of 1,2,...,nl in C1
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
cell等同于[]
'''
mode1=model.feasRelaxS(0, True, False, True)
# De = G.degree()
# LeafNodes = [5, 6, 9, 12, 14, 16, 19, 22, 24, 31, 33, 35, 37, 40, 43, 44, 45, 46, 48, 50, 52, 54, 55, 56]
# IJ = NodeInf
# Pij_hat = abs(np.linalg.inv(In[1:, :]) * Pi_max[1:])
# Qij_hat = abs(np.linalg.inv(In[1:, :]) * Qi_max[1:])
# Al_uij = np.zeros((len(LeafNodes), 2))
# A_l = np.zeros((len(LeafNodes), 2))
# A_l = A_l.tolist()
# Path_L = np.zeros((len(LeafNodes), 1))
# Path_L = Path_L.tolist()
# for i in range(len(LeafNodes)):
#     lt = LeafNodes[i]
#     Path = nx.shortest_path(G, source=lt, target=1)
#     A_l[i] = np.zeros((2, 2, len(Path) - 1))
#     print(Path)
#     for j in range(len(Path) - 1):
#         for k in range(L):
#             if (IJ[k, 0] == Path[j] and IJ[k, 1] == Path[j + 1]) or (IJ[k, 1] == Path[j] and IJ[k, 0] == Path[j + 1]):
#                 Path_L[i] = Path
#                 # print( Path_L)
#                 AE = A_l[i]
#                 for j1 in 2:
#                     for j2 in 2:
#                         AE[j1, j2, j] = np.diag([1] * 2) - 2 / v_min * np.transpose([r[k], x[k]]).dot([Pij_hat[k], Qij_hat[k]])
#     Al_uij_temp=np.diag([1] * 2)
#     for m in range(1,len(Path)-1):
#         Al_uij_temp=Al_uij_temp*AE[:, :, m]
#     Al_uij[i,:]
                    
                # 遍历2*2

# if (IJ[k,0]==Path[j] and IJ[k,1]==Path[j+1])or(IJ[k,1]==Path[j] and IJ[k,0]==Path[j+1])

# LeafNodes=
# node is hard
'''
Cons_Limits=[I_max^2>=l_ij>=0, v(2:end)>=v_min];
Cons=[Cons,Cons_Limits];
display('Constraints on voltage and current limits completed!')
% Cons_Limits
'''
for i in range(L):
    model.addConstr(l_ij[i] <= I_max ** 2, name='约束2.1' + str(i))
    model.addConstr(l_ij[i] >= 0, name='约束2.2' + str(i))
for i in range(1, N):
    model.addConstr(v[i] >= v_min, name='约束2.3' + str(i))

print(Inn)
# for i in range(len(N_loads)):
#     for j in range(1):
#         x1 = In[i, 0] * P_ij[0]
#         y1 = Inn[i, 0] * r[0] * l_ij[0]
#         for l in range(1, L):
#             x1 = x1 + In[i, l] * P_ij[l]
#             y1 = y1 + Inn[i, l] * r[l] * l_ij[l]
#         model.addConstr(x1 - y1 == P_shed[i] - P_load[i], name='约束1.1' + str(i))
#         # x1 = x1+In[i,l]*P_ij[l]
#         # y1 = y1+Inn[i,l]*r[l]*l_ij[l] model.addConstr(sum(In[i,l]*P_ij[l])-sum(Inn[i,l]*r[l]*l_ij[l])==P_shed[i]-P_load[i],name='约束1.1'+str(i))
# k1 = 0
# print('bbbbbbbbbbbbbbbbbb')
# print(In)
# for i in range(len(N_loads), N):
#     for j in range(1):
#         x1 = 0
#         y1 = 0
#         for l in range(L):
#             x1 = x1 + In[i, l] * P_ij[l]
#             y1 = y1 + Inn[i, l] * r[l] * l_ij[l]
#         model.addConstr(x1 - y1 == g_subs_P[k1], name='约束1.2' + str(i))
#         k1 = k1 + 1
# k1 = 0
# # K=x*l_ij
# for i in range(len(N_loads)):
#     for j in range(1):
#         x1 = 0
#         y1 = 0
#         for l in range(L):
#             x1 = x1 + In[i, l] * Q_ij[l]
#             y1 = y1 + Inn[i, l] * x[l] * l_ij[l]
#         model.addConstr(x1 - y1 == Q_shed[i] - Q_load[i], name='约束1.3' + str(i))
# # print(k1)
# for i in range(len(N_loads), N):
#     for j in range(1):
#         x1 = 0
#         y1 = 0
#         for l in range(L):
#             x1 = x1 + In[i, l] * Q_ij[l]
#             y1 = y1 + Inn[i, l] * x[l] * l_ij[l]
#         model.addConstr(x1 - y1 == g_subs_Q[k1], name='约束1.4' + str(i))
#         k1 = k1 + 1
# 
# for i in range(len(N_loads)):
#     model.addConstr(Q_shed[i] == P_shed[i] * math.sqrt(1 - n ** 2) / n, name='约束1.5' + str(i))
# 
# for i in range(len(N_loads)):
#     model.addConstr(P_shed[i] >= 0, name='约束1.6' + str(i))
# 
# for i in range(L):
#     model.addConstr(0 <= l_ij[i], name='约束2.6' + str(i))
#     model.addConstr(l_ij[i] <= y_ij[i] * I_max * I_max, name='约束2.7' + str(i))

#######################################################################
'''
Cons_V=[];
Cons_V=[Cons_V, v(1)==v_max];
Cons_V=[Cons_V, In'*v == 2*r.*P_ij+2*x.*Q_ij-(r.^2+x.^2).*l_ij];
Cons=[Cons,Cons_V];
'''

model.addConstr(v[0] == v_max, name='约束2.4' + str(i))
# print(In)
Iu = np.transpose(In)
for i in range(len(Iu)):
    for j in range(1):
        x1 = 0
        y1 = 0
        for l in range(N):
            x1 = x1 + Iu[i, l] * v[l]
        y1 = 2 * r[i] * P_ij[i] + 2 * x[i] * Q_ij[i] - (r[i] * r[i] + x[i] * x[i]) * l_ij[i]
        model.addConstr(x1 - y1 == 0, name='约束2.5' + str(i))

'''
Cons_SOC=[];
for k=1:L
    i=I(k);%�׽ڵ�
    Cons_SOC=[Cons_SOC,(l_ij(k)+v(i)).^2 >= 4*P_ij(k).^2 + 4*Q_ij(k).^2 + (l_ij(k)-v(i)).^2];
end
Cons=[Cons,Cons_SOC];
display('Constraints on SOCP relaxation completed!')
% Cons_SOC
'''
for l in range(L):
    sl = I[l]  # sl is the node index of starting node of line l
    x2 = (l_ij[l] + v[sl]) * (l_ij[l] + v[sl])
    y2 = 4 * P_ij[l] * P_ij[l] + 4 * Q_ij[l] * Q_ij[l]
    z2 = (l_ij[l] - v[sl]) * (l_ij[l] - v[sl])
    model.addConstr(x2 >= y2 + z2, name='约束2.6' + str(i))
'''
Cons_PF=[In*P_ij - Inn*(r.*l_ij) == Pi, In*Q_ij - Inn*(x.*l_ij) == Qi];
Cons=[Cons,Cons_PF];
display('Constraints on power flow caculation completed!')
% Cons_PF
'''

# K = r * l_ij
for i in range(N):
    for j in range(1):
        x1 = 0
        y1 = 0
        for l in range(L):
            x1 = x1 + In[i, l] * P_ij[l]
            y1 = y1 + Inn[i, l] * r[l] * l_ij[l]
        model.addConstr(x1 - y1 == Pi[i], name='约束2.7' + str(i))

# J = x * l_ij
for i in range(N):
    for j in range(1):
        x1 = 0
        y1 = 0
        for l in range(L):
            x1 = x1 + In[i, l] * Q_ij[l]
            y1 = y1 + Inn[i, l] * x[l] * l_ij[l]
        model.addConstr(x1 - y1 == Qi[i], name='约束2.8' + str(i))

'''
if WithSvolt==1
    P_ij_s=sdpvar(L,1); % the solution of the Linear DistFlow model, to constitue Svolt
    Q_ij_s=sdpvar(L,1); % the solution of the Linear DistFlow model
    v_s=sdpvar(N,1); % the solution of the Linear DistFlow model

    Cons_Svolt=[];
    Cons_Svolt=[Cons_Svolt, v_max>=v_s(2:end)];
    Cons_Svolt=[Cons_Svolt, v_s(1)==v_max];
    Cons_Svolt=[Cons_Svolt, In'*v_s == 2*r.*P_ij_s+2*x.*Q_ij_s];
    Cons_Svolt=[Cons_Svolt, In(2:end,:)*P_ij_s == Pi(2:end), In(2:end,:)*Q_ij_s == Qi(2:end)];
    Cons=[Cons, Cons_Svolt];
    display('Constraints on s��S_{volt} completed!')
end
'''
if WithSvolt == 1:
    P_ij_s = model.addMVar((L, 1), vtype=GRB.CONTINUOUS, name=' P_ij_s')
    Q_ij_s = model.addMVar((L, 1), vtype=GRB.CONTINUOUS, name='Q_ij_s ')
    v_s = model.addMVar((N, 1), vtype=GRB.CONTINUOUS, name=' v_s')

    for i in range(1, N):
        model.addConstr(v_max >= v_s[i], name='约束3.1' + str(i))

    model.addConstr(v_max == v_s[1], name='约束3.2' + str(i))

    Iu = np.transpose(In)
    for i in range(len(Iu)):
        for j in range(1):
            x1 = 0
            y1 = 0
            for l in range(N):
                x1 = x1 + Iu[i, l] * v_s[l]
            y1 = 2 * r[i] * P_ij_s[i] + 2 * x[i] * Q_ij_s[i]
            model.addConstr(x1 == y1, name='约束3.3' + str(i))

    for i in range(1, N):
        for j in range(1):
            x1 = 0
            y1 = 0
            for l in range(L):
                x1 = x1 + In[i, l] * P_ij_s[l]
            y1 = Pi[i]
            model.addConstr(x1 == y1, name='约束3.4' + str(i))

    for i in range(1, N):
        for j in range(1):
            x1 = 0
            y1 = 0
            for l in range(L):
                x1 = x1 + In[i, l] * Q_ij_s[l]
            y1 = Qi[i]
            model.addConstr(x1 == y1, name='约束3.5' + str(i))

# if WithSvolt==1:
# obj
'''
Pr_sub=0.9;
Pr_pv=[0.8 0.7 0.6 0.7 0.8];
C=Pr_pv*Pi([find(judge==0)])+Pr_sub*Pi(1);
'''
# Pr_sub = 0.9
# Pr_pv = np.transpose([0.8, 0.7, 0.6, 0.7, 0.8])
# Obj = 0
# or i in range(N):
# Obj = Obj +Pi[i]
# M=1e8
# Obj_inv=0
# Obj_ope=0
# for i in range(L):
# Obj_inv=Obj_inv+Cost[i]*r[i]
# for i in range(len(N_loads)):
# Obj_ope=Obj_ope+M*P_ij[i]
# Obj_inv=sum(Cost*y_ij)
# Obj_ope=M*sum(P_shed)
# Obj=Obj_inv+Obj_ope


#OBJ
for i in range(N):
    if judge[i]==0:
        Obj=Obj+Pr_pv[index]*Pi[i]*Pr_p
        index=index+1
   
Obj=Obj+Pr_sub*Pi[0]*Pr_p
#Obj = (0.8*Pi[12]+0.7*Pi[16]+0.6*Pi[18]+0.7*Pi[22]+0.8*Pi[23]+0.9*Pi[0])*19.3
#Obj_inv = 0
#Obj_ope = 0
#for i in range(L):
    #Obj_inv = Obj_inv + Cost[i] * y_ij[i]
#for i in range(len(N_loads)):
    #Obj_ope = Obj_ope + M * P_shed[i]
# Obj_inv=sum(Cost*y_ij)
# Obj_ope=M*sum(P_shed)
#Obj = Obj_inv + Obj_ope
model.setObjective(Obj,GRB.MAXIMIZE)
# 求解
# model.setParam('outPutFlag', 0)  # 不输出求解日志
# model.optimize()
# model.computeIIS()
# model.write("ogg.ilp")

mode1
model.optimize()
print(model.objVal)


model.write("model.sol")
#for v in model.getVars:
    #print(v.varName, v.x)
# for var in model.getVars():
# print(f"{var.varName}: {round(var.X, 3)}")
model.write("ouo.lp")