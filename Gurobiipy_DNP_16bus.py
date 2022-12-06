# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 20:35:38 2022

@author: 29951
"""

from gurobipy import *
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

'''
def incidence(self, vertices, edges):
    self.vertices = vertices
    self.edges = edges
    self.liste = [[0 for i in range(vertices)] for i in range(vertices)]

    # print(self.liste)

    for i in range(0, vertices):
        for j in range(0, len(edges)):
            if edges[i][j - 1] >= vertices or edges[i][j - 1] < 0 or edges[i][j - 1] >= vertices or edges[i][j - 1] < 0:
                print("Index out of range")
            return
        self.liste[edges[0][j + 1]][edges[1][j + 1]] = 1
        self.liste[edges[1][j + 1]][edges[0][j + 1]] = 1
    for x in range(0, vertices):
        row = ""
        for y in range(0, len(edges)):
            row = row + str(self.liste[x][y]) + " "
    print(row)
    return row
'''
def myincidence(s,t):
    MaxNode = max(max(s), max(t))
    I = np.zeros((MaxNode,len(s)))
    for j in range(len(s)):
        I[s[j]-1, j]= 1
        I[t[j]-1, j] = -1
    return I
#参数部分
N=16 # number of load nodes
L=33 # number of dis. lines
Sbase=1e6  # base value of complex power, unit:VA
Ubase=10e3  # base value of voltage, unit:V, i.e. this is a 10 kV distribution network
Ibase=Sbase/Ubase/1.732  # unit: A
Zbase=Ubase/Ibase/1.732  # unit: Ω
path = r'C:\Users\29951\PycharmProjects\pythonProject4\case data\16bus33lines.xlsx'
frame = pd.read_excel(path)
LineInf=frame.iloc[:,5:]
NodeInf=frame.iloc[:17,:4]
LineInf=np.array(LineInf)
NodeInf=np.array(NodeInf)
print(LineInf)
'''
LineInf=xlsread('case data\16bus33lines.xlsx','F3:L35');
NodeInf=xlsread('case data\16bus33lines.xlsx','A3:D18');
s=LineInf(:,2);
t=LineInf(:,3);
N_subs=[13,14,15,16];  % Subs nodes
N_loads=1:12; % Load nodes
v_min=0.95^2; % unit: p.u.
v_max=1.05^2; % unit: p.u.
I_max=LineInf(:,5)*1e6/Sbase; % max current in distribution line, unit: p.u. I_max≈S_max in p.u. because U≈1 (p.u.)
M=1e8;
% line investment cost denoted by 
Cost=LineInf(:,7).*LineInf(:,4);
% make load (MVA) become P_load Q_load assuming a power factor n.
S=NodeInf(N_loads,4);
n=0.8;  % power factor
P_load=S*n;
Q_load=S*sqrt(1-n^2);
% make impedance z become r and x assuming a rx rate n_rx
z=LineInf(:,4).*LineInf(:,6)/Zbase; % line impedance, unit:p.u. 
n_rx=1;
r=z/sqrt(1+n_rx^2);
x=r/n_rx;
% calculate the maximum complex power in each distribution line
S_max=sqrt(v_max)*I_max;
'''

s=LineInf[1:34,1]
t=LineInf[1:34,2]
N_subs=[13,14,15,16]  #Subs nodes
N_loads=[1,2,3,4,5,6,7,8,9,10,11,12] # Load nodes？可能为随机数？
v_min=float(0.95**2) # unit: p.u.
v_max=float(1.05**2) # unit: p.u.
I_max=LineInf[1:34,4]*1e6/Sbase # max current in distribution line, unit: p.u. I_max≈S_max in p.u. because U≈1 (p.u.)
#print(I_max)
#for i in range(30):
    #I_max[i]=1e6/Sbase*I_max[i]
#print(I_max)
M=1e8
# line investment cost denoted by
Cost=LineInf[1:34,6]*LineInf[1:34,3]
##print(Cost)
# make load (MVA) become P_load Q_load assuming a power factor n.
S=NodeInf[N_loads,3]
#print(S)
n=0.8  # power factor
P_load=S*n
Q_load=S*(math.sqrt(1-n**2))
# make impedance z become r and x assuming a rx rate n_rx
z=LineInf[1:34,3]*LineInf[1:34,5]/Zbase # line impedance, unit:p.u.
#print(z)

n_rx=1
r=z/math.sqrt(1+n_rx**2)
print(r[3])
x=r/n_rx
# calculate the maximum complex power in each distribution line
S_max=math.sqrt(v_max)*I_max


#图论部分
'''
G=graph(s,t);  % G=(V,E), G:graph V:vertex E:edge
In=myincidence(s,t);  % 节支关联矩阵，node-branch incidence matrix
% I=I(:,idxOut);   % make line index follow the order of s and t
Inn=In;
Inn(Inn>0)=0;    % Inn is the negative part of I
% Plot the Graph
figure;
p=plot(G,'Layout','force');
p.XData=NodeInf(:,2);
p.YData=NodeInf(:,3);
hold on;
labelnode(p,N_subs,{'13','14','15','16'});
highlight(p,N_loads,'NodeColor','y','Markersize',20,'NodeFontSize',20);
highlight(p,N_subs,'Marker','s','NodeColor','c','Markersize',30,'NodeFontSize',40);
highlight(p,s,t,'EdgeColor','k','LineStyle','-.','LineWidth',2,'EdgeFontSize',8);
text(p.XData, p.YData, p.NodeLabel,'HorizontalAlignment', 'center','FontSize', 15); % put nodes' label in right position.
p.NodeLabel={};
hold off;
'''
edges = pd.DataFrame()
#print(s)
#print(t)
edges['sources'] = s
edges['targets'] = t
G = nx.from_pandas_edgelist(edges,source='sources',target='targets')
nx.draw(G, with_labels=True,pos=None, arrows=True)
#plt.savefig("undirected_graph.png")
plt.show()
In=myincidence(s,t)
#print(In)
#print('bbbbbbbbbbbbbbbbbb')
Inn = np.zeros((N,L))

for i in range(len(In)):
    for j in range(len(In[0])):
        Inn[i,j]=In[i,j]
#Inn=In
# Inn[Inn>0]=0    # Inn is the negative part of I
for i in range(len(Inn)):
    for j in range(len(Inn[0])):
        if Inn[i][j]>0:
            Inn[i][j]=0
        else:
            Inn[i][j]=Inn[i][j]


#print(Inn)



#画图部分最后补
#print('bbbbbbbbbbbbbbbbbb')
In=myincidence(s,t)
#print(In)
'''
figure;
p=plot(G,'Layout','force');
p.XData=NodeInf(:,2);
p.YData=NodeInf(:,3);
hold on;
labelnode(p,N_subs,{'13','14','15','16'});
highlight(p,N_loads,'NodeColor','y','Markersize',20,'NodeFontSize',20);
highlight(p,N_subs,'Marker','s','NodeColor','c','Markersize',30,'NodeFontSize',40);
highlight(p,s,t,'EdgeColor','k','LineStyle','-.','LineWidth',2,'EdgeFontSize',8);
text(p.XData, p.YData, p.NodeLabel,'HorizontalAlignment', 'center','FontSize', 15); % put nodes' label in right position.
p.NodeLabel={};
hold off;
'''

#变量声明
'''
v_i=sdpvar(N,1, 'full'); % v=|U|^2=U·U*, "*" means conjugate
l_ij=sdpvar(L,1, 'full'); % l=|I|^2=I·I*
P_ij=sdpvar(L,1, 'full'); % P_ij,i
Q_ij=sdpvar(L,1, 'full'); % Q_ij,i
P_shed=sdpvar(length(N_loads),1, 'full'); % shedded load
Q_shed=sdpvar(length(N_loads),1, 'full'); % shedded load
g_subs_P=sdpvar(length(N_subs),1,'full'); % generation from substations
g_subs_Q=sdpvar(length(N_subs),1,'full'); % generation from substations
y_ij=binvar(L,1,'full'); % decision variable denoting whether this line should be invested
'''
model = Model('mip')


v_i = model.addMVar((N,1),vtype=GRB.CONTINUOUS,name = 'v_i')
l_ij= model.addMVar((L,1),vtype=GRB.CONTINUOUS,name ='l_ij')
P_ij= model.addMVar((L,1),vtype=GRB.CONTINUOUS,name ='P_ij')
Q_ij= model.addMVar((L,1),vtype=GRB.CONTINUOUS,name ='Q_ij')
P_shed= model.addMVar((len(N_loads), 1),vtype=GRB.CONTINUOUS,name ='P_shed')
Q_shed= model.addMVar((len(N_loads), 1),vtype=GRB.CONTINUOUS,name ='Q_shed')
g_subs_P= model.addMVar((len(N_subs), 1),vtype=GRB.CONTINUOUS,name ='g_subs_P')
g_subs_Q= model.addMVar((len(N_subs), 1),vtype=GRB.CONTINUOUS,name ='g_subs_Q')
y_ij= model.addMVar((L,1),vtype=GRB.BINARY,name ='y_ij')


#约束

# %% 1. Power balance
# % S_ij=s_i+\sum_(h:h→i)(S_hi-z_hi·l_hi) for any (i,j)∈E,
# % denoted by node-branch incidence matrix I
#Cons_S=[[In*P_ij-Inn*(r*l_ij)==np.transpose([-(P_load-P_shed),g_subs_P])],[In*Q_ij-Inn*(x*l_ij)==np.transpose([-(Q_load-Q_shed),g_subs_Q])],[P_shed>=0,Q_shed==P_shed*math.sqrt(1-n^2)/n]]
#model.addConstr(In.dot(P_ij)-Inn.dot(r*l_ij)==np.transpose([-(P_load-P_shed),g_subs_P]))
#model.addConstr(In.dot(Q_ij)-Inn.dot(x*l_ij)==np.transpose([-(Q_load-Q_shed),g_subs_Q]))
#model.addConstr(Q_shed==P_shed*math.sqrt(1-n**2)/n)
#model.addConstr(P_shed>=0)
#In 16*33,N=16,L=33,r=33*1
#model.addConstr(In@P_ij-Inn@(r*l_ij)==[[-(P_load-P_shed)],[g_subs_P]])
#约束1 功率约束
#J=[]
#for i in range(L):
    #j[i]=r[i]*l_ij[i]
#J=r*l_ij
#print(type(J)#)
#for i in range(len(N_loads)):
    #for j in range(1):
        #for l in range(L):
            #model.addConstr(sum(In[i,l]*P_ij[l])-sum(Inn[i,l]*r[l]*l_ij[l])==P_shed[i]-P_load[i],name='约束1.1'+str(i))
            #x1 = x1+In[i,l]*P_ij[l]
            #y1 = y1+Inn[i,l]*r[l]*l_ij[l] model.addConstr(sum(In[i,l]*P_ij[l])-sum(Inn[i,l]*r[l]*l_ij[l])==P_shed[i]-P_load[i],name='约束1.1'+str(i))
#for i in range(len(N_loads)):
    #model.addConstr(P_shed[i]==0)
print(Inn)
for i in range(len(N_loads)):
    for j in range(1):
        x1 = In[i,0]*P_ij[0]
        y1 = Inn[i,0]*r[0]*l_ij[0]
        for l in range(1,L):
            x1 = x1+In[i,l]*P_ij[l]
            y1 = y1+Inn[i,l]*r[l]*l_ij[l]
        model.addConstr(x1-y1==P_shed[i]-P_load[i],name='约束1.1'+str(i))
            #x1 = x1+In[i,l]*P_ij[l]
            #y1 = y1+Inn[i,l]*r[l]*l_ij[l] model.addConstr(sum(In[i,l]*P_ij[l])-sum(Inn[i,l]*r[l]*l_ij[l])==P_shed[i]-P_load[i],name='约束1.1'+str(i))
k1=0
print('bbbbbbbbbbbbbbbbbb')
print(In)
for i in range(len(N_loads),N):
    for j in range(1):
        x1=0
        y1=0
        for l in range(L):
            x1 = x1+In[i,l]*P_ij[l]
            y1 = y1+Inn[i,l]*r[l]*l_ij[l]
        model.addConstr(x1-y1==g_subs_P[k1],name='约束1.2'+str(i)) 
        k1=k1+1
k1=0        
#K=x*l_ij
for i in range(len(N_loads)):
    for j in range(1):
        x1=0
        y1=0
        for l in range(L):
            x1 = x1+In[i,l]*Q_ij[l]
            y1 = y1+Inn[i,l]*x[l]*l_ij[l]
        model.addConstr(x1-y1==Q_shed[i]-Q_load[i],name='约束1.3'+str(i))
#print(k1)
for i in range(len(N_loads),N):
    for j in range(1):
        x1=0
        y1=0
        for l in range(L):
            x1 = x1+In[i,l]*Q_ij[l]
            y1 = y1+Inn[i,l]*x[l]*l_ij[l]
        model.addConstr(x1-y1==g_subs_Q[k1],name='约束1.4'+str(i))
        k1=k1+1

for i in range(len(N_loads)):
        model.addConstr(Q_shed[i]==P_shed[i]*math.sqrt(1-n**2)/n,name='约束1.5'+str(i))

for i in range(len(N_loads)):
        model.addConstr(P_shed[i]>=0,name='约束1.6'+str(i))


#model.addConstr(In[i,j]*P_ij[i,1]-Inn[i,j]*t[i,1])
#model.addConstr(In@P_ij-Inn@(r*l_ij)==[[-(P_load-P_shed)],[g_subs_P]])
#model.addConstr(In@Q_ij-Inn@(x*l_ij)==[[-(Q_load-Q_shed)],[g_subs_Q]])
#model.addConstr(Q_shed==P_shed*math.sqrt(1-n**2)/n)
#model.addConstr(P_shed>=0)
# %% 2. Voltage Calculation
# % v_i-v_j=2Re(z_ij·S_ij*)+(r.^2+x.^2).*l_ij=2(r·P_ij,i+x·Q_ij,i)+(r.^2+x.^2).*l_ij
# % → |v_i-v_j-2(r·P_ij+x·Q_ij)-(r.^2+x.^2).*l_ij|≤(1-y_ij)*M
# Cons_S.append([v_i(N_subs)==v_max])
# Cons_S.append([abs(np.transpose(In)*v_i-2*r*P_ij-2*x*Q_ij+z**2.*l_ij)<=(1-y_ij)*M])#小心转置

#model.addConstr(v_i(N_subs)==v_max)
#model.addConstr(abs(np.transpose(In).dot(v_i)-2*r*P_ij-2*x*Q_ij+z*z*l_ij)<=(1-y_ij)*M)#1-y_ij)*M可能有问题
for i in range(len(N_subs)):
    model.addConstr(v_i[i+len(N_loads)]==v_max,name='约束2.1'+str(i))
k1=0
Iu=np.transpose(In)
#print(Iu)
for i in range(len(Iu)):
    for j in range(1):
        x1=0
        y1=0
        for l in range(N):
            x1 = x1+Iu[i,l]*v_i[l]
        y1 = 2*r[i]*P_ij[i]+2*x[i]*Q_ij[i]-z[i]*z[i]*l_ij[i]
        z1 = (1-y_ij[i])*M
        model.addConstr(x1-y1<=z1,name='约束2.2'+str(i)) 
        model.addConstr(y1-x1<=z1,name='约束2.3'+str(i)) 


# Cons=[Cons_S,Cons_V]
# %% 3. Voltage limits
# % v_min<=v<=v_max
#Cons_S.append([v_min<=v_i<=v_max])
#model.addConstr(v_min<=v_i<=v_max)
for i in range(N):
    model.addConstr(v_min<=v_i[i],name='约束2.4'+str(i))
    model.addConstr(v_i[i]<=v_max,name='约束2.5'+str(i))
# 4. l_ij<=l_ij_max (I_max.^2)
#Cons_S.append([0<=l_ij<=y_ij*I_max**2])
#odel.addConstr(0<=l_ij<=y_ij*(I_max*I_max))
for i in range(L):
    model.addConstr(0<=l_ij[i],name='约束2.6'+str(i))
    model.addConstr(l_ij[i]<=y_ij[i]*I_max[i]*I_max[i],name='约束2.7'+str(i))
# Cons=[Cons, Cons_l]
#5. l_ij<=S_ij^2/v_i, for any (i,j)
# Cons_SOC= []
# for l in range(L):
#     sl=s[l] # sl is the node index of starting node of line l
#     Cons_SOC.append(list([(l_ij(l)+v_i(sl))**2>=4*(P_ij(l)**2+Q_ij(l)**2)+(l_ij(l)-v_i(sl))**2]))
# Cons_S.extend(Cons_SOC)
# Cons = np.matrix(Cons_S)
# Cons = np.transpose(Cons)
# model.addConstr(Cons)
print(s)
for l in range(L):
     sl=s[l] # sl is the node index of starting node of line l
     x2 = (l_ij[l]+v_i[sl])*(l_ij[l]+v_i[sl])
     y2 = 4*P_ij[l]*P_ij[l]+4*Q_ij[l]*Q_ij[l]
     z2 = (l_ij[l]-v_i[sl])*(l_ij[l]-v_i[sl])
     model.addConstr(x2>=y2+z2,name='约束2.8'+str(i))
'''
Cons=[];
%% 1. Power balance
% S_ij=s_i+\sum_(h:h→i)(S_hi-z_hi·l_hi) for any (i,j)∈E, 
% denoted by node-branch incidence matrix I
Cons_S=[];
Cons_S=[In*P_ij-Inn*(r.*l_ij)==[-(P_load-P_shed);g_subs_P],In*Q_ij-Inn*(x.*l_ij)==[-(Q_load-Q_shed);g_subs_Q], P_shed>=0,Q_shed==P_shed*sqrt(1-n^2)/n];
Cons=[Cons,Cons_S];
%% 2. Voltage Calculation
% v_i-v_j=2Re(z_ij·S_ij*)+(r.^2+x.^2).*l_ij=2(r·P_ij,i+x·Q_ij,i)+(r.^2+x.^2).*l_ij
% → |v_i-v_j-2(r·P_ij+x·Q_ij)-(r.^2+x.^2).*l_ij|≤(1-y_ij)*M
Cons_V=[v_i(N_subs)==v_max];
Cons_V=[Cons_V,abs(In'*v_i-2*r.*P_ij-2*x.*Q_ij+z.^2.*l_ij)<=(1-y_ij)*M];
Cons=[Cons,Cons_V];
%% 3. Voltage limits
% v_min<=v<=v_max
Cons=[Cons,v_min<=v_i<=v_max];
%% 4. l_ij<=l_ij_max (I_max.^2)
Cons_l=[0<=l_ij<=y_ij.*I_max.^2];
Cons=[Cons, Cons_l];
%% 5. l_ij<=S_ij^2/v_i, for any (i,j)
Cons_SOC=[];
for l=1:L
    sl=s(l); % sl is the node index of starting node of line l
    Cons_SOC=[Cons_SOC,(l_ij(l)+v_i(sl)).^2>=4*(P_ij(l).^2+Q_ij(l).^2)+(l_ij(l)-v_i(sl)).^2];
end
Cons=[Cons, Cons_SOC];
'''
#目标约束
'''
Obj_inv=sum(Cost.*y_ij);
Obj_ope=M*sum(P_shed);
Obj=Obj_inv+Obj_ope;
'''
Obj_inv=0
Obj_ope=0
for i in range(L):
    Obj_inv=Obj_inv+Cost[i]*y_ij[i]
for i in range(len(N_loads)):
    Obj_ope=Obj_ope+M*P_shed[i]
#Obj_inv=sum(Cost*y_ij)
#Obj_ope=M*sum(P_shed)
Obj=Obj_inv+Obj_ope
model.setObjective(Obj,GRB.MINIMIZE)

#求解
#model.setParam('outPutFlag', 0)  # 不输出求解日志
model.optimize()
model.write("out.lp")
#model.feasRelaxS(0, True, False, True)

#model.computeIIS()
#model.write("o.ilp")

print('obj=', model.objVal)
for var in model.getVars():
   print(f"{var.varName}: {round(var.X, 3)}")
