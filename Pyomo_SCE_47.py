import docplex.mp.model as cpx
#opt_model = cpx.Model(name="MIP Model")
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from gurobipy import *
import pyomo.environ as pyo
from pyomo.core import ConstraintList
import pyomo.kernel as pk

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
I_max =560.98/Ibase  # 12MW? 560.98/Ibase
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
Pr_p=0.89
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
w = 1j
z = r + x * w
z_c = r - x * w

#建立具体模型
#声明变量
model = pyo.ConcreteModel()
model.u = pyo.Var(range(N))
model.v = pyo.Var(range(N))
model.P_ij = pyo.Var(range(L))
model.Q_ij = pyo.Var(range(L))
model.l_ij = pyo.Var(range(L))
model.Pi = pyo.Var(range(N))
model.Qi = pyo.Var(range(N))


#Constraint 1
Eta = 2.8
Pi_max = np.zeros((N, 1))
Pi_max = Pi_max.tolist()
Qi_max = np.zeros((N, 1))
Qi_max = Qi_max.tolist()
zeros = np.zeros((N, 1))
zeros = zeros.tolist()
model.cons1 = pyo.ConstraintList()
# model.cons2 = pyo.ConstraintList()
# model.cons3 = pyo.ConstraintList()
# model.cons4 = pyo.ConstraintList()
# model.cons5 = pyo.ConstraintList()
# model.cons6 = pyo.ConstraintList()
# model.cons7 = pyo.ConstraintList()
# model.cons8 = pyo.ConstraintList()
# model.cons9 = pyo.ConstraintList()
# model.cons10 = pyo.ConstraintList()
# model.cons11 = pyo.ConstraintList()
# model.cons12 = pyo.ConstraintList()

for i in range(N):
    if judge[i] == 1:
        model.cons1.add(expr=model.Pi[i] == 0)
        Qi_max[i] = So[i] * Eta
        model.cons1.add(expr=model.Qi[i] >= 0)
        model.cons1.add(expr=model.Qi[i] <= Qi_max[i])
    elif judge[i] == 2:
        Pi_max[i] = -So[i] * 0.9
        Qi_max[i] = -So[i] * math.sqrt(1 - 0.9 ** 2)
        model.cons1.add(expr=model.Pi[i] == Pi_max[i])
        model.cons1.add(expr=model.Qi[i] == Qi_max[i])
    elif judge[i] == 0:
        Pi_max[i] = So[i] * Eta
        model.cons1.add(expr=model.Pi[i] >= 0)
        model.cons1.add(expr=model.Pi[i] <= Pi_max[i])
        model.cons1.add(expr=model.Qi[i] == 0)
    elif judge[i] == 3:
        model.cons1.add(expr=model.Pi[i] >= -So[i])
        model.cons1.add(expr=model.Pi[i] <= So[i])
        model.cons1.add(expr=model.Qi[i] >= -So[i] * 0.5)
        model.cons1.add(expr=model.Qi[i] <= So[i] * 0.5)


#cosnstraint 2
model.cons2 = pyo.ConstraintList()
for i in range(L):
    model.cons2.add(expr=model.l_ij[i] <= I_max ** 2)
    model.cons2.add(expr=model.l_ij[i] >= 0)
for i in range(1, N):
    model.cons2.add(expr=model.v[i] >= v_min)


#cosnstraint 3
model.cons3 = pyo.ConstraintList()

model.cons3.add(expr=model.v[0] == v_max)
Iu = np.transpose(In)
Iu = Iu.tolist()
In = In.tolist()
Inn = Inn.tolist()
r = r.tolist()
x = x.tolist()
#print(In[1][2])
for i in range(L):
    for j in range(1):
        x1 = 0
        y1 = 0
        for l in range(N):
            x1 = x1 + Iu[i][l] * model.v[l]
        y1 = 2 * r[i] * model.P_ij[i] + 2 * x[i] * model.Q_ij[i] - (r[i] * r[i] + x[i] * x[i]) * model.l_ij[i]
        model.cons3.add(expr=x1 - y1 == 0)
#model.Constraint1 = pyo.Constraint(expr=3 * model.x1 + 4 * model.x2 >= 1)
#cosnstraint 4
model.cons4 = pyo.ConstraintList()
for l in range(L):
    sl = I[l]  # sl is the node index of starting node of line l
    x2 = (model.l_ij[l] + model.v[sl]) * (model.l_ij[l] + model.v[sl])
    y2 = 4 * model.P_ij[l] * model.P_ij[l] + 4 * model.Q_ij[l] * model.Q_ij[l]
    z2 = (model.l_ij[l] - model.v[sl]) * (model.l_ij[l] - model.v[sl])
    model.cons4.add(expr=x2 >= y2 + z2)

#cosnstraint 5
model.cons5 = pyo.ConstraintList()
for i in range(N):
    for j in range(1):
        x1 = 0
        y1 = 0
        for l in range(L):
            x1 = x1 + In[i][l] * model.P_ij[l]
            y1 = y1 + Inn[i][l] * r[l] * model.l_ij[l]
        model.cons4.add(expr=x1 - y1 == model.Pi[i])

for i in range(N):
    for j in range(1):
        x1 = 0
        y1 = 0
        for l in range(L):
            x1 = x1 + In[i][l] * model.Q_ij[l]
            y1 = y1 + Inn[i][l] * x[l] * model.l_ij[l]
        model.cons4.add(expr=x1 - y1 == model.Qi[i])

#cosnstraint 6
'''
model.v = pyo.Var(range(N), domain=pyo.NonNegativeReals)
model.P_ij = pyo.Var(range(L), domain=pyo.NonNegativeReals)
'''
model.cons6 = pyo.ConstraintList()
if WithSvolt == 1:
    model.P_ij_s = pyo.Var(range(L))
    model.Q_ij_s = pyo.Var(range(L))
    model.v_s = pyo.Var(range(N))

    for i in range(1, N):
        model.cons6.add(expr=v_max >= model.v_s[i])

    model.cons6.add(expr=v_max == model.v_s[1])


    for i in range(L):
        for j in range(1):
            x1 = 0
            y1 = 0
            for l in range(N):
                x1 = x1 + Iu[i][l] * model.v_s[l]
            y1 = 2 * r[i] * model.P_ij_s[i] + 2 * x[i] * model.Q_ij_s[i]
            model.cons6.add(expr=x1 == y1)

    for i in range(1, N):
        for j in range(1):
            x1 = 0
            y1 = 0
            for l in range(L):
                x1 = x1 + In[i][l] * model.P_ij_s[l]
            y1 = model.Pi[i]
            model.cons6.add(expr=x1 == y1)

    for i in range(1, N):
        for j in range(1):
            x1 = 0
            y1 = 0
            for l in range(L):
                x1 = x1 + In[i][l] * model.Q_ij_s[l]
            y1 = model.Qi[i]
            model.cons6.add(expr=x1 == y1)
#

#OBJ
def ObjRule(model):
    obj=0
    index =0
    for i in range(N):
        if judge[i] == 0:
            obj = obj + Pr_pv[index] * model.Pi[i] * Pr_p
            index = index + 1

    obj = obj + Pr_sub * model.Pi[0] * Pr_p
    return obj

model.Obj = pyo.Objective(expr=ObjRule(model), sense=pyo.minimize)

model.pprint()
pyo.SolverFactory('gurobi').solve(model)
print('\nObj = ', model.Obj())

print('\nDecision Variables')

'''
# #建立具体模型
# model = pyo.ConcreteModel()
# # 声明决策变量
# model.x1 = pyo.Var(domain=pyo.NonNegativeReals)  # NonNegativeReals代表非负实数
# model.x2 = pyo.Var(domain=pyo.NonNegativeReals)
# # x是变量组成的向量，维度是2，非负实数
# # model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals))
# # 定义目标函数
# def ObjRule(model):
#     return 2 * model.x1 + 3 * model.x2
# 
# 
# # 生成目标函数
# model.Obj = pyo.Objective(expr=ObjRule(model), sense=pyo.minimize)
# 
# 
# # 定义约束条件
# model.Constraint1 = pyo.Constraint(expr=3 * model.x1 + 4 * model.x2 >= 1)
# 
# # 显示模型信息
# # model.pprint()
# # 求解
# pyo.SolverFactory('cplex', solver_io="python").solve(model).write()
# 
# print('\nObj = ', model.Obj())
# 
# print('\nDecision Variables')
# print('x1 = ', model.x1())
# print('x2 = ', model.x2())
# 
# 
# print('\nConstraints')
# print('Constraint1  = ', model.Constraint1())
# 
# # ----------------------------------------------------------
# #   Solution Information
# # ----------------------------------------------------------
# # Solution:
# # - number of solutions: 0
# #   number of solutions displayed: 0
# 
# # Obj =  0.6666666666666666
# 
# # Decision Variables
# # x1 =  0.3333333333333333
# # x2 =  0.0
# 
# # Constraints
# # Constraint1  =  1.0
'''