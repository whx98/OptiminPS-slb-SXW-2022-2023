function [Cons,Pimax,Qimax] = ConsOfSOCP(Pij,Qij,v,lij,Pi,Qi,s,Imax,vmax,vmin,LoadType,headNode,endNode,NodeBranchMat,LineNum,r,x,PenetrateRate,Eta,N)
[Part1,Pimax,Qimax] = ConsLoad(Pi,Qi,s,LoadType,PenetrateRate,Eta,N);
Part2 = ConsPowerFlow(Pij,Qij,lij,Pi,Qi,NodeBranchMat,r,x);
Part3 = ConsSOC(Pij,Qij,lij,v,LineNum,headNode);
Part4 = ConsVoltAndCurrentLimit(Pij,Qij,v,lij,Imax,vmax,vmin,NodeBranchMat,r,x);
Cons = [Part1,Part2,Part3,Part4];
end
% 第一部分：节点电压方程相角松弛的线性化约束
function [Part1,Pimax,Qimax] = ConsLoad(Pi,Qi,s,LoadType,PenetrateRate,Eta,N)
Part1 = [];
Pimax=zeros(N,1);
Qimax=zeros(N,1);
for i=1:N
    switch LoadType(i)
        case 0     %太阳能板
            Pimax(i)=s(i)*PenetrateRate;
            Part1=[Part1,0<=Pi(i)<=Pimax(i)];
            Part1=[Part1,0==Qi(i)];
        case 1     % 电容器
            Part1=[Part1,Pi(i)==0];
            Qimax(i)=s(i)*PenetrateRate;
            Part1=[Part1,0<=Qi(i)<=Qimax(i)];
        case 2     % 可控负载
            Pimax(i)=-s(i)*Eta;
            Qimax(i)=-s(i)*sqrt(1-Eta^2);
            Part1=[Part1,Pi(i)==Pimax(i)];
            Part1=[Part1,Qi(i)==Qimax(i)];
        otherwise  %松弛节点
            Part1=[Part1,-s(i)<=Pi(i)<=s(i)];
            Part1=[Part1,-s(i)*0.5<=Qi(i)<=s(i)*0.5];
    end
end
end

% 第二部分：节点潮流约束
function Part2 = ConsPowerFlow(Pij,Qij,lij,Pi,Qi,NodeBranchMat,r,x)
StartNode = NodeBranchMat;
StartNode(StartNode>0) = 0;
Part2 =[NodeBranchMat*Pij - StartNode*(r.*lij) == Pi, NodeBranchMat*Qij - StartNode*(x.*lij) == Qi];
end

% 第三部分：SOC约束
function Part3 = ConsSOC(Pij,Qij,lij,v,LineNum,headNode)
Part3 = [];
for k=1:LineNum
    i=headNode(k);% Starting node of line k
    Part3=[Part3,(lij(k) + v(i)).^2 >= 4*Pij(k).^2 + 4*Qij(k).^2 + (lij(k) - v(i)).^2];
end
end

% 第四部分：电压、电流约束
function Part4 = ConsVoltAndCurrentLimit(Pij,Qij,v,lij,Imax,vmax,vmin,NodeBranchMat,r,x)
ConsVoltEq = [NodeBranchMat'*v == 2*r.*Pij+2*x.*Qij-(r.^2+x.^2).*lij];
ConsVoltIneq = [v(1)==vmax,v(2:end)>=vmin];
ConsCur = [Imax^2>=lij>=0];
Part4 = [ConsVoltEq,ConsVoltIneq,ConsCur];
end