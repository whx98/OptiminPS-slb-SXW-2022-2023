%% define incidence function about node-branch incidence matrix IN.
function IN =myincidence(I,J)
% Inputs: 对某条支路，I为起始节点，J为末节点
% Output: IN the incidence matrix IN(I(j),j)=1 and I(J(j),j)=-1 if line j is starting from I(j) to J(j)
MaxNode=max(max(I),max(J));
IN=zeros(MaxNode,length(I));
for j=1:length(I)
    IN(I(j),j)=1; % 流出为正
    IN(J(j),j)=-1; % 流入为负
end
end
